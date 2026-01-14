/*-------------------------------------------------------------------------
 *
 * driver.go
 *    Browser driver using chromedp for headless Chrome automation
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/browser/driver.go
 *
 *-------------------------------------------------------------------------
 */

package browser

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/chromedp/cdproto/cdp"
	"github.com/chromedp/cdproto/network"
	"github.com/chromedp/cdproto/page"
	"github.com/chromedp/chromedp"
)

/* Driver manages browser contexts and sessions */
type Driver struct {
	mu             sync.RWMutex
	contexts       map[string]*BrowserContext
	allocCtx       context.Context
	allocCancel    context.CancelFunc
	defaultTimeout time.Duration
	config         *BrowserConfig
}

/* BrowserContext represents a browser session */
type BrowserContext struct {
	Ctx        context.Context
	Cancel     context.CancelFunc
	SessionID  string
	LastURL    string
	LastTitle  string
	UserAgent  string
	Viewport   Viewport
	Cookies    []*network.Cookie
	CreatedAt  time.Time
	LastAccess time.Time
}

/* Viewport represents browser viewport dimensions */
type Viewport struct {
	Width  int
	Height int
}

/* NewDriver creates a new browser driver with default configuration */
func NewDriver() (*Driver, error) {
	return NewDriverWithConfig(DefaultConfig())
}

/* NewDriverWithConfig creates a new browser driver with custom configuration */
func NewDriverWithConfig(config *BrowserConfig) (*Driver, error) {
	if config == nil {
		config = DefaultConfig()
	}
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid browser configuration: %w", err)
	}

	opts := append(chromedp.DefaultExecAllocatorOptions[:],
		chromedp.Flag("headless", true),
		chromedp.Flag("disable-gpu", !config.EnableGPU),
		chromedp.Flag("no-sandbox", !config.SandboxMode),
		chromedp.Flag("disable-dev-shm-usage", true),
		chromedp.Flag("disable-blink-features", "AutomationControlled"),
		chromedp.UserAgent(config.DefaultUserAgent),
	)

	if config.DisableWebSec {
		opts = append(opts, chromedp.Flag("disable-web-security", true))
	}
	if config.IgnoreCertErrors {
		opts = append(opts, chromedp.Flag("ignore-certificate-errors", true))
	}

	allocCtx, allocCancel := chromedp.NewExecAllocator(context.Background(), opts...)

	return &Driver{
		contexts:       make(map[string]*BrowserContext),
		allocCtx:       allocCtx,
		allocCancel:    allocCancel,
		defaultTimeout: config.DefaultTimeout,
		config:         config,
	}, nil
}

/* GetOrCreateContext gets or creates a browser context for a session */
func (d *Driver) GetOrCreateContext(sessionID string, viewport Viewport, userAgent string) (*BrowserContext, error) {
	d.mu.Lock()
	defer d.mu.Unlock()

	/* Check if config is initialized */
	if d.config == nil {
		return nil, fmt.Errorf("browser driver configuration not initialized")
	}

	/* Check max sessions limit */
	if len(d.contexts) >= d.config.MaxSessions {
		return nil, fmt.Errorf("maximum number of browser sessions reached: %d/%d", len(d.contexts), d.config.MaxSessions)
	}

	if ctx, exists := d.contexts[sessionID]; exists {
		ctx.LastAccess = time.Now()
		return ctx, nil
	}

	opts := []chromedp.ContextOption{}

	browserCtx, cancel := chromedp.NewContext(d.allocCtx, opts...)

	ctx := &BrowserContext{
		Ctx:        browserCtx,
		Cancel:     cancel,
		SessionID:  sessionID,
		Viewport:   viewport,
		UserAgent:  userAgent,
		CreatedAt:  time.Now(),
		LastAccess: time.Now(),
	}

	d.contexts[sessionID] = ctx

	/* Initialize browser and set viewport */
	actions := []chromedp.Action{
		chromedp.EmulateViewport(int64(viewport.Width), int64(viewport.Height)),
	}

	if err := chromedp.Run(browserCtx, chromedp.Tasks(actions)); err != nil {
		cancel()
		delete(d.contexts, sessionID)
		return nil, fmt.Errorf("browser context initialization failed (session_id='%s'): %w. This may indicate chromedp/chrome is not available", sessionID, err)
	}

	/* Test browser is responsive */
	var testResult string
	if err := chromedp.Run(browserCtx, chromedp.Evaluate(`"test"`, &testResult)); err != nil {
		cancel()
		delete(d.contexts, sessionID)
		return nil, fmt.Errorf("browser context health check failed (session_id='%s'): %w", sessionID, err)
	}

	return ctx, nil
}

/* GetContext gets an existing browser context */
func (d *Driver) GetContext(sessionID string) (*BrowserContext, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	ctx, exists := d.contexts[sessionID]
	if !exists {
		return nil, fmt.Errorf("browser driver context not found: session_id='%s'", sessionID)
	}

	ctx.LastAccess = time.Now()
	return ctx, nil
}

/* Navigate navigates to a URL */
func (d *Driver) Navigate(ctx *BrowserContext, url string, waitUntil string) (string, string, error) {
	timeout := d.config.NavigateTimeout
	if waitUntil == "networkidle" || waitUntil == "networkidle0" || waitUntil == "networkidle2" {
		timeout = d.config.NavigateTimeout + 30*time.Second
	}

	navCtx, cancel := context.WithTimeout(ctx.Ctx, timeout)
	defer cancel()

	var title, finalURL string

	actions := []chromedp.Action{
		chromedp.Navigate(url),
	}

	switch waitUntil {
	case "load":
		actions = append(actions, chromedp.WaitVisible("body", chromedp.ByQuery))
	case "domcontentloaded":
		actions = append(actions, chromedp.ActionFunc(func(ctx context.Context) error {
			_, err := page.AddScriptToEvaluateOnNewDocument("").Do(ctx)
			return err
		}))
	case "networkidle", "networkidle0":
		actions = append(actions, chromedp.WaitReady("body", chromedp.ByQuery))
		actions = append(actions, chromedp.Sleep(500*time.Millisecond))
	case "networkidle2":
		actions = append(actions, chromedp.WaitReady("body", chromedp.ByQuery))
		actions = append(actions, chromedp.Sleep(1*time.Second))
	default:
		actions = append(actions, chromedp.WaitVisible("body", chromedp.ByQuery))
	}

	actions = append(actions,
		chromedp.Title(&title),
		chromedp.Location(&finalURL),
	)

	if err := chromedp.Run(navCtx, chromedp.Tasks(actions)); err != nil {
		return "", "", fmt.Errorf("browser driver navigation failed: url='%s', wait_until='%s', error=%w", url, waitUntil, err)
	}

	ctx.LastURL = finalURL
	ctx.LastTitle = title

	return title, finalURL, nil
}

/* Click clicks on an element */
func (d *Driver) Click(ctx *BrowserContext, selector string, waitForNavigation bool) error {
	actionCtx, cancel := context.WithTimeout(ctx.Ctx, d.defaultTimeout)
	defer cancel()

	actions := []chromedp.Action{
		chromedp.WaitVisible(selector, chromedp.ByQuery),
		chromedp.ScrollIntoView(selector, chromedp.ByQuery),
		chromedp.Click(selector, chromedp.ByQuery),
	}

	if waitForNavigation {
		actions = append(actions, chromedp.WaitReady("body", chromedp.ByQuery))
	}

	if err := chromedp.Run(actionCtx, chromedp.Tasks(actions)); err != nil {
		return fmt.Errorf("browser driver click failed: selector='%s', wait_for_navigation=%v, error=%w", selector, waitForNavigation, err)
	}

	return nil
}

/* Type types text into an element */
func (d *Driver) Type(ctx *BrowserContext, selector string, text string, clear bool) error {
	actionCtx, cancel := context.WithTimeout(ctx.Ctx, d.defaultTimeout)
	defer cancel()

	actions := []chromedp.Action{
		chromedp.WaitVisible(selector, chromedp.ByQuery),
		chromedp.ScrollIntoView(selector, chromedp.ByQuery),
	}

	if clear {
		actions = append(actions, chromedp.Clear(selector, chromedp.ByQuery))
	}

	actions = append(actions, chromedp.SendKeys(selector, text, chromedp.ByQuery))

	if err := chromedp.Run(actionCtx, chromedp.Tasks(actions)); err != nil {
		return fmt.Errorf("browser driver type failed: selector='%s', text_length=%d, clear=%v, error=%w", selector, len(text), clear, err)
	}

	return nil
}

/* Extract extracts data from elements */
func (d *Driver) Extract(ctx *BrowserContext, selector string, extractType string) (interface{}, error) {
	actionCtx, cancel := context.WithTimeout(ctx.Ctx, d.defaultTimeout)
	defer cancel()

	var result interface{}

	switch extractType {
	case "text":
		var text string
		if err := chromedp.Run(actionCtx,
			chromedp.WaitVisible(selector, chromedp.ByQuery),
			chromedp.Text(selector, &text, chromedp.ByQuery),
		); err != nil {
			return nil, fmt.Errorf("browser driver extract text failed: selector='%s', error=%w", selector, err)
		}
		result = text

	case "html":
		var html string
		if err := chromedp.Run(actionCtx,
			chromedp.WaitVisible(selector, chromedp.ByQuery),
			chromedp.OuterHTML(selector, &html, chromedp.ByQuery),
		); err != nil {
			return nil, fmt.Errorf("browser driver extract html failed: selector='%s', error=%w", selector, err)
		}
		result = html

	case "attribute":
		var attr string
		attrName := "href"
		if err := chromedp.Run(actionCtx,
			chromedp.WaitVisible(selector, chromedp.ByQuery),
			chromedp.AttributeValue(selector, attrName, &attr, nil, chromedp.ByQuery),
		); err != nil {
			return nil, fmt.Errorf("browser driver extract attribute failed: selector='%s', error=%w", selector, err)
		}
		result = attr

	case "all":
		var nodes []*cdp.Node
		if err := chromedp.Run(actionCtx,
			chromedp.WaitVisible(selector, chromedp.ByQuery),
			chromedp.Nodes(selector, &nodes, chromedp.ByQuery),
		); err != nil {
			return nil, fmt.Errorf("browser driver extract all failed: selector='%s', error=%w", selector, err)
		}

		results := make([]map[string]interface{}, 0, len(nodes))
		for _, node := range nodes {
			item := map[string]interface{}{
				"text":      node.NodeValue,
				"node_name": node.NodeName,
				"node_type": node.NodeType,
			}
			if node.Attributes != nil {
				attrs := make(map[string]string)
				for i := 0; i < len(node.Attributes); i += 2 {
					if i+1 < len(node.Attributes) {
						attrs[node.Attributes[i]] = node.Attributes[i+1]
					}
				}
				item["attributes"] = attrs
			}
			results = append(results, item)
		}
		result = results

	default:
		return nil, fmt.Errorf("browser driver extract failed: extract_type='%s', validation_error='unknown extract type. valid types: text, html, attribute, all'", extractType)
	}

	return result, nil
}

/* Screenshot takes a screenshot */
func (d *Driver) Screenshot(ctx *BrowserContext, fullPage bool) ([]byte, error) {
	actionCtx, cancel := context.WithTimeout(ctx.Ctx, d.defaultTimeout)
	defer cancel()

	var buf []byte

	if fullPage {
		if err := chromedp.Run(actionCtx, chromedp.FullScreenshot(&buf, 90)); err != nil {
			return nil, fmt.Errorf("browser driver full screenshot failed: error=%w", err)
		}
	} else {
		if err := chromedp.Run(actionCtx, chromedp.CaptureScreenshot(&buf)); err != nil {
			return nil, fmt.Errorf("browser driver screenshot failed: error=%w", err)
		}
	}

	return buf, nil
}

/* ExecuteScript executes JavaScript */
func (d *Driver) ExecuteScript(ctx *BrowserContext, script string) (interface{}, error) {
	actionCtx, cancel := context.WithTimeout(ctx.Ctx, d.defaultTimeout)
	defer cancel()

	var result interface{}

	if err := chromedp.Run(actionCtx,
		chromedp.Evaluate(script, &result),
	); err != nil {
		return nil, fmt.Errorf("browser driver execute script failed: script_length=%d, error=%w", len(script), err)
	}

	return result, nil
}

/* GetCookies gets cookies for a URL */
func (d *Driver) GetCookies(ctx *BrowserContext, url string) ([]*network.Cookie, error) {
	actionCtx, cancel := context.WithTimeout(ctx.Ctx, 10*time.Second)
	defer cancel()

	var cookies []*network.Cookie

	if err := chromedp.Run(actionCtx,
		chromedp.ActionFunc(func(c context.Context) error {
			var err error
			cookies, err = network.GetCookies().WithURLs([]string{url}).Do(c)
			return err
		}),
	); err != nil {
		return nil, fmt.Errorf("browser driver get cookies failed: url='%s', error=%w", url, err)
	}

	ctx.Cookies = cookies
	return cookies, nil
}

/* SetCookies sets cookies */
func (d *Driver) SetCookies(ctx *BrowserContext, cookies []map[string]interface{}) error {
	actionCtx, cancel := context.WithTimeout(ctx.Ctx, 10*time.Second)
	defer cancel()

	networkCookies := make([]*network.CookieParam, 0, len(cookies))

	for _, cookie := range cookies {
		name, _ := cookie["name"].(string)
		value, _ := cookie["value"].(string)
		domain, _ := cookie["domain"].(string)
		path, _ := cookie["path"].(string)
		if path == "" {
			path = "/"
		}

		networkCookie := &network.CookieParam{
			Name:   name,
			Value:  value,
			Domain: domain,
			Path:   path,
		}

		if httpOnly, ok := cookie["httpOnly"].(bool); ok {
			networkCookie.HTTPOnly = httpOnly
		}
		if secure, ok := cookie["secure"].(bool); ok {
			networkCookie.Secure = secure
		}
		if sameSite, ok := cookie["sameSite"].(string); ok {
			switch strings.ToLower(sameSite) {
			case "strict":
				networkCookie.SameSite = network.CookieSameSiteStrict
			case "lax":
				networkCookie.SameSite = network.CookieSameSiteLax
			case "none":
				networkCookie.SameSite = network.CookieSameSiteNone
			}
		}

		networkCookies = append(networkCookies, networkCookie)
	}

	if err := chromedp.Run(actionCtx,
		chromedp.ActionFunc(func(c context.Context) error {
			return network.SetCookies(networkCookies).Do(c)
		}),
	); err != nil {
		return fmt.Errorf("browser driver set cookies failed: cookies_count=%d, error=%w", len(cookies), err)
	}

	return nil
}

/* WaitForElement waits for an element to appear */
func (d *Driver) WaitForElement(ctx *BrowserContext, selector string, timeout time.Duration) error {
	actionCtx, cancel := context.WithTimeout(ctx.Ctx, timeout)
	defer cancel()

	if err := chromedp.Run(actionCtx,
		chromedp.WaitVisible(selector, chromedp.ByQuery),
	); err != nil {
		return fmt.Errorf("browser driver wait for element failed: selector='%s', timeout=%v, error=%w", selector, timeout, err)
	}

	return nil
}

/* Scrape scrapes multiple elements */
func (d *Driver) Scrape(ctx *BrowserContext, selectors map[string]string) (map[string]interface{}, error) {
	actionCtx, cancel := context.WithTimeout(ctx.Ctx, d.defaultTimeout)
	defer cancel()

	results := make(map[string]interface{})

	for key, selector := range selectors {
		var text string
		if err := chromedp.Run(actionCtx,
			chromedp.WaitVisible(selector, chromedp.ByQuery),
			chromedp.Text(selector, &text, chromedp.ByQuery),
		); err != nil {
			results[key] = nil
			continue
		}
		results[key] = text
	}

	return results, nil
}

/* CloseContext closes a browser context */
func (d *Driver) CloseContext(sessionID string) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	ctx, exists := d.contexts[sessionID]
	if !exists {
		return nil
	}

	ctx.Cancel()
	delete(d.contexts, sessionID)

	return nil
}

/* CloseAll closes all browser contexts */
func (d *Driver) CloseAll() {
	d.mu.Lock()
	defer d.mu.Unlock()

	for _, ctx := range d.contexts {
		ctx.Cancel()
	}

	d.contexts = make(map[string]*BrowserContext)
	d.allocCancel()
}

/* GetScreenshotBase64 returns screenshot as base64 string */
func GetScreenshotBase64(data []byte) string {
	return base64.StdEncoding.EncodeToString(data)
}

/* CookiesToJSON converts cookies to JSON */
func CookiesToJSON(cookies []*network.Cookie) ([]map[string]interface{}, error) {
	result := make([]map[string]interface{}, 0, len(cookies))

	for _, cookie := range cookies {
		item := map[string]interface{}{
			"name":     cookie.Name,
			"value":    cookie.Value,
			"domain":   cookie.Domain,
			"path":     cookie.Path,
			"httpOnly": cookie.HTTPOnly,
			"secure":   cookie.Secure,
			"sameSite": cookie.SameSite.String(),
		}

		if cookie.Expires > 0 {
			item["expires"] = cookie.Expires
		}

		result = append(result, item)
	}

	return result, nil
}

/* JSONToCookies converts JSON to cookie maps */
func JSONToCookies(data []byte) ([]map[string]interface{}, error) {
	var cookies []map[string]interface{}
	if err := json.Unmarshal(data, &cookies); err != nil {
		return nil, fmt.Errorf("json to cookies conversion failed: error=%w", err)
	}
	return cookies, nil
}
