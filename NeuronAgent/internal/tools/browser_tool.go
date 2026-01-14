/*-------------------------------------------------------------------------
 *
 * browser_tool.go
 *    Web browser automation tool with DOM interaction
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/tools/browser_tool.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/browser"
	"github.com/neurondb/NeuronAgent/internal/db"
)

type BrowserTool struct {
	allowedURLs map[string]bool
	timeout     time.Duration
	maxPages    int
	driver      *browser.Driver
	queries     *db.Queries
	db          *db.DB
}

func NewBrowserTool() *BrowserTool {
	driver, err := browser.NewDriver()
	if err != nil {
		/* If driver initialization fails, create tool without driver (will fail on use with error message) */
		return &BrowserTool{
			allowedURLs: make(map[string]bool),
			timeout:     60 * time.Second,
			maxPages:    10,
			driver:      nil,
		}
	}

	return &BrowserTool{
		allowedURLs: make(map[string]bool),
		timeout:     60 * time.Second,
		maxPages:    10,
		driver:      driver,
	}
}

/* SetDB sets database connection for session management */
func (t *BrowserTool) SetDB(queries *db.Queries, database *db.DB) {
	t.queries = queries
	t.db = database
}

/* GetDriver returns the browser driver (for cleanup) */
func (t *BrowserTool) GetDriver() *browser.Driver {
	return t.driver
}

/* SetDriver sets the browser driver (for external initialization) */
func (t *BrowserTool) SetDriver(driver *browser.Driver) {
	t.driver = driver
}

/* Cleanup shuts down all browser contexts */
func (t *BrowserTool) Cleanup() {
	if t.driver != nil {
		t.driver.CloseAll()
	}
}

func (t *BrowserTool) Execute(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	if t.driver == nil {
		return "", fmt.Errorf("browser tool execution failed: tool_name='%s', handler_type='browser', error='browser driver not initialized'", tool.Name)
	}

	action, ok := args["action"].(string)
	if !ok {
		argKeys := make([]string, 0, len(args))
		for k := range args {
			argKeys = append(argKeys, k)
		}
		return "", fmt.Errorf("browser tool execution failed: tool_name='%s', handler_type='browser', args_count=%d, arg_keys=[%v], validation_error='action parameter is required and must be a string'",
			tool.Name, len(args), argKeys)
	}

	ctx, cancel := context.WithTimeout(ctx, t.timeout)
	defer cancel()

	switch action {
	case "navigate":
		return t.navigate(ctx, tool, args)
	case "click":
		return t.click(ctx, tool, args)
	case "type":
		return t.typeText(ctx, tool, args)
	case "extract":
		return t.extract(ctx, tool, args)
	case "screenshot":
		return t.screenshot(ctx, tool, args)
	case "execute_script":
		return t.executeScript(ctx, tool, args)
	case "get_cookies":
		return t.getCookies(ctx, tool, args)
	case "set_cookies":
		return t.setCookies(ctx, tool, args)
	case "scrape":
		return t.scrape(ctx, tool, args)
	case "wait_for_element":
		return t.waitForElement(ctx, tool, args)
	default:
		return "", fmt.Errorf("browser tool execution failed: tool_name='%s', handler_type='browser', action='%s', validation_error='unknown action. valid actions: navigate, click, type, extract, screenshot, execute_script, get_cookies, set_cookies, scrape, wait_for_element'",
			tool.Name, action)
	}
}

func (t *BrowserTool) navigate(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	/* Validate driver */
	if t.driver == nil {
		return "", fmt.Errorf("browser driver not initialized")
	}

	url, ok := args["url"].(string)
	if !ok {
		return "", fmt.Errorf("browser tool navigate failed: tool_name='%s', handler_type='browser', action='navigate', validation_error='url parameter is required and must be a string'",
			tool.Name)
	}

	if !t.isURLAllowed(url) {
		return "", fmt.Errorf("browser tool navigate failed: tool_name='%s', handler_type='browser', action='navigate', url='%s', validation_error='URL not in allowlist'",
			tool.Name, url)
	}

	sessionID, _ := args["session_id"].(string)
	if sessionID == "" {
		sessionID = uuid.New().String()
	}

	waitUntil := "load"
	if w, ok := args["wait_until"].(string); ok {
		waitUntil = w
	}

	viewport := browser.Viewport{Width: 1920, Height: 1080}
	if vw, ok := args["viewport_width"].(float64); ok {
		viewport.Width = int(vw)
	}
	if vh, ok := args["viewport_height"].(float64); ok {
		viewport.Height = int(vh)
	}

	userAgent := ""
	if ua, ok := args["user_agent"].(string); ok {
		userAgent = ua
	}

	/* Get or create browser context */
	browserCtx, err := t.driver.GetOrCreateContext(sessionID, viewport, userAgent)
	if err != nil {
		return "", fmt.Errorf("browser tool navigate failed: tool_name='%s', handler_type='browser', action='navigate', url='%s', session_id='%s', error=%w",
			tool.Name, url, sessionID, err)
	}

	/* Save session to database if DB is available */
	if t.queries != nil && t.db != nil {
		t.saveBrowserSession(ctx, sessionID, browserCtx, url)
	}

	/* Navigate */
	title, finalURL, err := t.driver.Navigate(browserCtx, url, waitUntil)
	if err != nil {
		return "", fmt.Errorf("browser tool navigate failed: tool_name='%s', handler_type='browser', action='navigate', url='%s', wait_until='%s', error=%w",
			tool.Name, url, waitUntil, err)
	}

	result := map[string]interface{}{
		"action":     "navigate",
		"url":        url,
		"url_final":  finalURL,
		"session_id": sessionID,
		"wait_until": waitUntil,
		"title":      title,
		"status":     "success",
	}

	jsonResult, err := json.Marshal(result)
	if err != nil {
		return "", fmt.Errorf("browser tool navigate result marshaling failed: tool_name='%s', handler_type='browser', action='navigate', url='%s', wait_until='%s', error=%w",
			tool.Name, url, waitUntil, err)
	}

	return string(jsonResult), nil
}

func (t *BrowserTool) click(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	selector, ok := args["selector"].(string)
	if !ok {
		return "", fmt.Errorf("browser tool click failed: tool_name='%s', handler_type='browser', action='click', validation_error='selector parameter is required and must be a string'",
			tool.Name)
	}

	sessionID, _ := args["session_id"].(string)
	if sessionID == "" {
		return "", fmt.Errorf("browser tool click failed: tool_name='%s', handler_type='browser', action='click', validation_error='session_id is required for click action'",
			tool.Name)
	}

	waitForNavigation := false
	if w, ok := args["wait_for_navigation"].(bool); ok {
		waitForNavigation = w
	}

	/* Get browser context */
	browserCtx, err := t.driver.GetContext(sessionID)
	if err != nil {
		return "", fmt.Errorf("browser tool click failed: tool_name='%s', handler_type='browser', action='click', session_id='%s', error=%w",
			tool.Name, sessionID, err)
	}

	/* Click */
	if err := t.driver.Click(browserCtx, selector, waitForNavigation); err != nil {
		return "", fmt.Errorf("browser tool click failed: tool_name='%s', handler_type='browser', action='click', selector='%s', error=%w",
			tool.Name, selector, err)
	}

	result := map[string]interface{}{
		"action":              "click",
		"selector":            selector,
		"session_id":          sessionID,
		"wait_for_navigation": waitForNavigation,
		"status":              "success",
	}

	jsonResult, err := json.Marshal(result)
	if err != nil {
		return "", fmt.Errorf("browser tool click result marshaling failed: tool_name='%s', handler_type='browser', action='click', selector='%s', error=%w",
			tool.Name, selector, err)
	}

	return string(jsonResult), nil
}

func (t *BrowserTool) typeText(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	selector, ok := args["selector"].(string)
	if !ok {
		return "", fmt.Errorf("browser tool type failed: tool_name='%s', handler_type='browser', action='type', validation_error='selector parameter is required and must be a string'",
			tool.Name)
	}

	text, ok := args["text"].(string)
	if !ok {
		return "", fmt.Errorf("browser tool type failed: tool_name='%s', handler_type='browser', action='type', validation_error='text parameter is required and must be a string'",
			tool.Name)
	}

	sessionID, _ := args["session_id"].(string)
	if sessionID == "" {
		return "", fmt.Errorf("browser tool type failed: tool_name='%s', handler_type='browser', action='type', validation_error='session_id is required for type action'",
			tool.Name)
	}

	clear := false
	if c, ok := args["clear"].(bool); ok {
		clear = c
	}

	/* Get browser context */
	browserCtx, err := t.driver.GetContext(sessionID)
	if err != nil {
		return "", fmt.Errorf("browser tool type failed: tool_name='%s', handler_type='browser', action='type', session_id='%s', error=%w",
			tool.Name, sessionID, err)
	}

	/* Type text */
	if err := t.driver.Type(browserCtx, selector, text, clear); err != nil {
		return "", fmt.Errorf("browser tool type failed: tool_name='%s', handler_type='browser', action='type', selector='%s', text_length=%d, error=%w",
			tool.Name, selector, len(text), err)
	}

	result := map[string]interface{}{
		"action":     "type",
		"selector":   selector,
		"text":       text,
		"session_id": sessionID,
		"clear":      clear,
		"status":     "success",
	}

	jsonResult, err := json.Marshal(result)
	if err != nil {
		return "", fmt.Errorf("browser tool type result marshaling failed: tool_name='%s', handler_type='browser', action='type', selector='%s', text_length=%d, error=%w",
			tool.Name, selector, len(text), err)
	}

	return string(jsonResult), nil
}

func (t *BrowserTool) extract(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	selector, ok := args["selector"].(string)
	if !ok {
		return "", fmt.Errorf("browser tool extract failed: tool_name='%s', handler_type='browser', action='extract', validation_error='selector parameter is required and must be a string'",
			tool.Name)
	}

	extractType := "text"
	if et, ok := args["extract_type"].(string); ok {
		extractType = et
	}

	sessionID, _ := args["session_id"].(string)
	if sessionID == "" {
		return "", fmt.Errorf("browser tool extract failed: tool_name='%s', handler_type='browser', action='extract', validation_error='session_id is required for extract action'",
			tool.Name)
	}

	/* Get browser context */
	browserCtx, err := t.driver.GetContext(sessionID)
	if err != nil {
		return "", fmt.Errorf("browser tool extract failed: tool_name='%s', handler_type='browser', action='extract', session_id='%s', error=%w",
			tool.Name, sessionID, err)
	}

	/* Extract */
	value, err := t.driver.Extract(browserCtx, selector, extractType)
	if err != nil {
		return "", fmt.Errorf("browser tool extract failed: tool_name='%s', handler_type='browser', action='extract', selector='%s', extract_type='%s', error=%w",
			tool.Name, selector, extractType, err)
	}

	result := map[string]interface{}{
		"action":       "extract",
		"selector":     selector,
		"extract_type": extractType,
		"session_id":   sessionID,
		"value":        value,
		"status":       "success",
	}

	jsonResult, err := json.Marshal(result)
	if err != nil {
		return "", fmt.Errorf("browser tool extract result marshaling failed: tool_name='%s', handler_type='browser', action='extract', selector='%s', extract_type='%s', error=%w",
			tool.Name, selector, extractType, err)
	}

	return string(jsonResult), nil
}

func (t *BrowserTool) screenshot(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	sessionID, _ := args["session_id"].(string)
	if sessionID == "" {
		return "", fmt.Errorf("browser tool screenshot failed: tool_name='%s', handler_type='browser', action='screenshot', validation_error='session_id is required for screenshot action'",
			tool.Name)
	}

	fullPage := false
	if fp, ok := args["full_page"].(bool); ok {
		fullPage = fp
	}

	/* Get browser context */
	browserCtx, err := t.driver.GetContext(sessionID)
	if err != nil {
		return "", fmt.Errorf("browser tool screenshot failed: tool_name='%s', handler_type='browser', action='screenshot', session_id='%s', error=%w",
			tool.Name, sessionID, err)
	}

	/* Take screenshot */
	screenshotData, err := t.driver.Screenshot(browserCtx, fullPage)
	if err != nil {
		return "", fmt.Errorf("browser tool screenshot failed: tool_name='%s', handler_type='browser', action='screenshot', full_page=%v, error=%w",
			tool.Name, fullPage, err)
	}

	screenshotB64 := browser.GetScreenshotBase64(screenshotData)

	/* Save to database if available */
	if t.queries != nil && t.db != nil {
		t.saveBrowserSnapshot(ctx, sessionID, browserCtx.LastURL, browserCtx.LastTitle, screenshotB64, screenshotData)
	}

	result := map[string]interface{}{
		"action":         "screenshot",
		"session_id":     sessionID,
		"full_page":      fullPage,
		"screenshot_b64": screenshotB64,
		"status":         "success",
	}

	jsonResult, err := json.Marshal(result)
	if err != nil {
		return "", fmt.Errorf("browser tool screenshot result marshaling failed: tool_name='%s', handler_type='browser', action='screenshot', session_id='%s', error=%w",
			tool.Name, sessionID, err)
	}

	return string(jsonResult), nil
}

func (t *BrowserTool) executeScript(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	script, ok := args["script"].(string)
	if !ok {
		return "", fmt.Errorf("browser tool execute_script failed: tool_name='%s', handler_type='browser', action='execute_script', validation_error='script parameter is required and must be a string'",
			tool.Name)
	}

	sessionID, _ := args["session_id"].(string)
	if sessionID == "" {
		return "", fmt.Errorf("browser tool execute_script failed: tool_name='%s', handler_type='browser', action='execute_script', validation_error='session_id is required for execute_script action'",
			tool.Name)
	}

	/* Get browser context */
	browserCtx, err := t.driver.GetContext(sessionID)
	if err != nil {
		return "", fmt.Errorf("browser tool execute_script failed: tool_name='%s', handler_type='browser', action='execute_script', session_id='%s', error=%w",
			tool.Name, sessionID, err)
	}

	/* Execute script */
	result, err := t.driver.ExecuteScript(browserCtx, script)
	if err != nil {
		return "", fmt.Errorf("browser tool execute_script failed: tool_name='%s', handler_type='browser', action='execute_script', script_length=%d, error=%w",
			tool.Name, len(script), err)
	}

	response := map[string]interface{}{
		"action":     "execute_script",
		"script":     script,
		"session_id": sessionID,
		"result":     result,
		"status":     "success",
	}

	jsonResult, err := json.Marshal(response)
	if err != nil {
		return "", fmt.Errorf("browser tool execute_script result marshaling failed: tool_name='%s', handler_type='browser', action='execute_script', script_length=%d, error=%w",
			tool.Name, len(script), err)
	}

	return string(jsonResult), nil
}

func (t *BrowserTool) getCookies(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	sessionID, _ := args["session_id"].(string)
	if sessionID == "" {
		return "", fmt.Errorf("browser tool get_cookies failed: tool_name='%s', handler_type='browser', action='get_cookies', validation_error='session_id is required for get_cookies action'",
			tool.Name)
	}

	url, _ := args["url"].(string)
	if url == "" {
		/* Try to get URL from browser context */
		browserCtx, err := t.driver.GetContext(sessionID)
		if err == nil && browserCtx.LastURL != "" {
			url = browserCtx.LastURL
		}
	}

	/* Get browser context */
	browserCtx, err := t.driver.GetContext(sessionID)
	if err != nil {
		return "", fmt.Errorf("browser tool get_cookies failed: tool_name='%s', handler_type='browser', action='get_cookies', session_id='%s', error=%w",
			tool.Name, sessionID, err)
	}

	/* Get cookies */
	cookies, err := t.driver.GetCookies(browserCtx, url)
	if err != nil {
		return "", fmt.Errorf("browser tool get_cookies failed: tool_name='%s', handler_type='browser', action='get_cookies', url='%s', error=%w",
			tool.Name, url, err)
	}

	cookiesJSON, err := browser.CookiesToJSON(cookies)
	if err != nil {
		return "", fmt.Errorf("browser tool get_cookies conversion failed: tool_name='%s', handler_type='browser', action='get_cookies', error=%w",
			tool.Name, err)
	}

	result := map[string]interface{}{
		"action":     "get_cookies",
		"session_id": sessionID,
		"url":        url,
		"cookies":    cookiesJSON,
		"status":     "success",
	}

	jsonResult, err := json.Marshal(result)
	if err != nil {
		return "", fmt.Errorf("browser tool get_cookies result marshaling failed: tool_name='%s', handler_type='browser', action='get_cookies', session_id='%s', error=%w",
			tool.Name, sessionID, err)
	}

	return string(jsonResult), nil
}

func (t *BrowserTool) setCookies(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	cookies, ok := args["cookies"].([]interface{})
	if !ok {
		return "", fmt.Errorf("browser tool set_cookies failed: tool_name='%s', handler_type='browser', action='set_cookies', validation_error='cookies parameter is required and must be an array'",
			tool.Name)
	}

	sessionID, _ := args["session_id"].(string)
	if sessionID == "" {
		return "", fmt.Errorf("browser tool set_cookies failed: tool_name='%s', handler_type='browser', action='set_cookies', validation_error='session_id is required for set_cookies action'",
			tool.Name)
	}

	/* Convert cookies to map format */
	cookieMaps := make([]map[string]interface{}, 0, len(cookies))
	for _, cookie := range cookies {
		if cookieMap, ok := cookie.(map[string]interface{}); ok {
			cookieMaps = append(cookieMaps, cookieMap)
		}
	}

	/* Get browser context */
	browserCtx, err := t.driver.GetContext(sessionID)
	if err != nil {
		return "", fmt.Errorf("browser tool set_cookies failed: tool_name='%s', handler_type='browser', action='set_cookies', session_id='%s', error=%w",
			tool.Name, sessionID, err)
	}

	/* Set cookies */
	if err := t.driver.SetCookies(browserCtx, cookieMaps); err != nil {
		return "", fmt.Errorf("browser tool set_cookies failed: tool_name='%s', handler_type='browser', action='set_cookies', cookies_count=%d, error=%w",
			tool.Name, len(cookieMaps), err)
	}

	result := map[string]interface{}{
		"action":      "set_cookies",
		"session_id":  sessionID,
		"cookies_set": len(cookieMaps),
		"status":      "success",
	}

	jsonResult, err := json.Marshal(result)
	if err != nil {
		return "", fmt.Errorf("browser tool set_cookies result marshaling failed: tool_name='%s', handler_type='browser', action='set_cookies', session_id='%s', cookies_count=%d, error=%w",
			tool.Name, sessionID, len(cookieMaps), err)
	}

	return string(jsonResult), nil
}

/* scrape scrapes multiple elements from the page */
func (t *BrowserTool) scrape(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	selectors, ok := args["selectors"].(map[string]interface{})
	if !ok {
		return "", fmt.Errorf("browser tool scrape failed: tool_name='%s', handler_type='browser', action='scrape', validation_error='selectors parameter is required and must be a map'",
			tool.Name)
	}

	sessionID, _ := args["session_id"].(string)
	if sessionID == "" {
		return "", fmt.Errorf("browser tool scrape failed: tool_name='%s', handler_type='browser', action='scrape', validation_error='session_id is required for scrape action'",
			tool.Name)
	}

	/* Get browser context */
	browserCtx, err := t.driver.GetContext(sessionID)
	if err != nil {
		return "", fmt.Errorf("browser tool scrape failed: tool_name='%s', handler_type='browser', action='scrape', session_id='%s', error=%w",
			tool.Name, sessionID, err)
	}

	/* Convert selectors to map[string]string */
	selectorMap := make(map[string]string)
	for key, value := range selectors {
		if str, ok := value.(string); ok {
			selectorMap[key] = str
		}
	}

	/* Scrape */
	results, err := t.driver.Scrape(browserCtx, selectorMap)
	if err != nil {
		return "", fmt.Errorf("browser tool scrape failed: tool_name='%s', handler_type='browser', action='scrape', selectors_count=%d, error=%w",
			tool.Name, len(selectorMap), err)
	}

	result := map[string]interface{}{
		"action":     "scrape",
		"session_id": sessionID,
		"data":       results,
		"status":     "success",
	}

	jsonResult, err := json.Marshal(result)
	if err != nil {
		return "", fmt.Errorf("browser tool scrape result marshaling failed: tool_name='%s', handler_type='browser', action='scrape', error=%w",
			tool.Name, err)
	}

	return string(jsonResult), nil
}

/* waitForElement waits for an element to appear */
func (t *BrowserTool) waitForElement(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	selector, ok := args["selector"].(string)
	if !ok {
		return "", fmt.Errorf("browser tool wait_for_element failed: tool_name='%s', handler_type='browser', action='wait_for_element', validation_error='selector parameter is required and must be a string'",
			tool.Name)
	}

	sessionID, _ := args["session_id"].(string)
	if sessionID == "" {
		return "", fmt.Errorf("browser tool wait_for_element failed: tool_name='%s', handler_type='browser', action='wait_for_element', validation_error='session_id is required for wait_for_element action'",
			tool.Name)
	}

	timeout := t.timeout
	if to, ok := args["timeout"].(float64); ok {
		timeout = time.Duration(to) * time.Second
	}

	/* Get browser context */
	browserCtx, err := t.driver.GetContext(sessionID)
	if err != nil {
		return "", fmt.Errorf("browser tool wait_for_element failed: tool_name='%s', handler_type='browser', action='wait_for_element', session_id='%s', error=%w",
			tool.Name, sessionID, err)
	}

	/* Wait for element */
	if err := t.driver.WaitForElement(browserCtx, selector, timeout); err != nil {
		return "", fmt.Errorf("browser tool wait_for_element failed: tool_name='%s', handler_type='browser', action='wait_for_element', selector='%s', timeout=%v, error=%w",
			tool.Name, selector, timeout, err)
	}

	result := map[string]interface{}{
		"action":     "wait_for_element",
		"selector":   selector,
		"session_id": sessionID,
		"timeout":    timeout.Seconds(),
		"status":     "success",
	}

	jsonResult, err := json.Marshal(result)
	if err != nil {
		return "", fmt.Errorf("browser tool wait_for_element result marshaling failed: tool_name='%s', handler_type='browser', action='wait_for_element', selector='%s', error=%w",
			tool.Name, selector, err)
	}

	return string(jsonResult), nil
}

/* saveBrowserSession saves browser session to database */
func (t *BrowserTool) saveBrowserSession(ctx context.Context, sessionID string, browserCtx *browser.BrowserContext, url string) {
	if t.queries == nil || t.db == nil {
		return
	}

	cookiesJSON, _ := browser.CookiesToJSON(browserCtx.Cookies)
	cookiesBytes, _ := json.Marshal(cookiesJSON)

	query := `
		INSERT INTO neurondb_agent.browser_sessions 
		(session_id, current_url, cookies, user_agent, viewport_width, viewport_height, updated_at)
		VALUES ($1, $2, $3::jsonb, $4, $5, $6, NOW())
		ON CONFLICT (session_id) 
		DO UPDATE SET 
			current_url = EXCLUDED.current_url,
			cookies = EXCLUDED.cookies,
			user_agent = EXCLUDED.user_agent,
			viewport_width = EXCLUDED.viewport_width,
			viewport_height = EXCLUDED.viewport_height,
			updated_at = NOW()
	`

	_, err := t.db.ExecContext(ctx, query, sessionID, url, cookiesBytes, browserCtx.UserAgent, browserCtx.Viewport.Width, browserCtx.Viewport.Height)
	if err != nil {
		/* Log error but don't fail the operation */
		fmt.Printf("browser tool save session failed: session_id='%s', error=%v\n", sessionID, err)
	}
}

/* saveBrowserSnapshot saves screenshot to database */
func (t *BrowserTool) saveBrowserSnapshot(ctx context.Context, sessionID string, url string, title string, screenshotB64 string, screenshotData []byte) {
	if t.queries == nil || t.db == nil {
		return
	}

	/* Get browser session ID */
	var browserSessionID uuid.UUID
	query := `SELECT id FROM neurondb_agent.browser_sessions WHERE session_id = $1`
	err := t.db.GetContext(ctx, &browserSessionID, query, sessionID)
	if err != nil {
		/* Session might not exist yet, skip snapshot save */
		return
	}

	insertQuery := `
		INSERT INTO neurondb_agent.browser_snapshots 
		(session_id, browser_session_id, url, screenshot_data, screenshot_b64, page_title)
		VALUES ($1, $2, $3, $4, $5, $6)
	`

	_, err = t.db.ExecContext(ctx, insertQuery, sessionID, browserSessionID, url, screenshotData, screenshotB64, title)
	if err != nil {
		/* Log error but don't fail the operation */
		fmt.Printf("browser tool save snapshot failed: session_id='%s', error=%v\n", sessionID, err)
	}
}

func (t *BrowserTool) isURLAllowed(url string) bool {
	if len(t.allowedURLs) == 0 {
		return true
	}

	if t.allowedURLs[url] {
		return true
	}

	for allowedURL := range t.allowedURLs {
		if strings.HasPrefix(url, allowedURL) {
			return true
		}
	}

	return false
}

func (t *BrowserTool) Validate(args map[string]interface{}, schema map[string]interface{}) error {
	return ValidateArgs(args, schema)
}
