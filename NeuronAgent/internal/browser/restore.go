/*-------------------------------------------------------------------------
 *
 * restore.go
 *    Browser session restore from database
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/browser/restore.go
 *
 *-------------------------------------------------------------------------
 */

package browser

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	"github.com/chromedp/cdproto/network"
	"github.com/chromedp/chromedp"
	"github.com/google/uuid"
)

/* SessionData represents browser session data from database */
type SessionData struct {
	SessionID      string
	AgentID        uuid.UUID
	CurrentURL     string
	Cookies        string
	LocalStorage   string
	SessionStorage string
	UserAgent      string
	ViewportWidth  int
	ViewportHeight int
	CreatedAt      time.Time
	UpdatedAt      time.Time
}

/* RestoreSession attempts to restore a browser session from database */
func (d *Driver) RestoreSession(db interface{}, sessionID string) (*BrowserContext, error) {
	/* Try to query session from database */
	var session SessionData

	/* Type assert to get database interface */
	type Querier interface {
		QueryRowContext(ctx context.Context, query string, args ...interface{}) *sql.Row
	}

	dbQuerier, ok := db.(Querier)
	if !ok {
		return nil, fmt.Errorf("invalid database interface")
	}

	query := `
		SELECT session_id, agent_id, current_url, 
		       COALESCE(cookies::text, '[]') as cookies,
		       COALESCE(local_storage::text, '{}') as local_storage,
		       COALESCE(session_storage::text, '{}') as session_storage,
		       COALESCE(user_agent, '') as user_agent,
		       viewport_width, viewport_height, created_at, updated_at
		FROM neurondb_agent.browser_sessions
		WHERE session_id = $1 
		AND (expires_at IS NULL OR expires_at > NOW())
	`

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	err := dbQuerier.QueryRowContext(ctx, query, sessionID).Scan(
		&session.SessionID,
		&session.AgentID,
		&session.CurrentURL,
		&session.Cookies,
		&session.LocalStorage,
		&session.SessionStorage,
		&session.UserAgent,
		&session.ViewportWidth,
		&session.ViewportHeight,
		&session.CreatedAt,
		&session.UpdatedAt,
	)

	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("session not found in database: %s", sessionID)
		}
		return nil, fmt.Errorf("failed to restore session from database: %w", err)
	}

	/* Create new browser context with restored settings */
	viewport := Viewport{
		Width:  session.ViewportWidth,
		Height: session.ViewportHeight,
	}
	if viewport.Width == 0 {
		viewport.Width = 1920
	}
	if viewport.Height == 0 {
		viewport.Height = 1080
	}

	userAgent := session.UserAgent
	if userAgent == "" && d.config != nil {
		userAgent = d.config.DefaultUserAgent
	}

	browserCtx, err := d.GetOrCreateContext(sessionID, viewport, userAgent)
	if err != nil {
		return nil, fmt.Errorf("failed to create browser context for restored session: %w", err)
	}

	/* Restore cookies */
	if session.Cookies != "" && session.Cookies != "[]" {
		var cookiesData []map[string]interface{}
		if err := json.Unmarshal([]byte(session.Cookies), &cookiesData); err == nil {
			for _, cookieData := range cookiesData {
				name, _ := cookieData["name"].(string)
				value, _ := cookieData["value"].(string)
				domain, _ := cookieData["domain"].(string)
				path, _ := cookieData["path"].(string)

				if name != "" && value != "" {
					/* Set cookie in browser */
					_ = chromedp.Run(browserCtx.Ctx, chromedp.ActionFunc(func(c context.Context) error {
						expr := network.SetCookie(name, value)
						if domain != "" {
							expr = expr.WithDomain(domain)
						}
						if path != "" {
							expr = expr.WithPath(path)
						}
						return expr.Do(c)
					}))
				}
			}
		}
	}

	/* Restore local storage */
	if session.LocalStorage != "" && session.LocalStorage != "{}" {
		var localStorageData map[string]string
		if err := json.Unmarshal([]byte(session.LocalStorage), &localStorageData); err == nil {
			for key, value := range localStorageData {
				script := fmt.Sprintf(`localStorage.setItem(%q, %q)`, key, value)
				_ = chromedp.Run(browserCtx.Ctx, chromedp.Evaluate(script, nil))
			}
		}
	}

	/* Restore session storage */
	if session.SessionStorage != "" && session.SessionStorage != "{}" {
		var sessionStorageData map[string]string
		if err := json.Unmarshal([]byte(session.SessionStorage), &sessionStorageData); err == nil {
			for key, value := range sessionStorageData {
				script := fmt.Sprintf(`sessionStorage.setItem(%q, %q)`, key, value)
				_ = chromedp.Run(browserCtx.Ctx, chromedp.Evaluate(script, nil))
			}
		}
	}

	/* Navigate to last URL if exists */
	if session.CurrentURL != "" && session.CurrentURL != "about:blank" {
		_ = chromedp.Run(browserCtx.Ctx, chromedp.Navigate(session.CurrentURL))
		browserCtx.LastURL = session.CurrentURL
	}

	return browserCtx, nil
}

/* GetOrRestoreContext tries to restore session from DB, otherwise creates new */
func (d *Driver) GetOrRestoreContext(db interface{}, sessionID string, viewport Viewport, userAgent string) (*BrowserContext, error) {
	d.mu.RLock()
	if ctx, exists := d.contexts[sessionID]; exists {
		d.mu.RUnlock()
		ctx.LastAccess = time.Now()
		return ctx, nil
	}
	d.mu.RUnlock()

	/* Try to restore from database */
	if db != nil {
		ctx, err := d.RestoreSession(db, sessionID)
		if err == nil {
			return ctx, nil
		}
		/* If restoration failed, fall through to create new context */
	}

	/* Create new context */
	return d.GetOrCreateContext(sessionID, viewport, userAgent)
}

/* SaveSession saves current browser session state to database */
func (d *Driver) SaveSession(db interface{}, sessionID string, agentID uuid.UUID) error {
	ctx, err := d.GetContext(sessionID)
	if err != nil {
		return err
	}

	/* Get current cookies */
	var cookies []*network.Cookie
	err = chromedp.Run(ctx.Ctx, chromedp.ActionFunc(func(c context.Context) error {
		cookies, err = network.GetCookies().Do(c)
		return err
	}))

	cookiesJSON := "[]"
	if err == nil && len(cookies) > 0 {
		cookiesData := make([]map[string]interface{}, 0, len(cookies))
		for _, cookie := range cookies {
			cookiesData = append(cookiesData, map[string]interface{}{
				"name":   cookie.Name,
				"value":  cookie.Value,
				"domain": cookie.Domain,
				"path":   cookie.Path,
			})
		}
		if b, err := json.Marshal(cookiesData); err == nil {
			cookiesJSON = string(b)
		}
	}

	/* Get local storage */
	var localStorageJSON string
	_ = chromedp.Run(ctx.Ctx, chromedp.Evaluate(`JSON.stringify(Object.assign({}, localStorage))`, &localStorageJSON))
	if localStorageJSON == "" {
		localStorageJSON = "{}"
	}

	/* Get session storage */
	var sessionStorageJSON string
	_ = chromedp.Run(ctx.Ctx, chromedp.Evaluate(`JSON.stringify(Object.assign({}, sessionStorage))`, &sessionStorageJSON))
	if sessionStorageJSON == "" {
		sessionStorageJSON = "{}"
	}

	/* Get current URL and title */
	var currentURL, currentTitle string
	_ = chromedp.Run(ctx.Ctx,
		chromedp.Evaluate(`window.location.href`, &currentURL),
		chromedp.Evaluate(`document.title`, &currentTitle),
	)

	/* Type assert to get database interface */
	type Execer interface {
		ExecContext(ctx context.Context, query string, args ...interface{}) (sql.Result, error)
	}

	dbExecer, ok := db.(Execer)
	if !ok {
		return fmt.Errorf("invalid database interface")
	}

	query := `
		INSERT INTO neurondb_agent.browser_sessions 
			(session_id, agent_id, current_url, cookies, local_storage, session_storage, 
			 user_agent, viewport_width, viewport_height, created_at, updated_at)
		VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6::jsonb, $7, $8, $9, NOW(), NOW())
		ON CONFLICT (session_id) 
		DO UPDATE SET 
			current_url = EXCLUDED.current_url,
			cookies = EXCLUDED.cookies,
			local_storage = EXCLUDED.local_storage,
			session_storage = EXCLUDED.session_storage,
			updated_at = NOW()
	`

	saveCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	_, err = dbExecer.ExecContext(saveCtx, query,
		sessionID, agentID, currentURL, cookiesJSON, localStorageJSON, sessionStorageJSON,
		ctx.UserAgent, ctx.Viewport.Width, ctx.Viewport.Height)

	return err
}
