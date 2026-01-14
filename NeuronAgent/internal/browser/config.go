/*-------------------------------------------------------------------------
 *
 * config.go
 *    Browser configuration options
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/browser/config.go
 *
 *-------------------------------------------------------------------------
 */

package browser

import (
	"time"
)

/* BrowserConfig holds browser driver configuration options */
type BrowserConfig struct {
	/* Timeout settings */
	DefaultTimeout  time.Duration /* Default timeout for operations (default: 60s) */
	NavigateTimeout time.Duration /* Timeout for page navigation (default: 90s) */
	ScriptTimeout   time.Duration /* Timeout for script execution (default: 30s) */
	WaitTimeout     time.Duration /* Timeout for element waiting (default: 30s) */

	/* Session settings */
	MaxSessions          int           /* Maximum concurrent browser sessions (default: 10) */
	SessionIdleTimeout   time.Duration /* Idle timeout before session cleanup (default: 24h) */
	SessionCleanupPeriod time.Duration /* How often to run session cleanup (default: 1h) */

	/* Page settings */
	MaxPages         int  /* Maximum pages per session (default: 5) */
	EnableJavaScript bool /* Enable JavaScript execution (default: true) */
	EnableImages     bool /* Enable image loading (default: true) */
	EnableCSS        bool /* Enable CSS loading (default: true) */

	/* Screenshot settings */
	MaxScreenshots         int           /* Maximum screenshots per session (default: 100) */
	ScreenshotQuality      int           /* Screenshot quality 1-100 (default: 90) */
	ScreenshotTimeout      time.Duration /* Timeout for screenshot capture (default: 10s) */
	ScreenshotStoreToDBMax int64         /* Max screenshot size to store in DB in bytes (default: 10MB) */

	/* Browser pool settings */
	PoolSize          int           /* Number of browser instances to pool (default: 3) */
	PoolMaxIdleTime   time.Duration /* Max idle time for pooled browsers (default: 5m) */
	PoolReuseContexts bool          /* Reuse browser contexts (default: true) */

	/* Security settings */
	AllowedDomains []string /* Whitelist of allowed domains (empty = all allowed) */
	BlockedDomains []string /* Blacklist of blocked domains */
	SandboxMode    bool     /* Enable sandbox mode (default: true) */

	/* Performance settings */
	MaxMemoryMB      int  /* Maximum memory per browser instance in MB (default: 512) */
	EnableGPU        bool /* Enable GPU acceleration (default: false for headless) */
	EnableDevTools   bool /* Enable DevTools protocol (default: true) */
	DisableWebSec    bool /* Disable web security (default: false) */
	IgnoreCertErrors bool /* Ignore certificate errors (default: false) */

	/* User agent and viewport */
	DefaultUserAgent     string /* Default user agent string */
	DefaultViewportW     int    /* Default viewport width (default: 1920) */
	DefaultViewportH     int    /* Default viewport height (default: 1080) */
	EmulateDeviceMetrics bool   /* Emulate device metrics (default: true) */
}

/* DefaultConfig returns default browser configuration */
func DefaultConfig() *BrowserConfig {
	return &BrowserConfig{
		/* Timeouts */
		DefaultTimeout:  60 * time.Second,
		NavigateTimeout: 90 * time.Second,
		ScriptTimeout:   30 * time.Second,
		WaitTimeout:     30 * time.Second,

		/* Sessions */
		MaxSessions:          10,
		SessionIdleTimeout:   24 * time.Hour,
		SessionCleanupPeriod: 1 * time.Hour,

		/* Pages */
		MaxPages:         5,
		EnableJavaScript: true,
		EnableImages:     true,
		EnableCSS:        true,

		/* Screenshots */
		MaxScreenshots:         100,
		ScreenshotQuality:      90,
		ScreenshotTimeout:      10 * time.Second,
		ScreenshotStoreToDBMax: 10 * 1024 * 1024, // 10MB

		/* Browser pool */
		PoolSize:          3,
		PoolMaxIdleTime:   5 * time.Minute,
		PoolReuseContexts: true,

		/* Security */
		AllowedDomains: []string{},
		BlockedDomains: []string{},
		SandboxMode:    true,

		/* Performance */
		MaxMemoryMB:      512,
		EnableGPU:        false,
		EnableDevTools:   true,
		DisableWebSec:    false,
		IgnoreCertErrors: false,

		/* User agent and viewport */
		DefaultUserAgent:     "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
		DefaultViewportW:     1920,
		DefaultViewportH:     1080,
		EmulateDeviceMetrics: true,
	}
}

/* ProductionConfig returns recommended production configuration */
func ProductionConfig() *BrowserConfig {
	cfg := DefaultConfig()
	cfg.MaxSessions = 20
	cfg.PoolSize = 5
	cfg.MaxMemoryMB = 256
	cfg.EnableImages = false                     // Save bandwidth
	cfg.ScreenshotStoreToDBMax = 5 * 1024 * 1024 // 5MB
	return cfg
}

/* DevelopmentConfig returns configuration for development */
func DevelopmentConfig() *BrowserConfig {
	cfg := DefaultConfig()
	cfg.MaxSessions = 5
	cfg.PoolSize = 1
	cfg.EnableDevTools = true
	cfg.IgnoreCertErrors = true
	return cfg
}

/* Validate validates the configuration */
func (c *BrowserConfig) Validate() error {
	if c.MaxSessions < 1 {
		c.MaxSessions = 10
	}
	if c.MaxPages < 1 {
		c.MaxPages = 5
	}
	if c.PoolSize < 1 {
		c.PoolSize = 1
	}
	if c.DefaultTimeout < time.Second {
		c.DefaultTimeout = 60 * time.Second
	}
	if c.ScreenshotQuality < 1 || c.ScreenshotQuality > 100 {
		c.ScreenshotQuality = 90
	}
	return nil
}
