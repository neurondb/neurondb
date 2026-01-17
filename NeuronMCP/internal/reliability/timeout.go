/*-------------------------------------------------------------------------
 *
 * timeout.go
 *    Per-tool timeout configuration for NeuronMCP
 *
 * Provides per-tool timeout configuration and management to ensure
 * tools don't run indefinitely.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/reliability/timeout.go
 *
 *-------------------------------------------------------------------------
 */

package reliability

import (
	"time"
)

/* TimeoutManager manages per-tool timeouts */
type TimeoutManager struct {
	defaultTimeout time.Duration
	toolTimeouts   map[string]time.Duration
}

/* NewTimeoutManager creates a new timeout manager */
func NewTimeoutManager(defaultTimeout time.Duration, toolTimeouts map[string]int) *TimeoutManager {
	tm := &TimeoutManager{
		defaultTimeout: defaultTimeout,
		toolTimeouts:   make(map[string]time.Duration),
	}

	/* Convert tool timeouts from seconds to duration */
	for tool, seconds := range toolTimeouts {
		tm.toolTimeouts[tool] = time.Duration(seconds) * time.Second
	}

	return tm
}

/* GetTimeout returns the timeout for a specific tool */
func (tm *TimeoutManager) GetTimeout(toolName string) time.Duration {
	if tm == nil {
		return 60 * time.Second /* Default fallback */
	}

	/* Check for tool-specific timeout */
	if timeout, exists := tm.toolTimeouts[toolName]; exists {
		return timeout
	}

	/* Return default timeout */
	if tm.defaultTimeout > 0 {
		return tm.defaultTimeout
	}

	return 60 * time.Second /* Final fallback */
}

/* SetToolTimeout sets a timeout for a specific tool */
func (tm *TimeoutManager) SetToolTimeout(toolName string, timeout time.Duration) {
	if tm == nil {
		return
	}
	if tm.toolTimeouts == nil {
		tm.toolTimeouts = make(map[string]time.Duration)
	}
	tm.toolTimeouts[toolName] = timeout
}

/* GetDefaultTimeout returns the default timeout */
func (tm *TimeoutManager) GetDefaultTimeout() time.Duration {
	if tm == nil || tm.defaultTimeout == 0 {
		return 60 * time.Second
	}
	return tm.defaultTimeout
}



