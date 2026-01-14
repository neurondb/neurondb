/*-------------------------------------------------------------------------
 *
 * timeout.go
 *    Timeout middleware for per-tool configurable timeouts
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/middleware/timeout.go
 *
 *-------------------------------------------------------------------------
 */

package middleware

import (
	"context"
	"fmt"
	"time"

	"github.com/neurondb/NeuronMCP/internal/logging"
)

/* TimeoutMiddleware enforces per-tool execution timeouts */
type TimeoutMiddleware struct {
	logger          *logging.Logger
	defaultTimeout  time.Duration
	toolTimeouts    map[string]time.Duration
	enableOverrides bool
}

/* TimeoutConfig configures timeout behavior */
type TimeoutConfig struct {
	DefaultTimeout  time.Duration         // Default timeout for all tools
	ToolTimeouts    map[string]time.Duration // Per-tool timeout overrides
	EnableOverrides bool                  // Allow tools to override via params
}

/* NewTimeoutMiddleware creates a new timeout middleware */
func NewTimeoutMiddleware(logger *logging.Logger, config TimeoutConfig) *TimeoutMiddleware {
	if config.DefaultTimeout == 0 {
		config.DefaultTimeout = 30 * time.Second
	}
	if config.ToolTimeouts == nil {
		config.ToolTimeouts = make(map[string]time.Duration)
	}
	
	if config.ToolTimeouts["load_dataset"] == 0 {
		config.ToolTimeouts["load_dataset"] = 300 * time.Second // 5 minutes
	}
	if config.ToolTimeouts["train_model"] == 0 {
		config.ToolTimeouts["train_model"] = 600 * time.Second // 10 minutes
	}
	if config.ToolTimeouts["create_hnsw_index"] == 0 {
		config.ToolTimeouts["create_hnsw_index"] = 180 * time.Second // 3 minutes
	}
	if config.ToolTimeouts["vector_search"] == 0 {
		config.ToolTimeouts["vector_search"] = 10 * time.Second
	}
	if config.ToolTimeouts["generate_embeddings"] == 0 {
		config.ToolTimeouts["generate_embeddings"] = 60 * time.Second
	}

	return &TimeoutMiddleware{
		logger:          logger,
		defaultTimeout:  config.DefaultTimeout,
		toolTimeouts:    config.ToolTimeouts,
		enableOverrides: config.EnableOverrides,
	}
}

/* Execute enforces timeout for tool execution */
func (m *TimeoutMiddleware) Execute(ctx context.Context, params map[string]interface{}, next MiddlewareFunc) (interface{}, error) {
	/* Determine timeout for this request */
	timeout := m.defaultTimeout
	
	/* Check for tool-specific timeout */
	if toolName, ok := params["_tool_name"].(string); ok {
		if toolTimeout, exists := m.toolTimeouts[toolName]; exists {
			timeout = toolTimeout
		}
	}
	
	/* Allow override via params if enabled */
	if m.enableOverrides {
		if timeoutParam, ok := params["_timeout"].(float64); ok && timeoutParam > 0 {
			requestedTimeout := time.Duration(timeoutParam) * time.Second
			/* Cap at 10 minutes for safety */
			if requestedTimeout <= 600*time.Second {
				timeout = requestedTimeout
				m.logger.Debug("Using request-specified timeout", map[string]interface{}{
					"timeout": timeout,
					"params":  params,
				})
			} else {
				m.logger.Warn("Requested timeout exceeds maximum, using default", map[string]interface{}{
					"requested": requestedTimeout,
					"maximum":   600 * time.Second,
					"using":     timeout,
				})
			}
		}
	}

	/* Create timeout context */
	timeoutCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	/* Execute with timeout */
	resultChan := make(chan interface{}, 1)
	errChan := make(chan error, 1)

	go func() {
		/* Check if context is already cancelled before starting */
		select {
		case <-timeoutCtx.Done():
			/* Context already cancelled, don't start execution */
			return
		default:
		}
		
		result, err := next(timeoutCtx, params)
		
		/* Check context again before sending result */
		select {
		case <-timeoutCtx.Done():
			/* Context was cancelled, don't send result */
			return
		default:
			if err != nil {
				select {
				case errChan <- err:
				case <-timeoutCtx.Done():
					/* Context cancelled while sending error */
				}
			} else {
				select {
				case resultChan <- result:
				case <-timeoutCtx.Done():
					/* Context cancelled while sending result */
				}
			}
		}
	}()

	select {
	case result := <-resultChan:
		return result, nil
	case err := <-errChan:
		return nil, err
	case <-timeoutCtx.Done():
		m.logger.Error("Tool execution timeout", timeoutCtx.Err(), map[string]interface{}{
			"timeout": timeout,
			"params":  params,
		})
		return nil, fmt.Errorf("tool execution timeout after %v: %w", timeout, timeoutCtx.Err())
	}
}

/* Name returns the middleware name */
func (m *TimeoutMiddleware) Name() string {
	return "timeout"
}

/* SetToolTimeout sets timeout for a specific tool */
func (m *TimeoutMiddleware) SetToolTimeout(toolName string, timeout time.Duration) {
	m.toolTimeouts[toolName] = timeout
}

/* GetToolTimeout gets timeout for a specific tool */
func (m *TimeoutMiddleware) GetToolTimeout(toolName string) time.Duration {
	if timeout, exists := m.toolTimeouts[toolName]; exists {
		return timeout
	}
	return m.defaultTimeout
}



