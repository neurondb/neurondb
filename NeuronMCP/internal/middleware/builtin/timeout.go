/*-------------------------------------------------------------------------
 *
 * timeout.go
 *    Timeout middleware for NeuronMCP
 *
 * Provides request timeout management middleware that enforces maximum
 * execution time for MCP requests with configurable timeout values.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/middleware/builtin/timeout.go
 *
 *-------------------------------------------------------------------------
 */

package builtin

import (
	"context"
	"fmt"
	"time"

	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/middleware"
	"github.com/neurondb/NeuronMCP/internal/reliability"
)

/* TimeoutMiddleware adds timeout to requests */
type TimeoutMiddleware struct {
	timeoutManager *reliability.TimeoutManager
	defaultTimeout time.Duration
	logger         *logging.Logger
}

/* NewTimeoutMiddleware creates a new timeout middleware */
func NewTimeoutMiddleware(timeout time.Duration, logger *logging.Logger) *TimeoutMiddleware {
	return &TimeoutMiddleware{
		defaultTimeout: timeout,
		logger:         logger,
	}
}

/* NewTimeoutMiddlewareWithManager creates a new timeout middleware with per-tool timeouts */
func NewTimeoutMiddlewareWithManager(timeoutManager *reliability.TimeoutManager, logger *logging.Logger) *TimeoutMiddleware {
	return &TimeoutMiddleware{
		timeoutManager: timeoutManager,
		logger:         logger,
	}
}

/* Name returns the middleware name */
func (m *TimeoutMiddleware) Name() string {
	return "timeout"
}

/* Order returns the execution order */
func (m *TimeoutMiddleware) Order() int {
	return 3
}

/* Enabled returns whether the middleware is enabled */
func (m *TimeoutMiddleware) Enabled() bool {
	return true
}

/* Execute executes the middleware */
func (m *TimeoutMiddleware) Execute(ctx context.Context, req *middleware.MCPRequest, next middleware.Handler) (*middleware.MCPResponse, error) {
	/* Determine timeout based on tool name if available */
	var timeout time.Duration
	if m.timeoutManager != nil && req.Method == "tools/call" {
		if toolName, ok := req.Params["name"].(string); ok {
			timeout = m.timeoutManager.GetTimeout(toolName)
		} else {
			timeout = m.timeoutManager.GetDefaultTimeout()
		}
	} else if m.timeoutManager != nil {
		timeout = m.timeoutManager.GetDefaultTimeout()
	} else {
		timeout = m.defaultTimeout
	}

	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	done := make(chan *middleware.MCPResponse, 1)
	errChan := make(chan error, 1)

	/* Use a goroutine that properly handles cancellation and cleanup */
	go func() {
		/* Ensure goroutine exits when context is cancelled */
		defer func() {
			/* Recover from any panics */
			if r := recover(); r != nil {
				/* Panic occurred, log it and send error */
				panicErr := fmt.Errorf("panic in timeout middleware: %v", r)
				if m.logger != nil {
					m.logger.Warn("Panic recovered in timeout middleware", map[string]interface{}{
						"method":     req.Method,
						"panic_value": fmt.Sprintf("%v", r),
					})
				}
				/* Try to send error, but don't block if context is cancelled */
				select {
				case errChan <- panicErr:
				case <-ctx.Done():
					/* Context cancelled, error already logged */
				default:
					/* Channel full, error already logged */
				}
			}
		}()
		
		/* Check if context is already cancelled before starting */
		select {
		case <-ctx.Done():
			/* Context already cancelled, don't start execution */
			return
		default:
		}
		
		resp, err := next(ctx, req)
		
		/* Check context again before sending result */
		select {
		case <-ctx.Done():
			/* Context was cancelled, don't send result */
			/* Goroutine will exit naturally */
			return
		default:
			if err != nil {
				select {
				case errChan <- err:
				case <-ctx.Done():
					/* Context cancelled while sending error - goroutine exits */
					return
				}
			} else {
				select {
				case done <- resp:
				case <-ctx.Done():
					/* Context cancelled while sending response - goroutine exits */
					return
				}
			}
		}
	}()

	select {
	case resp := <-done:
		return resp, nil
	case err := <-errChan:
		return nil, err
	case <-ctx.Done():
		toolName := ""
		if req.Method == "tools/call" {
			if name, ok := req.Params["name"].(string); ok {
				toolName = name
			}
		}
		m.logger.Warn("Request timeout", map[string]interface{}{
			"method":    req.Method,
			"tool_name": toolName,
			"timeout":   timeout,
		})
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: fmt.Sprintf("Request timeout after %v", timeout)},
			},
			IsError: true,
			Metadata: map[string]interface{}{
				"error_code": "TIMEOUT",
				"timeout":    timeout.String(),
			},
		}, nil
	}
}

