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
)

/* TimeoutMiddleware adds timeout to requests */
type TimeoutMiddleware struct {
	timeout time.Duration
	logger  *logging.Logger
}

/* NewTimeoutMiddleware creates a new timeout middleware */
func NewTimeoutMiddleware(timeout time.Duration, logger *logging.Logger) *TimeoutMiddleware {
	return &TimeoutMiddleware{
		timeout: timeout,
		logger:  logger,
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
	ctx, cancel := context.WithTimeout(ctx, m.timeout)
	defer cancel()

	done := make(chan *middleware.MCPResponse, 1)
	errChan := make(chan error, 1)

	go func() {
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
			return
		default:
			if err != nil {
				select {
				case errChan <- err:
				case <-ctx.Done():
					/* Context cancelled while sending error */
				}
			} else {
				select {
				case done <- resp:
				case <-ctx.Done():
					/* Context cancelled while sending response */
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
		m.logger.Warn("Request timeout", map[string]interface{}{
			"method":  req.Method,
			"timeout": m.timeout,
		})
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: fmt.Sprintf("Request timeout after %v", m.timeout)},
			},
			IsError: true,
		}, nil
	}
}

