package builtin

import (
	"context"
	"fmt"
	"time"

	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/middleware"
)

// TimeoutMiddleware adds timeout to requests
type TimeoutMiddleware struct {
	timeout time.Duration
	logger  *logging.Logger
}

// NewTimeoutMiddleware creates a new timeout middleware
func NewTimeoutMiddleware(timeout time.Duration, logger *logging.Logger) *TimeoutMiddleware {
	return &TimeoutMiddleware{
		timeout: timeout,
		logger:  logger,
	}
}

// Name returns the middleware name
func (m *TimeoutMiddleware) Name() string {
	return "timeout"
}

// Order returns the execution order
func (m *TimeoutMiddleware) Order() int {
	return 3
}

// Enabled returns whether the middleware is enabled
func (m *TimeoutMiddleware) Enabled() bool {
	return true
}

// Execute executes the middleware
func (m *TimeoutMiddleware) Execute(ctx context.Context, req *middleware.MCPRequest, next middleware.Handler) (*middleware.MCPResponse, error) {
	ctx, cancel := context.WithTimeout(ctx, m.timeout)
	defer cancel()

	done := make(chan *middleware.MCPResponse, 1)
	errChan := make(chan error, 1)

	go func() {
		resp, err := next(ctx)
		if err != nil {
			errChan <- err
			return
		}
		done <- resp
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

