package builtin

import (
	"context"

	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/middleware"
)

// ErrorHandlingMiddleware handles errors
type ErrorHandlingMiddleware struct {
	logger          *logging.Logger
	enableErrorStack bool
}

// NewErrorHandlingMiddleware creates a new error handling middleware
func NewErrorHandlingMiddleware(logger *logging.Logger, enableStack bool) *ErrorHandlingMiddleware {
	return &ErrorHandlingMiddleware{
		logger:            logger,
		enableErrorStack: enableStack,
	}
}

// Name returns the middleware name
func (m *ErrorHandlingMiddleware) Name() string {
	return "error-handling"
}

// Order returns the execution order
func (m *ErrorHandlingMiddleware) Order() int {
	return 100
}

// Enabled returns whether the middleware is enabled
func (m *ErrorHandlingMiddleware) Enabled() bool {
	return true
}

// Execute executes the middleware
func (m *ErrorHandlingMiddleware) Execute(ctx context.Context, req *middleware.MCPRequest, next middleware.Handler) (*middleware.MCPResponse, error) {
	resp, err := next(ctx)
	if err != nil {
		errorMsg := err.Error()
		var stack string
		if m.enableErrorStack {
			// In Go, we'd need to get stack trace differently
			stack = ""
		}

		m.logger.Error("Unhandled error", err, map[string]interface{}{
			"method": req.Method,
			"params": req.Params,
		})

		text := "Error: " + errorMsg
		if stack != "" {
			text += "\n" + stack
		}

		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: text},
			},
			IsError: true,
			Metadata: map[string]interface{}{
				"error": map[string]interface{}{
					"message": errorMsg,
					"stack":   stack,
				},
			},
		}, nil
	}

	return resp, nil
}

