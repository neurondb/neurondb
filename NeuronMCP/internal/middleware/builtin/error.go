/*-------------------------------------------------------------------------
 *
 * error.go
 *    Error handling middleware for NeuronMCP
 *
 * Provides error handling middleware that catches and logs unhandled errors
 * in MCP request processing with configurable stack trace support.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/middleware/builtin/error.go
 *
 *-------------------------------------------------------------------------
 */

package builtin

import (
	"context"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/middleware"
	"github.com/neurondb/NeuronMCP/internal/observability"
	"github.com/neurondb/NeuronMCP/internal/reliability"
)

/* ErrorHandlingMiddleware handles errors */
type ErrorHandlingMiddleware struct {
	logger          *logging.Logger
	enableErrorStack bool
}

/* NewErrorHandlingMiddleware creates a new error handling middleware */
func NewErrorHandlingMiddleware(logger *logging.Logger, enableStack bool) *ErrorHandlingMiddleware {
	return &ErrorHandlingMiddleware{
		logger:            logger,
		enableErrorStack: enableStack,
	}
}

/* Name returns the middleware name */
func (m *ErrorHandlingMiddleware) Name() string {
	return "error-handling"
}

/* Order returns the execution order */
func (m *ErrorHandlingMiddleware) Order() int {
	return 100
}

/* Enabled returns whether the middleware is enabled */
func (m *ErrorHandlingMiddleware) Enabled() bool {
	return true
}

/* Execute executes the middleware */
func (m *ErrorHandlingMiddleware) Execute(ctx context.Context, req *middleware.MCPRequest, next middleware.Handler) (*middleware.MCPResponse, error) {
	resp, err := next(ctx, req)
	if err != nil {
		/* Get request ID from context */
		requestID := ""
		if reqID, ok := observability.GetRequestIDFromContext(ctx); ok {
			requestID = reqID.String()
		}

		/* Classify error using taxonomy */
		classifier := reliability.NewErrorClassifier()
		structuredErr := classifier.ClassifyError(err, requestID)

		/* Log error with structured information */
		m.logger.Error("Unhandled error", err, map[string]interface{}{
			"method":     req.Method,
			"error_code": string(structuredErr.Code),
			"request_id": requestID,
		})

		/* Never expose raw driver errors - use structured error message */
		errorText := structuredErr.Message
		if structuredErr.Details != nil && len(structuredErr.Details) > 0 {
			/* Add relevant details if available */
			if toolName, ok := req.Params["name"].(string); ok {
				errorText += fmt.Sprintf(" (tool: %s)", toolName)
			}
		}

		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: errorText},
			},
			IsError: true,
			Metadata: map[string]interface{}{
				"error_code": string(structuredErr.Code),
				"message":    structuredErr.Message,
				"request_id": requestID,
			},
		}, nil
	}

	/* Also check if response indicates an error */
	if resp != nil && resp.IsError {
		/* Get request ID from context */
		requestID := ""
		if reqID, ok := observability.GetRequestIDFromContext(ctx); ok {
			requestID = reqID.String()
		}

		/* Ensure metadata has error code */
		if resp.Metadata == nil {
			resp.Metadata = make(map[string]interface{})
		}
		if _, hasCode := resp.Metadata["error_code"]; !hasCode {
			resp.Metadata["error_code"] = "UNKNOWN_ERROR"
		}
		if requestID != "" {
			resp.Metadata["request_id"] = requestID
		}
	}

	return resp, nil
}

