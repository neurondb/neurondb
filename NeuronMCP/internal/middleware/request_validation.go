/*-------------------------------------------------------------------------
 *
 * request_validation.go
 *    Request validation middleware for pre-execution parameter validation
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/middleware/request_validation.go
 *
 *-------------------------------------------------------------------------
 */

package middleware

import (
	"context"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/validation"
)

/* RequestValidationMiddleware validates all incoming requests before execution */
type RequestValidationMiddleware struct {
	logger *logging.Logger
}

/* NewRequestValidationMiddleware creates a new request validation middleware */
func NewRequestValidationMiddleware(logger *logging.Logger) *RequestValidationMiddleware {
	return &RequestValidationMiddleware{
		logger: logger,
	}
}

/* Execute validates request parameters before tool execution */
func (m *RequestValidationMiddleware) Execute(ctx context.Context, params map[string]interface{}, next MiddlewareFunc) (interface{}, error) {
	/* Validate context has not expired */
	if err := validation.ValidateContextDeadline(ctx); err != nil {
		m.logger.Error("Request validation failed: context deadline exceeded", err, params)
		return nil, fmt.Errorf("request timeout: %w", err)
	}

	/* Validate request ID exists (for tracing) */
	if requestID, ok := params["_request_id"].(string); ok {
		if err := validation.ValidateRequired(requestID, "request_id"); err != nil {
			m.logger.Warn("Request missing request_id", map[string]interface{}{
				"params": params,
			})
		}
	}

	/* Validate tool name */
	if toolName, ok := params["_tool_name"].(string); ok {
		if err := validation.ValidateRequired(toolName, "tool_name"); err != nil {
			m.logger.Error("Request validation failed: missing tool name", err, params)
			return nil, fmt.Errorf("tool name is required: %w", err)
		}
		if err := validation.ValidateMaxLength(toolName, "tool_name", 255); err != nil {
			m.logger.Error("Request validation failed: tool name too long", err, params)
			return nil, fmt.Errorf("tool name too long: %w", err)
		}
	}

	/* Validate no null bytes in string parameters (security) */
	for key, value := range params {
		if str, ok := value.(string); ok {
			if err := validation.ValidateNoNullBytes(str, key); err != nil {
				m.logger.Error("Request validation failed: null bytes detected", err, map[string]interface{}{
					"parameter": key,
					"params":    params,
				})
				return nil, fmt.Errorf("invalid parameter %s: contains null bytes", key)
			}
		}
	}

	m.logger.Debug("Request validation passed", map[string]interface{}{
		"params": params,
	})

	return next(ctx, params)
}

/* Name returns the middleware name */
func (m *RequestValidationMiddleware) Name() string {
	return "request_validation"
}



