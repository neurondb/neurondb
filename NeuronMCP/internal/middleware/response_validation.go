/*-------------------------------------------------------------------------
 *
 * response_validation.go
 *    Response schema validation middleware to ensure outputs match declared schemas
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/middleware/response_validation.go
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

/* ResponseValidationMiddleware validates tool responses against expected schemas */
type ResponseValidationMiddleware struct {
	logger *logging.Logger
}

/* NewResponseValidationMiddleware creates a new response validation middleware */
func NewResponseValidationMiddleware(logger *logging.Logger) *ResponseValidationMiddleware {
	return &ResponseValidationMiddleware{
		logger: logger,
	}
}

/* Execute validates response data after tool execution */
func (m *ResponseValidationMiddleware) Execute(ctx context.Context, params map[string]interface{}, next MiddlewareFunc) (interface{}, error) {
	result, err := next(ctx, params)
	if err != nil {
		return result, err
	}

	/* Validate response is not nil */
	if result == nil {
		m.logger.Warn("Response validation warning: nil result", map[string]interface{}{
			"params": params,
		})
		return result, nil
	}

	/* Validate response structure */
	switch r := result.(type) {
	case map[string]interface{}:
		/* Validate content field exists for successful responses */
		if isError, ok := r["isError"].(bool); !ok || !isError {
			if content, hasContent := r["content"]; !hasContent {
				m.logger.Warn("Response validation warning: missing content field", map[string]interface{}{
					"params": params,
					"result": result,
				})
			} else {
				/* Validate content is not empty for successful responses */
				if contentList, ok := content.([]interface{}); ok && len(contentList) == 0 {
					m.logger.Warn("Response validation warning: empty content list", map[string]interface{}{
						"params": params,
					})
				}
			}
		}

		/* Validate error responses have error field */
		if isError, ok := r["isError"].(bool); ok && isError {
			if _, hasError := r["error"]; !hasError {
				m.logger.Error("Response validation failed: error response missing error field", nil, map[string]interface{}{
					"params": params,
					"result": result,
				})
				return nil, fmt.Errorf("invalid error response: missing error field")
			}
		}

		/* Validate JSON content if present */
		if content, ok := r["content"]; ok {
			if contentList, ok := content.([]interface{}); ok {
				for i, item := range contentList {
					if itemMap, ok := item.(map[string]interface{}); ok {
						/* Validate text content */
						if text, hasText := itemMap["text"]; hasText {
							if textStr, ok := text.(string); ok {
								if err := validation.ValidateNoNullBytes(textStr, fmt.Sprintf("content[%d].text", i)); err != nil {
									m.logger.Error("Response validation failed: null bytes in text content", err, map[string]interface{}{
										"index":  i,
										"params": params,
									})
									return nil, fmt.Errorf("invalid response content: null bytes detected at index %d", i)
								}
							}
						}
					}
				}
			}
		}

	case []interface{}:
		/* Validate array responses are not empty (warning only) */
		if len(r) == 0 {
			m.logger.Debug("Response validation: empty array result", map[string]interface{}{
				"params": params,
			})
		}
	}

	m.logger.Debug("Response validation passed", map[string]interface{}{
		"params": params,
	})

	return result, nil
}

/* Name returns the middleware name */
func (m *ResponseValidationMiddleware) Name() string {
	return "response_validation"
}



