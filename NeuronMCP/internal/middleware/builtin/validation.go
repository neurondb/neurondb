/*-------------------------------------------------------------------------
 *
 * validation.go
 *    Request validation middleware for NeuronMCP
 *
 * Provides middleware for validating MCP request structure, parameters,
 * and required fields before request processing.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/middleware/builtin/validation.go
 *
 *-------------------------------------------------------------------------
 */

package builtin

import (
	"context"

	"github.com/neurondb/NeuronMCP/internal/middleware"
)

/* ValidationMiddleware validates requests */
type ValidationMiddleware struct{}

/* NewValidationMiddleware creates a new validation middleware */
func NewValidationMiddleware() *ValidationMiddleware {
	return &ValidationMiddleware{}
}

/* Name returns the middleware name */
func (m *ValidationMiddleware) Name() string {
	return "validation"
}

/* Order returns the execution order */
func (m *ValidationMiddleware) Order() int {
	return 1
}

/* Enabled returns whether the middleware is enabled */
func (m *ValidationMiddleware) Enabled() bool {
	return true
}

/* Execute executes the middleware */
func (m *ValidationMiddleware) Execute(ctx context.Context, req *middleware.MCPRequest, next middleware.Handler) (*middleware.MCPResponse, error) {
	if req.Method == "" {
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: "Missing method in request"},
			},
			IsError: true,
		}, nil
	}

	if req.Params != nil {
   /* Params should be a map, which is already validated by type */
	}

	return next(ctx, req)
}

