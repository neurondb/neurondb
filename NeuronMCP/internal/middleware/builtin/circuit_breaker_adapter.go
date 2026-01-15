/*-------------------------------------------------------------------------
 *
 * circuit_breaker_adapter.go
 *    Adapter to use circuit breaker middleware with MCP middleware interface
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/middleware/builtin/circuit_breaker_adapter.go
 *
 *-------------------------------------------------------------------------
 */

package builtin

import (
	"context"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/middleware"
	mw "github.com/neurondb/NeuronMCP/internal/middleware"
)

/* CircuitBreakerAdapter adapts circuit breaker middleware to MCP middleware interface */
type CircuitBreakerAdapter struct {
	cb     *mw.CircuitBreakerMiddleware
	logger *logging.Logger
}

/* NewCircuitBreakerAdapter creates a new circuit breaker adapter */
func NewCircuitBreakerAdapter(logger *logging.Logger, config mw.CircuitBreakerConfig) *CircuitBreakerAdapter {
	return &CircuitBreakerAdapter{
		cb:     mw.NewCircuitBreakerMiddleware(logger, config),
		logger: logger,
	}
}

/* Name returns the middleware name */
func (a *CircuitBreakerAdapter) Name() string {
	return "circuit_breaker"
}

/* Order returns the middleware order */
func (a *CircuitBreakerAdapter) Order() int {
	return 6 /* After retry, before error handling */
}

/* Enabled returns whether the middleware is enabled */
func (a *CircuitBreakerAdapter) Enabled() bool {
	return true
}

/* Execute executes the circuit breaker middleware */
func (a *CircuitBreakerAdapter) Execute(ctx context.Context, req *middleware.MCPRequest, next middleware.Handler) (*middleware.MCPResponse, error) {
	/* Only apply circuit breaker to tool calls */
	if req.Method != "tools/call" {
		return next(ctx, req)
	}

	/* Extract tool name */
	toolName := "unknown"
	if name, ok := req.Params["name"].(string); ok {
		toolName = name
	}

	/* Check if circuit breaker allows execution */
	params := map[string]interface{}{
		"_tool_name": toolName,
	}

	/* Use the circuit breaker's canExecute check */
	/* We need to wrap the next handler to work with the circuit breaker's signature */
	result, err := a.cb.Execute(ctx, params, func(ctx context.Context, params map[string]interface{}) (interface{}, error) {
		/* Call the next handler */
		resp, err := next(ctx, req)
		if err != nil {
			return nil, err
		}
		if resp.IsError {
			return nil, fmt.Errorf("request failed: %v", resp.Content)
		}
		return resp, nil
	})

	if err != nil {
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: fmt.Sprintf("Circuit breaker error: %v", err)},
			},
			IsError: true,
		}, nil
	}

	/* Convert result back to MCPResponse */
	if resp, ok := result.(*middleware.MCPResponse); ok {
		return resp, nil
	}

	return next(ctx, req)
}








