/*-------------------------------------------------------------------------
 *
 * resource_quota_adapter.go
 *    Adapter to use resource quota middleware with MCP middleware interface
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/middleware/builtin/resource_quota_adapter.go
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

/* ResourceQuotaAdapter adapts resource quota middleware to MCP middleware interface */
type ResourceQuotaAdapter struct {
	rq     *mw.ResourceQuotaMiddleware
	logger *logging.Logger
}

/* NewResourceQuotaAdapter creates a new resource quota adapter */
func NewResourceQuotaAdapter(logger *logging.Logger, config mw.ResourceQuotaConfig) *ResourceQuotaAdapter {
	return &ResourceQuotaAdapter{
		rq:     mw.NewResourceQuotaMiddleware(logger, config),
		logger: logger,
	}
}

/* Name returns the middleware name */
func (a *ResourceQuotaAdapter) Name() string {
	return "resource_quota"
}

/* Order returns the middleware order */
func (a *ResourceQuotaAdapter) Order() int {
	return 7 /* After circuit breaker, before error handling */
}

/* Enabled returns whether the middleware is enabled */
func (a *ResourceQuotaAdapter) Enabled() bool {
	return true
}

/* Execute executes the resource quota middleware */
func (a *ResourceQuotaAdapter) Execute(ctx context.Context, req *middleware.MCPRequest, next middleware.Handler) (*middleware.MCPResponse, error) {
	/* Only apply resource quota to tool calls */
	if req.Method != "tools/call" {
		return next(ctx, req)
	}

	/* Extract tool name and arguments */
	toolName := "unknown"
	if name, ok := req.Params["name"].(string); ok {
		toolName = name
	}

	/* Convert request params to tool params format */
	params := make(map[string]interface{})
	if arguments, ok := req.Params["arguments"].(map[string]interface{}); ok {
		for k, v := range arguments {
			params[k] = v
		}
	}
	params["_tool_name"] = toolName

	/* Use the resource quota middleware */
	result, err := a.rq.Execute(ctx, params, func(ctx context.Context, params map[string]interface{}) (interface{}, error) {
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
				{Type: "text", Text: fmt.Sprintf("Resource quota error: %v", err)},
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







