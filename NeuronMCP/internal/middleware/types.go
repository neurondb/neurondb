/*-------------------------------------------------------------------------
 *
 * types.go
 *    Middleware type definitions
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/middleware/types.go
 *
 *-------------------------------------------------------------------------
 */

package middleware

import "context"

/* MiddlewareFunc is the function signature for middleware next handlers */
type MiddlewareFunc func(ctx context.Context, params map[string]interface{}) (interface{}, error)

/* MCPRequest represents an MCP request */
type MCPRequest struct {
	Method   string                 `json:"method"`
	Params   map[string]interface{} `json:"params,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

/* MCPResponse represents an MCP response */
type MCPResponse struct {
	Content  []ContentBlock         `json:"content,omitempty"`
	IsError  bool                   `json:"isError,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

/* ContentBlock represents a content block in an MCP response */
type ContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

/* Handler is the function signature for handlers */
type Handler func(ctx context.Context, req *MCPRequest) (*MCPResponse, error)

/* Middleware interface for MCP middleware */
type Middleware interface {
	Execute(ctx context.Context, req *MCPRequest, next Handler) (*MCPResponse, error)
	Name() string
	Order() int
	Enabled() bool
}

