/*-------------------------------------------------------------------------
 *
 * http_handlers.go
 *    HTTP request routing for NeuronMCP server
 *
 * Routes HTTP requests through middleware and to appropriate handlers
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/server/http_handlers.go
 *
 *-------------------------------------------------------------------------
 */

package server

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/middleware"
)

/* contextKey for storing HTTP metadata in context */
type contextKey string

const httpMetadataKey contextKey = "http_metadata"

/* getHTTPMetadataFromContext retrieves HTTP metadata from context if available */
func getHTTPMetadataFromContext(ctx context.Context) map[string]interface{} {
	if md := ctx.Value(httpMetadataKey); md != nil {
		if metadata, ok := md.(map[string]interface{}); ok {
			return metadata
		}
	}
	return nil
}

/* HandleHTTPRequest routes an HTTP MCP request through middleware to the appropriate handler */
func (s *Server) HandleHTTPRequest(ctx context.Context, mcpReq *middleware.MCPRequest) (*middleware.MCPResponse, error) {
	/* Validate server state */
	if s == nil {
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: "Server instance is nil"},
			},
			IsError: true,
			Metadata: map[string]interface{}{
				"error_code": "SERVER_ERROR",
			},
		}, nil
	}

	if s.middleware == nil {
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: "Middleware manager is not initialized"},
			},
			IsError: true,
			Metadata: map[string]interface{}{
				"error_code": "SERVER_ERROR",
			},
		}, nil
	}

	/* Validate request */
	if mcpReq == nil {
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: "Request is nil"},
			},
			IsError: true,
			Metadata: map[string]interface{}{
				"error_code": "INVALID_REQUEST",
			},
		}, nil
	}

	if mcpReq.Method == "" {
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: "Method is required"},
			},
			IsError: true,
			Metadata: map[string]interface{}{
				"error_code": "INVALID_PARAMS",
			},
		}, nil
	}

	/* Check context cancellation */
	select {
	case <-ctx.Done():
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: "Request context cancelled"},
			},
			IsError: true,
			Metadata: map[string]interface{}{
				"error_code": "REQUEST_CANCELLED",
			},
		}, ctx.Err()
	default:
	}

	/* Store HTTP metadata in context so handlers that execute middleware can access it */
	if mcpReq.Metadata != nil {
		ctx = context.WithValue(ctx, httpMetadataKey, mcpReq.Metadata)
	}

	/* Route to appropriate handler based on method */
	switch mcpReq.Method {
	case "initialize":
		return s.handleHTTPInitialize(ctx, mcpReq)
	case "tools/list":
		return s.handleHTTPListTools(ctx, mcpReq)
	case "tools/call":
		return s.handleHTTPCallTool(ctx, mcpReq)
	case "tools/search":
		return s.handleHTTPSearchTools(ctx, mcpReq)
	case "tools/call_batch":
		return s.handleHTTPCallBatch(ctx, mcpReq)
	case "resources/list":
		return s.handleHTTPListResources(ctx, mcpReq)
	case "resources/read":
		return s.handleHTTPReadResource(ctx, mcpReq)
	case "prompts/list":
		return s.handleHTTPListPrompts(ctx, mcpReq)
	case "prompts/get":
		return s.handleHTTPGetPrompt(ctx, mcpReq)
	case "sampling/createMessage":
		return s.handleHTTPCreateMessage(ctx, mcpReq)
	case "progress/get":
		return s.handleHTTPGetProgress(ctx, mcpReq)
	case "health/check":
		return s.handleHTTPHealthCheck(ctx, mcpReq)
	default:
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: fmt.Sprintf("Method not found: %s", mcpReq.Method)},
			},
			IsError: true,
			Metadata: map[string]interface{}{
				"error_code": "METHOD_NOT_FOUND",
			},
		}, nil
	}
}

/* handleHTTPInitialize handles initialize requests */
func (s *Server) handleHTTPInitialize(ctx context.Context, mcpReq *middleware.MCPRequest) (*middleware.MCPResponse, error) {
	/* Convert params to JSON */
	paramsJSON, _ := json.Marshal(mcpReq.Params)
	
	/* Call MCP server's initialize handler */
	result, err := s.mcpServer.HandleInitialize(ctx, paramsJSON)
	if err != nil {
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: err.Error()},
			},
			IsError: true,
		}, nil
	}

	/* Convert result to MCP response */
	resultJSON, _ := json.Marshal(result)
	return &middleware.MCPResponse{
		Content: []middleware.ContentBlock{
			{Type: "text", Text: string(resultJSON)},
		},
		Metadata: map[string]interface{}{
			"result": result,
		},
	}, nil
}

/* handleHTTPListTools handles tools/list requests */
func (s *Server) handleHTTPListTools(ctx context.Context, mcpReq *middleware.MCPRequest) (*middleware.MCPResponse, error) {
	/* Convert params to JSON and call handler (doesn't execute middleware internally) */
	paramsJSON, _ := json.Marshal(mcpReq.Params)
	result, err := s.handleListTools(ctx, paramsJSON)
	if err != nil {
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: err.Error()},
			},
			IsError: true,
		}, nil
	}

	/* Convert result to MCP response */
	resultJSON, _ := json.Marshal(result)
	return &middleware.MCPResponse{
		Content: []middleware.ContentBlock{
			{Type: "text", Text: string(resultJSON)},
		},
		Metadata: map[string]interface{}{
			"result": result,
		},
	}, nil
}

/* handleHTTPCallTool handles tools/call requests */
func (s *Server) handleHTTPCallTool(ctx context.Context, mcpReq *middleware.MCPRequest) (*middleware.MCPResponse, error) {
	/* Validate params */
	if mcpReq.Params == nil {
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: "Parameters are required for tools/call"},
			},
			IsError: true,
			Metadata: map[string]interface{}{
				"error_code": "INVALID_PARAMS",
			},
		}, nil
	}

	/* Convert params to JSON for handler */
	paramsJSON, err := json.Marshal(mcpReq.Params)
	if err != nil {
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: fmt.Sprintf("Failed to marshal parameters: %v", err)},
			},
			IsError: true,
			Metadata: map[string]interface{}{
				"error_code": "INVALID_PARAMS",
			},
		}, nil
	}
	
	/* Call handler which already executes middleware internally */
	result, err := s.handleCallTool(ctx, paramsJSON)
	if err != nil {
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: err.Error()},
			},
			IsError: true,
			Metadata: map[string]interface{}{
				"error_code": "EXECUTION_ERROR",
			},
		}, nil
	}

	/* Check if result is already a middleware response */
	if mcpResp, ok := result.(*middleware.MCPResponse); ok {
		return mcpResp, nil
	}

	/* Convert result to MCP response */
	resultJSON, err := json.Marshal(result)
	if err != nil {
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: fmt.Sprintf("Failed to marshal result: %v", err)},
			},
			IsError: true,
			Metadata: map[string]interface{}{
				"error_code": "SERIALIZATION_ERROR",
			},
		}, nil
	}

	return &middleware.MCPResponse{
		Content: []middleware.ContentBlock{
			{Type: "text", Text: string(resultJSON)},
		},
		Metadata: map[string]interface{}{
			"result": result,
		},
	}, nil
}

/* handleHTTPSearchTools handles tools/search requests */
func (s *Server) handleHTTPSearchTools(ctx context.Context, mcpReq *middleware.MCPRequest) (*middleware.MCPResponse, error) {
	paramsJSON, _ := json.Marshal(mcpReq.Params)
	result, err := s.handleSearchTools(ctx, paramsJSON)
	if err != nil {
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: err.Error()},
			},
			IsError: true,
		}, nil
	}

	resultJSON, _ := json.Marshal(result)
	return &middleware.MCPResponse{
		Content: []middleware.ContentBlock{
			{Type: "text", Text: string(resultJSON)},
		},
		Metadata: map[string]interface{}{
			"result": result,
		},
	}, nil
}

/* handleHTTPCallBatch handles tools/call_batch requests */
func (s *Server) handleHTTPCallBatch(ctx context.Context, mcpReq *middleware.MCPRequest) (*middleware.MCPResponse, error) {
	paramsJSON, err := json.Marshal(mcpReq.Params)
	if err != nil {
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: fmt.Sprintf("Failed to marshal parameters: %v", err)},
			},
			IsError: true,
			Metadata: map[string]interface{}{
				"error_code": "INVALID_PARAMS",
			},
		}, nil
	}

	result, err := s.handleCallBatch(ctx, paramsJSON)
	if err != nil {
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: err.Error()},
			},
			IsError: true,
			Metadata: map[string]interface{}{
				"error_code": "EXECUTION_ERROR",
			},
		}, nil
	}

	if mcpResp, ok := result.(*middleware.MCPResponse); ok {
		return mcpResp, nil
	}

	resultJSON, err := json.Marshal(result)
	if err != nil {
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: fmt.Sprintf("Failed to marshal result: %v", err)},
			},
			IsError: true,
			Metadata: map[string]interface{}{
				"error_code": "SERIALIZATION_ERROR",
			},
		}, nil
	}

	return &middleware.MCPResponse{
		Content: []middleware.ContentBlock{
			{Type: "text", Text: string(resultJSON)},
		},
		Metadata: map[string]interface{}{
			"result": result,
		},
	}, nil
}

/* handleHTTPListResources handles resources/list requests */
func (s *Server) handleHTTPListResources(ctx context.Context, mcpReq *middleware.MCPRequest) (*middleware.MCPResponse, error) {
	paramsJSON, _ := json.Marshal(mcpReq.Params)
	result, err := s.handleListResources(ctx, paramsJSON)
	if err != nil {
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: err.Error()},
			},
			IsError: true,
		}, nil
	}

	resultJSON, _ := json.Marshal(result)
	return &middleware.MCPResponse{
		Content: []middleware.ContentBlock{
			{Type: "text", Text: string(resultJSON)},
		},
		Metadata: map[string]interface{}{
			"result": result,
		},
	}, nil
}

/* handleHTTPReadResource handles resources/read requests */
func (s *Server) handleHTTPReadResource(ctx context.Context, mcpReq *middleware.MCPRequest) (*middleware.MCPResponse, error) {
	paramsJSON, _ := json.Marshal(mcpReq.Params)
	result, err := s.handleReadResource(ctx, paramsJSON)
	if err != nil {
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: err.Error()},
			},
			IsError: true,
		}, nil
	}

	resultJSON, _ := json.Marshal(result)
	return &middleware.MCPResponse{
		Content: []middleware.ContentBlock{
			{Type: "text", Text: string(resultJSON)},
		},
		Metadata: map[string]interface{}{
			"result": result,
		},
	}, nil
}

/* handleHTTPListPrompts handles prompts/list requests */
func (s *Server) handleHTTPListPrompts(ctx context.Context, mcpReq *middleware.MCPRequest) (*middleware.MCPResponse, error) {
	paramsJSON, _ := json.Marshal(mcpReq.Params)
	result, err := s.handleListPrompts(ctx, paramsJSON)
	if err != nil {
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: err.Error()},
			},
			IsError: true,
		}, nil
	}

	resultJSON, _ := json.Marshal(result)
	return &middleware.MCPResponse{
		Content: []middleware.ContentBlock{
			{Type: "text", Text: string(resultJSON)},
		},
		Metadata: map[string]interface{}{
			"result": result,
		},
	}, nil
}

/* handleHTTPGetPrompt handles prompts/get requests */
func (s *Server) handleHTTPGetPrompt(ctx context.Context, mcpReq *middleware.MCPRequest) (*middleware.MCPResponse, error) {
	paramsJSON, _ := json.Marshal(mcpReq.Params)
	result, err := s.handleGetPrompt(ctx, paramsJSON)
	if err != nil {
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: err.Error()},
			},
			IsError: true,
		}, nil
	}

	resultJSON, _ := json.Marshal(result)
	return &middleware.MCPResponse{
		Content: []middleware.ContentBlock{
			{Type: "text", Text: string(resultJSON)},
		},
		Metadata: map[string]interface{}{
			"result": result,
		},
	}, nil
}

/* handleHTTPCreateMessage handles sampling/createMessage requests */
func (s *Server) handleHTTPCreateMessage(ctx context.Context, mcpReq *middleware.MCPRequest) (*middleware.MCPResponse, error) {
	paramsJSON, _ := json.Marshal(mcpReq.Params)
	result, err := s.handleCreateMessage(ctx, paramsJSON)
	if err != nil {
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: err.Error()},
			},
			IsError: true,
		}, nil
	}

	resultJSON, _ := json.Marshal(result)
	return &middleware.MCPResponse{
		Content: []middleware.ContentBlock{
			{Type: "text", Text: string(resultJSON)},
		},
		Metadata: map[string]interface{}{
			"result": result,
		},
	}, nil
}

/* handleHTTPGetProgress handles progress/get requests */
func (s *Server) handleHTTPGetProgress(ctx context.Context, mcpReq *middleware.MCPRequest) (*middleware.MCPResponse, error) {
	paramsJSON, _ := json.Marshal(mcpReq.Params)
	result, err := s.handleGetProgress(ctx, paramsJSON)
	if err != nil {
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: err.Error()},
			},
			IsError: true,
		}, nil
	}

	resultJSON, _ := json.Marshal(result)
	return &middleware.MCPResponse{
		Content: []middleware.ContentBlock{
			{Type: "text", Text: string(resultJSON)},
		},
		Metadata: map[string]interface{}{
			"result": result,
		},
	}, nil
}

/* handleHTTPHealthCheck handles health/check requests */
func (s *Server) handleHTTPHealthCheck(ctx context.Context, mcpReq *middleware.MCPRequest) (*middleware.MCPResponse, error) {
	paramsJSON, _ := json.Marshal(mcpReq.Params)
	result, err := s.handleHealthCheck(ctx, paramsJSON)
	if err != nil {
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: err.Error()},
			},
			IsError: true,
		}, nil
	}

	resultJSON, _ := json.Marshal(result)
	return &middleware.MCPResponse{
		Content: []middleware.ContentBlock{
			{Type: "text", Text: string(resultJSON)},
		},
		Metadata: map[string]interface{}{
			"result": result,
		},
	}, nil
}

