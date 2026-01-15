/*-------------------------------------------------------------------------
 *
 * prompt_handlers.go
 *    Prompt handler setup for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/server/prompt_handlers.go
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

/* setupPromptHandlers sets up prompt-related MCP handlers */
func (s *Server) setupPromptHandlers() {
  /* List prompts handler */
	s.mcpServer.SetHandler("prompts/list", s.handleListPrompts)

  /* Get prompt handler */
	s.mcpServer.SetHandler("prompts/get", s.handleGetPrompt)
}

/* handleListPrompts handles the prompts/list request */
func (s *Server) handleListPrompts(ctx context.Context, params json.RawMessage) (interface{}, error) {
	if s == nil {
		return nil, fmt.Errorf("server instance is nil")
	}
	if s.middleware == nil {
		return nil, fmt.Errorf("middleware manager is not initialized")
	}
	if s.prompts == nil {
		return nil, fmt.Errorf("prompts manager is not initialized")
	}
	
	mcpReq := &middleware.MCPRequest{
		Method:   "prompts/list",
		Params:   make(map[string]interface{}),
		Metadata: getHTTPMetadataFromContext(ctx), /* Include HTTP metadata for auth middleware */
	}

	return s.middleware.Execute(ctx, mcpReq, func(ctx context.Context, _ *middleware.MCPRequest) (*middleware.MCPResponse, error) {
		result, err := s.prompts.HandleListPrompts(ctx, params)
		if err != nil {
			return &middleware.MCPResponse{
				Content: []middleware.ContentBlock{
					{Type: "text", Text: fmt.Sprintf("Error: %v", err)},
				},
				IsError: true,
			}, nil
		}

		resultJSON, _ := json.MarshalIndent(result, "", "  ")
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: string(resultJSON)},
			},
		}, nil
	})
}

/* handleGetPrompt handles the prompts/get request */
func (s *Server) handleGetPrompt(ctx context.Context, params json.RawMessage) (interface{}, error) {
	if s == nil {
		return nil, fmt.Errorf("server instance is nil")
	}
	if s.middleware == nil {
		return nil, fmt.Errorf("middleware manager is not initialized")
	}
	if s.prompts == nil {
		return nil, fmt.Errorf("prompts manager is not initialized")
	}
	
	mcpReq := &middleware.MCPRequest{
		Method:   "prompts/get",
		Params:   make(map[string]interface{}),
		Metadata: getHTTPMetadataFromContext(ctx), /* Include HTTP metadata for auth middleware */
	}

	return s.middleware.Execute(ctx, mcpReq, func(ctx context.Context, _ *middleware.MCPRequest) (*middleware.MCPResponse, error) {
		result, err := s.prompts.HandleGetPrompt(ctx, params)
		if err != nil {
			return &middleware.MCPResponse{
				Content: []middleware.ContentBlock{
					{Type: "text", Text: fmt.Sprintf("Error: %v", err)},
				},
				IsError: true,
			}, nil
		}

		resultJSON, _ := json.MarshalIndent(result, "", "  ")
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: string(resultJSON)},
			},
		}, nil
	})
}

