/*-------------------------------------------------------------------------
 *
 * progress_handlers.go
 *    Progress handler setup for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/server/progress_handlers.go
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

/* setupProgressHandlers sets up progress MCP handlers */
func (s *Server) setupProgressHandlers() {
  /* Get progress handler */
	s.mcpServer.SetHandler("progress/get", s.handleGetProgress)
}

/* handleGetProgress handles the progress/get request */
func (s *Server) handleGetProgress(ctx context.Context, params json.RawMessage) (interface{}, error) {
	if s == nil {
		return nil, fmt.Errorf("server instance is nil")
	}
	if s.middleware == nil {
		return nil, fmt.Errorf("middleware manager is not initialized")
	}
	if s.progress == nil {
		return nil, fmt.Errorf("progress tracker is not initialized")
	}
	
	mcpReq := &middleware.MCPRequest{
		Method: "progress/get",
		Params: make(map[string]interface{}),
	}

	return s.middleware.Execute(ctx, mcpReq, func(ctx context.Context, _ *middleware.MCPRequest) (*middleware.MCPResponse, error) {
		result, err := s.progress.HandleGetProgress(ctx, params)
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












