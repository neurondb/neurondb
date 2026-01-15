/*-------------------------------------------------------------------------
 *
 * batch_handlers.go
 *    Batch handler setup for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/server/batch_handlers.go
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

/* setupBatchHandlers sets up batch MCP handlers */
func (s *Server) setupBatchHandlers() {
  /* Batch tool calls handler */
	s.mcpServer.SetHandler("tools/call_batch", s.handleCallBatch)
}

/* handleCallBatch handles the tools/call_batch request */
func (s *Server) handleCallBatch(ctx context.Context, params json.RawMessage) (interface{}, error) {
	if s == nil {
		return nil, fmt.Errorf("server instance is nil")
	}
	if s.middleware == nil {
		return nil, fmt.Errorf("middleware manager is not initialized")
	}
	if s.batch == nil {
		return nil, fmt.Errorf("batch processor is not initialized")
	}
	
	mcpReq := &middleware.MCPRequest{
		Method:   "tools/call_batch",
		Params:   make(map[string]interface{}),
		Metadata: getHTTPMetadataFromContext(ctx), /* Include HTTP metadata for auth middleware */
	}

	return s.middleware.Execute(ctx, mcpReq, func(ctx context.Context, _ *middleware.MCPRequest) (*middleware.MCPResponse, error) {
		result, err := s.batch.HandleCallBatch(ctx, params)
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












