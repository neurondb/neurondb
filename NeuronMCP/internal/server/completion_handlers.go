/*-------------------------------------------------------------------------
 *
 * completion_handlers.go
 *    Completion handler setup for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/server/completion_handlers.go
 *
 *-------------------------------------------------------------------------
 */

package server

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/middleware"
	"github.com/neurondb/NeuronMCP/pkg/mcp"
)

/* setupCompletionHandlers sets up completion-related MCP handlers */
func (s *Server) setupCompletionHandlers() {
	/* Complete handler */
	s.mcpServer.SetHandler("completion/complete", s.handleComplete)
}

/* handleComplete handles the completion/complete request */
func (s *Server) handleComplete(ctx context.Context, params json.RawMessage) (interface{}, error) {
	if s == nil {
		return nil, fmt.Errorf("server instance is nil")
	}
	if s.middleware == nil {
		return nil, fmt.Errorf("middleware manager is not initialized")
	}
	if s.completionManager == nil {
		return nil, fmt.Errorf("completion manager is not initialized")
	}

	if len(params) == 0 {
		return nil, fmt.Errorf("completion/complete request parameters are required: received nil or empty params")
	}

	var req mcp.CompletionRequest
	if err := json.Unmarshal(params, &req); err != nil {
		return nil, fmt.Errorf("failed to parse completion/complete request: %w", err)
	}

	/* Validate request */
	if req.Ref.Type == "" {
		return nil, fmt.Errorf("completion/complete: reference type is required")
	}

	mcpReq := &middleware.MCPRequest{
		Method:   "completion/complete",
		Params: map[string]interface{}{
			"ref":      req.Ref,
			"argument": req.Argument,
			"context":  req.Context,
		},
		Metadata: getHTTPMetadataFromContext(ctx), /* Include HTTP metadata for auth middleware */
	}

	/* Execute through middleware for auth/validation */
	var completionResult *mcp.CompletionResponse
	_, err := s.middleware.Execute(ctx, mcpReq, func(ctx context.Context, _ *middleware.MCPRequest) (*middleware.MCPResponse, error) {
		/* Get the actual completion result */
		result, completionErr := s.completionManager.Complete(ctx, req)
		if completionErr != nil {
			return &middleware.MCPResponse{
				Content: []middleware.ContentBlock{
					{Type: "text", Text: fmt.Sprintf("Error: %v", completionErr)},
				},
				IsError: true,
			}, completionErr
		}

		/* Store result for return after middleware processing */
		completionResult = result

		/* Return success through middleware (for logging/audit) */
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: "Completion successful"},
			},
			Metadata: map[string]interface{}{
				"completion_result": result,
			},
		}, nil
	})

	/* If middleware returned an error, return it */
	if err != nil {
		return nil, err
	}

	/* Return CompletionResponse directly as per MCP spec */
	return completionResult, nil
}
