package server

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/neurondb/NeuronMCP/pkg/mcp"
)

// setupResourceHandlers sets up resource-related MCP handlers
func (s *Server) setupResourceHandlers() {
	// List resources handler
	s.mcpServer.SetHandler("resources/list", s.handleListResources)

	// Read resource handler
	s.mcpServer.SetHandler("resources/read", s.handleReadResource)
}

// handleListResources handles the resources/list request
func (s *Server) handleListResources(ctx context.Context, params json.RawMessage) (interface{}, error) {
	definitions := s.resources.ListResources()
	
	mcpDefs := make([]mcp.ResourceDefinition, len(definitions))
	for i, def := range definitions {
		mcpDefs[i] = mcp.ResourceDefinition{
			URI:         def.URI,
			Name:        def.Name,
			Description: def.Description,
			MimeType:    def.MimeType,
		}
	}
	
	return mcp.ListResourcesResponse{Resources: mcpDefs}, nil
}

// handleReadResource handles the resources/read request
func (s *Server) handleReadResource(ctx context.Context, params json.RawMessage) (interface{}, error) {
	var req mcp.ReadResourceRequest
	if err := json.Unmarshal(params, &req); err != nil {
		return nil, fmt.Errorf("failed to parse read resource request: %w", err)
	}

	resp, err := s.resources.HandleResource(ctx, req.URI)
	if err != nil {
		return nil, fmt.Errorf("failed to read resource %s: %w", req.URI, err)
	}

	mcpContents := make([]mcp.ResourceContent, len(resp.Contents))
	for i, content := range resp.Contents {
		mcpContents[i] = mcp.ResourceContent{
			URI:      content.URI,
			MimeType: content.MimeType,
			Text:     content.Text,
		}
	}

	return mcp.ReadResourceResponse{Contents: mcpContents}, nil
}

