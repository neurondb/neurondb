/*-------------------------------------------------------------------------
 *
 * resource_handlers.go
 *    Database operations
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/server/resource_handlers.go
 *
 *-------------------------------------------------------------------------
 */

package server

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/neurondb/NeuronMCP/internal/resources"
	"github.com/neurondb/NeuronMCP/pkg/mcp"
)

/* setupResourceHandlers sets up resource-related MCP handlers */
func (s *Server) setupResourceHandlers() {
  /* List resources handler */
	s.mcpServer.SetHandler("resources/list", s.handleListResources)

  /* Read resource handler */
	s.mcpServer.SetHandler("resources/read", s.handleReadResource)
}

/* handleListResources handles the resources/list request */
func (s *Server) handleListResources(ctx context.Context, params json.RawMessage) (interface{}, error) {
	if s == nil {
		return nil, fmt.Errorf("server instance is nil")
	}
	if s.resources == nil {
		return nil, fmt.Errorf("resources manager is not initialized")
	}
	if s.logger == nil {
		return nil, fmt.Errorf("logger is not initialized")
	}

	startTime := time.Now()
	method := "resources/list"

	/* Track request */
	if s.metricsCollector != nil {
		s.metricsCollector.IncrementRequest(method)
		defer func() {
			s.metricsCollector.AddDuration(time.Since(startTime))
		}()
	}

	definitions := s.resources.ListResources()
	
	mcpDefs := make([]mcp.ResourceDefinition, 0, len(definitions))
	for i, def := range definitions {
		/* Validate resource definition */
		if def.URI == "" {
			if s.logger != nil {
				s.logger.Warn("Skipping resource with empty URI", map[string]interface{}{
					"index":       i,
					"name":        def.Name,
					"description": def.Description,
				})
			}
			continue
		}
		if def.Name == "" {
			if s.logger != nil {
				s.logger.Warn("Skipping resource with empty name", map[string]interface{}{
					"index": i,
					"uri":   def.URI,
				})
			}
			continue
		}
		
		mcpDefs = append(mcpDefs, mcp.ResourceDefinition{
			URI:         def.URI,
			Name:        def.Name,
			Description: def.Description,
			MimeType:    def.MimeType,
		})
	}

	if s.logger != nil {
		s.logger.Debug("Resources list requested", map[string]interface{}{
			"total_resources": len(mcpDefs),
		})
	}
	
	return mcp.ListResourcesResponse{Resources: mcpDefs}, nil
}

/* handleReadResource handles the resources/read request */
func (s *Server) handleReadResource(ctx context.Context, params json.RawMessage) (interface{}, error) {
	if s == nil {
		return nil, fmt.Errorf("server instance is nil")
	}
	if s.resources == nil {
		return nil, fmt.Errorf("resources manager is not initialized")
	}
	if s.logger == nil {
		return nil, fmt.Errorf("logger is not initialized")
	}

	startTime := time.Now()
	method := "resources/read"

	/* Track request */
	if s.metricsCollector != nil {
		s.metricsCollector.IncrementRequest(method)
		defer func() {
			s.metricsCollector.AddDuration(time.Since(startTime))
		}()
	}

	if params == nil || len(params) == 0 {
		if s.metricsCollector != nil {
			s.metricsCollector.IncrementError(method, "PARSE_ERROR")
		}
		return nil, fmt.Errorf("resources/read request parameters are required: received nil or empty params")
	}

	var req mcp.ReadResourceRequest
	if err := json.Unmarshal(params, &req); err != nil {
		if s.metricsCollector != nil {
			s.metricsCollector.IncrementError(method, "PARSE_ERROR")
		}
		return nil, fmt.Errorf("resources/read: failed to parse request parameters: params_length=%d, params_preview='%s', error=%w (invalid JSON format or missing required fields)", len(params), string(params[:min(100, len(params))]), err)
	}

	/* Validate URI */
	if req.URI == "" {
		if s.metricsCollector != nil {
			s.metricsCollector.IncrementError(method, "VALIDATION_ERROR")
		}
		return nil, fmt.Errorf("resources/read: URI parameter is required and cannot be empty: received empty URI in request")
	}

	/* Validate URI format (basic validation) */
	if len(req.URI) > 1000 {
		if s.metricsCollector != nil {
			s.metricsCollector.IncrementError(method, "VALIDATION_ERROR")
		}
		return nil, fmt.Errorf("resources/read: URI parameter too long: uri_length=%d, max_length=1000, uri_preview='%s'", len(req.URI), req.URI[:min(100, len(req.URI))])
	}

	/* URI should not contain control characters */
	if strings.ContainsAny(req.URI, "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x7f") {
		if s.metricsCollector != nil {
			s.metricsCollector.IncrementError(method, "VALIDATION_ERROR")
		}
		return nil, fmt.Errorf("resources/read: URI parameter contains invalid control characters: uri='%s'", req.URI)
	}

	resp, err := s.resources.HandleResource(ctx, req.URI)
	if err != nil {
		errorType := "UNKNOWN_ERROR"
		if _, ok := err.(*resources.ResourceNotFoundError); ok {
			errorType = "RESOURCE_NOT_FOUND"
		}
		if s.metricsCollector != nil {
			s.metricsCollector.IncrementError(method, errorType)
		}
		return nil, fmt.Errorf("resources/read: failed to read resource: uri='%s', error=%w", req.URI, err)
	}

	if resp == nil {
		if s.metricsCollector != nil {
			s.metricsCollector.IncrementError(method, "UNKNOWN_ERROR")
		}
		return nil, fmt.Errorf("resources/read: resource handler returned nil response: uri='%s'", req.URI)
	}

	mcpContents := make([]mcp.ResourceContent, 0, len(resp.Contents))
	for i, content := range resp.Contents {
		if content.URI == "" {
			if s.logger != nil {
				s.logger.Warn("Skipping resource content with empty URI", map[string]interface{}{
					"index": i,
					"mime_type": content.MimeType,
				})
			}
			continue
		}
		mcpContents = append(mcpContents, mcp.ResourceContent{
			URI:      content.URI,
			MimeType: content.MimeType,
			Text:     content.Text,
		})
	}

	if len(mcpContents) == 0 {
		if s.metricsCollector != nil {
			s.metricsCollector.IncrementError(method, "EMPTY_RESPONSE")
		}
		return nil, fmt.Errorf("resources/read: resource returned no content: uri='%s'", req.URI)
	}

	if s.logger != nil {
		s.logger.Debug("Resource read successful", map[string]interface{}{
			"uri":            req.URI,
			"content_count":  len(mcpContents),
		})
	}

	return mcp.ReadResourceResponse{Contents: mcpContents}, nil
}

