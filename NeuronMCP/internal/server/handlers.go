/*-------------------------------------------------------------------------
 *
 * handlers.go
 *    MCP request handlers for NeuronMCP server
 *
 * Provides handlers for MCP protocol requests including tools/list,
 * tools/call, resources/list, and resources/read operations.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/server/handlers.go
 *
 *-------------------------------------------------------------------------
 */

package server

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/neurondb/NeuronMCP/internal/middleware"
	"github.com/neurondb/NeuronMCP/internal/tools"
	"github.com/neurondb/NeuronMCP/pkg/mcp"
)

/* min returns the minimum of two integers */
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

/* setupToolHandlers sets up tool-related MCP handlers */
func (s *Server) setupToolHandlers() {
  /* List tools handler */
	s.mcpServer.SetHandler("tools/list", s.handleListTools)

  /* Call tool handler */
	s.mcpServer.SetHandler("tools/call", s.handleCallTool)

  /* Search tools handler */
	s.mcpServer.SetHandler("tools/search", s.handleSearchTools)
}

/* handleListTools handles the tools/list request */
func (s *Server) handleListTools(ctx context.Context, params json.RawMessage) (interface{}, error) {
	if s == nil {
		return nil, fmt.Errorf("server instance is nil")
	}
	if s.toolRegistry == nil {
		return nil, fmt.Errorf("tool registry is not initialized")
	}
	if s.logger == nil {
		return nil, fmt.Errorf("logger is not initialized")
	}

	startTime := time.Now()
	method := "tools/list"
	
	if s.metricsCollector != nil {
		s.metricsCollector.IncrementRequest(method)
		defer func() {
			s.metricsCollector.AddDuration(time.Since(startTime))
		}()
	}
	
	definitions := s.toolRegistry.GetAllDefinitions()
	filtered := s.filterToolsByFeatures(definitions)
	
	/* Log tool count for debugging */
	s.logger.Info("Tools list requested", map[string]interface{}{
		"total_tools":     len(definitions),
		"filtered_tools":  len(filtered),
		"filtered_out":    len(definitions) - len(filtered),
	})
	
	/* Validate and filter out any tools with invalid names or schemas */
	validTools := make([]mcp.ToolDefinition, 0, len(filtered))
	for _, def := range filtered {
		/* Skip tools with empty names */
		if def.Name == "" {
			s.logger.Warn("Skipping tool with empty name", map[string]interface{}{
				"description": def.Description,
			})
			continue
		}
		
		/* Validate tool name format (must be valid identifier) */
		if len(def.Name) > 100 {
			s.logger.Warn("Skipping tool with name too long", map[string]interface{}{
				"tool_name": def.Name,
				"name_length": len(def.Name),
			})
			continue
		}
		
		/* Ensure inputSchema is valid - Claude Desktop requires type: "object" */
		if def.InputSchema == nil {
			s.logger.Warn("Tool has nil inputSchema, using default", map[string]interface{}{
				"tool_name": def.Name,
			})
			def.InputSchema = map[string]interface{}{
				"type":       "object",
				"properties": map[string]interface{}{},
			}
		} else {
			/* Ensure type field is set - required by Claude Desktop */
			if _, hasType := def.InputSchema["type"]; !hasType {
				s.logger.Warn("Tool inputSchema missing type field, adding type: object", map[string]interface{}{
					"tool_name": def.Name,
				})
				/* Create new map to preserve order and add type first */
				newSchema := make(map[string]interface{})
				newSchema["type"] = "object"
				for k, v := range def.InputSchema {
					newSchema[k] = v
				}
				def.InputSchema = newSchema
			} else if def.InputSchema["type"] != "object" {
				/* Log warning if type is not "object" but don't change it (might be intentional) */
				s.logger.Warn("Tool inputSchema has non-object type", map[string]interface{}{
					"tool_name": def.Name,
					"type":      def.InputSchema["type"],
				})
			}
			/* Ensure properties field exists */
			if _, hasProperties := def.InputSchema["properties"]; !hasProperties {
				def.InputSchema["properties"] = map[string]interface{}{}
			}
		}
		
		mcpTool := mcp.ToolDefinition{
			Name:         def.Name,
			Description:  def.Description,
			InputSchema:  def.InputSchema,
			OutputSchema: def.OutputSchema,
			Version:      def.Version,
			Deprecated:   def.Deprecated,
			Deprecation:  def.Deprecation,
		}
		validTools = append(validTools, mcpTool)
	}
	
	s.logger.Info("Tools list response prepared", map[string]interface{}{
		"valid_tools": len(validTools),
		"total_requested": len(filtered),
	})
	
	return mcp.ListToolsResponse{Tools: validTools}, nil
}

/* handleCallTool handles the tools/call request */
func (s *Server) handleCallTool(ctx context.Context, params json.RawMessage) (interface{}, error) {
	if s == nil {
		return nil, fmt.Errorf("server instance is nil")
	}
	if s.toolRegistry == nil {
		return nil, fmt.Errorf("tool registry is not initialized")
	}
	if s.middleware == nil {
		return nil, fmt.Errorf("middleware manager is not initialized")
	}
	if s.logger == nil {
		return nil, fmt.Errorf("logger is not initialized")
	}

	startTime := time.Now()
	method := "tools/call"
	
	/* Track request */
	if s.metricsCollector != nil {
		s.metricsCollector.IncrementRequest(method)
	}
	
	var req mcp.CallToolRequest
	if params == nil || len(params) == 0 {
		return nil, fmt.Errorf("tools/call request parameters are required: received nil or empty params")
	}
	if err := json.Unmarshal(params, &req); err != nil {
		if s.metricsCollector != nil {
			s.metricsCollector.IncrementError(method, "PARSE_ERROR")
			s.metricsCollector.AddDuration(time.Since(startTime))
		}
		return nil, fmt.Errorf("failed to parse tools/call request: params_length=%d, params_preview='%s', error=%w (invalid JSON format or missing required fields)", len(params), string(params[:min(100, len(params))]), err)
	}

	if req.Name == "" {
		return nil, fmt.Errorf("tool name is required in tools/call request: received empty name, params=%v", req)
	}

	mcpReq := &middleware.MCPRequest{
		Method: "tools/call",
		Params: map[string]interface{}{
			"name":           req.Name,
			"arguments":      req.Arguments,
			"dryRun":         req.DryRun,
			"idempotencyKey": req.IdempotencyKey,
			"requireConfirm": req.RequireConfirm,
		},
		Metadata: getHTTPMetadataFromContext(ctx), /* Include HTTP metadata for auth middleware */
	}

	resp, err := s.middleware.Execute(ctx, mcpReq, func(ctx context.Context, _ *middleware.MCPRequest) (*middleware.MCPResponse, error) {
		return s.executeTool(ctx, req.Name, req.Arguments, req.DryRun, req.IdempotencyKey, req.RequireConfirm)
	})
	
	/* Track metrics */
	if s.metricsCollector != nil {
		duration := time.Since(startTime)
		s.metricsCollector.AddDuration(duration)
		s.metricsCollector.IncrementTool(req.Name)
		if err != nil || (resp != nil && resp.IsError) {
			errorType := "UNKNOWN_ERROR"
			if resp != nil && resp.Metadata != nil {
				if code, ok := resp.Metadata["error_code"].(string); ok {
					errorType = code
				}
			}
			s.metricsCollector.IncrementError(method, errorType)
		}
	}
	
	return resp, err
}

/* executeTool executes a tool and returns the response */
func (s *Server) executeTool(ctx context.Context, toolName string, arguments map[string]interface{}, dryRun bool, idempotencyKey string, requireConfirm bool) (*middleware.MCPResponse, error) {
	if s == nil {
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: "Server instance is nil"},
			},
			IsError: true,
		}, nil
	}
	if s.toolRegistry == nil {
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: "Tool registry is not initialized"},
			},
			IsError: true,
		}, nil
	}
	if s.logger == nil {
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: "Logger is not initialized"},
			},
			IsError: true,
		}, nil
	}
	if toolName == "" {
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: fmt.Sprintf("Tool name is required and cannot be empty: received empty string, arguments_count=%d", len(arguments))},
			},
			IsError: true,
		}, nil
	}
	if arguments == nil {
		arguments = make(map[string]interface{})
	}

	tool := s.toolRegistry.GetTool(toolName)
	if tool == nil {
		availableTools := s.toolRegistry.GetAllDefinitions()
		toolNames := make([]string, 0, len(availableTools))
		for _, def := range availableTools {
			toolNames = append(toolNames, def.Name)
		}
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: fmt.Sprintf("Tool not found: tool_name='%s', arguments_count=%d, available_tools_count=%d, available_tools=%v", toolName, len(arguments), len(availableTools), toolNames)},
			},
			IsError: true,
		}, nil
	}

  /* Log tool execution start */
	s.logger.Info("Executing tool", map[string]interface{}{
		"tool_name":       toolName,
		"arguments_count": len(arguments),
		"dry_run":         dryRun,
		"idempotency_key": idempotencyKey,
		"require_confirm": requireConfirm,
	})

	/* Handle dry run mode */
	if dryRun {
		dryRunExecutor := tools.NewDryRunExecutor(tool)
		dryRunResult, err := dryRunExecutor.Execute(ctx, arguments)
		if err != nil {
			return s.formatToolError(dryRunResult), nil
		}
		resultJSON, _ := json.MarshalIndent(dryRunResult.Data, "", "  ")
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: string(resultJSON)},
			},
			Metadata: map[string]interface{}{
				"dryRun": true,
				"tool":   toolName,
			},
		}, nil
	}

	/* Check if confirmation is required */
	if requireConfirm && tools.RequiresConfirmation(toolName) {
		/* Check if confirmation is provided in arguments */
		if confirmed, ok := arguments["confirmed"].(bool); !ok || !confirmed {
			return &middleware.MCPResponse{
				Content: []middleware.ContentBlock{
					{Type: "text", Text: fmt.Sprintf("Confirmation required for tool '%s' - set 'confirmed' parameter to true", toolName)},
				},
				IsError: true,
				Metadata: map[string]interface{}{
					"error_code": "CONFIRMATION_REQUIRED",
					"tool":       toolName,
				},
			}, nil
		}
	}

	/* Handle idempotency - check cache for existing result */
	if idempotencyKey != "" {
		if s.idempotencyCache == nil {
			s.logger.Warn("Idempotency cache is not initialized, skipping cache check", map[string]interface{}{
				"idempotency_key": idempotencyKey,
				"tool_name":       toolName,
			})
		} else {
			s.logger.Debug(fmt.Sprintf("Tool execution with idempotency key: %s", idempotencyKey), nil)
			
			/* Check if we have a cached result for this idempotency key */
			if cachedResult, found := s.idempotencyCache.Get(idempotencyKey); found {
			s.logger.Info("Returning cached result for idempotency key", map[string]interface{}{
				"idempotency_key": idempotencyKey,
				"tool_name":       toolName,
			})
			
			/* Convert cached mcp.ToolResult to middleware.MCPResponse */
			content := []middleware.ContentBlock{}
			if cachedResult.Content != nil {
				for _, c := range cachedResult.Content {
					content = append(content, middleware.ContentBlock{
						Type: c.Type,
						Text: c.Text,
					})
				}
			}
			
			return &middleware.MCPResponse{
				Content:  content,
				IsError:  cachedResult.IsError,
				Metadata: map[string]interface{}{
					"cached":        true,
					"idempotency_key": idempotencyKey,
					"tool":          toolName,
				},
			}, nil
			}
		}
	}

	result, err := tool.Execute(ctx, arguments)
	if err != nil {
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: fmt.Sprintf("Tool execution error: tool_name='%s', arguments_count=%d, arguments=%v, error=%v", toolName, len(arguments), arguments, err)},
			},
			IsError: true,
		}, nil
	}

	/* Format the result */
	response, formatErr := s.formatToolResult(result)
	if formatErr != nil {
		return response, formatErr
	}
	
	/* Cache the result if idempotency key is provided */
	if idempotencyKey != "" && s.idempotencyCache != nil {
		/* Convert middleware.MCPResponse to mcp.ToolResult for caching */
		cachedResult := &mcp.ToolResult{
			Content: make([]mcp.ContentBlock, len(response.Content)),
			IsError: response.IsError,
		}
		for i, c := range response.Content {
			cachedResult.Content[i] = mcp.ContentBlock{
				Type: c.Type,
				Text: c.Text,
			}
		}
		
		s.idempotencyCache.Set(idempotencyKey, cachedResult)
		s.logger.Debug("Cached result for idempotency key", map[string]interface{}{
			"idempotency_key": idempotencyKey,
			"tool_name":       toolName,
		})
	}

	return response, nil
}

/* formatToolResult formats a tool result as an MCP response */
func (s *Server) formatToolResult(result *tools.ToolResult) (*middleware.MCPResponse, error) {
	if result == nil {
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: "Tool result is nil"},
			},
			IsError: true,
		}, nil
	}
	if !result.Success {
		return s.formatToolError(result), nil
	}

	/* Validate output against schema if tool has output schema */
	/* Note: Basic validation is performed by the tool itself. Full schema validation
	 * against the tool's OutputSchema would require additional JSON schema validation.
	 * This is a future enhancement for strict output validation.
	 */
	
	resultJSON, _ := json.MarshalIndent(result.Data, "", "  ")
	return &middleware.MCPResponse{
		Content: []middleware.ContentBlock{
			{Type: "text", Text: string(resultJSON)},
		},
		Metadata: result.Metadata,
	}, nil
}

/* formatToolError formats a tool error as an MCP response */
func (s *Server) formatToolError(result *tools.ToolResult) *middleware.MCPResponse {
	errorText := "Unknown error"
	errorMetadata := make(map[string]interface{})
	
	if result.Error != nil {
		errorText = result.Error.Message
		errorMetadata["message"] = result.Error.Message
		if result.Error.Code != "" {
			errorMetadata["code"] = result.Error.Code
		}
		if result.Error.Details != nil {
			errorMetadata["details"] = result.Error.Details
		}
	}
	
	return &middleware.MCPResponse{
		Content: []middleware.ContentBlock{
			{Type: "text", Text: fmt.Sprintf("Error: %s", errorText)},
		},
		IsError: true,
		Metadata: errorMetadata,
	}
}

/* handleSearchTools handles the tools/search request */
func (s *Server) handleSearchTools(ctx context.Context, params json.RawMessage) (interface{}, error) {
	if s == nil {
		return nil, fmt.Errorf("server instance is nil")
	}
	if s.toolRegistry == nil {
		return nil, fmt.Errorf("tool registry is not initialized")
	}

	var req struct {
		Query    string `json:"query,omitempty"`
		Category string `json:"category,omitempty"`
	}
	if params != nil && len(params) > 0 {
		if err := json.Unmarshal(params, &req); err != nil {
			return nil, fmt.Errorf("failed to parse tools/search request: %w", err)
		}
	}

	definitions := s.toolRegistry.Search(req.Query, req.Category)
	
	mcpTools := make([]mcp.ToolDefinition, len(definitions))
	for i, def := range definitions {
		mcpTools[i] = mcp.ToolDefinition{
			Name:         def.Name,
			Description:  def.Description,
			InputSchema:  def.InputSchema,
			OutputSchema: def.OutputSchema,
			Version:      def.Version,
			Deprecated:   def.Deprecated,
			Deprecation:  def.Deprecation,
		}
	}
	
	return mcp.ListToolsResponse{Tools: mcpTools}, nil
}

