/*-------------------------------------------------------------------------
 *
 * test_helpers.go
 *    Test helper utilities
 *
 * Provides common utilities for testing NeuronMCP functionality.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/test/test_helpers.go
 *
 *-------------------------------------------------------------------------
 */

package test

import (
	"context"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/server"
	"github.com/neurondb/NeuronMCP/internal/tools"
	"github.com/neurondb/NeuronMCP/pkg/mcp"
)

/* TestServer wraps a server for testing */
type TestServer struct {
	Server       *server.Server
	ToolRegistry *tools.ToolRegistry
	DB           *database.Database
}

/* NewTestServer creates a new test server */
func NewTestServer() (*TestServer, error) {
	s, err := server.NewServer()
	if err != nil {
		return nil, fmt.Errorf("failed to create test server: %w", err)
	}

	/* Get tool registry and database from server using getter methods */
	return &TestServer{
		Server:       s,
		ToolRegistry: s.GetToolRegistry(),
		DB:           s.GetDatabase(),
	}, nil
}

/* CallTool calls a tool via the server */
func (ts *TestServer) CallTool(ctx context.Context, toolName string, arguments map[string]interface{}) (*mcp.ToolResult, error) {
	if ts.Server == nil {
		return nil, fmt.Errorf("server is nil")
	}
	
	/* Get tool registry from server */
	if ts.ToolRegistry == nil {
		ts.ToolRegistry = ts.Server.GetToolRegistry()
		if ts.ToolRegistry == nil {
			return nil, fmt.Errorf("tool registry not available")
		}
	}
	
	/* Get the tool from registry */
	tool := ts.ToolRegistry.GetTool(toolName)
	if tool == nil {
		return nil, fmt.Errorf("tool not found: %s", toolName)
	}
	
	/* Execute the tool */
	result, err := tool.Execute(ctx, arguments)
	if err != nil {
		return nil, fmt.Errorf("tool execution failed: %w", err)
	}
	
	/* Convert tools.ToolResult to mcp.ToolResult */
	mcpResult := &mcp.ToolResult{
		Content: make([]mcp.ContentBlock, 0),
		IsError: !result.Success,
	}
	
	/* Convert result data to JSON content */
	if result.Data != nil {
		/* For now, we'll create a simple text content block */
		/* In a full implementation, we'd properly serialize the data */
		contentJSON := fmt.Sprintf("%v", result.Data)
		mcpResult.Content = append(mcpResult.Content, mcp.ContentBlock{
			Type: "text",
			Text: contentJSON,
		})
	}
	
	return mcpResult, nil
}

/* ListTools lists all available tools */
func (ts *TestServer) ListTools(ctx context.Context) ([]mcp.ToolDefinition, error) {
	if ts.Server == nil {
		return nil, fmt.Errorf("server is nil")
	}
	
	/* Get tool definitions from registry */
	if ts.ToolRegistry == nil {
		ts.ToolRegistry = ts.Server.GetToolRegistry()
		if ts.ToolRegistry == nil {
			return nil, fmt.Errorf("tool registry not available")
		}
	}
	
	definitions := ts.ToolRegistry.GetAllDefinitions()
	
	/* Convert to mcp.ToolDefinition */
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
	
	return mcpTools, nil
}

/* ValidateToolOutput validates tool output against its schema */
func ValidateToolOutput(tool tools.Tool, output interface{}) error {
	schema := tool.OutputSchema()
	if schema == nil {
		return nil // No schema to validate against
	}

	valid, errors := tools.ValidateOutput(output, schema)
	if !valid {
		return fmt.Errorf("output validation failed: %v", errors)
	}

	return nil
}

/* AssertToolVersion asserts that a tool has the expected version */
func AssertToolVersion(tool tools.Tool, expectedVersion string) error {
	actualVersion := tool.Version()
	if actualVersion != expectedVersion {
		return fmt.Errorf("tool version mismatch: expected %s, got %s", expectedVersion, actualVersion)
	}
	return nil
}

