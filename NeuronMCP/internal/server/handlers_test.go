package server

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/neurondb/NeuronMCP/internal/config"
	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/tools"
	"github.com/neurondb/NeuronMCP/pkg/mcp"
)

func TestServerSetup(t *testing.T) {
	cfgMgr := config.NewConfigManager()
	_, err := cfgMgr.Load("")
	if err != nil {
		t.Fatalf("Failed to load config: %v", err)
	}

	logger := logging.NewLogger(cfgMgr.GetLoggingConfig())
	if logger == nil {
		t.Fatal("Failed to create logger")
	}

	db := database.NewDatabase()
	if db == nil {
		t.Fatal("Failed to create database")
	}

	toolRegistry := tools.NewToolRegistry(db, logger)
	if toolRegistry == nil {
		t.Fatal("Failed to create tool registry")
	}

	// This should not panic or crash
	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("RegisterAllTools panicked: %v", r)
			}
		}()
		tools.RegisterAllTools(toolRegistry, db, logger)
	}()

	// Verify tools are registered
	definitions := toolRegistry.GetAllDefinitions()
	if len(definitions) == 0 {
		t.Fatal("No tools registered - this indicates a real problem")
	}

	// Check that we have expected tools
	toolNames := make(map[string]bool)
	for _, def := range definitions {
		if def.Name == "" {
			t.Error("Tool definition has empty name")
		}
		toolNames[def.Name] = true
	}

	expectedTools := []string{"vector_search", "train_model", "predict", "cluster_data"}
	for _, expected := range expectedTools {
		if !toolNames[expected] {
			t.Errorf("Missing expected tool: %s", expected)
		}
	}
}

func TestToolRegistry(t *testing.T) {
	cfgMgr := config.NewConfigManager()
	_, err := cfgMgr.Load("")
	if err != nil {
		t.Fatalf("Failed to load config: %v", err)
	}

	logger := logging.NewLogger(cfgMgr.GetLoggingConfig())
	if logger == nil {
		t.Fatal("Failed to create logger")
	}

	db := database.NewDatabase()
	if db == nil {
		t.Fatal("Failed to create database")
	}

	toolRegistry := tools.NewToolRegistry(db, logger)
	if toolRegistry == nil {
		t.Fatal("Failed to create tool registry")
	}

	// This should not panic
	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("RegisterAllTools panicked: %v", r)
			}
		}()
		tools.RegisterAllTools(toolRegistry, db, logger)
	}()

	// Test that nonexistent tool returns nil
	tool := toolRegistry.GetTool("nonexistent_tool")
	if tool != nil {
		t.Error("GetTool() should return nil for nonexistent tool")
	}

	// Test that existing tool is found
	tool = toolRegistry.GetTool("vector_search")
	if tool == nil {
		t.Fatal("GetTool() should return tool for 'vector_search'")
	}

	// Test with empty string - should not crash
	tool = toolRegistry.GetTool("")
	if tool != nil {
		t.Error("GetTool() should return nil for empty string")
	}

	// Test with nil-like behavior - should not crash
	tool = toolRegistry.GetTool("\x00")
	if tool != nil {
		t.Error("GetTool() should return nil for invalid tool name")
	}
}

func TestHandleListTools(t *testing.T) {
	cfgMgr := config.NewConfigManager()
	_, err := cfgMgr.Load("")
	if err != nil {
		t.Fatalf("Failed to load config: %v", err)
	}

	logger := logging.NewLogger(cfgMgr.GetLoggingConfig())
	db := database.NewDatabase()
	toolRegistry := tools.NewToolRegistry(db, logger)
	tools.RegisterAllTools(toolRegistry, db, logger)

	srv, err := NewServer()
	if err != nil {
		t.Fatalf("Failed to create server: %v", err)
	}
	defer srv.Stop()

	ctx := context.Background()

	// Test with valid empty params
	params := json.RawMessage("{}")
	result, err := srv.handleListTools(ctx, params)
	if err != nil {
		t.Fatalf("handleListTools() returned error: %v", err)
	}

	listResp, ok := result.(mcp.ListToolsResponse)
	if !ok {
		t.Fatalf("handleListTools() returned wrong type: %T", result)
	}

	if len(listResp.Tools) == 0 {
		t.Error("handleListTools() returned no tools")
	}

	// Test with invalid JSON - should return error, not crash
	invalidParams := json.RawMessage("{invalid json}")
	result, err = srv.handleListTools(ctx, invalidParams)
	// Note: handleListTools doesn't parse params, so it might not error
	// But it should not crash
	if err != nil {
		t.Logf("handleListTools() with invalid JSON returned error (expected): %v", err)
	}

	// Test with nil params - should not crash
	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("handleListTools panicked with nil params: %v", r)
			}
		}()
		_, _ = srv.handleListTools(ctx, nil)
	}()
}

func TestHandleCallTool_ErrorConditions(t *testing.T) {
	cfgMgr := config.NewConfigManager()
	_, err := cfgMgr.Load("")
	if err != nil {
		t.Fatalf("Failed to load config: %v", err)
	}

	srv, err := NewServer()
	if err != nil {
		t.Fatalf("Failed to create server: %v", err)
	}
	defer srv.Stop()

	ctx := context.Background()

	// Test with invalid JSON - should return error
	invalidParams := json.RawMessage("{invalid json}")
	_, err = srv.handleCallTool(ctx, invalidParams)
	if err == nil {
		t.Error("handleCallTool() should return error for invalid JSON")
	}

	// Test with missing tool name - should return error
	missingNameParams := json.RawMessage(`{"arguments": {}}`)
	_, err = srv.handleCallTool(ctx, missingNameParams)
	if err == nil {
		t.Error("handleCallTool() should return error for missing tool name")
	}

	// Test with empty tool name - should return error
	emptyNameParams := json.RawMessage(`{"name": "", "arguments": {}}`)
	_, err = srv.handleCallTool(ctx, emptyNameParams)
	if err == nil {
		t.Error("handleCallTool() should return error for empty tool name")
	}

	// Test with nonexistent tool - should return error response, not crash
	nonexistentParams := json.RawMessage(`{"name": "nonexistent_tool_xyz", "arguments": {}}`)
	result, err := srv.handleCallTool(ctx, nonexistentParams)
	if err != nil {
		t.Fatalf("handleCallTool() should not return error for nonexistent tool, but return error response: %v", err)
	}
	// Should return an error response, not nil
	if result == nil {
		t.Error("handleCallTool() should return error response for nonexistent tool, not nil")
	}

	// Test with nil params - should not crash
	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("handleCallTool panicked with nil params: %v", r)
			}
		}()
		_, _ = srv.handleCallTool(ctx, nil)
	}()

	// Test with malformed arguments - should handle gracefully
	malformedParams := json.RawMessage(`{"name": "vector_search", "arguments": "not an object"}`)
	_, err = srv.handleCallTool(ctx, malformedParams)
	// May or may not error depending on JSON parsing, but should not crash
	if err != nil {
		t.Logf("handleCallTool() with malformed arguments returned error: %v", err)
	}
}

func TestExecuteTool_ErrorConditions(t *testing.T) {
	cfgMgr := config.NewConfigManager()
	_, err := cfgMgr.Load("")
	if err != nil {
		t.Fatalf("Failed to load config: %v", err)
	}

	srv, err := NewServer()
	if err != nil {
		t.Fatalf("Failed to create server: %v", err)
	}
	defer srv.Stop()

	ctx := context.Background()

	// Test with empty tool name - should return error response
	resp, err := srv.executeTool(ctx, "", map[string]interface{}{})
	if err != nil {
		t.Fatalf("executeTool() should not return error for empty name, but error response: %v", err)
	}
	if resp == nil {
		t.Fatal("executeTool() should return error response for empty name")
	}
	if !resp.IsError {
		t.Error("executeTool() should return error response for empty name")
	}

	// Test with nonexistent tool - should return error response
	resp, err = srv.executeTool(ctx, "nonexistent_tool_abc", map[string]interface{}{})
	if err != nil {
		t.Fatalf("executeTool() should not return error for nonexistent tool: %v", err)
	}
	if resp == nil {
		t.Fatal("executeTool() should return error response for nonexistent tool")
	}
	if !resp.IsError {
		t.Error("executeTool() should return error response for nonexistent tool")
	}

	// Test with nil arguments - should not crash
	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("executeTool panicked with nil arguments: %v", r)
			}
		}()
		_, _ = srv.executeTool(ctx, "vector_search", nil)
	}()

	// Test with invalid tool name containing null bytes - should not crash
	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("executeTool panicked with invalid tool name: %v", r)
			}
		}()
		_, _ = srv.executeTool(ctx, "vector_search\x00", map[string]interface{}{})
	}()
}

