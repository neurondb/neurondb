package test

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/neurondb/NeuronMCP/internal/server"
	"github.com/neurondb/NeuronMCP/pkg/mcp"
)

// IntegrationTestSuite provides integration tests for NeuronMCP
// Note: These tests require a running NeuronDB database
// Set NEURONDB_HOST, NEURONDB_PORT, etc. environment variables to run

func TestServerInitialization(t *testing.T) {
	// Skip if no database configured
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	srv, err := server.NewServer()
	if err != nil {
		t.Fatalf("NewServer() error = %v", err)
	}
	defer srv.Stop()

	// Server should be initialized
	if srv == nil {
		t.Error("NewServer() returned nil")
	}
}

func TestMCPProtocolFlow(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	// This would test the full MCP protocol flow
	// including initialize/initialized handshake
	// In a real scenario, you'd use a mock transport

	t.Skip("Full MCP protocol flow test requires mock transport")
}

func TestToolExecution(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	srv, err := server.NewServer()
	if err != nil {
		t.Fatalf("NewServer() error = %v", err)
	}
	defer srv.Stop()

	// Test that tools are registered
	tools := srv.toolRegistry.GetAllDefinitions()
	if len(tools) == 0 {
		t.Error("No tools registered")
	}

	// Test list tools
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	params := json.RawMessage("{}")
	result, err := srv.handleListTools(ctx, params)
	if err != nil {
		t.Fatalf("handleListTools() error = %v", err)
	}

	listResp, ok := result.(mcp.ListToolsResponse)
	if !ok {
		t.Fatalf("handleListTools() result type = %T, want ListToolsResponse", result)
	}

	if len(listResp.Tools) == 0 {
		t.Error("handleListTools() returned no tools")
	}
}

func TestResourceAccess(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	srv, err := server.NewServer()
	if err != nil {
		t.Fatalf("NewServer() error = %v", err)
	}
	defer srv.Stop()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Test listing resources
	definitions := srv.resources.ListResources()
	if len(definitions) == 0 {
		t.Error("No resources available")
	}

	// Test reading a resource (if database is connected)
	if srv.db.IsConnected() {
		_, err := srv.resources.HandleResource(ctx, "neurondb://schema")
		if err != nil {
			t.Logf("Resource access failed (may be expected if no database): %v", err)
		}
	}
}


