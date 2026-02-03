package mcp

import (
	"testing"
)

func TestMCPClient_NewClient(t *testing.T) {
	tests := []struct {
		name    string
		config  MCPConfig
		wantErr bool
	}{
		{
			name: "valid echo command",
			config: MCPConfig{
				Command: "echo",
				Args:    []string{"test"},
			},
			wantErr: true, // Echo is not an MCP server, so initialization will fail
		},
		{
			name: "invalid command",
			config: MCPConfig{
				Command: "nonexistent-command-xyz-123",
				Args:    []string{},
			},
			wantErr: true,
		},
		{
			name: "empty command",
			config: MCPConfig{
				Command: "",
				Args:    []string{},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, err := NewClient(tt.config)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewClient() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if client != nil {
				defer client.Close()
			}
		})
	}
}

func TestMCPClient_Close(t *testing.T) {
	// Test closing a client that was never created
	t.Run("close nil client", func(t *testing.T) {
		var client *Client
		// Should not panic
		if client != nil {
			client.Close()
		}
	})
}

func TestMCPClient_IsAlive(t *testing.T) {
	config := MCPConfig{
		Command: "echo",
		Args:    []string{"test"},
	}

	client, err := NewClient(config)
	if err == nil {
		defer client.Close()

		// Client should be alive if created successfully
		if !client.IsAlive() {
			t.Error("Expected client to be alive")
		}

		// Close and check again
		client.Close()
		// After close, IsAlive should return false
		if client.IsAlive() {
			t.Error("Expected client to be dead after close")
		}
	}
}

func TestMCPClient_ListTools(t *testing.T) {
	// This test requires an actual MCP server
	// For now, we'll test error handling
	config := MCPConfig{
		Command: "echo",
		Args:    []string{"test"},
	}

	client, err := NewClient(config)
	if err == nil {
		defer client.Close()

		_, err := client.ListTools()
		// Should fail because echo is not an MCP server
		if err == nil {
			t.Error("Expected error when listing tools from non-MCP server")
		}
	}
}

func TestMCPClient_CallTool(t *testing.T) {
	// This test requires an actual MCP server
	// For now, we'll test error handling
	config := MCPConfig{
		Command: "echo",
		Args:    []string{"test"},
	}

	client, err := NewClient(config)
	if err == nil {
		defer client.Close()

		_, err := client.CallTool("test_tool", map[string]interface{}{})
		// Should fail because echo is not an MCP server
		if err == nil {
			t.Error("Expected error when calling tool on non-MCP server")
		}
	}
}
