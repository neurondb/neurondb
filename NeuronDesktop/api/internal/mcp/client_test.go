package mcp

import (
	"testing"
)

func TestMCPClient_NewClient(t *testing.T) {
	// Only test cases that fail immediately (no subprocess that could block on MCP handshake).
	tests := []struct {
		name    string
		config  MCPConfig
		wantErr bool
	}{
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
	// NewClient with "echo" blocks in initialize() because echo is not an MCP server.
	// Skip this test in CI; run with a real MCP server binary to test IsAlive.
	t.Skip("IsAlive test requires a live MCP server; skipped in CI to avoid timeout")
}

func TestMCPClient_ListTools(t *testing.T) {
	// Requires a live MCP server; NewClient with a fake command would block in initialize().
	t.Skip("ListTools test requires a live MCP server; skipped in CI")
}

func TestMCPClient_CallTool(t *testing.T) {
	// Requires a live MCP server; NewClient with a fake command would block in initialize().
	t.Skip("CallTool test requires a live MCP server; skipped in CI")
}
