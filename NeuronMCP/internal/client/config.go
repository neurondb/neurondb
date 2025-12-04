package client

import (
	"encoding/json"
	"fmt"
	"os"
)

// LoadConfig loads MCP configuration from Claude Desktop format
func LoadConfig(configPath, serverName string) (*MCPConfig, error) {
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("configuration file not found: %w", err)
	}

	var configData map[string]interface{}
	if err := json.Unmarshal(data, &configData); err != nil {
		return nil, fmt.Errorf("failed to parse configuration: %w", err)
	}

	// Claude Desktop format: { "mcpServers": { "server_name": { ... } } }
	mcpServers, ok := configData["mcpServers"].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid configuration format: missing 'mcpServers'")
	}

	serverConfig, ok := mcpServers[serverName].(map[string]interface{})
	if !ok {
		// List available servers
		var available []string
		for k := range mcpServers {
			available = append(available, k)
		}
		return nil, fmt.Errorf("server '%s' not found in configuration. Available servers: %v", serverName, available)
	}

	// Extract command
	command, ok := serverConfig["command"].(string)
	if !ok {
		return nil, fmt.Errorf("server '%s' missing 'command' field", serverName)
	}

	// Extract environment variables
	env := make(map[string]string)
	if envMap, ok := serverConfig["env"].(map[string]interface{}); ok {
		for k, v := range envMap {
			if str, ok := v.(string); ok {
				env[k] = str
			}
		}
	}

	// Extract arguments (if any)
	var args []string
	if argsList, ok := serverConfig["args"].([]interface{}); ok {
		for _, arg := range argsList {
			if str, ok := arg.(string); ok {
				args = append(args, str)
			}
		}
	}

	return &MCPConfig{
		Command: command,
		Env:     env,
		Args:    args,
	}, nil
}

