package client

import (
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/neurondb/NeuronMCP/pkg/mcp"
)

// MCPConfig represents the configuration for an MCP server
type MCPConfig struct {
	Command string
	Env     map[string]string
	Args    []string
}

// GetEnv returns environment variables, merging with current environment
func (c *MCPConfig) GetEnv() map[string]string {
	env := make(map[string]string)
	// Copy current environment
	for _, e := range os.Environ() {
		// Split on first '='
		for i := 0; i < len(e); i++ {
			if e[i] == '=' {
				env[e[:i]] = e[i+1:]
				break
			}
		}
	}
	// Override with config env
	for k, v := range c.Env {
		env[k] = v
	}
	return env
}

// MCPClient is a client for communicating with MCP servers
type MCPClient struct {
	config      *MCPConfig
	verbose     bool
	transport   *ClientTransport
	initialized bool
}

// NewMCPClient creates a new MCP client
func NewMCPClient(config *MCPConfig, verbose bool) (*MCPClient, error) {
	return &MCPClient{
		config:  config,
		verbose: verbose,
	}, nil
}

// Connect connects to the MCP server
func (c *MCPClient) Connect() error {
	if c.transport != nil {
		return fmt.Errorf("already connected")
	}

	// Create transport
	transport, err := NewClientTransport(c.config.Command, c.config.GetEnv(), c.config.Args)
	if err != nil {
		return fmt.Errorf("failed to create transport: %w", err)
	}

	c.transport = transport

	// Start server process
	if c.verbose {
		fmt.Printf("Starting MCP server: %s\n", c.config.Command)
	}

	if err := c.transport.Start(); err != nil {
		return fmt.Errorf("failed to start transport: %w", err)
	}

	// Initialize connection
	return c.initialize()
}

// initialize initializes the MCP connection
func (c *MCPClient) initialize() error {
	if c.initialized {
		return nil
	}

	// Send initialize request
	initRequest := &mcp.JSONRPCRequest{
		JSONRPC: "2.0",
		ID:      json.RawMessage(`"` + generateID() + `"`),
		Method:  "initialize",
		Params: json.RawMessage(`{
			"protocolVersion": "2025-06-18",
			"capabilities": {},
			"clientInfo": {
				"name": "neurondb-mcp-client",
				"version": "1.0.0"
			}
		}`),
	}

	response, err := c.transport.SendRequest(initRequest)
	if err != nil {
		return fmt.Errorf("initialize failed: %w", err)
	}

	if response.Error != nil {
		return fmt.Errorf("initialize error: %s (code: %d)", response.Error.Message, response.Error.Code)
	}

	if c.verbose {
		fmt.Println("MCP connection initialized")
	}

	// Send initialized notification
	notification := map[string]interface{}{
		"jsonrpc": "2.0",
		"method":  "notifications/initialized",
		"params":  map[string]interface{}{},
	}

	if err := c.transport.SendNotification(notification); err != nil {
		// Notification errors are not fatal
		if c.verbose {
			fmt.Printf("Warning: failed to send initialized notification: %v\n", err)
		}
	}

	c.initialized = true
	return nil
}

// Disconnect disconnects from the MCP server
func (c *MCPClient) Disconnect() {
	if c.transport != nil {
		if c.verbose {
			fmt.Println("Disconnecting from MCP server")
		}
		c.transport.Stop()
		c.transport = nil
		c.initialized = false
	}
}

// ListTools lists available tools
func (c *MCPClient) ListTools() (map[string]interface{}, error) {
	request := &mcp.JSONRPCRequest{
		JSONRPC: "2.0",
		ID:      json.RawMessage(`"` + generateID() + `"`),
		Method:  "tools/list",
		Params:  json.RawMessage("{}"),
	}

	response, err := c.transport.SendRequest(request)
	if err != nil {
		return nil, err
	}

	if response.Error != nil {
		return map[string]interface{}{
			"error": fmt.Sprintf("%s (code: %d)", response.Error.Message, response.Error.Code),
		}, nil
	}

	if response.Result != nil {
		if resultMap, ok := response.Result.(map[string]interface{}); ok {
			return resultMap, nil
		}
		// If result is not a map, wrap it
		return map[string]interface{}{
			"result": response.Result,
		}, nil
	}

	return map[string]interface{}{}, nil
}

// CallTool calls a tool with the given name and arguments
func (c *MCPClient) CallTool(toolName string, arguments map[string]interface{}) (map[string]interface{}, error) {
	params := map[string]interface{}{
		"name":      toolName,
		"arguments": arguments,
	}

	paramsJSON, err := json.Marshal(params)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal params: %w", err)
	}

	request := &mcp.JSONRPCRequest{
		JSONRPC: "2.0",
		ID:      json.RawMessage(`"` + generateID() + `"`),
		Method:  "tools/call",
		Params:  json.RawMessage(paramsJSON),
	}

	response, err := c.transport.SendRequest(request)
	if err != nil {
		return nil, err
	}

	if response.Error != nil {
		return map[string]interface{}{
			"error": fmt.Sprintf("%s (code: %d)", response.Error.Message, response.Error.Code),
		}, nil
	}

	if response.Result != nil {
		if resultMap, ok := response.Result.(map[string]interface{}); ok {
			return resultMap, nil
		}
		// If result is not a map, wrap it
		return map[string]interface{}{
			"result": response.Result,
		}, nil
	}

	return map[string]interface{}{}, nil
}

// ExecuteCommand executes a command string
func (c *MCPClient) ExecuteCommand(commandStr string) (map[string]interface{}, error) {
	// Parse command
	toolName, arguments, err := ParseCommand(commandStr)
	if err != nil {
		return map[string]interface{}{
			"error": fmt.Sprintf("Failed to parse command: %v", err),
		}, nil
	}

	// Handle special commands
	if toolName == "list_tools" {
		return c.ListTools()
	}

	if toolName == "resources/list" {
		return c.ListResources()
	}

	if toolName == "resources/read" {
		uri, ok := arguments["uri"].(string)
		if !ok {
			return map[string]interface{}{
				"error": "Missing 'uri' parameter for resources/read",
			}, nil
		}
		return c.ReadResource(uri)
	}

	// Call tool
	return c.CallTool(toolName, arguments)
}

// ListResources lists available resources
func (c *MCPClient) ListResources() (map[string]interface{}, error) {
	request := &mcp.JSONRPCRequest{
		JSONRPC: "2.0",
		ID:      json.RawMessage(`"` + generateID() + `"`),
		Method:  "resources/list",
		Params:  json.RawMessage("{}"),
	}

	response, err := c.transport.SendRequest(request)
	if err != nil {
		return nil, err
	}

	if response.Error != nil {
		return map[string]interface{}{
			"error": fmt.Sprintf("%s (code: %d)", response.Error.Message, response.Error.Code),
		}, nil
	}

	if response.Result != nil {
		if resultMap, ok := response.Result.(map[string]interface{}); ok {
			return resultMap, nil
		}
		return map[string]interface{}{
			"result": response.Result,
		}, nil
	}

	return map[string]interface{}{}, nil
}

// ReadResource reads a resource by URI
func (c *MCPClient) ReadResource(uri string) (map[string]interface{}, error) {
	params := map[string]interface{}{
		"uri": uri,
	}

	paramsJSON, err := json.Marshal(params)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal params: %w", err)
	}

	request := &mcp.JSONRPCRequest{
		JSONRPC: "2.0",
		ID:      json.RawMessage(`"` + generateID() + `"`),
		Method:  "resources/read",
		Params:  json.RawMessage(paramsJSON),
	}

	response, err := c.transport.SendRequest(request)
	if err != nil {
		return nil, err
	}

	if response.Error != nil {
		return map[string]interface{}{
			"error": fmt.Sprintf("%s (code: %d)", response.Error.Message, response.Error.Code),
		}, nil
	}

	if response.Result != nil {
		if resultMap, ok := response.Result.(map[string]interface{}); ok {
			return resultMap, nil
		}
		return map[string]interface{}{
			"result": response.Result,
		}, nil
	}

	return map[string]interface{}{}, nil
}

func generateID() string {
	return fmt.Sprintf("%d", time.Now().UnixNano())
}

