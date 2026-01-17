package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
	"syscall"
	"time"
)

/* Client wraps an MCP server process and provides JSON-RPC communication */
type Client struct {
	cmd         *exec.Cmd
	stdin       io.WriteCloser
	stdout      *bufio.Reader
	stdoutPipe  io.ReadCloser /* Keep reference to underlying pipe for cleanup */
	stderr      io.ReadCloser
	mu          sync.Mutex
	requests    map[string]chan *JSONRPCResponse
	nextID      int64
	ctx         context.Context
	cancel      context.CancelFunc
	closed      bool
	closeMu     sync.Mutex
}

/* MCPConfig defines how to spawn an MCP server */
type MCPConfig struct {
	Command string            `json:"command"`
	Args    []string          `json:"args,omitempty"`
	Env     map[string]string `json:"env,omitempty"`
}

/* NewClient creates a new MCP client and spawns the server process */
func NewClient(config MCPConfig) (*Client, error) {
	ctx, cancel := context.WithCancel(context.Background())

	cmd := exec.CommandContext(ctx, config.Command, config.Args...)

	cmd.Env = os.Environ()
	for k, v := range config.Env {
		cmd.Env = append(cmd.Env, fmt.Sprintf("%s=%s", k, v))
	}

	stdin, err := cmd.StdinPipe()
	if err != nil {
		cancel()
		return nil, fmt.Errorf("failed to create stdin pipe: %w", err)
	}

	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		cancel()
		stdin.Close()
		return nil, fmt.Errorf("failed to create stdout pipe: %w", err)
	}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		cancel()
		stdin.Close()
		stdoutPipe.Close()
		return nil, fmt.Errorf("failed to create stderr pipe: %w", err)
	}

	client := &Client{
		cmd:        cmd,
		stdin:      stdin,
		stdout:     bufio.NewReader(stdoutPipe),
		stdoutPipe: stdoutPipe,
		stderr:     stderr,
		requests:   make(map[string]chan *JSONRPCResponse),
		ctx:        ctx,
		cancel:     cancel,
		closed:     false,
	}

	if err := cmd.Start(); err != nil {
		cancel()
		stdin.Close()
		stdoutPipe.Close()
		stderr.Close()
		return nil, fmt.Errorf("failed to start MCP server: %w", err)
	}

	go client.readResponses()

	if err := client.initialize(); err != nil {
		client.Close()
		return nil, fmt.Errorf("failed to initialize MCP connection: %w", err)
	}

	/* Start goroutine to read stderr to prevent blocking */
	go client.readStderr()

	return client, nil
}

/* readStderr reads from stderr to prevent pipe blocking */
func (c *Client) readStderr() {
	defer func() {
		/* Ensure stderr is closed when goroutine exits */
		if c.stderr != nil {
			c.stderr.Close()
		}
	}()

	buf := make([]byte, 4096)
	for {
		select {
		case <-c.ctx.Done():
			return
		default:
			n, err := c.stderr.Read(buf)
			if err != nil {
				if err == io.EOF {
					return
				}
				/* Check if we're closed */
				c.closeMu.Lock()
				closed := c.closed
				c.closeMu.Unlock()
				if closed {
					return
				}
				return
			}
			/* Log stderr output for debugging (could be enhanced to use proper logger) */
			if n > 0 {
				/* Silently consume stderr to prevent pipe blocking */
				_ = buf[:n]
			}
		}
	}
}

/* initialize performs the MCP initialize handshake (like Claude Desktop) */
func (c *Client) initialize() error {
	req := InitializeRequest{
		ProtocolVersion: ProtocolVersion,
		Capabilities: map[string]interface{}{
			"tools":     map[string]interface{}{},
			"resources": map[string]interface{}{},
			"prompts":   map[string]interface{}{},
			"sampling":  map[string]interface{}{},
		},
		ClientInfo: map[string]interface{}{
			"name":    "neurondesk",
			"version": "2.1.0",
		},
	}

	resp, err := c.Call("initialize", req)
	if err != nil {
		return fmt.Errorf("initialize failed: %w", err)
	}

	if err := c.Notify("notifications/initialized", nil); err != nil {
		return fmt.Errorf("initialized notification failed: %w", err)
	}

	var initResp InitializeResponse
	respBytes, err := json.Marshal(resp)
	if err != nil {
		return fmt.Errorf("failed to marshal initialize response: %w", err)
	}
	if err := json.Unmarshal(respBytes, &initResp); err != nil {
		return fmt.Errorf("invalid initialize response: %w", err)
	}

	return nil
}

/* Call sends a JSON-RPC request and waits for a response */
func (c *Client) Call(method string, params interface{}) (interface{}, error) {
	c.mu.Lock()
	id := fmt.Sprintf("%d", c.nextID)
	c.nextID++
	responseChan := make(chan *JSONRPCResponse, 1)
	c.requests[id] = responseChan
	c.mu.Unlock()

	defer func() {
		c.mu.Lock()
		delete(c.requests, id)
		c.mu.Unlock()
	}()

	req := JSONRPCRequest{
		JSONRPC: "2.0",
		ID:      json.RawMessage(fmt.Sprintf(`"%s"`, id)),
		Method:  method,
	}

	if params != nil {
		paramsBytes, err := json.Marshal(params)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal params: %w", err)
		}
		req.Params = paramsBytes
	}

	reqBytes, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	header := fmt.Sprintf("Content-Length: %d\r\n\r\n", len(reqBytes))
	if _, err := c.stdin.Write([]byte(header)); err != nil {
		return nil, fmt.Errorf("failed to write header: %w", err)
	}
	if _, err := c.stdin.Write(reqBytes); err != nil {
		return nil, fmt.Errorf("failed to write request: %w", err)
	}

	select {
	case resp := <-responseChan:
		if resp.Error != nil {
			return nil, fmt.Errorf("MCP error: %s (code: %d)", resp.Error.Message, resp.Error.Code)
		}
		return resp.Result, nil
	case <-time.After(30 * time.Second):
		return nil, fmt.Errorf("request timeout")
	case <-c.ctx.Done():
		return nil, fmt.Errorf("client closed")
	}
}

/* Notify sends a JSON-RPC notification (no response expected) */
func (c *Client) Notify(method string, params interface{}) error {
	req := map[string]interface{}{
		"jsonrpc": "2.0",
		"method":  method,
	}

	if params != nil {
		req["params"] = params
	}

	reqBytes, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("failed to marshal notification: %w", err)
	}

	header := fmt.Sprintf("Content-Length: %d\r\n\r\n", len(reqBytes))
	if _, err := c.stdin.Write([]byte(header)); err != nil {
		return fmt.Errorf("failed to write header: %w", err)
	}
	if _, err := c.stdin.Write(reqBytes); err != nil {
		return fmt.Errorf("failed to write notification: %w", err)
	}

	return nil
}

/* readResponses reads JSON-RPC responses from stdout */
func (c *Client) readResponses() {
	defer func() {
		/* Ensure stdout pipe is closed when goroutine exits */
		if c.stdoutPipe != nil {
			c.stdoutPipe.Close()
		}
	}()

	for {
		/* Check context cancellation */
		select {
		case <-c.ctx.Done():
			c.mu.Lock()
			for _, ch := range c.requests {
				select {
				case ch <- nil:
				default:
				}
			}
			c.mu.Unlock()
			return
		default:
		}

		var contentLength int
		for {
			line, err := c.stdout.ReadString('\n')
			if err != nil {
				if err == io.EOF {
					return
				}
				/* Check if we're closed */
				c.closeMu.Lock()
				closed := c.closed
				c.closeMu.Unlock()
				if closed {
					return
				}
				c.mu.Lock()
				for _, ch := range c.requests {
					select {
					case ch <- nil:
					default:
					}
				}
				c.mu.Unlock()
				return
			}

			line = strings.TrimRight(line, "\r\n")

			if line == "" {
				break
			}

			if strings.HasPrefix(strings.ToLower(line), "content-length:") {
				fmt.Sscanf(line, "Content-Length: %d", &contentLength)
			}
		}

		if contentLength <= 0 {
			continue
		}

		body := make([]byte, contentLength)
		if _, err := io.ReadFull(c.stdout, body); err != nil {
			if err == io.EOF {
				return
			}
			continue
		}

		var resp JSONRPCResponse
		if err := json.Unmarshal(body, &resp); err != nil {
			continue
		}

		if len(resp.ID) > 0 {
			id := string(resp.ID)
			if len(id) >= 2 && id[0] == '"' && id[len(id)-1] == '"' {
				id = id[1 : len(id)-1]
			}

			c.mu.Lock()
			ch, ok := c.requests[id]
			c.mu.Unlock()

			if ok {
				select {
				case ch <- &resp:
				default:
				}
			}
		}
	}
}

/* ListTools lists available tools from the MCP server */
func (c *Client) ListTools() (*ListToolsResponse, error) {
	resp, err := c.Call("tools/list", nil)
	if err != nil {
		return nil, err
	}

	respBytes, err := json.Marshal(resp)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal tools/list response: %w", err)
	}
	var toolsResp ListToolsResponse
	if err := json.Unmarshal(respBytes, &toolsResp); err != nil {
		return nil, fmt.Errorf("failed to parse tools/list response: %w", err)
	}

	return &toolsResp, nil
}

/* CallTool calls a tool on the MCP server */
func (c *Client) CallTool(name string, arguments map[string]interface{}) (*ToolResult, error) {
	req := CallToolRequest{
		Name:      name,
		Arguments: arguments,
	}

	resp, err := c.Call("tools/call", req)
	if err != nil {
		return nil, err
	}

	respBytes, err := json.Marshal(resp)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal tools/call response: %w", err)
	}
	var toolResp ToolResult
	if err := json.Unmarshal(respBytes, &toolResp); err != nil {
		return nil, fmt.Errorf("failed to parse tools/call response: %w", err)
	}

	return &toolResp, nil
}

/* ListResources lists available resources */
func (c *Client) ListResources() (*ListResourcesResponse, error) {
	resp, err := c.Call("resources/list", nil)
	if err != nil {
		return nil, err
	}

	respBytes, err := json.Marshal(resp)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal resources/list response: %w", err)
	}
	var resourcesResp ListResourcesResponse
	if err := json.Unmarshal(respBytes, &resourcesResp); err != nil {
		return nil, fmt.Errorf("failed to parse resources/list response: %w", err)
	}

	return &resourcesResp, nil
}

/* ReadResource reads a resource from the MCP server */
func (c *Client) ReadResource(uri string) (*ReadResourceResponse, error) {
	req := ReadResourceRequest{
		URI: uri,
	}

	resp, err := c.Call("resources/read", req)
	if err != nil {
		return nil, err
	}

	respBytes, err := json.Marshal(resp)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal resources/read response: %w", err)
	}
	var resourceResp ReadResourceResponse
	if err := json.Unmarshal(respBytes, &resourceResp); err != nil {
		return nil, fmt.Errorf("failed to parse resources/read response: %w", err)
	}

	return &resourceResp, nil
}

/* ListPrompts lists available prompts from the MCP server */
func (c *Client) ListPrompts() (*ListPromptsResponse, error) {
	resp, err := c.Call("prompts/list", nil)
	if err != nil {
		return nil, err
	}

	respBytes, err := json.Marshal(resp)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal prompts/list response: %w", err)
	}
	var promptsResp ListPromptsResponse
	if err := json.Unmarshal(respBytes, &promptsResp); err != nil {
		return nil, fmt.Errorf("failed to parse prompts/list response: %w", err)
	}

	return &promptsResp, nil
}

/* GetPrompt gets a prompt from the MCP server */
func (c *Client) GetPrompt(name string, arguments map[string]interface{}) (*GetPromptResponse, error) {
	req := GetPromptRequest{
		Name:      name,
		Arguments: arguments,
	}

	resp, err := c.Call("prompts/get", req)
	if err != nil {
		return nil, err
	}

	respBytes, err := json.Marshal(resp)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal prompts/get response: %w", err)
	}
	var promptResp GetPromptResponse
	if err := json.Unmarshal(respBytes, &promptResp); err != nil {
		return nil, fmt.Errorf("failed to parse prompts/get response: %w", err)
	}

	return &promptResp, nil
}

/* CreateMessage uses MCP sampling/createMessage for chat (like Claude Desktop) */
func (c *Client) CreateMessage(messages []map[string]interface{}, model string, temperature float64) (interface{}, error) {
	params := map[string]interface{}{
		"messages": messages,
	}
	if model != "" {
		params["model"] = model
	}
	if temperature > 0 {
		params["temperature"] = temperature
	}

	resp, err := c.Call("sampling/createMessage", params)
	if err != nil {
		return nil, err
	}

	return resp, nil
}

/* Close shuts down the MCP client and process */
func (c *Client) Close() error {
	c.closeMu.Lock()
	if c.closed {
		c.closeMu.Unlock()
		return nil
	}
	c.closed = true
	c.closeMu.Unlock()

	/* Cancel context to signal goroutines to stop */
	c.cancel()

	/* Close stdin first to signal the process to stop */
	if c.stdin != nil {
		c.stdin.Close()
	}

	/* Close stdout pipe */
	if c.stdoutPipe != nil {
		c.stdoutPipe.Close()
	}

	/* Close stderr pipe */
	if c.stderr != nil {
		c.stderr.Close()
	}

	/* Wait for goroutines to finish (with timeout) */
	done := make(chan struct{})
	go func() {
		/* Give goroutines a moment to finish */
		time.Sleep(100 * time.Millisecond)
		close(done)
	}()

	/* Signal process to stop */
	if c.cmd != nil && c.cmd.Process != nil {
		c.cmd.Process.Signal(os.Interrupt)
		select {
		case <-done:
		case <-time.After(100 * time.Millisecond):
		}

		/* If process is still running, kill it */
		if c.cmd.ProcessState == nil || !c.cmd.ProcessState.Exited() {
			c.cmd.Process.Kill()
		}
		c.cmd.Wait()
	}

	return nil
}

/* IsAlive checks if the MCP process is still running */
func (c *Client) IsAlive() bool {
	if c.cmd == nil || c.cmd.Process == nil {
		return false
	}

	err := c.cmd.Process.Signal(syscall.Signal(0))
	return err == nil
}
