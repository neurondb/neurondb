package client

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"

	"github.com/neurondb/NeuronMCP/pkg/mcp"
)

// ClientTransport handles MCP communication over stdio for clients
type ClientTransport struct {
	command string
	env     map[string]string
	args    []string
	process *exec.Cmd
	stdin   io.WriteCloser
	stdout  *bufio.Reader
	stderr  io.ReadCloser
}

// NewClientTransport creates a new client transport
func NewClientTransport(command string, env map[string]string, args []string) (*ClientTransport, error) {
	return &ClientTransport{
		command: command,
		env:     env,
		args:    args,
	}, nil
}

// Start starts the MCP server process
func (t *ClientTransport) Start() error {
	if t.process != nil {
		return fmt.Errorf("transport already started")
	}

	// Build command
	cmd := exec.Command(t.command, t.args...)

	// Set environment
	var env []string
	for _, e := range os.Environ() {
		env = append(env, e)
	}
	for k, v := range t.env {
		env = append(env, fmt.Sprintf("%s=%s", k, v))
	}
	cmd.Env = env

	// Setup stdio
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return fmt.Errorf("failed to create stdin pipe: %w", err)
	}
	t.stdin = stdin

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("failed to create stdout pipe: %w", err)
	}
	t.stdout = bufio.NewReader(stdout)

	stderr, err := cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("failed to create stderr pipe: %w", err)
	}
	t.stderr = stderr

	// Start process
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start process: %w", err)
	}

	t.process = cmd
	return nil
}

// Stop stops the MCP server process
func (t *ClientTransport) Stop() {
	if t.process != nil {
		if t.stdin != nil {
			t.stdin.Close()
		}
		if t.process.Process != nil {
			t.process.Process.Kill()
			t.process.Wait()
		}
		t.process = nil
	}
}

// SendRequest sends a request and waits for response
func (t *ClientTransport) SendRequest(request *mcp.JSONRPCRequest) (*mcp.JSONRPCResponse, error) {
	if t.process == nil {
		return nil, fmt.Errorf("transport not started")
	}

	// Serialize request
	requestJSON, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize request: %w", err)
	}

	// Send Content-Length header + body (standard MCP format)
	header := fmt.Sprintf("Content-Length: %d\r\n\r\n", len(requestJSON))
	if _, err := t.stdin.Write([]byte(header)); err != nil {
		return nil, fmt.Errorf("failed to write header: %w", err)
	}
	if _, err := t.stdin.Write(requestJSON); err != nil {
		return nil, fmt.Errorf("failed to write request: %w", err)
	}

	// Read response
	return t.readResponse(request.ID)
}

// SendNotification sends a notification (no response expected)
func (t *ClientTransport) SendNotification(notification map[string]interface{}) error {
	if t.process == nil {
		return fmt.Errorf("transport not started")
	}

	// Serialize notification
	notificationJSON, err := json.Marshal(notification)
	if err != nil {
		return fmt.Errorf("failed to serialize notification: %w", err)
	}

	// Send Content-Length header + body
	header := fmt.Sprintf("Content-Length: %d\r\n\r\n", len(notificationJSON))
	if _, err := t.stdin.Write([]byte(header)); err != nil {
		return fmt.Errorf("failed to write header: %w", err)
	}
	if _, err := t.stdin.Write(notificationJSON); err != nil {
		return fmt.Errorf("failed to write notification: %w", err)
	}

	return nil
}

// readResponse reads a JSON-RPC response from stdout
// Supports both Content-Length format and Claude Desktop format (JSON directly)
func (t *ClientTransport) readResponse(expectedID json.RawMessage) (*mcp.JSONRPCResponse, error) {
	if t.process == nil {
		return nil, fmt.Errorf("transport not started")
	}

	// Match response ID with request ID (skip notifications)
	maxAttempts := 10
	for attempt := 0; attempt < maxAttempts; attempt++ {
		// Read first line to determine format
		firstLine, err := t.stdout.ReadString('\n')
		if err != nil {
			return nil, fmt.Errorf("failed to read response: %w", err)
		}

		firstLine = strings.TrimRight(firstLine, "\r\n")

		var response mcp.JSONRPCResponse

		// Claude Desktop format: JSON directly (starts with '{')
		if strings.HasPrefix(firstLine, "{") {
			if err := json.Unmarshal([]byte(firstLine), &response); err != nil {
				return nil, fmt.Errorf("failed to parse JSON response: %w", err)
			}
		} else {
			// Standard MCP format: Content-Length headers
			// First line is a header, continue reading headers
			headerLines := []string{firstLine}

			for {
				line, err := t.stdout.ReadString('\n')
				if err != nil {
					return nil, fmt.Errorf("failed to read header: %w", err)
				}
				line = strings.TrimRight(line, "\r\n")
				if line == "" {
					break
				}
				headerLines = append(headerLines, line)
			}

			// Parse Content-Length
			var contentLength int
			for _, line := range headerLines {
				lineLower := strings.ToLower(line)
				if strings.HasPrefix(lineLower, "content-length:") {
					if _, err := fmt.Sscanf(line, "Content-Length: %d", &contentLength); err != nil {
						if _, err := fmt.Sscanf(line, "content-length: %d", &contentLength); err != nil {
							return nil, fmt.Errorf("invalid Content-Length header: %s", line)
						}
					}
					break
				}
			}

			if contentLength <= 0 {
				return nil, fmt.Errorf("missing or invalid Content-Length header")
			}

			// Read body
			body := make([]byte, contentLength)
			if _, err := io.ReadFull(t.stdout, body); err != nil {
				return nil, fmt.Errorf("failed to read response body: %w", err)
			}

			// Parse JSON response
			if err := json.Unmarshal(body, &response); err != nil {
				return nil, fmt.Errorf("failed to parse JSON response: %w", err)
			}
		}

		// Check if this is a notification (no ID) - skip it and continue
		if len(response.ID) == 0 {
			if attempt < maxAttempts-1 {
				continue
			}
			// If this is the last attempt and it's a notification, return it
			return &response, nil
		}

		// Check if ID matches
		if string(response.ID) == string(expectedID) {
			return &response, nil
		}

		// ID doesn't match - try reading more responses
		if attempt < maxAttempts-1 {
			continue
		}
	}

	// Should not reach here, but return error if we do
	return nil, fmt.Errorf("failed to find matching response after %d attempts", maxAttempts)
}

