package mcp

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
)

// StdioTransport handles MCP communication over stdio
type StdioTransport struct {
	stdin  *bufio.Reader
	stdout *bufio.Writer
	stderr io.Writer
}

// NewStdioTransport creates a new stdio transport
func NewStdioTransport() *StdioTransport {
	// Use a buffered writer for stdout to enable flushing
	stdoutWriter := bufio.NewWriter(os.Stdout)
	return &StdioTransport{
		stdin:  bufio.NewReader(os.Stdin),
		stdout: stdoutWriter,
		stderr: os.Stderr,
	}
}

// ReadMessage reads a JSON-RPC message from stdin
func (t *StdioTransport) ReadMessage() (*JSONRPCRequest, error) {
	t.WriteError(fmt.Errorf("DEBUG: ReadMessage() called, starting to read headers"))
	// Read headers
	var contentLength int
	headerLines := 0
	maxHeaders := 10 // Prevent infinite loop
	
	for headerLines < maxHeaders {
		t.WriteError(fmt.Errorf("DEBUG: Reading header line %d", headerLines))
		line, err := t.stdin.ReadString('\n')
		t.WriteError(fmt.Errorf("DEBUG: Read header line: %q, err=%v", line, err))
		if err != nil {
			if err == io.EOF {
				// If we got EOF while reading headers and haven't found Content-Length,
				// this means the connection closed
				if contentLength == 0 {
					return nil, io.EOF
				}
				// If we have Content-Length but got EOF, it's still EOF
				return nil, io.EOF
			}
			return nil, fmt.Errorf("failed to read header: %w", err)
		}
		headerLines++

		// Remove trailing newline/carriage return
		line = strings.TrimRight(line, "\r\n")
		
		// Check if the first line is JSON (starts with '{')
		// If so, Claude Desktop is sending JSON directly without Content-Length headers
		if headerLines == 1 && strings.HasPrefix(strings.TrimSpace(line), "{") {
			t.WriteError(fmt.Errorf("DEBUG: First line is JSON (no Content-Length headers), parsing directly"))
			// Parse the JSON directly
			return ParseRequest([]byte(line))
		}
		
		// Empty line indicates end of headers
		if line == "" {
			break
		}

		// Parse Content-Length
		lineLower := strings.ToLower(line)
		if strings.HasPrefix(lineLower, "content-length:") {
			// Try both capitalized and lowercase
			if _, err := fmt.Sscanf(line, "Content-Length: %d", &contentLength); err != nil {
				if _, err := fmt.Sscanf(line, "content-length: %d", &contentLength); err != nil {
					return nil, fmt.Errorf("invalid Content-Length header: %s", line)
				}
			}
		}
		// Skip other headers (Content-Type, etc.)
	}

	if contentLength <= 0 {
		// This can happen if we read an empty line before getting Content-Length
		// or if there's malformed input. Return error but don't treat as fatal.
		t.WriteError(fmt.Errorf("DEBUG: No valid Content-Length found after %d headers", headerLines))
		return nil, fmt.Errorf("missing or invalid Content-Length header")
	}

	t.WriteError(fmt.Errorf("DEBUG: Headers parsed, contentLength=%d, reading body", contentLength))
	// Read message body
	body := make([]byte, contentLength)
	if _, err := io.ReadFull(t.stdin, body); err != nil {
		if err == io.EOF {
			return nil, io.EOF
		}
		return nil, fmt.Errorf("failed to read message body: %w", err)
	}

	return ParseRequest(body)
}

// WriteMessage writes a JSON-RPC message to stdout
func (t *StdioTransport) WriteMessage(resp *JSONRPCResponse) error {
	data, err := SerializeResponse(resp)
	if err != nil {
		return fmt.Errorf("failed to serialize response: %w", err)
	}

	t.WriteError(fmt.Errorf("DEBUG: Writing response: %s", string(data)))

	// Claude Desktop expects JSON directly without Content-Length headers
	// Write JSON followed by newline
	if _, err := t.stdout.Write(data); err != nil {
		return fmt.Errorf("failed to write body: %w", err)
	}
	
	// Add newline after JSON
	if _, err := t.stdout.Write([]byte("\n")); err != nil {
		return fmt.Errorf("failed to write newline: %w", err)
	}

	// Flush stdout to ensure message is sent immediately
	if err := t.stdout.Flush(); err != nil {
		return fmt.Errorf("failed to flush stdout: %w", err)
	}

	t.WriteError(fmt.Errorf("DEBUG: Response written and flushed"))

	return nil
}

// WriteNotification writes a JSON-RPC notification (no response expected)
func (t *StdioTransport) WriteNotification(method string, params interface{}) error {
	notification := map[string]interface{}{
		"jsonrpc": "2.0",
		"method":  method,
	}
	
	if params != nil {
		notification["params"] = params
	}
	
	data, err := json.Marshal(notification)
	if err != nil {
		return fmt.Errorf("failed to serialize notification: %w", err)
	}

	t.WriteError(fmt.Errorf("DEBUG: Writing notification: %s", string(data)))

	// Claude Desktop expects JSON directly without Content-Length headers
	// Write JSON followed by newline
	if _, err := t.stdout.Write(data); err != nil {
		return fmt.Errorf("failed to write body: %w", err)
	}
	
	// Add newline after JSON
	if _, err := t.stdout.Write([]byte("\n")); err != nil {
		return fmt.Errorf("failed to write newline: %w", err)
	}

	// Flush stdout to ensure message is sent immediately
	if err := t.stdout.Flush(); err != nil {
		return fmt.Errorf("failed to flush stdout: %w", err)
	}

	t.WriteError(fmt.Errorf("DEBUG: Notification written and flushed"))

	return nil
}

// WriteError writes an error to stderr
func (t *StdioTransport) WriteError(err error) {
	fmt.Fprintf(t.stderr, "Error: %v\n", err)
}

