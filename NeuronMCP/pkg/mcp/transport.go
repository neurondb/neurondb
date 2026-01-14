/*-------------------------------------------------------------------------
 *
 * transport.go
 *    Database operations
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/pkg/mcp/transport.go
 *
 *-------------------------------------------------------------------------
 */

package mcp

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
)

/* StdioTransport handles MCP communication over stdio */
type StdioTransport struct {
	stdin          *bufio.Reader
	stdout         *bufio.Writer
	stderr         io.Writer
	maxRequestSize int64
	clientUsesHeaders bool /* Track if client uses Content-Length headers */
}

/* NewStdioTransport creates a new stdio transport */
func NewStdioTransport() *StdioTransport {
	return NewStdioTransportWithMaxSize(0) /* Default: unlimited */
}

/* NewStdioTransportWithMaxSize creates a new stdio transport with max request size */
func NewStdioTransportWithMaxSize(maxRequestSize int64) *StdioTransport {
	/* Use a small buffered writer for stdout
	 * MCP protocol requires precise control over when data is sent
	 * Small buffer (1KB) ensures quick flushing while maintaining efficiency
	 * Note: Must flush after each message to ensure immediate transmission
	 */
	return &StdioTransport{
		stdin:             bufio.NewReader(os.Stdin),
		stdout:            bufio.NewWriterSize(os.Stdout, 1024), /* 1KB buffer for efficiency with immediate flushing */
		stderr:            os.Stderr,
		maxRequestSize:    maxRequestSize,
		clientUsesHeaders: true, /* Default to Content-Length headers (MCP standard) */
	}
}

/* ReadMessage reads a JSON-RPC message from stdin */
func (t *StdioTransport) ReadMessage() (*JSONRPCRequest, error) {
  /* Read headers */
	var contentLength int
	headerLines := 0
 	maxHeaders := 10 /* Prevent infinite loop */
	
	for headerLines < maxHeaders {
		line, err := t.stdin.ReadString('\n')
		if err != nil {
			if err == io.EOF {
     /* If we got EOF while reading headers and haven't found Content-Length, */
     /* this means the connection closed */
				if contentLength == 0 {
					return nil, io.EOF
				}
     /* If we have Content-Length but got EOF, it's still EOF */
				return nil, io.EOF
			}
			return nil, fmt.Errorf("failed to read header: %w", err)
		}
		headerLines++

   /* Remove trailing newline/carriage return */
		line = strings.TrimRight(line, "\r\n")
		
   /* Backward compatibility: Check if the first line is JSON (starts with '{') */
   /* Standard MCP protocol always uses Content-Length headers */
   /* Claude Desktop sends with Content-Length, but we support direct JSON for compatibility */
		if headerLines == 1 && strings.HasPrefix(strings.TrimSpace(line), "{") {
			/* Client doesn't use headers - we'll respond without headers too */
			t.clientUsesHeaders = false
			
			/* Enforce maximum request size for JSON without Content-Length */
			if t.maxRequestSize > 0 && int64(len(line)) > t.maxRequestSize {
				return nil, fmt.Errorf("request size %d exceeds maximum allowed size %d bytes", len(line), t.maxRequestSize)
			}
			
			/* Parse the JSON directly */
			return ParseRequest([]byte(line))
		}
		
   /* Empty line indicates end of headers */
		if line == "" {
			break
		}

   /* Parse Content-Length */
		lineLower := strings.ToLower(line)
		if strings.HasPrefix(lineLower, "content-length:") {
    /* Try both capitalized and lowercase */
			if _, err := fmt.Sscanf(line, "Content-Length: %d", &contentLength); err != nil {
				if _, err := fmt.Sscanf(line, "content-length: %d", &contentLength); err != nil {
					return nil, fmt.Errorf("invalid Content-Length header: %s", line)
				}
			}
		}
   /* Skip other headers (Content-Type, etc.) */
	}

	if contentLength <= 0 {
   /* This can happen if we read an empty line before getting Content-Length */
   /* or if there's malformed input. Return error but don't treat as fatal. */
		return nil, fmt.Errorf("missing or invalid Content-Length header")
	}

	/* Client uses Content-Length headers */
	t.clientUsesHeaders = true
	
	/* Enforce maximum request size */
	if t.maxRequestSize > 0 && int64(contentLength) > t.maxRequestSize {
		return nil, fmt.Errorf("request size %d exceeds maximum allowed size %d bytes", contentLength, t.maxRequestSize)
	}
	
	/* Read message body */
	body := make([]byte, contentLength)
	if _, err := io.ReadFull(t.stdin, body); err != nil {
		if err == io.EOF {
			return nil, io.EOF
		}
		return nil, fmt.Errorf("failed to read message body: %w", err)
	}

	return ParseRequest(body)
}

/* WriteMessage writes a JSON-RPC message to stdout */
func (t *StdioTransport) WriteMessage(resp *JSONRPCResponse) error {
	if t == nil {
		return fmt.Errorf("transport is nil")
	}
	if t.stdout == nil {
		return fmt.Errorf("stdout writer is nil")
	}
	if resp == nil {
		return fmt.Errorf("cannot write nil response")
	}
	
	data, err := SerializeResponse(resp)
	if err != nil {
		return fmt.Errorf("failed to serialize response: %w", err)
	}
	if data == nil || len(data) == 0 {
		return fmt.Errorf("serialized response is empty")
	}

  /* MCP Protocol Header Handling:
   * The MCP specification requires Content-Length headers for proper message framing.
   * This implementation automatically detects whether the client uses headers by examining
   * the first line of incoming requests. If the client sends Content-Length headers,
   * we respond with headers. If the client sends direct JSON, we respond with direct JSON.
   * 
   * Environment Variable Override:
   * NEURONMCP_FORCE_NO_HEADERS=true can be set to force headerless mode for compatibility
   * with older clients that don't support the full MCP specification.
   * 
   * Note: Claude Desktop and other modern MCP clients properly support Content-Length headers
   * per the MCP specification. The automatic detection ensures compatibility with both
   * standard-compliant and legacy clients.
   */
	forceNoHeaders := os.Getenv("NEURONMCP_FORCE_NO_HEADERS") == "true"
	
	if forceNoHeaders {
		/* Only disable headers if explicitly forced via environment variable */
		t.clientUsesHeaders = false
	}
	
	if t.clientUsesHeaders {
    /* MCP protocol format: Content-Length: <len>\r\n\r\n<body> */
    /* Per MCP spec: headers must end with \r\n\r\n (CRLF CRLF) */
		header := fmt.Sprintf("Content-Length: %d\r\n\r\n", len(data))
		headerBytes := []byte(header)
		
    /* Write header first, then body, then flush */
		if _, err := t.stdout.Write(headerBytes); err != nil {
			return fmt.Errorf("failed to write header (data_length=%d): %w", len(data), err)
		}
	}
	
  /* Write JSON body */
	if _, err := t.stdout.Write(data); err != nil {
		return fmt.Errorf("failed to write body (data_length=%d, using_headers=%v): %w", len(data), t.clientUsesHeaders, err)
	}
	
  /* When sending without headers (direct JSON), add a newline for compatibility
   * Some clients expect responses to end with a newline when using direct JSON format
   */
	if !t.clientUsesHeaders {
		if _, err := t.stdout.Write([]byte("\n")); err != nil {
			return fmt.Errorf("failed to write newline: %w", err)
		}
	}

  /* CRITICAL: Flush immediately after writing to ensure message is sent
   * Without flushing, the buffer might hold the data and cause protocol issues
   */
	if err := t.stdout.Flush(); err != nil {
		return fmt.Errorf("failed to flush stdout: %w", err)
	}

	return nil
}

/* WriteNotification writes a JSON-RPC notification (no response expected) */
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

  /* WORKAROUND: Check environment variable for forced no headers */
	forceNoHeaders := os.Getenv("NEURONMCP_FORCE_NO_HEADERS") == "true"
	
  /* Match client's request format (notifications follow same format) */
	if t.clientUsesHeaders && !forceNoHeaders {
    /* MCP protocol format: Content-Length: <len>\r\n\r\n<body> */
		header := fmt.Sprintf("Content-Length: %d\r\n\r\n", len(data))
		headerBytes := []byte(header)
		if _, err := t.stdout.Write(headerBytes); err != nil {
			return fmt.Errorf("failed to write header: %w", err)
		}
	}
	
  /* Write JSON body */
	if _, err := t.stdout.Write(data); err != nil {
		return fmt.Errorf("failed to write body: %w", err)
	}
	
  /* When sending without headers (direct JSON), add a newline for compatibility */
	if !t.clientUsesHeaders || forceNoHeaders {
		if _, err := t.stdout.Write([]byte("\n")); err != nil {
			return fmt.Errorf("failed to write newline: %w", err)
		}
	}

  /* CRITICAL: Flush immediately after writing to ensure message is sent */
	if err := t.stdout.Flush(); err != nil {
		return fmt.Errorf("failed to flush stdout: %w", err)
	}

	return nil
}

/* WriteError writes an error to stderr (only in debug mode) */
func (t *StdioTransport) WriteError(err error) {
  /* Only write debug errors if DEBUG environment variable is set */
  /* This prevents stderr pollution in production */
	if os.Getenv("NEURONDB_DEBUG") == "true" {
		fmt.Fprintf(t.stderr, "DEBUG: %v\n", err)
	}
}

