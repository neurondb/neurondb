package mcp

import (
	"bytes"
	"encoding/json"
	"fmt"
)

const ProtocolVersion = "2025-06-18"

// ParseRequest parses a JSON-RPC request
func ParseRequest(data []byte) (*JSONRPCRequest, error) {
	var req JSONRPCRequest
	if err := json.Unmarshal(data, &req); err != nil {
		return nil, fmt.Errorf("failed to parse JSON-RPC request: %w", err)
	}

	if req.JSONRPC != "2.0" {
		return nil, fmt.Errorf("invalid JSON-RPC version: %s", req.JSONRPC)
	}

	return &req, nil
}

// CreateResponse creates a JSON-RPC response
func CreateResponse(id json.RawMessage, result interface{}) *JSONRPCResponse {
	return &JSONRPCResponse{
		JSONRPC: "2.0",
		ID:      id,
		Result:  result,
	}
}

// CreateErrorResponse creates a JSON-RPC error response
func CreateErrorResponse(id json.RawMessage, code int, message string, data interface{}) *JSONRPCResponse {
	return &JSONRPCResponse{
		JSONRPC: "2.0",
		ID:      id,
		Error: &JSONRPCError{
			Code:    code,
			Message: message,
			Data:    data,
		},
	}
}

// Standard JSON-RPC error codes
const (
	ErrCodeParseError     = -32700
	ErrCodeInvalidRequest = -32600
	ErrCodeMethodNotFound = -32601
	ErrCodeInvalidParams  = -32602
	ErrCodeInternalError  = -32603
)

// MCP-specific error codes
const (
	ErrCodeToolNotFound    = -32001
	ErrCodeResourceNotFound = -32002
	ErrCodeExecutionError  = -32003
)

// SerializeResponse serializes a JSON-RPC response to JSON
func SerializeResponse(resp *JSONRPCResponse) ([]byte, error) {
	return json.Marshal(resp)
}

// ValidateRequest validates a JSON-RPC request
func ValidateRequest(req *JSONRPCRequest) error {
	if req.JSONRPC != "2.0" {
		return fmt.Errorf("invalid JSON-RPC version")
	}
	if req.Method == "" {
		return fmt.Errorf("method is required")
	}
	// Note: ID is optional for notifications, but required for requests
	// We'll handle this in the server
	return nil
}

// IsNotification checks if a request is a notification (no ID)
func IsNotification(req *JSONRPCRequest) bool {
	return len(req.ID) == 0 || bytes.Equal(req.ID, []byte("null"))
}

