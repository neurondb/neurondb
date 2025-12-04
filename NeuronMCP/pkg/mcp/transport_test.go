package mcp

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"testing"
)

func TestStdioTransport_ReadMessage(t *testing.T) {
	// Create a test message
	message := map[string]interface{}{
		"jsonrpc": "2.0",
		"id":      1,
		"method":  "test",
		"params":  map[string]interface{}{},
	}
	messageJSON, err := json.Marshal(message)
	if err != nil {
		t.Fatalf("Failed to marshal message: %v", err)
	}
	messageStr := string(messageJSON)

	// Create input with Content-Length header
	input := fmt.Sprintf("Content-Length: %d\r\nContent-Type: application/json\r\n\r\n%s", len(messageJSON), messageStr)

	transport := &StdioTransport{
		stdin:  bufio.NewReader(strings.NewReader(input)),
		stdout: bufio.NewWriter(&bytes.Buffer{}),
		stderr: &bytes.Buffer{},
	}

	req, err := transport.ReadMessage()
	if err != nil {
		t.Fatalf("ReadMessage() error = %v", err)
	}

	if req == nil {
		t.Fatal("ReadMessage() returned nil request")
	}

	if req.Method != "test" {
		t.Errorf("ReadMessage() method = %v, want test", req.Method)
	}
}

func TestStdioTransport_ReadMessage_InvalidContentLength(t *testing.T) {
	// Test with invalid Content-Length header
	input := "Content-Length: invalid\r\n\r\n{}"

	transport := &StdioTransport{
		stdin:  bufio.NewReader(strings.NewReader(input)),
		stdout: bufio.NewWriter(&bytes.Buffer{}),
		stderr: &bytes.Buffer{},
	}

	_, err := transport.ReadMessage()
	if err == nil {
		t.Error("ReadMessage() should return error for invalid Content-Length")
	}
}

func TestStdioTransport_ReadMessage_MissingContentLength(t *testing.T) {
	// Test with missing Content-Length header
	input := "Content-Type: application/json\r\n\r\n{}"

	transport := &StdioTransport{
		stdin:  bufio.NewReader(strings.NewReader(input)),
		stdout: bufio.NewWriter(&bytes.Buffer{}),
		stderr: &bytes.Buffer{},
	}

	_, err := transport.ReadMessage()
	if err == nil {
		t.Error("ReadMessage() should return error for missing Content-Length")
	}
}

func TestStdioTransport_ReadMessage_InvalidJSON(t *testing.T) {
	// Test with invalid JSON body
	input := "Content-Length: 10\r\n\r\n{invalid}"

	transport := &StdioTransport{
		stdin:  bufio.NewReader(strings.NewReader(input)),
		stdout: bufio.NewWriter(&bytes.Buffer{}),
		stderr: &bytes.Buffer{},
	}

	_, err := transport.ReadMessage()
	if err == nil {
		t.Error("ReadMessage() should return error for invalid JSON")
	}
}

func TestStdioTransport_ReadMessage_ShortBody(t *testing.T) {
	// Test with Content-Length larger than actual body
	input := "Content-Length: 100\r\n\r\n{}"

	transport := &StdioTransport{
		stdin:  bufio.NewReader(strings.NewReader(input)),
		stdout: bufio.NewWriter(&bytes.Buffer{}),
		stderr: &bytes.Buffer{},
	}

	_, err := transport.ReadMessage()
	if err == nil {
		t.Error("ReadMessage() should return error when body is shorter than Content-Length")
	}
}

func TestStdioTransport_ReadMessage_EOF(t *testing.T) {
	transport := &StdioTransport{
		stdin:  bufio.NewReader(strings.NewReader("")),
		stdout: bufio.NewWriter(&bytes.Buffer{}),
		stderr: &bytes.Buffer{},
	}

	_, err := transport.ReadMessage()
	if err != io.EOF {
		t.Errorf("ReadMessage() error = %v, want EOF", err)
	}
}

func TestStdioTransport_ReadMessage_JSONDirect(t *testing.T) {
	// Test reading JSON directly (without Content-Length headers)
	message := map[string]interface{}{
		"jsonrpc": "2.0",
		"id":      1,
		"method":  "test",
	}
	messageJSON, err := json.Marshal(message)
	if err != nil {
		t.Fatalf("Failed to marshal message: %v", err)
	}

	transport := &StdioTransport{
		stdin:  bufio.NewReader(strings.NewReader(string(messageJSON))),
		stdout: bufio.NewWriter(&bytes.Buffer{}),
		stderr: &bytes.Buffer{},
	}

	req, err := transport.ReadMessage()
	if err != nil {
		t.Fatalf("ReadMessage() error = %v", err)
	}

	if req == nil {
		t.Fatal("ReadMessage() returned nil request")
	}

	if req.Method != "test" {
		t.Errorf("ReadMessage() method = %v, want test", req.Method)
	}
}

func TestStdioTransport_WriteMessage(t *testing.T) {
	var buf bytes.Buffer
	transport := &StdioTransport{
		stdin:  bufio.NewReader(strings.NewReader("")),
		stdout: bufio.NewWriter(&buf),
		stderr: &bytes.Buffer{},
	}

	resp := CreateResponse(json.RawMessage("1"), map[string]string{"test": "value"})
	if resp == nil {
		t.Fatal("CreateResponse() returned nil")
	}

	err := transport.WriteMessage(resp)
	if err != nil {
		t.Fatalf("WriteMessage() error = %v", err)
	}

	// Flush the buffer to get the output
	if err := transport.stdout.Flush(); err != nil {
		t.Fatalf("Failed to flush stdout: %v", err)
	}

	output := buf.String()
	if output == "" {
		t.Fatal("WriteMessage() produced no output")
	}

	// Should contain JSON
	if !strings.Contains(output, "jsonrpc") {
		t.Error("WriteMessage() should include jsonrpc in output")
	}
}

func TestStdioTransport_WriteMessage_NilResponse(t *testing.T) {
	var buf bytes.Buffer
	transport := &StdioTransport{
		stdin:  bufio.NewReader(strings.NewReader("")),
		stdout: bufio.NewWriter(&buf),
		stderr: &bytes.Buffer{},
	}

	// Should not crash with nil response
	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("WriteMessage panicked with nil response: %v", r)
			}
		}()
		err := transport.WriteMessage(nil)
		if err == nil {
			t.Error("WriteMessage() should return error for nil response")
		}
	}()
}

func TestStdioTransport_WriteNotification(t *testing.T) {
	var buf bytes.Buffer
	transport := &StdioTransport{
		stdin:  bufio.NewReader(strings.NewReader("")),
		stdout: bufio.NewWriter(&buf),
		stderr: &bytes.Buffer{},
	}

	err := transport.WriteNotification("test/notification", map[string]string{"test": "value"})
	if err != nil {
		t.Fatalf("WriteNotification() error = %v", err)
	}

	// Flush the buffer to get the output
	if err := transport.stdout.Flush(); err != nil {
		t.Fatalf("Failed to flush stdout: %v", err)
	}

	output := buf.String()
	if output == "" {
		t.Fatal("WriteNotification() produced no output")
	}

	// Should contain method
	if !strings.Contains(output, "method") {
		t.Error("WriteNotification() should include method in JSON")
	}
}

func TestStdioTransport_WriteNotification_EmptyMethod(t *testing.T) {
	var buf bytes.Buffer
	transport := &StdioTransport{
		stdin:  bufio.NewReader(strings.NewReader("")),
		stdout: bufio.NewWriter(&buf),
		stderr: &bytes.Buffer{},
	}

	// Should not crash with empty method
	err := transport.WriteNotification("", nil)
	if err != nil {
		t.Logf("WriteNotification() with empty method returned error: %v", err)
	} else {
		// Flush if no error
		_ = transport.stdout.Flush()
	}
}

func TestStdioTransport_WriteNotification_NilParams(t *testing.T) {
	var buf bytes.Buffer
	transport := &StdioTransport{
		stdin:  bufio.NewReader(strings.NewReader("")),
		stdout: bufio.NewWriter(&buf),
		stderr: &bytes.Buffer{},
	}

	// Should not crash with nil params
	err := transport.WriteNotification("test/notification", nil)
	if err != nil {
		t.Fatalf("WriteNotification() error with nil params = %v", err)
	}

	// Flush the buffer to get the output
	if err := transport.stdout.Flush(); err != nil {
		t.Fatalf("Failed to flush stdout: %v", err)
	}

	output := buf.String()
	if output == "" {
		t.Fatal("WriteNotification() produced no output")
	}
}

