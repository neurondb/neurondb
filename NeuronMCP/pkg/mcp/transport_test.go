/*-------------------------------------------------------------------------
 *
 * transport_test.go
 *    Database operations
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/pkg/mcp/transport_test.go
 *
 *-------------------------------------------------------------------------
 */

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
  /* Create a test message */
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

  /* Create input with Content-Length header */
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
  /* Test with invalid Content-Length header */
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
  /* Test with missing Content-Length header */
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
  /* Test with invalid JSON body */
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
  /* Test with Content-Length larger than actual body */
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
  /* Test reading JSON directly (without Content-Length headers) */
	message := map[string]interface{}{
		"jsonrpc": "2.0",
		"id":      1,
		"method":  "test",
	}
	messageJSON, err := json.Marshal(message)
	if err != nil {
		t.Fatalf("Failed to marshal message: %v", err)
	}

	/* Add newline so ReadString can read it properly */
	messageWithNewline := string(messageJSON) + "\n"
	
	transport := &StdioTransport{
		stdin:  bufio.NewReader(strings.NewReader(messageWithNewline)),
		stdout: bufio.NewWriter(&bytes.Buffer{}),
		stderr: &bytes.Buffer{},
	}

	req, err := transport.ReadMessage()
	/* EOF is acceptable for this test since we're reading JSON directly */
	if err != nil && err != io.EOF {
		t.Fatalf("ReadMessage() error = %v", err)
	}
	/* If we got EOF, the message should still have been parsed */
	if err == io.EOF && req == nil {
		t.Fatalf("ReadMessage() returned EOF but should have parsed JSON first")
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

  /* Flush the buffer to get the output */
	if err := transport.stdout.Flush(); err != nil {
		t.Fatalf("Failed to flush stdout: %v", err)
	}

	output := buf.String()
	if output == "" {
		t.Fatal("WriteMessage() produced no output")
	}

  /* Should start with Content-Length header per MCP specification */
	if !strings.HasPrefix(output, "Content-Length:") {
		t.Error("WriteMessage() should start with Content-Length header")
	}

  /* Should contain JSON body after headers */
	if !strings.Contains(output, "jsonrpc") {
		t.Error("WriteMessage() should include jsonrpc in output")
	}
	
  /* Verify Content-Length header format: Content-Length: <number>\r\n\r\n */
	lines := strings.Split(output, "\r\n")
	if len(lines) < 3 {
		t.Error("WriteMessage() should have Content-Length header followed by empty line")
	}
	if !strings.HasPrefix(lines[0], "Content-Length:") {
		t.Error("First line should be Content-Length header")
	}
	if lines[1] != "" {
		t.Error("Second line should be empty (end of headers)")
	}
}

func TestStdioTransport_WriteMessage_NilResponse(t *testing.T) {
	var buf bytes.Buffer
	transport := &StdioTransport{
		stdin:  bufio.NewReader(strings.NewReader("")),
		stdout: bufio.NewWriter(&buf),
		stderr: &bytes.Buffer{},
	}

  /* Should not crash with nil response */
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

  /* Flush the buffer to get the output */
	if err := transport.stdout.Flush(); err != nil {
		t.Fatalf("Failed to flush stdout: %v", err)
	}

	output := buf.String()
	if output == "" {
		t.Fatal("WriteNotification() produced no output")
	}

  /* Should start with Content-Length header per MCP specification */
	if !strings.HasPrefix(output, "Content-Length:") {
		t.Error("WriteNotification() should start with Content-Length header")
	}

  /* Should contain method */
	if !strings.Contains(output, "method") {
		t.Error("WriteNotification() should include method in JSON")
	}
	
  /* Verify Content-Length header format: Content-Length: <number>\r\n\r\n */
	lines := strings.Split(output, "\r\n")
	if len(lines) < 3 {
		t.Error("WriteNotification() should have Content-Length header followed by empty line")
	}
	if !strings.HasPrefix(lines[0], "Content-Length:") {
		t.Error("First line should be Content-Length header")
	}
	if lines[1] != "" {
		t.Error("Second line should be empty (end of headers)")
	}
}

func TestStdioTransport_WriteNotification_EmptyMethod(t *testing.T) {
	var buf bytes.Buffer
	transport := &StdioTransport{
		stdin:  bufio.NewReader(strings.NewReader("")),
		stdout: bufio.NewWriter(&buf),
		stderr: &bytes.Buffer{},
	}

  /* Should not crash with empty method */
	err := transport.WriteNotification("", nil)
	if err != nil {
		t.Logf("WriteNotification() with empty method returned error: %v", err)
	} else {
   /* Flush if no error */
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

  /* Should not crash with nil params */
	err := transport.WriteNotification("test/notification", nil)
	if err != nil {
		t.Fatalf("WriteNotification() error with nil params = %v", err)
	}

  /* Flush the buffer to get the output */
	if err := transport.stdout.Flush(); err != nil {
		t.Fatalf("Failed to flush stdout: %v", err)
	}

	output := buf.String()
	if output == "" {
		t.Fatal("WriteNotification() produced no output")
	}
}

