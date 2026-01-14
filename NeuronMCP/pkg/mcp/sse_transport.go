/*-------------------------------------------------------------------------
 *
 * sse_transport.go
 *    Server-Sent Events transport for MCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/pkg/mcp/sse_transport.go
 *
 *-------------------------------------------------------------------------
 */

package mcp

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

/* SSETransport handles MCP communication over Server-Sent Events */
type SSETransport struct {
	w       http.ResponseWriter
	flusher http.Flusher
}

/* NewSSETransport creates a new SSE transport */
func NewSSETransport(w http.ResponseWriter) (*SSETransport, error) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		return nil, fmt.Errorf("response writer does not support flushing")
	}

	/* Set SSE headers */
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	return &SSETransport{
		w:       w,
		flusher: flusher,
	}, nil
}

/* WriteEvent writes an SSE event */
func (t *SSETransport) WriteEvent(event string, data interface{}) error {
	dataJSON, err := json.Marshal(data)
	if err != nil {
		return fmt.Errorf("failed to marshal event data: %w", err)
	}

	/* Write SSE format: event: <event>\ndata: <data>\n\n */
	_, err = fmt.Fprintf(t.w, "event: %s\ndata: %s\n\n", event, string(dataJSON))
	if err != nil {
		return fmt.Errorf("failed to write SSE event: %w", err)
	}

	t.flusher.Flush()
	return nil
}

/* WriteMessage writes a JSON-RPC message as SSE */
func (t *SSETransport) WriteMessage(resp *JSONRPCResponse) error {
	return t.WriteEvent("message", resp)
}

/* WriteProgress writes a progress update */
func (t *SSETransport) WriteProgress(progressID string, progress float64, message string) error {
	data := map[string]interface{}{
		"id":       progressID,
		"progress": progress,
		"message":  message,
		"timestamp": time.Now().Unix(),
	}
	return t.WriteEvent("progress", data)
}

/* WriteError writes an error event */
func (t *SSETransport) WriteError(err error) error {
	data := map[string]interface{}{
		"error":     err.Error(),
		"timestamp": time.Now().Unix(),
	}
	return t.WriteEvent("error", data)
}

/* Close closes the SSE connection */
func (t *SSETransport) Close() error {
	/* Send close event */
	t.WriteEvent("close", map[string]interface{}{"message": "connection closed"})
	return nil
}

