/*-------------------------------------------------------------------------
 *
 * streaming.go
 *    Streaming response handler for MCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/pkg/mcp/streaming.go
 *
 *-------------------------------------------------------------------------
 */

package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
)

/* StreamHandler is a function that handles streaming */
type StreamHandler func(ctx context.Context, writer StreamWriter) error

/* StreamWriter writes streaming data */
type StreamWriter interface {
	Write(data []byte) error
	WriteJSON(v interface{}) error
	WriteProgress(progressID string, progress float64, message string) error
	WriteError(err error) error
	Flush() error
	Close() error
}

/* streamWriter implements StreamWriter */
type streamWriter struct {
	w       io.Writer
	flusher interface{ Flush() }
}

/* NewStreamWriter creates a new stream writer */
func NewStreamWriter(w io.Writer) StreamWriter {
	sw := &streamWriter{w: w}
	if f, ok := w.(interface{ Flush() }); ok {
		sw.flusher = f
	}
	return sw
}

/* Write writes data to the stream */
func (sw *streamWriter) Write(data []byte) error {
	_, err := sw.w.Write(data)
	return err
}

/* WriteJSON writes JSON to the stream */
func (sw *streamWriter) WriteJSON(v interface{}) error {
	data, err := json.Marshal(v)
	if err != nil {
		return fmt.Errorf("failed to marshal JSON: %w", err)
	}
	return sw.Write(data)
}

/* WriteProgress writes a progress update */
func (sw *streamWriter) WriteProgress(progressID string, progress float64, message string) error {
	data := map[string]interface{}{
		"type":      "progress",
		"id":        progressID,
		"progress":  progress,
		"message":   message,
	}
	return sw.WriteJSON(data)
}

/* WriteError writes an error */
func (sw *streamWriter) WriteError(err error) error {
	data := map[string]interface{}{
		"type":  "error",
		"error": err.Error(),
	}
	return sw.WriteJSON(data)
}

/* Flush flushes the stream */
func (sw *streamWriter) Flush() error {
	if sw.flusher != nil {
		sw.flusher.Flush()
	}
	return nil
}

/* Close closes the stream */
func (sw *streamWriter) Close() error {
	return nil
}

/* StreamToolExecution streams tool execution progress */
func StreamToolExecution(ctx context.Context, toolName string, handler StreamHandler, writer StreamWriter) error {
	/* Send start event */
	startData := map[string]interface{}{
		"type": "tool_start",
		"tool":  toolName,
	}
	if err := writer.WriteJSON(startData); err != nil {
		return err
	}
	writer.Flush()

	/* Execute handler */
	if err := handler(ctx, writer); err != nil {
		writer.WriteError(err)
		return err
	}

	/* Send complete event */
	completeData := map[string]interface{}{
		"type": "tool_complete",
		"tool":  toolName,
	}
	if err := writer.WriteJSON(completeData); err != nil {
		return err
	}

	return writer.Flush()
}

/* StreamCompletion streams completion tokens */
func StreamCompletion(ctx context.Context, handler StreamHandler, writer StreamWriter) error {
	return handler(ctx, writer)
}












