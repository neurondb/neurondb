/*-------------------------------------------------------------------------
 *
 * completions.go
 *    Completion generation for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/sampling/completions.go
 *
 *-------------------------------------------------------------------------
 */

package sampling

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
)

/* StreamWriter is an interface for writing streaming responses */
type StreamWriter interface {
	Write(data []byte) error
	Flush() error
}

/* CreateMessageStream creates a streaming completion */
func (m *Manager) CreateMessageStream(ctx context.Context, req SamplingRequest, writer StreamWriter) error {
	if ctx == nil {
		return fmt.Errorf("context cannot be nil")
	}

	if writer == nil {
		return fmt.Errorf("stream writer cannot be nil")
	}

	/* For now, we'll do a non-streaming call and simulate streaming */
	/* In a full implementation, this would use SSE or similar */
	
	response, err := m.CreateMessage(ctx, req)
	if err != nil {
		/* Send error to stream */
		errorData := map[string]interface{}{
			"type":  "error",
			"error": err.Error(),
		}
		errorJSON, _ := json.Marshal(errorData)
		writer.Write(errorJSON)
		writer.Flush()
		return fmt.Errorf("failed to create message: %w", err)
	}

	/* Simulate streaming by sending chunks */
	content := response.Content
	if content == "" {
		/* Send empty content notification */
		emptyData := map[string]interface{}{
			"type":    "content",
			"content": "",
		}
		emptyJSON, _ := json.Marshal(emptyData)
		if err := writer.Write(emptyJSON); err != nil {
			return fmt.Errorf("failed to write empty content: %w", err)
		}
	} else {
		chunkSize := 20 /* Reasonable chunk size for streaming */
		
		for i := 0; i < len(content); i += chunkSize {
			/* Check context cancellation */
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
			}

			end := i + chunkSize
			if end > len(content) {
				end = len(content)
			}
			
			chunk := content[i:end]
			chunkData := map[string]interface{}{
				"type":    "content",
				"content": chunk,
				"index":   i,
			}
			
			chunkJSON, err := json.Marshal(chunkData)
			if err != nil {
				return fmt.Errorf("failed to marshal chunk: %w", err)
			}

			if err := writer.Write(chunkJSON); err != nil {
				return fmt.Errorf("failed to write chunk at index %d: %w", i, err)
			}
			
			if err := writer.Flush(); err != nil {
				return fmt.Errorf("failed to flush at index %d: %w", i, err)
			}
		}
	}

	/* Send done signal */
	doneData := map[string]interface{}{
		"type": "done",
	}
	doneJSON, err := json.Marshal(doneData)
	if err != nil {
		return fmt.Errorf("failed to marshal done signal: %w", err)
	}

	if err := writer.Write(doneJSON); err != nil {
		return fmt.Errorf("failed to write done signal: %w", err)
	}

	return writer.Flush()
}

/* StreamCompletion streams a completion to an io.Writer */
func (m *Manager) StreamCompletion(ctx context.Context, req SamplingRequest, w io.Writer) error {
	writer := &streamWriter{w: w}
	return m.CreateMessageStream(ctx, req, writer)
}

/* streamWriter implements StreamWriter for io.Writer */
type streamWriter struct {
	w io.Writer
}

func (sw *streamWriter) Write(data []byte) error {
	_, err := sw.w.Write(data)
	if err != nil {
		return err
	}
	_, err = sw.w.Write([]byte("\n"))
	return err
}

func (sw *streamWriter) Flush() error {
	if f, ok := sw.w.(interface{ Flush() error }); ok {
		return f.Flush()
	}
	return nil
}

