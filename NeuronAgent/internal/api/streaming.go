/*-------------------------------------------------------------------------
 *
 * streaming.go
 *    Server-Sent Events (SSE) streaming for NeuronAgent API
 *
 * Provides HTTP streaming functionality for real-time agent responses
 * using Server-Sent Events protocol.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/api/streaming.go
 *
 *-------------------------------------------------------------------------
 */

package api

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/agent"
)

/* StreamResponse streams agent responses chunk by chunk */
func StreamResponse(w http.ResponseWriter, r *http.Request, runtime *agent.Runtime, sessionIDStr string, userMessage string) {
	/* Set headers for streaming */
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no") // Disable nginx buffering

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	/* Parse session ID */
	sessionID, err := uuid.Parse(sessionIDStr)
	if err != nil {
		sendSSE(w, flusher, "error", map[string]interface{}{
			"error": "invalid session_id",
		})
		return
	}

	/* Execute agent with streaming */
	callback := func(chunk string, eventType string) error {
		/* Check if client disconnected */
		if r.Context().Err() != nil {
			return r.Context().Err()
		}

		switch eventType {
		case "chunk":
			sendSSE(w, flusher, "chunk", map[string]interface{}{
				"content": chunk,
			})
		case "tool_calls":
			sendSSE(w, flusher, "tool_calls", map[string]interface{}{
				"data": chunk,
			})
		case "tool_results":
			sendSSE(w, flusher, "tool_results", map[string]interface{}{
				"data": chunk,
			})
		case "done":
			/* Done will be sent after ExecuteStream completes */
		}
		return nil
	}

	state, err := runtime.ExecuteStream(r.Context(), sessionID, userMessage, callback)
	if err != nil {
		sendSSE(w, flusher, "error", map[string]interface{}{
			"error": err.Error(),
		})
		return
	}

	/* Send completion */
	sendSSE(w, flusher, "done", map[string]interface{}{
		"tokens_used":  state.TokensUsed,
		"tool_calls":   state.ToolCalls,
		"tool_results": state.ToolResults,
	})
}

func sendSSE(w http.ResponseWriter, flusher http.Flusher, event string, data interface{}) {
	jsonData, err := json.Marshal(data)
	if err != nil {
		return
	}

	fmt.Fprintf(w, "event: %s\n", event)
	fmt.Fprintf(w, "data: %s\n\n", jsonData)
	flusher.Flush()
}
