package api

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/agent"
)

// StreamResponse streams agent responses chunk by chunk
func StreamResponse(w http.ResponseWriter, r *http.Request, runtime *agent.Runtime, sessionIDStr string, userMessage string) {
	// Set headers for streaming
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no") // Disable nginx buffering

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	// Parse session ID
	sessionID, err := uuid.Parse(sessionIDStr)
	if err != nil {
		sendSSE(w, flusher, "error", map[string]interface{}{
			"error": "invalid session_id",
		})
		return
	}

	// Execute agent with streaming
	// Note: This is a simplified version - full implementation would stream LLM output
	state, err := runtime.Execute(r.Context(), sessionID, userMessage)
	if err != nil {
		sendSSE(w, flusher, "error", map[string]interface{}{
			"error": err.Error(),
		})
		return
	}

	// Stream response in chunks
	response := state.FinalAnswer
	chunkSize := 50 // Characters per chunk

	for i := 0; i < len(response); i += chunkSize {
		end := i + chunkSize
		if end > len(response) {
			end = len(response)
		}

		chunk := response[i:end]
		sendSSE(w, flusher, "chunk", map[string]interface{}{
			"content": chunk,
		})

		// Check if client disconnected
		if r.Context().Err() != nil {
			return
		}
	}

	// Send completion
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


