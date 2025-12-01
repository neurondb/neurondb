package api

import (
	"net/http"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	"github.com/neurondb/NeuronAgent/internal/agent"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true // Allow all origins in development
	},
}

// HandleWebSocket handles WebSocket connections for streaming agent responses
func HandleWebSocket(runtime *agent.Runtime) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()

		// Get session ID from query parameter
		sessionIDStr := r.URL.Query().Get("session_id")
		sessionID, err := uuid.Parse(sessionIDStr)
		if err != nil {
			conn.WriteJSON(map[string]string{"error": "invalid session_id"})
			return
		}

		// Read messages from client
		for {
			var msg map[string]interface{}
			if err := conn.ReadJSON(&msg); err != nil {
				break
			}

			content, ok := msg["content"].(string)
			if !ok {
				conn.WriteJSON(map[string]string{"error": "invalid message format"})
				continue
			}

			// Execute agent
			state, err := runtime.Execute(r.Context(), sessionID, content)
			if err != nil {
				conn.WriteJSON(map[string]string{"error": err.Error()})
				continue
			}

			// Stream response
			response := map[string]interface{}{
				"type":     "response",
				"content":  state.FinalAnswer,
				"complete": true,
			}

			if err := conn.WriteJSON(response); err != nil {
				break
			}
		}
	}
}

