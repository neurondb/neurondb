/*-------------------------------------------------------------------------
 *
 * websocket.go
 *    WebSocket handler for NeuronAgent API
 *
 * Provides WebSocket support for real-time agent communication and
 * streaming responses.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/api/websocket.go
 *
 *-------------------------------------------------------------------------
 */

package api

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	"github.com/neurondb/NeuronAgent/internal/agent"
	"github.com/neurondb/NeuronAgent/internal/auth"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/metrics"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true /* Allow all origins in development */
	},
	HandshakeTimeout: 10 * time.Second,
}

const (
	/* WebSocket connection timeouts */
	writeWait      = 10 * time.Second
	pongWait       = 60 * time.Second
	pingPeriod     = (pongWait * 9) / 10
	maxMessageSize = 512 * 1024 /* 512KB */
)

/* connectionState tracks the state of a WebSocket connection */
type connectionState struct {
	conn      *websocket.Conn
	sessionID uuid.UUID
	apiKey    *db.APIKey
	ctx       context.Context
	cancel    context.CancelFunc
	mu        sync.Mutex
	closed    bool
}

/* HandleWebSocket handles WebSocket connections for streaming agent responses */
func HandleWebSocket(runtime *agent.Runtime, keyManager *auth.APIKeyManager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		requestID := GetRequestID(r.Context())
		logCtx := metrics.WithLogContext(r.Context(), requestID, "", "", "", "")

		/* Authenticate before upgrading connection */
		apiKey, err := authenticateWebSocket(r, keyManager, logCtx)
		if err != nil {
			metrics.WarnWithContext(logCtx, "WebSocket authentication failed", map[string]interface{}{
				"error": err.Error(),
			})
			http.Error(w, "Authentication failed", http.StatusUnauthorized)
			return
		}

		/* Get session ID from query parameter */
		sessionIDStr := r.URL.Query().Get("session_id")
		if sessionIDStr == "" {
			http.Error(w, "session_id is required", http.StatusBadRequest)
			return
		}

		sessionID, err := uuid.Parse(sessionIDStr)
		if err != nil {
			http.Error(w, "invalid session_id format", http.StatusBadRequest)
			return
		}

		/* Upgrade connection */
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			metrics.WarnWithContext(logCtx, "WebSocket upgrade failed", map[string]interface{}{
				"error": err.Error(),
			})
			return
		}

		/* Set connection parameters */
		conn.SetReadDeadline(time.Now().Add(pongWait))
		conn.SetReadLimit(maxMessageSize)
		conn.SetPongHandler(func(string) error {
			conn.SetReadDeadline(time.Now().Add(pongWait))
			return nil
		})

		/* Create connection state */
		ctx, cancel := context.WithCancel(r.Context())
		state := &connectionState{
			conn:      conn,
			sessionID: sessionID,
			apiKey:    apiKey,
			ctx:       ctx,
			cancel:    cancel,
		}

		/* Start ping goroutine */
		go state.pingLoop()

		/* Handle connection */
		state.handleMessages(runtime, logCtx)

		/* Cleanup */
		state.close()
	}
}

/* authenticateWebSocket authenticates WebSocket connection */
func authenticateWebSocket(r *http.Request, keyManager *auth.APIKeyManager, logCtx context.Context) (*db.APIKey, error) {
	/* Try to get API key from query parameter first */
	apiKeyStr := r.URL.Query().Get("api_key")
	if apiKeyStr == "" {
		/* Try Authorization header */
		authHeader := r.Header.Get("Authorization")
		if authHeader != "" {
			parts := strings.Fields(authHeader)
			if len(parts) == 2 && (parts[0] == "Bearer" || parts[0] == "ApiKey") {
				apiKeyStr = parts[1]
			}
		}
	}

	if apiKeyStr == "" {
		return nil, fmt.Errorf("API key is required")
	}

	/* Validate API key */
	apiKey, err := keyManager.ValidateAPIKey(r.Context(), apiKeyStr)
	if err != nil {
		return nil, err
	}

	metrics.DebugWithContext(logCtx, "WebSocket authenticated", map[string]interface{}{
		"key_prefix": apiKey.KeyPrefix,
		"key_id":     apiKey.ID.String(),
	})

	return apiKey, nil
}

/* pingLoop sends periodic ping messages to keep connection alive */
func (s *connectionState) pingLoop() {
	ticker := time.NewTicker(pingPeriod)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			s.mu.Lock()
			if s.closed {
				s.mu.Unlock()
				return
			}
			s.conn.SetWriteDeadline(time.Now().Add(writeWait))
			if err := s.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				s.mu.Unlock()
				return
			}
			s.mu.Unlock()
		case <-s.ctx.Done():
			return
		}
	}
}

/* handleMessages handles incoming messages from the client */
func (s *connectionState) handleMessages(runtime *agent.Runtime, logCtx context.Context) {
	messageQueue := make(chan map[string]interface{}, 10)
	
	/* Start message reader goroutine */
	go func() {
		defer close(messageQueue)
		for {
			var msg map[string]interface{}
			if err := s.conn.ReadJSON(&msg); err != nil {
				if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
					metrics.WarnWithContext(logCtx, "WebSocket read error", map[string]interface{}{
						"error": err.Error(),
					})
				}
				return
			}
			select {
			case messageQueue <- msg:
			case <-s.ctx.Done():
				return
			}
		}
	}()

	/* Process messages */
	for {
		select {
		case msg, ok := <-messageQueue:
			if !ok {
				/* Channel closed, connection lost */
				return
			}

			content, ok := msg["content"].(string)
			if !ok {
				s.sendError("invalid message format: content field is required and must be a string")
				continue
			}

			/* Execute agent with streaming */
			callback := func(chunk string, eventType string) error {
				s.mu.Lock()
				defer s.mu.Unlock()
				
				if s.closed {
					return context.Canceled
				}

				s.conn.SetWriteDeadline(time.Now().Add(writeWait))
				if err := s.conn.WriteJSON(map[string]interface{}{
					"type":    eventType,
					"content": chunk,
				}); err != nil {
					return err
				}
				return nil
			}

			state, err := runtime.ExecuteStream(s.ctx, s.sessionID, content, callback)
			if err != nil {
				metrics.WarnWithContext(logCtx, "Agent execution failed", map[string]interface{}{
					"error":      err.Error(),
					"session_id": s.sessionID.String(),
				})
				s.sendError(err.Error())
				continue
			}

			/* Send final response */
			response := map[string]interface{}{
				"type":         "response",
				"content":      state.FinalAnswer,
				"complete":     true,
				"tokens_used":  state.TokensUsed,
				"tool_calls":   state.ToolCalls,
				"tool_results": state.ToolResults,
			}

			s.mu.Lock()
			if !s.closed {
				s.conn.SetWriteDeadline(time.Now().Add(writeWait))
				if err := s.conn.WriteJSON(response); err != nil {
					s.mu.Unlock()
					metrics.WarnWithContext(logCtx, "Failed to send final response", map[string]interface{}{
						"error": err.Error(),
					})
					return
				}
			}
			s.mu.Unlock()

		case <-s.ctx.Done():
			return
		}
	}
}

/* sendError sends an error message to the client */
func (s *connectionState) sendError(errorMsg string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	if s.closed {
		return
	}

	s.conn.SetWriteDeadline(time.Now().Add(writeWait))
	_ = s.conn.WriteJSON(map[string]interface{}{
		"type":  "error",
		"error": errorMsg,
	})
}

/* close closes the WebSocket connection */
func (s *connectionState) close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	if s.closed {
		return
	}
	
	s.closed = true
	s.cancel()
	
	/* Send close message */
	s.conn.SetWriteDeadline(time.Now().Add(writeWait))
	_ = s.conn.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""))
	
	/* Close connection */
	_ = s.conn.Close()
}
