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
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	"github.com/neurondb/NeuronAgent/internal/agent"
	"github.com/neurondb/NeuronAgent/internal/auth"
	"github.com/neurondb/NeuronAgent/internal/config"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/metrics"
)

/* createUpgrader creates a WebSocket upgrader with origin checking */
func createUpgrader(cfg *config.Config) websocket.Upgrader {
	allowedOrigins := cfg.Auth.WebSocketAllowedOrigins
	if len(allowedOrigins) == 0 {
		/* Fallback to CORS origins if WebSocket origins not set */
		allowedOrigins = cfg.Auth.AllowedOrigins
	}

	/* If still empty, check environment variable for development */
	if len(allowedOrigins) == 0 {
		if envOrigins := os.Getenv("WEBSOCKET_ALLOWED_ORIGINS"); envOrigins != "" {
			parts := strings.Split(envOrigins, ",")
			for _, part := range parts {
				trimmed := strings.TrimSpace(part)
				if trimmed != "" {
					allowedOrigins = append(allowedOrigins, trimmed)
				}
			}
		}
	}

	return websocket.Upgrader{
		CheckOrigin: func(r *http.Request) bool {
			origin := r.Header.Get("Origin")
			if origin == "" {
				/* No origin header - allow for non-browser clients */
				return true
			}

			/* If no origins configured, deny in production, allow in development */
			if len(allowedOrigins) == 0 {
				env := os.Getenv("ENV")
				if env == "production" || env == "prod" {
					metrics.WarnWithContext(r.Context(), "WebSocket connection denied: no allowed origins configured in production", map[string]interface{}{
						"origin": origin,
					})
					return false
				}
				/* Development mode - warn but allow */
				metrics.WarnWithContext(r.Context(), "WebSocket connection allowed without origin check (development mode)", map[string]interface{}{
					"origin": origin,
				})
				return true
			}

			/* Check against allowed origins */
			for _, allowed := range allowedOrigins {
				if origin == allowed {
					return true
				}
			}

			metrics.WarnWithContext(r.Context(), "WebSocket connection denied: origin not allowed", map[string]interface{}{
				"origin":          origin,
				"allowed_origins": allowedOrigins,
			})
			return false
		},
		HandshakeTimeout: 10 * time.Second,
	}
}

const (
	/* Maximum concurrent WebSocket connections to prevent resource exhaustion */
	maxWebSocketConnections = 1000
)

var wsConnectionCount int64

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
func HandleWebSocket(runtime *agent.Runtime, keyManager *auth.APIKeyManager, cfg *config.Config) http.HandlerFunc {
	upgrader := createUpgrader(cfg)

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

		/* Enforce connection limit before upgrading */
		if n := atomic.LoadInt64(&wsConnectionCount); n >= maxWebSocketConnections {
			metrics.WarnWithContext(logCtx, "WebSocket connection rejected: limit reached", map[string]interface{}{
				"current": n,
				"limit":   maxWebSocketConnections,
			})
			http.Error(w, "Service Unavailable: too many connections", http.StatusServiceUnavailable)
			return
		}
		atomic.AddInt64(&wsConnectionCount, 1)

		/* Upgrade connection */
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			atomic.AddInt64(&wsConnectionCount, -1)
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

		/* Ensure cleanup happens even if handleMessages panics */
		defer state.close()

		/* Handle connection */
		state.handleMessages(runtime, logCtx)
	}
}

/* authenticateWebSocket authenticates WebSocket connection */
func authenticateWebSocket(r *http.Request, keyManager *auth.APIKeyManager, logCtx context.Context) (*db.APIKey, error) {
	/* Prefer Authorization header over query parameter for security */
	apiKeyStr := ""
	authHeader := r.Header.Get("Authorization")
	if authHeader != "" {
		parts := strings.Fields(authHeader)
		if len(parts) == 2 && (parts[0] == "Bearer" || parts[0] == "ApiKey") {
			apiKeyStr = parts[1]
		}
	}

	/* Fallback to query parameter if header not present (for compatibility) */
	/* Note: Query parameters may be logged - prefer Authorization header */
	if apiKeyStr == "" {
		apiKeyStr = r.URL.Query().Get("api_key")
		if apiKeyStr != "" {
			keyPrefix := apiKeyStr
			if len(keyPrefix) > 8 {
				keyPrefix = keyPrefix[:8]
			}
			metrics.WarnWithContext(logCtx, "API key provided via query parameter (prefer Authorization header)", map[string]interface{}{
				"key_prefix": keyPrefix,
			})
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

	/* Ignore write errors when sending error messages - connection may be closing */
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
	atomic.AddInt64(&wsConnectionCount, -1)
	s.cancel()

	/* Send close message */
	/* Ignore errors during connection cleanup - connection may already be closed */
	s.conn.SetWriteDeadline(time.Now().Add(writeWait))
	_ = s.conn.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""))

	/* Close connection */
	/* Ignore close errors - connection may already be closed */
	_ = s.conn.Close()
}
