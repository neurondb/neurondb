package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
	"github.com/neurondb/NeuronDesktop/api/internal/logging"
)

/* logWebSocket logs a WebSocket message using structured logger if available, otherwise log.Printf */
func logWebSocket(logger *logging.Logger, level, message string, fields map[string]interface{}) {
	if logger != nil {
		switch level {
		case "debug":
			logger.Debug(message, fields)
		case "info":
			logger.Info(message, fields)
		case "warn":
			logger.Warn(message, fields)
		case "error":
			logger.Error(message, nil, fields)
		default:
			logger.Info(message, fields)
		}
	} else {
		fieldStr := ""
		if len(fields) > 0 {
			fieldStr = fmt.Sprintf(" %+v", fields)
		}
		log.Printf("[%s] %s%s", strings.ToUpper(level), message, fieldStr)
	}
}

/* newUpgrader creates a WebSocket upgrader with CORS checking */
func newUpgrader(allowedOrigins []string) websocket.Upgrader {
	return websocket.Upgrader{
		CheckOrigin: func(r *http.Request) bool {
			origin := r.Header.Get("Origin")
			if origin == "" {
				return true /* No origin header means same-origin request */
			}

			/* Check against allowed origins */
			for _, allowedOrigin := range allowedOrigins {
				if allowedOrigin == "*" {
					return true /* Allow all origins if configured */
				}
				if allowedOrigin == origin {
					return true
				}
				/* Support wildcard subdomains like *.example.com */
				if strings.HasPrefix(allowedOrigin, "*.") {
					domain := strings.TrimPrefix(allowedOrigin, "*.")
					if strings.HasSuffix(origin, domain) && len(origin) > len(domain) && origin[len(origin)-len(domain)-1] == '.' {
						return true
					}
				}
			}
			return false
		},
	}
}

/* Global upgrader for backwards compatibility (will be replaced with config-based one) */
var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true /* Allow all origins - will be replaced when handlers are updated */
	},
}

/* Helper functions */
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func keys(m map[string]interface{}) []string {
	result := make([]string, 0, len(m))
	for k := range m {
		result = append(result, k)
	}
	return result
}

/* safeWriteJSON safely writes JSON to WebSocket connection with error handling */
func safeWriteJSON(conn *websocket.Conn, data map[string]interface{}) bool {
	return safeWriteJSONWithMutex(conn, data, nil)
}

/* safeWriteJSONWithMutex safely writes JSON with optional mutex protection */
func safeWriteJSONWithMutex(conn *websocket.Conn, data map[string]interface{}, mu *sync.Mutex) bool {
	// Lock mutex if provided to prevent concurrent writes
	if mu != nil {
		mu.Lock()
		defer mu.Unlock()
	}
	
	// Set a longer write deadline for important messages (30 seconds)
	// This prevents timeouts during slow network conditions
	deadline := time.Now().Add(30 * time.Second)
	if err := conn.SetWriteDeadline(deadline); err != nil {
		/* Logger not available in this context, use log.Printf */
		log.Printf("[MCP] ERROR: Failed to set write deadline: %v", err)
		return false
	}
	
	if err := conn.WriteJSON(data); err != nil {
		/* Logger not available in this context, use log.Printf */
		log.Printf("[MCP] ERROR: WebSocket write error: %v", err)
		return false
	}
	
	// Reset write deadline to default after successful write
	conn.SetWriteDeadline(time.Time{})
	return true
}

/* MCPWebSocket handles WebSocket connections for MCP chat */
func (h *MCPHandlers) MCPWebSocket(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]

	logWebSocket(h.logger, "info", "WebSocket upgrade attempt", map[string]interface{}{"profile_id": profileID})

	/* Use CORS-aware upgrader */
	wsUpgrader := newUpgrader(h.corsConfig.AllowedOrigins)
	conn, err := wsUpgrader.Upgrade(w, r, nil)
	if err != nil {
		logWebSocket(h.logger, "error", "WebSocket upgrade failed", map[string]interface{}{"error": err.Error()})
		return
	}
	defer conn.Close()

	logWebSocket(h.logger, "info", "WebSocket connection established", map[string]interface{}{"profile_id": profileID})

	client, err := h.GetMCPManager().GetClient(r.Context(), profileID)
	if err != nil {
		logWebSocket(h.logger, "error", "Failed to get MCP client", map[string]interface{}{"profile_id": profileID, "error": err.Error()})
		safeWriteJSON(conn, map[string]interface{}{
			"type":  "error",
			"error": fmt.Sprintf("Failed to connect to MCP server: %v. Please check your profile configuration in Settings.", err),
		})
		return
	}

	if !client.IsAlive() {
		safeWriteJSON(conn, map[string]interface{}{
			"type":  "error",
			"error": "MCP server process is not running. Please check your MCP configuration and ensure the server command is correct.",
		})
		return
	}

	safeWriteJSON(conn, map[string]interface{}{
		"type":    "connected",
		"message": "Connected to MCP server successfully",
	})

	/* Create context for goroutine cancellation */
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()

	/* Mutex to protect WebSocket writes from concurrent access */
	var writeMu sync.Mutex

	go func() {
		defer func() {
			if r := recover(); r != nil {
				logWebSocket(h.logger, "error", "WebSocket reader panic recovered", map[string]interface{}{"panic": fmt.Sprintf("%v", r)})
				safeWriteJSONWithMutex(conn, map[string]interface{}{
					"type":  "error",
					"error": "Internal server error in WebSocket handler",
				}, &writeMu)
			}
			cancel() /* Cancel context when goroutine exits */
		}()

		conn.SetReadDeadline(time.Now().Add(60 * time.Second))

		for {
			/* Check context cancellation */
			select {
			case <-ctx.Done():
				return
			default:
			}

			var msg map[string]interface{}
			if err := conn.ReadJSON(&msg); err != nil {
				if netErr, ok := err.(interface{ Timeout() bool }); ok && netErr.Timeout() {
					logWebSocket(h.logger, "warn", "WebSocket read timeout", map[string]interface{}{"profile_id": profileID})
					safeWriteJSONWithMutex(conn, map[string]interface{}{
						"type":  "error",
						"error": "Connection timeout. Please refresh the page.",
					}, &writeMu)
					break
				}

				if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
					logWebSocket(h.logger, "warn", "WebSocket unexpected close", map[string]interface{}{"profile_id": profileID, "error": err.Error()})
					safeWriteJSONWithMutex(conn, map[string]interface{}{
						"type":  "error",
						"error": fmt.Sprintf("WebSocket connection closed unexpectedly: %v", err),
					}, &writeMu)
				} else if websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) {
					logWebSocket(h.logger, "info", "WebSocket closed normally", map[string]interface{}{"profile_id": profileID})
				} else {
					logWebSocket(h.logger, "error", "WebSocket read error", map[string]interface{}{"profile_id": profileID, "error": err.Error()})
					safeWriteJSONWithMutex(conn, map[string]interface{}{
						"type":  "error",
						"error": fmt.Sprintf("Failed to read message: %v", err),
					}, &writeMu)
				}
				break
			}

			conn.SetReadDeadline(time.Now().Add(60 * time.Second))

			logWebSocket(h.logger, "debug", "WebSocket received message", map[string]interface{}{"type": msg["type"]})

			msgType, _ := msg["type"].(string)
			switch msgType {
			case "tool_call":
				name, _ := msg["name"].(string)
				args, _ := msg["arguments"].(map[string]interface{})

				if name == "" {
					safeWriteJSONWithMutex(conn, map[string]interface{}{
						"type":  "error",
						"error": "Tool name is required for tool_call",
					}, &writeMu)
					continue
				}

				if !client.IsAlive() {
					safeWriteJSONWithMutex(conn, map[string]interface{}{
						"type":  "error",
						"error": "MCP server connection lost. Please reconnect.",
					}, &writeMu)
					continue
				}

				result, err := client.CallTool(name, args)
				if err != nil {
					logWebSocket(h.logger, "error", "Tool call error", map[string]interface{}{"tool": name, "error": err.Error()})
					safeWriteJSONWithMutex(conn, map[string]interface{}{
						"type":  "error",
						"error": fmt.Sprintf("Tool call failed: %v", err),
					}, &writeMu)
				} else {
					safeWriteJSONWithMutex(conn, map[string]interface{}{
						"type":   "tool_result",
						"result": result,
					}, &writeMu)
				}
			case "list_tools":
				tools, err := client.ListTools()
				if err != nil {
					safeWriteJSONWithMutex(conn, map[string]interface{}{
						"type":  "error",
						"error": err.Error(),
					}, &writeMu)
				} else {
					safeWriteJSONWithMutex(conn, map[string]interface{}{
						"type":  "tools",
						"tools": tools,
					}, &writeMu)
				}
			case "list_resources":
				resources, err := client.ListResources()
				if err != nil {
					safeWriteJSONWithMutex(conn, map[string]interface{}{
						"type":  "error",
						"error": err.Error(),
					}, &writeMu)
				} else {
					safeWriteJSONWithMutex(conn, map[string]interface{}{
						"type":      "resources",
						"resources": resources,
					}, &writeMu)
				}
			case "read_resource":
				uri, _ := msg["uri"].(string)
				if uri == "" {
					safeWriteJSONWithMutex(conn, map[string]interface{}{
						"type":  "error",
						"error": "uri is required for read_resource",
					}, &writeMu)
				} else {
					resource, err := client.ReadResource(uri)
					if err != nil {
						safeWriteJSONWithMutex(conn, map[string]interface{}{
							"type":  "error",
							"error": err.Error(),
						}, &writeMu)
					} else {
						safeWriteJSONWithMutex(conn, map[string]interface{}{
							"type":     "resource",
							"resource": resource,
						}, &writeMu)
					}
				}
			case "list_prompts":
				prompts, err := client.ListPrompts()
				if err != nil {
					safeWriteJSONWithMutex(conn, map[string]interface{}{
						"type":  "error",
						"error": err.Error(),
					}, &writeMu)
				} else {
					safeWriteJSONWithMutex(conn, map[string]interface{}{
						"type":    "prompts",
						"prompts": prompts,
					}, &writeMu)
				}
			case "get_prompt":
				name, _ := msg["name"].(string)
				args, _ := msg["arguments"].(map[string]interface{})
				if name == "" {
					safeWriteJSONWithMutex(conn, map[string]interface{}{
						"type":  "error",
						"error": "name is required for get_prompt",
					}, &writeMu)
				} else {
					prompt, err := client.GetPrompt(name, args)
					if err != nil {
						safeWriteJSONWithMutex(conn, map[string]interface{}{
							"type":  "error",
							"error": err.Error(),
						}, &writeMu)
					} else {
						safeWriteJSONWithMutex(conn, map[string]interface{}{
							"type":   "prompt",
							"prompt": prompt,
						}, &writeMu)
					}
				}
			case "chat":
				content, _ := msg["content"].(string)
				modelID, _ := msg["model_id"].(string)

				logWebSocket(h.logger, "debug", "Chat request received", map[string]interface{}{"content_length": len(content), "model_id": modelID})

				if content == "" {
					logWebSocket(h.logger, "warn", "Empty content in chat request", nil)
					if !safeWriteJSONWithMutex(conn, map[string]interface{}{
						"type":  "error",
						"error": "Message content is required",
					}, &writeMu) {
						logWebSocket(h.logger, "error", "Failed to send empty content error to client", nil)
					}
					continue
				}

				if modelID == "" {
					logWebSocket(h.logger, "warn", "Empty modelID in chat request", nil)
					if !safeWriteJSONWithMutex(conn, map[string]interface{}{
						"type":  "error",
						"error": "Model ID is required",
					}, &writeMu) {
						logWebSocket(h.logger, "error", "Failed to send empty modelID error to client", nil)
					}
					continue
				}

				logWebSocket(h.logger, "debug", "Getting model config", map[string]interface{}{"model_id": modelID})
				modelConfig, err := h.mcpManager.queries.GetModelConfig(r.Context(), modelID, false)
				if err != nil {
					logWebSocket(h.logger, "error", "Failed to get model config", map[string]interface{}{"model_id": modelID, "error": err.Error()})
					if !safeWriteJSONWithMutex(conn, map[string]interface{}{
						"type":  "error",
						"error": fmt.Sprintf("Model not found: %v", err),
					}, &writeMu) {
						logWebSocket(h.logger, "error", "Failed to send model not found error to client", nil)
					}
					continue
				}

				logWebSocket(h.logger, "debug", "Model config retrieved", map[string]interface{}{"model_name": modelConfig.ModelName})

				messages := []map[string]interface{}{
					{
						"role":    "user",
						"content": content,
					},
				}

				logWebSocket(h.logger, "debug", "Calling CreateMessage", map[string]interface{}{"model": modelConfig.ModelName, "temperature": 0.7})
				result, err := client.CreateMessage(messages, modelConfig.ModelName, 0.7)
				if err != nil {
					logWebSocket(h.logger, "error", "CreateMessage failed", map[string]interface{}{"error": err.Error()})
					if !safeWriteJSONWithMutex(conn, map[string]interface{}{
						"type":  "error",
						"error": fmt.Sprintf("Failed to create message via MCP: %v", err),
					}, &writeMu) {
						logWebSocket(h.logger, "error", "Failed to send CreateMessage error to client", nil)
					}
					continue
				}

				logWebSocket(h.logger, "debug", "CreateMessage result received", map[string]interface{}{"result_type": fmt.Sprintf("%T", result)})

				/* Parse MCP response and forward to client
				 * The response format from NeuronMCP CreateMessageResponse:
				 * {
				 *   "content": "string",
				 *   "model": "string",
				 *   "stopReason": "string (optional)",
				 *   "metadata": {} (optional)
				 * }
				 */
				resultMap, ok := result.(map[string]interface{})
				if !ok {
					logWebSocket(h.logger, "warn", "Unexpected result type", map[string]interface{}{"result_type": fmt.Sprintf("%T", result)})
					if !safeWriteJSONWithMutex(conn, map[string]interface{}{
						"type":    "assistant",
						"content": fmt.Sprintf("%v", result),
					}, &writeMu) {
						logWebSocket(h.logger, "error", "Failed to send fallback response to client", nil)
					}
					continue
				}

				var responseText string
				parsingPath := "unknown"

				/* First, try the simple case: direct "content" field as string (most common) */
				if contentStr, ok := resultMap["content"].(string); ok {
					responseText = contentStr
					parsingPath = "direct_content_string"
					logWebSocket(h.logger, "debug", "Parsed response using direct content string", map[string]interface{}{"length": len(responseText)})
				} else if contentBlocks, ok := resultMap["content"].([]interface{}); ok {
					/* Handle content as array of ContentBlock (less common) */
					parsingPath = "content_blocks_array"
					logWebSocket(h.logger, "debug", "Parsing content blocks array", map[string]interface{}{"count": len(contentBlocks)})
					for i, block := range contentBlocks {
						if blockMap, ok := block.(map[string]interface{}); ok {
							if text, ok := blockMap["text"].(string); ok {
								logWebSocket(h.logger, "debug", "Block text field found", map[string]interface{}{"block": i, "length": len(text)})
								/* Try to parse as nested JSON */
								var nestedContent map[string]interface{}
								if err := json.Unmarshal([]byte(text), &nestedContent); err == nil {
									logWebSocket(h.logger, "debug", "Successfully parsed nested JSON", map[string]interface{}{"block": i, "keys": keys(nestedContent)})
									if contentVal, ok := nestedContent["content"].(string); ok {
										logWebSocket(h.logger, "debug", "Extracted nested content", map[string]interface{}{"block": i, "length": len(contentVal)})
										responseText += contentVal
									} else {
										logWebSocket(h.logger, "debug", "No 'content' field in nested JSON, using raw text", map[string]interface{}{"block": i})
										responseText += text
									}
								} else {
									logWebSocket(h.logger, "debug", "Text is not JSON, using as-is", map[string]interface{}{"block": i})
									responseText += text
								}
							} else if blockType, _ := blockMap["type"].(string); blockType == "text" {
								if textContent, ok := blockMap["content"].(string); ok {
									logWebSocket(h.logger, "debug", "Text type block with content", map[string]interface{}{"block": i, "length": len(textContent)})
									responseText += textContent
								}
							}
						}
					}
				} else {
					/* Fallback: try to serialize the entire result */
					parsingPath = "fallback_json_serialize"
					logWebSocket(h.logger, "warn", "Could not extract content from result, attempting JSON serialization", map[string]interface{}{"result_keys": keys(resultMap)})
					if resultJSON, err := json.Marshal(resultMap); err == nil {
						responseText = string(resultJSON)
						logWebSocket(h.logger, "debug", "Fallback: Serialized entire result as JSON", map[string]interface{}{"length": len(responseText)})
					} else {
						responseText = fmt.Sprintf("%v", result)
						logWebSocket(h.logger, "debug", "Fallback: Converted result to string", map[string]interface{}{"length": len(responseText)})
					}
				}

				if responseText == "" {
					logWebSocket(h.logger, "error", "Empty response text after parsing", map[string]interface{}{"path": parsingPath, "result_keys": keys(resultMap)})
					if !safeWriteJSONWithMutex(conn, map[string]interface{}{
						"type":  "error",
						"error": "Received empty response from MCP server",
					}, &writeMu) {
						logWebSocket(h.logger, "error", "Failed to send empty response error to client", nil)
					}
					continue
				}

				preview := responseText
				if len(responseText) > 100 {
					preview = responseText[:100]
				}
				logWebSocket(h.logger, "debug", "Successfully parsed response", map[string]interface{}{"path": parsingPath, "length": len(responseText), "preview": preview})
				
				/* Send response to frontend with error handling and mutex protection */
				/* CRITICAL: Use mutex to prevent ping ticker from interfering with response write */
				if !safeWriteJSONWithMutex(conn, map[string]interface{}{
					"type":    "assistant",
					"content": responseText,
				}, &writeMu) {
					logWebSocket(h.logger, "error", "Failed to send assistant response to WebSocket client", nil)
					/* Try to send error message to client */
					safeWriteJSONWithMutex(conn, map[string]interface{}{
						"type":  "error",
						"error": "Failed to send response to client",
					}, &writeMu)
				} else {
					logWebSocket(h.logger, "debug", "Successfully sent assistant response to client", map[string]interface{}{"length": len(responseText)})
				}
			case "pong":
			default:
				safeWriteJSONWithMutex(conn, map[string]interface{}{
					"type":  "error",
					"error": "Unknown message type: " + msgType,
				}, &writeMu)
			}
		}
	}()

	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	conn.SetWriteDeadline(time.Now().Add(10 * time.Second))

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if !client.IsAlive() {
				logWebSocket(h.logger, "warn", "MCP client not alive, closing WebSocket", map[string]interface{}{"profile_id": profileID})
				safeWriteJSONWithMutex(conn, map[string]interface{}{
					"type":  "error",
					"error": "MCP server connection lost",
				}, &writeMu)
				return
			}

			/* Use mutex to prevent ping from interfering with response writes */
			writeMu.Lock()
			conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if err := conn.WriteJSON(map[string]interface{}{
				"type": "ping",
			}); err != nil {
				writeMu.Unlock()
				logWebSocket(h.logger, "error", "Failed to send ping", map[string]interface{}{"profile_id": profileID, "error": err.Error()})
				return
			}
			conn.SetWriteDeadline(time.Time{})
			writeMu.Unlock()
		}
	}
}

/* AgentWebSocket handles WebSocket connections for NeuronAgent */
func (h *AgentHandlers) AgentWebSocket(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]
	sessionID := r.URL.Query().Get("session_id")

	if sessionID == "" {
		http.Error(w, "session_id required", http.StatusBadRequest)
		return
	}

	/* Use CORS-aware upgrader */
	wsUpgrader := newUpgrader(h.corsConfig.AllowedOrigins)
	conn, err := wsUpgrader.Upgrade(w, r, nil)
	if err != nil {
		logWebSocket(h.logger, "error", "WebSocket upgrade failed", map[string]interface{}{"error": err.Error()})
		return
	}
	defer conn.Close()

	profile, err := h.GetQueries().GetProfile(r.Context(), profileID)
	if err != nil {
		safeWriteJSON(conn, map[string]interface{}{
			"type":  "error",
			"error": "Failed to get profile: " + err.Error(),
		})
		return
	}

	agentWSURL := profile.AgentEndpoint + "/ws?session_id=" + sessionID
	agentConn, _, err := websocket.DefaultDialer.Dial(agentWSURL, http.Header{
		"Authorization": []string{"Bearer " + profile.AgentAPIKey},
	})
	if err != nil {
		safeWriteJSON(conn, map[string]interface{}{
			"type":  "error",
			"error": "Failed to connect to agent: " + err.Error(),
		})
		return
	}
	defer agentConn.Close()

	/* Create context for goroutine cancellation */
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()

	done := make(chan struct{})

	go func() {
		defer close(done)
		defer cancel()
		for {
			select {
			case <-ctx.Done():
				return
			default:
			}

			var msg map[string]interface{}
			if err := conn.ReadJSON(&msg); err != nil {
				if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
					logWebSocket(h.logger, "error", "Agent WebSocket client read error", map[string]interface{}{"error": err.Error()})
				}
				break
			}
			if err := agentConn.WriteJSON(msg); err != nil {
				logWebSocket(h.logger, "error", "Agent WebSocket write error", map[string]interface{}{"error": err.Error()})
				break
			}
		}
	}()

	for {
		select {
		case <-ctx.Done():
			return
		case <-done:
			return
		default:
		}

		var msg map[string]interface{}
		if err := agentConn.ReadJSON(&msg); err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				logWebSocket(h.logger, "error", "Agent WebSocket agent read error", map[string]interface{}{"error": err.Error()})
			}
			break
		}
		if !safeWriteJSON(conn, msg) {
			logWebSocket(h.logger, "error", "Agent WebSocket client write error", nil)
			break
		}
	}

	<-done
}
