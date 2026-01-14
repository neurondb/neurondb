/*-------------------------------------------------------------------------
 *
 * http_transport.go
 *    HTTP transport for MCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/transport/http_transport.go
 *
 *-------------------------------------------------------------------------
 */

package transport

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/neurondb/NeuronMCP/pkg/mcp"
)

/* HTTPTransport handles MCP over HTTP */
type HTTPTransport struct {
	server *http.Server
	mcpServer *mcp.Server
}

/* NewHTTPTransport creates a new HTTP transport */
func NewHTTPTransport(addr string, mcpServer *mcp.Server, prometheusHandler http.Handler) *HTTPTransport {
	transport := &HTTPTransport{
		mcpServer: mcpServer,
	}

	mux := http.NewServeMux()
	
	/* MCP endpoint */
	mux.HandleFunc("/mcp", transport.handleMCP)
	
	/* SSE endpoint for streaming */
	mux.HandleFunc("/mcp/stream", transport.handleSSE)
	
	/* Health endpoint */
	mux.HandleFunc("/health", transport.handleHealth)
	
	/* Prometheus metrics endpoint */
	if prometheusHandler != nil {
		mux.Handle("/metrics", prometheusHandler)
	}

	transport.server = &http.Server{
		Addr:         addr,
		Handler:      mux,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	return transport
}

/* Start starts the HTTP server */
func (t *HTTPTransport) Start() error {
	return t.server.ListenAndServe()
}

/* Shutdown gracefully shuts down the HTTP server */
func (t *HTTPTransport) Shutdown(ctx context.Context) error {
	return t.server.Shutdown(ctx)
}

/* handleMCP handles MCP requests over HTTP */
func (t *HTTPTransport) handleMCP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req mcp.JSONRPCRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid JSON-RPC request: %v", err), http.StatusBadRequest)
		return
	}

	/* Create context from request */
	ctx := r.Context()

	/* Handle request using MCP server's internal handler */
	/* Note: We need to access the internal handleRequest method */
	/* For now, we'll create a response manually */
	resp := t.mcpServer.HandleRequest(ctx, &req)

	/* Write response */
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		http.Error(w, fmt.Sprintf("Failed to encode response: %v", err), http.StatusInternalServerError)
		return
	}
}

/* handleSSE handles Server-Sent Events for streaming */
func (t *HTTPTransport) handleSSE(w http.ResponseWriter, r *http.Request) {
	sseTransport, err := mcp.NewSSETransport(w)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to create SSE transport: %v", err), http.StatusInternalServerError)
		return
	}

	/* Send initial connection event */
	sseTransport.WriteEvent("connected", map[string]interface{}{
		"message": "SSE connection established",
	})

	/* Keep connection alive */
	/* In a full implementation, this would handle streaming requests */
	select {
	case <-r.Context().Done():
		sseTransport.Close()
		return
	}
}

/* handleHealth handles health check requests */
func (t *HTTPTransport) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status": "healthy",
		"timestamp": time.Now(),
	})
}

