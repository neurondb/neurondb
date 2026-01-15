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
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/neurondb/NeuronMCP/internal/middleware"
	"github.com/neurondb/NeuronMCP/pkg/mcp"
)

/* HTTPRequestHandler interface for handling HTTP MCP requests */
/* This interface breaks the import cycle between transport and server packages */
type HTTPRequestHandler interface {
	HandleHTTPRequest(ctx context.Context, mcpReq *middleware.MCPRequest) (*middleware.MCPResponse, error)
	GetConfig() HTTPConfigProvider
}

/* HTTPConfigProvider interface for accessing configuration */
type HTTPConfigProvider interface {
	GetServerSettings() HTTPServerSettingsProvider
}

/* HTTPServerSettingsProvider interface for accessing server settings */
type HTTPServerSettingsProvider interface {
	GetMaxRequestSize() *int
}

/* HTTPTransport handles MCP over HTTP */
type HTTPTransport struct {
	server          *http.Server
	mcpServer       *mcp.Server
	middleware      *middleware.Manager
	requestHandler  HTTPRequestHandler /* Use interface instead of concrete type */
	prometheusHandler http.Handler
	maxRequestSize  int64 /* Maximum request size in bytes */
	logger          interface{} /* Logger interface - will be set if available */
}

/* NewHTTPTransport creates a new HTTP transport */
func NewHTTPTransport(addr string, mcpServer *mcp.Server, middlewareManager *middleware.Manager, requestHandler HTTPRequestHandler, prometheusHandler http.Handler) *HTTPTransport {
	maxRequestSize := int64(0)
	if requestHandler != nil {
		if cfg := requestHandler.GetConfig(); cfg != nil {
			if serverSettings := cfg.GetServerSettings(); serverSettings != nil {
				if maxSize := serverSettings.GetMaxRequestSize(); maxSize != nil && *maxSize > 0 {
					maxRequestSize = int64(*maxSize)
				}
			}
		}
	}
	/* Default to 10MB if not configured */
	if maxRequestSize == 0 {
		maxRequestSize = 10 * 1024 * 1024
	}

	transport := &HTTPTransport{
		mcpServer:        mcpServer,
		middleware:       middlewareManager,
		requestHandler:  requestHandler,
		prometheusHandler: prometheusHandler,
		maxRequestSize:   maxRequestSize,
	}

	mux := http.NewServeMux()
	
	/* MCP endpoint with CORS support */
	mux.HandleFunc("/mcp", func(w http.ResponseWriter, r *http.Request) {
		/* Handle OPTIONS for CORS preflight */
		if r.Method == http.MethodOptions {
			w.Header().Set("Access-Control-Allow-Origin", "*")
			w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key, X-Request-ID")
			w.Header().Set("Access-Control-Max-Age", "3600")
			w.WriteHeader(http.StatusNoContent)
			return
		}
		transport.handleMCP(w, r)
	})
	
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
		MaxHeaderBytes: 1 << 20, /* 1MB max header size */
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
	/* Recover from panics */
	defer func() {
		if rec := recover(); rec != nil {
			t.writeJSONRPCError(w, nil, mcp.ErrCodeInternalError, fmt.Sprintf("Internal server error: %v", rec))
		}
	}()

	/* Validate HTTP method */
	if r.Method != http.MethodPost {
		t.writeJSONRPCError(w, nil, mcp.ErrCodeInvalidRequest, "Method not allowed. Only POST is supported.")
		return
	}

	/* Validate Content-Type */
	contentType := r.Header.Get("Content-Type")
	if contentType != "" && contentType != "application/json" && !strings.HasPrefix(contentType, "application/json;") {
		t.writeJSONRPCError(w, nil, mcp.ErrCodeInvalidRequest, "Content-Type must be application/json")
		return
	}

	/* Limit request body size */
	var reqBody io.Reader = r.Body
	if t.maxRequestSize > 0 {
		reqBody = io.LimitReader(r.Body, t.maxRequestSize+1) /* +1 to detect overflow */
	}

	/* Decode JSON-RPC request */
	var req mcp.JSONRPCRequest
	decoder := json.NewDecoder(reqBody)
	decoder.DisallowUnknownFields() /* Reject unknown fields for security */
	
	if err := decoder.Decode(&req); err != nil {
		var errorMsg string
		if err == io.EOF {
			errorMsg = "Request body is empty"
		} else if strings.Contains(err.Error(), "request body too large") {
			errorMsg = fmt.Sprintf("Request body exceeds maximum size of %d bytes", t.maxRequestSize)
		} else {
			errorMsg = fmt.Sprintf("Invalid JSON-RPC request: %v", err)
		}
		t.writeJSONRPCError(w, nil, mcp.ErrCodeParseError, errorMsg)
		return
	}

	/* Check if request exceeded size limit */
	if t.maxRequestSize > 0 {
		/* Try to read one more byte to detect overflow */
		var buf [1]byte
		if n, _ := reqBody.Read(buf[:]); n > 0 {
			t.writeJSONRPCError(w, req.ID, mcp.ErrCodeInvalidRequest, 
				fmt.Sprintf("Request body exceeds maximum size of %d bytes", t.maxRequestSize))
			return
		}
	}

	/* Validate JSON-RPC structure */
	if req.JSONRPC != "2.0" {
		t.writeJSONRPCError(w, req.ID, mcp.ErrCodeInvalidRequest, "jsonrpc must be '2.0'")
		return
	}

	if req.Method == "" {
		t.writeJSONRPCError(w, req.ID, mcp.ErrCodeInvalidRequest, "method is required")
		return
	}

	/* Create context from request with timeout handling */
	ctx := r.Context()
	
	/* Check if context is already cancelled */
	select {
	case <-ctx.Done():
		t.writeJSONRPCError(w, req.ID, mcp.ErrCodeInternalError, "Request context cancelled")
		return
	default:
	}

	/* Extract HTTP headers into metadata */
	metadata := make(map[string]interface{})
	for key, values := range r.Header {
		if len(values) > 0 {
			/* Store first value, or all values as array if multiple */
			/* Limit header value size to prevent abuse */
			if len(values) == 1 {
				value := values[0]
				if len(value) > 8192 { /* 8KB max per header value */
					value = value[:8192] + "... (truncated)"
				}
				metadata[key] = value
			} else {
				/* Limit number of header values */
				if len(values) > 10 {
					values = values[:10]
				}
				metadata[key] = values
			}
		}
	}

	/* Extract specific headers for easier access */
	if auth := r.Header.Get("Authorization"); auth != "" {
		metadata["authorization"] = auth
	}
	if apiKey := r.Header.Get("X-API-Key"); apiKey != "" {
		metadata["apiKey"] = apiKey
	}
	if requestID := r.Header.Get("X-Request-ID"); requestID != "" {
		metadata["requestId"] = requestID
	}

	/* Convert JSON-RPC params to map */
	var params map[string]interface{}
	if req.Params != nil {
		if err := json.Unmarshal(req.Params, &params); err != nil {
			/* If params is not an object, wrap it */
			params = map[string]interface{}{
				"params": req.Params,
			}
		}
	}

	/* Create middleware request */
	mcpReq := &middleware.MCPRequest{
		Method:   req.Method,
		Params:   params,
		Metadata: metadata,
	}

	/* Route through request handler */
	if t.requestHandler == nil {
		t.writeJSONRPCError(w, req.ID, mcp.ErrCodeInternalError, "Request handler not available")
		return
	}

	/* Handle request with timeout protection */
	mcpResp, err := t.requestHandler.HandleHTTPRequest(ctx, mcpReq)
	if err != nil {
		/* Check if error is due to context cancellation */
		if ctx.Err() == context.Canceled {
			t.writeJSONRPCError(w, req.ID, mcp.ErrCodeInternalError, "Request cancelled")
			return
		}
		if ctx.Err() == context.DeadlineExceeded {
			t.writeJSONRPCError(w, req.ID, mcp.ErrCodeInternalError, "Request timeout")
			return
		}
		t.writeJSONRPCError(w, req.ID, mcp.ErrCodeInternalError, err.Error())
		return
	}

	/* Validate response */
	if mcpResp == nil {
		t.writeJSONRPCError(w, req.ID, mcp.ErrCodeInternalError, "Handler returned nil response")
		return
	}

	/* Convert middleware response to JSON-RPC response */
	t.writeJSONRPCResponse(w, req.ID, mcpResp)
}

/* writeJSONRPCResponse writes a JSON-RPC response from middleware response */
func (t *HTTPTransport) writeJSONRPCResponse(w http.ResponseWriter, id json.RawMessage, mcpResp *middleware.MCPResponse) {
	/* Set response headers */
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	
	/* Add CORS headers if needed (can be configured later) */
	/* For now, allow all origins - should be configurable in production */
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key, X-Request-ID")

	if mcpResp.IsError {
		/* Extract error code and message */
		errorCode := mcp.ErrCodeInternalError
		errorMessage := "Internal error"
		
		if len(mcpResp.Content) > 0 {
			errorMessage = mcpResp.Content[0].Text
		}
		
		if mcpResp.Metadata != nil {
			if code, ok := mcpResp.Metadata["error_code"].(string); ok {
				/* Map error codes */
				switch code {
				case "METHOD_NOT_FOUND":
					errorCode = mcp.ErrCodeMethodNotFound
				case "INVALID_PARAMS", "INVALID_REQUEST":
					errorCode = mcp.ErrCodeInvalidParams
				case "UNAUTHORIZED", "AUTH_ERROR":
					errorCode = -32000 /* Custom error code for auth */
				case "PARSE_ERROR":
					errorCode = mcp.ErrCodeParseError
				case "SERVER_ERROR", "EXECUTION_ERROR", "SERIALIZATION_ERROR":
					errorCode = mcp.ErrCodeInternalError
				default:
					errorCode = mcp.ErrCodeInternalError
				}
			}
		}

		resp := mcp.JSONRPCResponse{
			JSONRPC: "2.0",
			ID:      id,
			Error: &mcp.JSONRPCError{
				Code:    errorCode,
				Message: errorMessage,
			},
		}

		/* Set HTTP status based on error code */
		httpStatus := http.StatusInternalServerError
		switch {
		case errorCode == -32000: /* Custom unauthorized code */
			httpStatus = http.StatusUnauthorized
		case errorCode == mcp.ErrCodeParseError:
			httpStatus = http.StatusBadRequest
		case errorCode == mcp.ErrCodeInvalidRequest || errorCode == mcp.ErrCodeInvalidParams:
			httpStatus = http.StatusBadRequest
		case errorCode == mcp.ErrCodeMethodNotFound:
			httpStatus = http.StatusNotFound
		default:
			httpStatus = http.StatusInternalServerError
		}
		w.WriteHeader(httpStatus)

		if err := json.NewEncoder(w).Encode(resp); err != nil {
			/* If encoding fails, try to write a simple error */
			http.Error(w, "Failed to encode error response", http.StatusInternalServerError)
		}
		return
	}

	/* Success response - extract result from metadata or content */
	var result interface{}
	if mcpResp.Metadata != nil {
		if res, ok := mcpResp.Metadata["result"]; ok {
			result = res
		}
	}

	/* If no result in metadata, try to parse content */
	if result == nil && len(mcpResp.Content) > 0 {
		/* Try to parse first content block as JSON */
		if err := json.Unmarshal([]byte(mcpResp.Content[0].Text), &result); err != nil {
			/* If not JSON, use content as text */
			result = mcpResp.Content[0].Text
		}
	}

	resp := mcp.JSONRPCResponse{
		JSONRPC: "2.0",
		ID:      id,
		Result:  result,
	}

	w.WriteHeader(http.StatusOK)
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		/* If encoding fails, try to write a simple error */
		http.Error(w, "Failed to encode response", http.StatusInternalServerError)
	}
}

/* writeJSONRPCError writes a JSON-RPC error response */
func (t *HTTPTransport) writeJSONRPCError(w http.ResponseWriter, id json.RawMessage, code int, message string) {
	w.Header().Set("Content-Type", "application/json")

		/* Set HTTP status based on error code */
		httpStatus := http.StatusInternalServerError
		switch {
		case code == -32000: /* Custom unauthorized code */
			httpStatus = http.StatusUnauthorized
		case code == mcp.ErrCodeParseError:
			httpStatus = http.StatusBadRequest
		case code == mcp.ErrCodeInvalidRequest || code == mcp.ErrCodeInvalidParams:
			httpStatus = http.StatusBadRequest
		case code == mcp.ErrCodeMethodNotFound:
			httpStatus = http.StatusNotFound
		default:
			httpStatus = http.StatusInternalServerError
		}
		w.WriteHeader(httpStatus)

	resp := mcp.JSONRPCResponse{
		JSONRPC: "2.0",
		ID:      id,
		Error: &mcp.JSONRPCError{
			Code:    code,
			Message: message,
		},
	}

	if err := json.NewEncoder(w).Encode(resp); err != nil {
		/* If encoding fails, try to write a simple error */
		http.Error(w, "Failed to encode error response", http.StatusInternalServerError)
	}
}

/* handleSSE handles Server-Sent Events for streaming */
func (t *HTTPTransport) handleSSE(w http.ResponseWriter, r *http.Request) {
	/* Recover from panics */
	defer func() {
		if rec := recover(); rec != nil {
			http.Error(w, fmt.Sprintf("Internal server error: %v", rec), http.StatusInternalServerError)
		}
	}()

	/* Set SSE headers */
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no") /* Disable nginx buffering */

	sseTransport, err := mcp.NewSSETransport(w)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to create SSE transport: %v", err), http.StatusInternalServerError)
		return
	}

	/* Send initial connection event */
	if err := sseTransport.WriteEvent("connected", map[string]interface{}{
		"message": "SSE connection established",
	}); err != nil {
		return
	}

	/* Keep connection alive and handle context cancellation */
	select {
	case <-r.Context().Done():
		sseTransport.Close()
		return
	}
}

/* handleHealth handles health check requests */
func (t *HTTPTransport) handleHealth(w http.ResponseWriter, r *http.Request) {
	/* Recover from panics */
	defer func() {
		if rec := recover(); rec != nil {
			http.Error(w, fmt.Sprintf("Internal server error: %v", rec), http.StatusInternalServerError)
		}
	}()

	w.Header().Set("Content-Type", "application/json")
	
	/* Check if request handler is available */
	healthy := true
	if t.requestHandler == nil {
		healthy = false
	}

	status := "healthy"
	if !healthy {
		status = "unhealthy"
		w.WriteHeader(http.StatusServiceUnavailable)
	}

	if err := json.NewEncoder(w).Encode(map[string]interface{}{
		"status":    status,
		"timestamp": time.Now().UTC().Format(time.RFC3339),
		"transport": "http",
	}); err != nil {
		http.Error(w, "Failed to encode health response", http.StatusInternalServerError)
	}
}

