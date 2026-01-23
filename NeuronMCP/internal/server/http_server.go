/*-------------------------------------------------------------------------
 *
 * http_server.go
 *    Standalone HTTP server for NeuronMCP metrics (runs in parallel with stdio)
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/server/http_server.go
 *
 *-------------------------------------------------------------------------
 */

package server

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/neurondb/NeuronMCP/internal/logging"
)

/* HTTPServer provides HTTP endpoints for health and metrics */
type HTTPServer struct {
	server            *http.Server
	prometheusHandler http.Handler
	logger            *logging.Logger
	metricsAPIKey     string
}

/* NewHTTPServer creates a new HTTP server for metrics */
func NewHTTPServer(addr string, prometheusHandler http.Handler) *HTTPServer {
	return NewHTTPServerWithLogger(addr, prometheusHandler, nil)
}

/* NewHTTPServerWithLogger creates a new HTTP server for metrics with optional logger */
func NewHTTPServerWithLogger(addr string, prometheusHandler http.Handler, logger *logging.Logger) *HTTPServer {
	metricsAPIKey := strings.TrimSpace(os.Getenv("NEURONMCP_METRICS_API_KEY"))

	mux := http.NewServeMux()

	/* Health endpoint – no auth */
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, `{"status":"ok"}`)
	})

	/* Prometheus metrics endpoint – always require NEURONMCP_METRICS_API_KEY */
	if prometheusHandler != nil {
		mux.Handle("/metrics", metricsAuthHandler(metricsAPIKey, prometheusHandler, logger))
	}

	h := &HTTPServer{
		server: &http.Server{
			Addr:         addr,
			Handler:      mux,
			ReadTimeout:  5 * time.Second,
			WriteTimeout: 10 * time.Second,
			IdleTimeout:  60 * time.Second,
		},
		prometheusHandler: prometheusHandler,
		logger:            logger,
		metricsAPIKey:     metricsAPIKey,
	}
	if metricsAPIKey == "" && logger != nil {
		logger.Warn("NEURONMCP_METRICS_API_KEY unset; /metrics will return 401 until set", nil)
	}
	return h
}

/* metricsAuthHandler wraps /metrics; requires NEURONMCP_METRICS_API_KEY (X-API-Key or Bearer) */
func metricsAuthHandler(expectKey string, next http.Handler, logger *logging.Logger) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if expectKey == "" {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnauthorized)
			fmt.Fprint(w, `{"error":"metrics disabled: set NEURONMCP_METRICS_API_KEY to enable /metrics"}`)
			return
		}
		token := ""
		if k := r.Header.Get("X-API-Key"); k != "" {
			token = strings.TrimSpace(k)
		} else if auth := r.Header.Get("Authorization"); strings.HasPrefix(auth, "Bearer ") {
			token = strings.TrimSpace(strings.TrimPrefix(auth, "Bearer "))
		}
		if token == "" {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnauthorized)
			fmt.Fprint(w, `{"error":"metrics requires X-API-Key or Authorization: Bearer"}`)
			return
		}
		if token != expectKey {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnauthorized)
			fmt.Fprint(w, `{"error":"invalid metrics API key"}`)
			return
		}
		next.ServeHTTP(w, r)
	})
}

/* Start starts the HTTP server in a goroutine */
func (h *HTTPServer) Start() {
	go func() {
		if err := h.server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			/* Log error - use logger if available, otherwise stderr (never write to stdout as it breaks MCP protocol) */
			if h.logger != nil {
				h.logger.Error("HTTP metrics server error", err, map[string]interface{}{
					"address": h.server.Addr,
				})
			} else {
				fmt.Fprintf(os.Stderr, "HTTP metrics server error: %v\n", err)
			}
		}
	}()
}

/* Shutdown gracefully shuts down the HTTP server */
func (h *HTTPServer) Shutdown(ctx context.Context) error {
	return h.server.Shutdown(ctx)
}



