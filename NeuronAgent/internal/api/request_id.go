/*-------------------------------------------------------------------------
 *
 * request_id.go
 *    Request ID middleware for NeuronAgent API
 *
 * Provides request ID generation and context management for tracking
 * requests across the API with correlation support.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/api/request_id.go
 *
 *-------------------------------------------------------------------------
 */

package api

import (
	"context"
	"net/http"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/metrics"
)

const requestIDKey contextKey = "request_id"

/* RequestIDMiddleware adds a unique request ID to each request */
func RequestIDMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestID := r.Header.Get("X-Request-ID")
		if requestID == "" {
			requestID = uuid.New().String()
		}

		/* Add to context */
		ctx := context.WithValue(r.Context(), requestIDKey, requestID)
		/* Also add to metrics log context */
		ctx = metrics.WithLogContext(ctx, requestID, "", "", "", "")
		r = r.WithContext(ctx)

		/* Add to response header */
		w.Header().Set("X-Request-ID", requestID)

		next.ServeHTTP(w, r)
	})
}

/* GetRequestID gets the request ID from context */
func GetRequestID(ctx context.Context) string {
	if id, ok := ctx.Value(requestIDKey).(string); ok {
		return id
	}
	return ""
}
