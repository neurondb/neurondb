package api

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/neurondb/NeuronAgent/internal/auth"
	"github.com/neurondb/NeuronAgent/internal/metrics"
)

type contextKey string

const apiKeyContextKey contextKey = "api_key"

// AuthMiddleware authenticates requests using API keys
func AuthMiddleware(keyManager *auth.APIKeyManager, rateLimiter *auth.RateLimiter) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Skip auth for health and metrics endpoints
			if r.URL.Path == "/health" || r.URL.Path == "/metrics" {
				next.ServeHTTP(w, r)
				return
			}

			// Get API key from header
			authHeader := r.Header.Get("Authorization")
			if authHeader == "" {
				requestID := GetRequestID(r.Context())
				respondError(w, WrapError(ErrUnauthorized, requestID))
				return
			}

			// Extract key (format: "Bearer <key>" or "ApiKey <key>")
			parts := strings.Fields(authHeader)
			if len(parts) != 2 {
				requestID := GetRequestID(r.Context())
				respondError(w, WrapError(ErrUnauthorized, requestID))
				return
			}

			key := parts[1]
			keyPrefix := key
			if len(keyPrefix) > 8 {
				keyPrefix = keyPrefix[:8]
			}
			fmt.Printf("[MIDDLEWARE] Extracted key: prefix=%s, len=%d\n", keyPrefix, len(key))

			// Validate key
			apiKey, err := keyManager.ValidateAPIKey(r.Context(), key)
			if err != nil {
				requestID := GetRequestID(r.Context())
				// Log the actual error for debugging
				prefix := key
				if len(prefix) > 8 {
					prefix = prefix[:8]
				}
				fmt.Printf("[MIDDLEWARE] Authentication failed: %v, prefix=%s\n", err, prefix)
				respondError(w, WrapError(ErrUnauthorized, requestID))
				return
			}
			fmt.Printf("[MIDDLEWARE] Authentication succeeded: prefix=%s\n", apiKey.KeyPrefix)

			// Check rate limit
			if !rateLimiter.CheckLimit(apiKey.ID.String(), apiKey.RateLimitPerMin) {
				requestID := GetRequestID(r.Context())
				respondError(w, WrapError(NewError(http.StatusTooManyRequests, "rate limit exceeded", nil), requestID))
				return
			}

			// Add API key to context
			ctx := context.WithValue(r.Context(), apiKeyContextKey, apiKey)
			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
}

// CORSMiddleware adds CORS headers
func CORSMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// LoggingMiddleware logs requests with structured logging and metrics
func LoggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		
		// Wrap response writer to capture status code
		wrapped := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}
		
		next.ServeHTTP(wrapped, r)
		
		duration := time.Since(start)
		
		// Record metrics
		endpoint := r.URL.Path
		metrics.RecordHTTPRequest(r.Method, endpoint, wrapped.statusCode, duration)
	})
}

type responseWriter struct {
	http.ResponseWriter
	statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}

