/*-------------------------------------------------------------------------
 *
 * middleware.go
 *    OpenTelemetry HTTP middleware for tracing
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/observability/middleware.go
 *
 *-------------------------------------------------------------------------
 */

package observability

import (
	"fmt"
	"net/http"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/propagation"
	semconv "go.opentelemetry.io/otel/semconv/v1.24.0"
	"go.opentelemetry.io/otel/trace"
)

/* TracingMiddleware adds OpenTelemetry tracing to HTTP handlers */
func TracingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		/* Extract trace context from headers */
		ctx := otel.GetTextMapPropagator().Extract(r.Context(), propagation.HeaderCarrier(r.Header))

		/* Start span */
		tracer := otel.Tracer("neurondb-agent/http")
		spanName := fmt.Sprintf("%s %s", r.Method, r.URL.Path)
		ctx, span := tracer.Start(ctx, spanName,
			trace.WithAttributes(
				semconv.HTTPMethodKey.String(r.Method),
				semconv.HTTPURLKey.String(r.URL.String()),
				semconv.HTTPRouteKey.String(r.URL.Path),
				attribute.String("http.scheme", r.URL.Scheme),
				attribute.String("http.host", r.Host),
				attribute.String("http.user_agent", r.UserAgent()),
				attribute.String("http.request_id", r.Header.Get("X-Request-ID")),
			),
			trace.WithSpanKind(trace.SpanKindServer),
		)
		defer span.End()

		/* Inject trace context into request */
		otel.GetTextMapPropagator().Inject(ctx, propagation.HeaderCarrier(r.Header))
		r = r.WithContext(ctx)

		/* Wrap response writer to capture status code */
		wrapped := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}

		/* Call next handler */
		next.ServeHTTP(wrapped, r)

		/* Record status code and status */
		span.SetAttributes(
			semconv.HTTPStatusCodeKey.Int(wrapped.statusCode),
		)
		if wrapped.statusCode >= 400 {
			span.SetStatus(codes.Error, http.StatusText(wrapped.statusCode))
		} else {
			span.SetStatus(codes.Ok, "")
		}
	})
}

/* responseWriter wraps http.ResponseWriter to capture status code */
type responseWriter struct {
	http.ResponseWriter
	statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}

func (rw *responseWriter) Write(b []byte) (int, error) {
	if rw.statusCode == 0 {
		rw.statusCode = http.StatusOK
	}
	return rw.ResponseWriter.Write(b)
}
