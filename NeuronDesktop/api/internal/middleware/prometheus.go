package middleware

import (
	"net/http"
	"time"

	"github.com/neurondb/NeuronDesktop/api/internal/metrics"
)

/* PrometheusMetricsMiddleware records Prometheus metrics for HTTP requests */
func PrometheusMetricsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		/* Wrap response writer to capture status code */
		wrapped := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}

		next.ServeHTTP(wrapped, r)

		duration := time.Since(start)

		/* Record metrics */
		endpoint := r.URL.Path
		metrics.RecordHTTPRequest(r.Method, endpoint, wrapped.statusCode, duration.Seconds())
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





