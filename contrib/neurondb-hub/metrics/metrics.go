package metrics

import (
	"net/http"
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	httpRequestsTotal   *prometheus.CounterVec
	httpRequestDuration *prometheus.HistogramVec
	agentCreationsTotal prometheus.Counter
	knowledgeIngestTotal prometheus.Counter
	authFailuresTotal   prometheus.Counter
)

// Register registers all Hub metrics. Call once at startup.
func Register() {
	httpRequestsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "neurondb_hub_http_requests_total",
			Help: "Total HTTP requests",
		},
		[]string{"method", "path", "status"},
	)
	httpRequestDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "neurondb_hub_http_request_duration_seconds",
			Help:    "HTTP request duration in seconds",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"method", "path"},
	)
	agentCreationsTotal = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "neurondb_hub_agent_creations_total",
			Help: "Total agent creations",
		},
	)
	knowledgeIngestTotal = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "neurondb_hub_knowledge_ingest_total",
			Help: "Total knowledge ingest operations",
		},
	)
	authFailuresTotal = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "neurondb_hub_auth_failures_total",
			Help: "Total auth failures",
		},
	)
}

// Middleware returns an http.Handler that records request count and duration.
func Middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		wrapped := &responseWriter{ResponseWriter: w, status: 200}
		next.ServeHTTP(wrapped, r)
		dur := time.Since(start).Seconds()
		path := r.URL.Path
		if path == "" {
			path = "/"
		}
		status := strconv.Itoa(wrapped.status)
		httpRequestsTotal.WithLabelValues(r.Method, path, status).Inc()
		httpRequestDuration.WithLabelValues(r.Method, path).Observe(dur)
	})
}

type responseWriter struct {
	http.ResponseWriter
	status int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.status = code
	rw.ResponseWriter.WriteHeader(code)
}

// Handler returns the Prometheus scrape handler for GET /metrics.
func Handler() http.Handler {
	return promhttp.Handler()
}

// CountAgentCreation increments the agent creation counter.
func CountAgentCreation() {
	if agentCreationsTotal != nil {
		agentCreationsTotal.Inc()
	}
}

// CountKnowledgeIngest increments the knowledge ingest counter.
func CountKnowledgeIngest() {
	if knowledgeIngestTotal != nil {
		knowledgeIngestTotal.Inc()
	}
}

// CountAuthFailure increments the auth failure counter.
func CountAuthFailure() {
	if authFailuresTotal != nil {
		authFailuresTotal.Inc()
	}
}
