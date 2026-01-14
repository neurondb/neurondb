/*-------------------------------------------------------------------------
 *
 * prometheus.go
 *    Database operations
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/metrics/prometheus.go
 *
 *-------------------------------------------------------------------------
 */

package metrics

import (
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	/* Request metrics */
	httpRequestsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "neurondb_agent_http_requests_total",
			Help: "Total number of HTTP requests",
		},
		[]string{"method", "endpoint", "status"},
	)

	httpRequestDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "neurondb_agent_http_request_duration_seconds",
			Help:    "HTTP request duration in seconds",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"method", "endpoint"},
	)

	/* Agent metrics */
	agentExecutionsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "neurondb_agent_executions_total",
			Help: "Total number of agent executions",
		},
		[]string{"agent_id", "status"},
	)

	agentExecutionDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "neurondb_agent_execution_duration_seconds",
			Help:    "Agent execution duration in seconds",
			Buckets: []float64{0.1, 0.5, 1, 2, 5, 10, 30, 60},
		},
		[]string{"agent_id"},
	)

	/* LLM metrics */
	llmCallsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "neurondb_agent_llm_calls_total",
			Help: "Total number of LLM calls",
		},
		[]string{"model", "status"},
	)

	llmTokensTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "neurondb_agent_llm_tokens_total",
			Help: "Total number of LLM tokens",
		},
		[]string{"model", "type"},
	)

	/* Memory metrics */
	memoryChunksStored = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "neurondb_agent_memory_chunks_stored_total",
			Help: "Total number of memory chunks stored",
		},
		[]string{"agent_id"},
	)

	memoryRetrievalsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "neurondb_agent_memory_retrievals_total",
			Help: "Total number of memory retrievals",
		},
		[]string{"agent_id"},
	)

	/* Tool metrics */
	toolExecutionsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "neurondb_agent_tool_executions_total",
			Help: "Total number of tool executions",
		},
		[]string{"tool_name", "status"},
	)

	toolExecutionDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "neurondb_agent_tool_execution_duration_seconds",
			Help:    "Tool execution duration in seconds",
			Buckets: []float64{0.01, 0.1, 0.5, 1, 5, 10, 30},
		},
		[]string{"tool_name"},
	)

	/* Job metrics */
	jobsQueued = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "neurondb_agent_jobs_queued",
			Help: "Number of jobs in queue",
		},
	)

	jobsProcessedTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "neurondb_agent_jobs_processed_total",
			Help: "Total number of jobs processed",
		},
		[]string{"type", "status"},
	)

	/* Database connection pool metrics */
	dbPoolOpenConns = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "neurondb_agent_db_pool_open_connections",
			Help: "Number of open database connections",
		},
		[]string{"database"},
	)

	dbPoolIdleConns = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "neurondb_agent_db_pool_idle_connections",
			Help: "Number of idle database connections",
		},
		[]string{"database"},
	)

	dbPoolInUseConns = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "neurondb_agent_db_pool_in_use_connections",
			Help: "Number of database connections in use",
		},
		[]string{"database"},
	)

	/* Embedding metrics */
	embeddingGenerationDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "neurondb_agent_embedding_generation_duration_seconds",
			Help:    "Embedding generation duration in seconds",
			Buckets: []float64{0.01, 0.05, 0.1, 0.5, 1, 2, 5},
		},
		[]string{"model"},
	)

	embeddingGenerationTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "neurondb_agent_embedding_generation_total",
			Help: "Total number of embedding generations",
		},
		[]string{"model", "status"},
	)

	/* Vector search metrics */
	vectorSearchDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "neurondb_agent_vector_search_duration_seconds",
			Help:    "Vector search duration in seconds",
			Buckets: []float64{0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1},
		},
		[]string{"agent_id"},
	)

	vectorSearchTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "neurondb_agent_vector_search_total",
			Help: "Total number of vector searches",
		},
		[]string{"agent_id", "status"},
	)

	/* Rate limiting metrics */
	rateLimitAllowed = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "neurondb_agent_rate_limit_allowed_total",
			Help: "Total number of requests allowed by rate limiter",
		},
		[]string{"key_id"},
	)

	rateLimitRejected = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "neurondb_agent_rate_limit_rejected_total",
			Help: "Total number of requests rejected by rate limiter",
		},
		[]string{"key_id"},
	)
)

/* RecordHTTPRequest records an HTTP request */
func RecordHTTPRequest(method, endpoint string, status int, duration time.Duration) {
	/* Convert status code to status class for better PromQL queries */
	statusClass := "unknown"
	if status >= 200 && status < 300 {
		statusClass = "2xx"
	} else if status >= 300 && status < 400 {
		statusClass = "3xx"
	} else if status >= 400 && status < 500 {
		statusClass = "4xx"
	} else if status >= 500 {
		statusClass = "5xx"
	}

	httpRequestsTotal.WithLabelValues(method, endpoint, statusClass).Inc()
	httpRequestDuration.WithLabelValues(method, endpoint).Observe(duration.Seconds())
}

/* RecordAgentExecution records an agent execution */
func RecordAgentExecution(agentID, status string, duration time.Duration) {
	agentExecutionsTotal.WithLabelValues(agentID, status).Inc()
	agentExecutionDuration.WithLabelValues(agentID).Observe(duration.Seconds())
}

/* RecordLLMCall records an LLM call */
func RecordLLMCall(model, status string, promptTokens, completionTokens int) {
	llmCallsTotal.WithLabelValues(model, status).Inc()
	llmTokensTotal.WithLabelValues(model, "prompt").Add(float64(promptTokens))
	llmTokensTotal.WithLabelValues(model, "completion").Add(float64(completionTokens))
}

/* RecordMemoryChunkStored records a memory chunk being stored */
func RecordMemoryChunkStored(agentID string) {
	memoryChunksStored.WithLabelValues(agentID).Inc()
}

/* RecordMemoryRetrieval records a memory retrieval */
func RecordMemoryRetrieval(agentID string) {
	memoryRetrievalsTotal.WithLabelValues(agentID).Inc()
}

/* RecordToolExecution records a tool execution */
func RecordToolExecution(toolName, status string, duration time.Duration) {
	toolExecutionsTotal.WithLabelValues(toolName, status).Inc()
	toolExecutionDuration.WithLabelValues(toolName).Observe(duration.Seconds())
}

/* RecordJobQueued records a job being queued */
func RecordJobQueued() {
	jobsQueued.Inc()
}

/* RecordJobProcessed records a job being processed */
func RecordJobProcessed(jobType, status string) {
	jobsProcessedTotal.WithLabelValues(jobType, status).Inc()
	jobsQueued.Dec()
}

/* RecordDBPoolStats records database connection pool statistics */
func RecordDBPoolStats(database string, openConns, idleConns, inUse int) {
	dbPoolOpenConns.WithLabelValues(database).Set(float64(openConns))
	dbPoolIdleConns.WithLabelValues(database).Set(float64(idleConns))
	dbPoolInUseConns.WithLabelValues(database).Set(float64(inUse))
}

/* RecordEmbeddingGeneration records embedding generation metrics */
func RecordEmbeddingGeneration(model, status string, duration time.Duration) {
	embeddingGenerationTotal.WithLabelValues(model, status).Inc()
	embeddingGenerationDuration.WithLabelValues(model).Observe(duration.Seconds())
}

/* RecordVectorSearch records vector search metrics */
func RecordVectorSearch(agentID, status string, duration time.Duration) {
	vectorSearchTotal.WithLabelValues(agentID, status).Inc()
	vectorSearchDuration.WithLabelValues(agentID).Observe(duration.Seconds())
}

/* RecordRateLimitAllowed records a rate limit allowance */
func RecordRateLimitAllowed(keyID string) {
	rateLimitAllowed.WithLabelValues(keyID).Inc()
}

/* RecordRateLimitRejected records a rate limit rejection */
func RecordRateLimitRejected(keyID string) {
	rateLimitRejected.WithLabelValues(keyID).Inc()
}

/* Handler returns the Prometheus metrics handler */
func Handler() http.Handler {
	return promhttp.Handler()
}
