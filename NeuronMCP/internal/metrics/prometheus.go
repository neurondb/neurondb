/*-------------------------------------------------------------------------
 *
 * prometheus.go
 *    Prometheus exporter for NeuronMCP metrics
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/metrics/prometheus.go
 *
 *-------------------------------------------------------------------------
 */

package metrics

import (
	"fmt"
	"net/http"
)

/* PrometheusExporter exports metrics in Prometheus format */
type PrometheusExporter struct {
	collector *Collector
}

/* NewPrometheusExporter creates a new Prometheus exporter */
func NewPrometheusExporter(collector *Collector) *PrometheusExporter {
	return &PrometheusExporter{
		collector: collector,
	}
}

/* Export exports metrics in Prometheus format */
func (e *PrometheusExporter) Export() string {
	metrics := e.collector.GetMetrics()
	
	output := fmt.Sprintf(`# HELP neurondb_mcp_requests_total Total number of requests
# TYPE neurondb_mcp_requests_total counter
neurondb_mcp_requests_total %d

# HELP neurondb_mcp_errors_total Total number of errors
# TYPE neurondb_mcp_errors_total counter
neurondb_mcp_errors_total %d

# HELP neurondb_mcp_request_duration_seconds Total request duration in seconds
# TYPE neurondb_mcp_request_duration_seconds counter
neurondb_mcp_request_duration_seconds %.6f

# HELP neurondb_mcp_request_duration_seconds_avg Average request duration in seconds
# TYPE neurondb_mcp_request_duration_seconds_avg gauge
neurondb_mcp_request_duration_seconds_avg %.6f
`,
		metrics.RequestCount,
		metrics.ErrorCount,
		metrics.TotalDuration.Seconds(),
		metrics.AverageDuration.Seconds(),
	)

	/* Add method counts */
	for method, count := range metrics.MethodCounts {
		output += fmt.Sprintf(`# HELP neurondb_mcp_method_requests_total Total requests per method
# TYPE neurondb_mcp_method_requests_total counter
neurondb_mcp_method_requests_total{method="%s"} %d
`, method, count)
	}

	/* Add tool counts */
	for tool, count := range metrics.ToolCounts {
		output += fmt.Sprintf(`# HELP neurondb_mcp_tool_requests_total Total requests per tool
# TYPE neurondb_mcp_tool_requests_total counter
neurondb_mcp_tool_requests_total{tool="%s"} %d
`, tool, count)
	}

	/* Add pool metrics if available */
	if poolStats := metrics.PoolStats; poolStats != nil {
		output += fmt.Sprintf(`# HELP neurondb_mcp_pool_connections_total Total pool connections
# TYPE neurondb_mcp_pool_connections_total gauge
neurondb_mcp_pool_connections_total %d

# HELP neurondb_mcp_pool_connections_active Active pool connections
# TYPE neurondb_mcp_pool_connections_active gauge
neurondb_mcp_pool_connections_active %d

# HELP neurondb_mcp_pool_connections_idle Idle pool connections
# TYPE neurondb_mcp_pool_connections_idle gauge
neurondb_mcp_pool_connections_idle %d

# HELP neurondb_mcp_pool_connections_max Maximum pool connections
# TYPE neurondb_mcp_pool_connections_max gauge
neurondb_mcp_pool_connections_max %d

# HELP neurondb_mcp_pool_utilization Pool utilization ratio
# TYPE neurondb_mcp_pool_utilization gauge
neurondb_mcp_pool_utilization %.4f
`,
			poolStats.TotalConnections,
			poolStats.ActiveConnections,
			poolStats.IdleConnections,
			poolStats.MaxConnections,
			poolStats.Utilization,
		)
	}

	return output
}

/* Handler returns an HTTP handler for Prometheus metrics */
func (e *PrometheusExporter) Handler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain; version=0.0.4")
		fmt.Fprint(w, e.Export())
	}
}

