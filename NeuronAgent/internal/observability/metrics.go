/*-------------------------------------------------------------------------
 *
 * metrics.go
 *    Advanced metrics collection for detailed analytics
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/observability/metrics.go
 *
 *-------------------------------------------------------------------------
 */

package observability

import (
	"context"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	/* Agent performance metrics */
	agentSuccessRate = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "neurondb_agent_success_rate",
			Help: "Agent success rate (0-1)",
		},
		[]string{"agent_id"},
	)

	agentAverageResponseTime = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "neurondb_agent_average_response_time_seconds",
			Help: "Average agent response time in seconds",
		},
		[]string{"agent_id"},
	)

	agentTokenEfficiency = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "neurondb_agent_token_efficiency",
			Help: "Tokens per successful execution",
		},
		[]string{"agent_id"},
	)

	/* Tool usage analytics */
	toolUsageFrequency = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "neurondb_agent_tool_usage_frequency_total",
			Help: "Total tool usage frequency by agent",
		},
		[]string{"tool_name", "agent_id"},
	)

	toolSuccessRate = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "neurondb_agent_tool_success_rate",
			Help: "Tool success rate (0-1)",
		},
		[]string{"tool_name"},
	)

	toolAverageExecutionTime = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "neurondb_agent_tool_average_execution_time_seconds",
			Help: "Average tool execution time in seconds",
		},
		[]string{"tool_name"},
	)

	/* Cost patterns */
	costPerAgent = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "neurondb_agent_cost_per_agent_total",
			Help: "Total cost per agent",
		},
		[]string{"agent_id", "cost_type"},
	)

	costPerSession = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "neurondb_agent_cost_per_session_total",
			Help: "Total cost per session",
		},
		[]string{"session_id", "agent_id"},
	)

	costPerUser = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "neurondb_agent_cost_per_user_total",
			Help: "Total cost per user",
		},
		[]string{"user_id", "agent_id"},
	)

	/* Quality metrics */
	qualityScoreAverage = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "neurondb_agent_quality_score_average",
			Help: "Average quality score (0-1)",
		},
		[]string{"agent_id"},
	)

	qualityScoreDistribution = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "neurondb_agent_quality_score_distribution",
			Help:    "Quality score distribution",
			Buckets: []float64{0.0, 0.2, 0.4, 0.6, 0.8, 1.0},
		},
		[]string{"agent_id"},
	)

	/* Context metrics */
	contextWindowUtilization = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "neurondb_agent_context_window_utilization",
			Help:    "Context window utilization (0-1)",
			Buckets: []float64{0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0},
		},
		[]string{"agent_id"},
	)

	memoryChunksRetrieved = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "neurondb_agent_memory_chunks_retrieved",
			Help:    "Number of memory chunks retrieved per query",
			Buckets: []float64{0, 1, 5, 10, 20, 50, 100},
		},
		[]string{"agent_id"},
	)
)

/* RecordAgentPerformance records agent performance metrics */
func RecordAgentPerformance(ctx context.Context, agentID string, success bool, duration time.Duration, tokensUsed int) {
	/* This would typically update counters and calculate rates */
	/* For now, we record raw metrics */
	agentAverageResponseTime.WithLabelValues(agentID).Set(duration.Seconds())
	if success && tokensUsed > 0 {
		agentTokenEfficiency.WithLabelValues(agentID).Set(float64(tokensUsed))
	}
}

/* RecordToolUsage records detailed tool usage analytics */
func RecordToolUsage(ctx context.Context, toolName, agentID string, success bool, duration time.Duration) {
	toolUsageFrequency.WithLabelValues(toolName, agentID).Inc()
	toolAverageExecutionTime.WithLabelValues(toolName).Set(duration.Seconds())
	if success {
		/* Update success rate (simplified - in production would use a rate calculator) */
		toolSuccessRate.WithLabelValues(toolName).Set(1.0)
	} else {
		toolSuccessRate.WithLabelValues(toolName).Set(0.0)
	}
}

/* RecordCost records cost metrics */
func RecordCost(ctx context.Context, agentID, sessionID, userID, costType string, cost float64) {
	costPerAgent.WithLabelValues(agentID, costType).Add(cost)
	if sessionID != "" {
		costPerSession.WithLabelValues(sessionID, agentID).Add(cost)
	}
	if userID != "" {
		costPerUser.WithLabelValues(userID, agentID).Add(cost)
	}
}

/* RecordQualityScore records quality score metrics */
func RecordQualityScore(ctx context.Context, agentID string, score float64) {
	qualityScoreAverage.WithLabelValues(agentID).Set(score)
	qualityScoreDistribution.WithLabelValues(agentID).Observe(score)
}

/* RecordContextMetrics records context window utilization */
func RecordContextMetrics(ctx context.Context, agentID string, tokensUsed, maxTokens int, memoryChunks int) {
	if maxTokens > 0 {
		utilization := float64(tokensUsed) / float64(maxTokens)
		contextWindowUtilization.WithLabelValues(agentID).Observe(utilization)
	}
	if memoryChunks > 0 {
		memoryChunksRetrieved.WithLabelValues(agentID).Observe(float64(memoryChunks))
	}
}


