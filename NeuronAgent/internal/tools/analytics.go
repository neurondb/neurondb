/*-------------------------------------------------------------------------
 *
 * analytics.go
 *    Tool usage analytics
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/tools/analytics.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
)

type ToolAnalytics struct {
	queries *db.Queries
}

/* NewToolAnalytics creates a new tool analytics tracker */
func NewToolAnalytics(queries *db.Queries) *ToolAnalytics {
	return &ToolAnalytics{queries: queries}
}

/* RecordUsage records tool usage */
func (a *ToolAnalytics) RecordUsage(ctx context.Context, agentID, sessionID *uuid.UUID, toolName string, executionTime time.Duration, success bool, errorMsg string, tokensUsed int, cost float64) error {
	query := `INSERT INTO neurondb_agent.tool_usage_logs 
		(agent_id, session_id, tool_name, execution_time_ms, success, error_message, tokens_used, cost, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())`

	executionTimeMs := int(executionTime.Milliseconds())

	_, err := a.queries.DB.ExecContext(ctx, query, agentID, sessionID, toolName, executionTimeMs, success, errorMsg, tokensUsed, cost)
	if err != nil {
		return fmt.Errorf("tool usage recording failed: tool_name='%s', execution_time_ms=%d, success=%v, error=%w",
			toolName, executionTimeMs, success, err)
	}

	return nil
}

/* GetUsageStats gets usage statistics for a tool */
func (a *ToolAnalytics) GetUsageStats(ctx context.Context, toolName string, startDate, endDate time.Time) (*ToolUsageStats, error) {
	query := `SELECT 
		COUNT(*) AS total_calls,
		SUM(CASE WHEN success THEN 1 ELSE 0 END) AS success_calls,
		SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) AS error_calls,
		AVG(execution_time_ms) AS avg_execution_time_ms,
		SUM(tokens_used) AS total_tokens,
		SUM(cost) AS total_cost
		FROM neurondb_agent.tool_usage_logs
		WHERE tool_name = $1 AND created_at BETWEEN $2 AND $3`

	var stats ToolUsageStats
	err := a.queries.DB.GetContext(ctx, &stats, query, toolName, startDate, endDate)
	if err != nil {
		return nil, fmt.Errorf("tool usage stats retrieval failed: tool_name='%s', start_date='%s', end_date='%s', error=%w",
			toolName, startDate.Format(time.RFC3339), endDate.Format(time.RFC3339), err)
	}

	stats.ToolName = toolName
	stats.StartDate = startDate
	stats.EndDate = endDate

	return &stats, nil
}

/* GetAgentToolStats gets tool usage stats for an agent */
func (a *ToolAnalytics) GetAgentToolStats(ctx context.Context, agentID uuid.UUID, startDate, endDate time.Time) ([]ToolUsageStats, error) {
	query := `SELECT 
		tool_name,
		COUNT(*) AS total_calls,
		SUM(CASE WHEN success THEN 1 ELSE 0 END) AS success_calls,
		SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) AS error_calls,
		AVG(execution_time_ms) AS avg_execution_time_ms,
		SUM(tokens_used) AS total_tokens,
		SUM(cost) AS total_cost
		FROM neurondb_agent.tool_usage_logs
		WHERE agent_id = $1 AND created_at BETWEEN $2 AND $3
		GROUP BY tool_name
		ORDER BY total_calls DESC`

	var stats []ToolUsageStats
	err := a.queries.DB.SelectContext(ctx, &stats, query, agentID, startDate, endDate)
	if err != nil {
		return nil, fmt.Errorf("agent tool stats retrieval failed: agent_id='%s', start_date='%s', end_date='%s', error=%w",
			agentID.String(), startDate.Format(time.RFC3339), endDate.Format(time.RFC3339), err)
	}

	return stats, nil
}

/* ToolUsageStats represents tool usage statistics */
type ToolUsageStats struct {
	ToolName           string   `db:"tool_name"`
	TotalCalls         int      `db:"total_calls"`
	SuccessCalls       int      `db:"success_calls"`
	ErrorCalls         int      `db:"error_calls"`
	AvgExecutionTimeMs *float64 `db:"avg_execution_time_ms"`
	TotalTokens        int      `db:"total_tokens"`
	TotalCost          float64  `db:"total_cost"`
	StartDate          time.Time
	EndDate            time.Time
}
