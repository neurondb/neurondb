/*-------------------------------------------------------------------------
 *
 * cost_tracker.go
 *    Cost tracking and budget management
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/cost_tracker.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
)

type CostTracker struct {
	queries *db.Queries
}

/* NewCostTracker creates a new cost tracker */
func NewCostTracker(queries *db.Queries) *CostTracker {
	return &CostTracker{queries: queries}
}

/* RecordCost records a cost for an agent/session */
func (c *CostTracker) RecordCost(ctx context.Context, agentID, sessionID uuid.UUID, costType string, tokens int, cost float64) error {
	query := `INSERT INTO neurondb_agent.cost_logs 
		(agent_id, session_id, cost_type, tokens_used, cost, created_at)
		VALUES ($1, $2, $3, $4, $5, NOW())`

	_, err := c.queries.DB.ExecContext(ctx, query, agentID, sessionID, costType, tokens, cost)
	if err != nil {
		return fmt.Errorf("cost recording failed: agent_id='%s', session_id='%s', cost_type='%s', tokens=%d, cost=%.4f, error=%w",
			agentID.String(), sessionID.String(), costType, tokens, cost, err)
	}

	return nil
}

/* GetCostSummary gets cost summary for an agent */
func (c *CostTracker) GetCostSummary(ctx context.Context, agentID uuid.UUID, startDate, endDate time.Time) (*CostSummary, error) {
	query := `SELECT 
		SUM(tokens_used) AS total_tokens,
		SUM(cost) AS total_cost,
		COUNT(*) AS request_count
		FROM neurondb_agent.cost_logs
		WHERE agent_id = $1 AND created_at BETWEEN $2 AND $3`

	var summary CostSummary
	err := c.queries.DB.GetContext(ctx, &summary, query, agentID, startDate, endDate)
	if err != nil {
		return nil, fmt.Errorf("cost summary retrieval failed: agent_id='%s', start_date='%s', end_date='%s', error=%w",
			agentID.String(), startDate.Format(time.RFC3339), endDate.Format(time.RFC3339), err)
	}

	summary.AgentID = agentID
	summary.StartDate = startDate
	summary.EndDate = endDate

	return &summary, nil
}

/* CheckBudget checks if agent is within budget */
func (c *CostTracker) CheckBudget(ctx context.Context, agentID uuid.UUID, budget float64) (bool, float64, error) {
	/* Get current period cost (last 30 days) */
	endDate := time.Now()
	startDate := endDate.AddDate(0, 0, -30)

	summary, err := c.GetCostSummary(ctx, agentID, startDate, endDate)
	if err != nil {
		return false, 0, err
	}

	withinBudget := summary.TotalCost < budget
	remaining := budget - summary.TotalCost
	if remaining < 0 {
		remaining = 0
	}

	return withinBudget, remaining, nil
}

/* EstimateCost estimates cost for a request */
func (c *CostTracker) EstimateCost(model string, promptTokens, completionTokens int) float64 {
	/* Simple cost estimation based on model */
	/* In production, this would use actual pricing data */
	costPer1KTokens := map[string]float64{
		"gpt-4":         0.03,
		"gpt-4-turbo":   0.01,
		"gpt-3.5-turbo": 0.002,
		"default":       0.01,
	}

	rate, ok := costPer1KTokens[model]
	if !ok {
		rate = costPer1KTokens["default"]
	}

	promptCost := (float64(promptTokens) / 1000.0) * rate
	completionCost := (float64(completionTokens) / 1000.0) * rate * 2 /* Completion usually costs more */

	return promptCost + completionCost
}

/* CostSummary represents a cost summary */
type CostSummary struct {
	AgentID      uuid.UUID `db:"agent_id"`
	TotalTokens  int       `db:"total_tokens"`
	TotalCost    float64   `db:"total_cost"`
	RequestCount int       `db:"request_count"`
	StartDate    time.Time
	EndDate      time.Time
}
