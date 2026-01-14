/*-------------------------------------------------------------------------
 *
 * budget_queries.go
 *    Database queries for budget management
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/db/budget_queries.go
 *
 *-------------------------------------------------------------------------
 */

package db

import (
	"context"
	"database/sql"
	"fmt"
	"time"

	"github.com/google/uuid"
)

/* Budget queries */
const (
	createBudgetQuery = `
		INSERT INTO neurondb_agent.agent_budgets 
		(agent_id, budget_amount, period_type, start_date, end_date, is_active, metadata)
		VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
		RETURNING id, created_at, updated_at`

	getBudgetQuery = `
		SELECT * FROM neurondb_agent.agent_budgets 
		WHERE agent_id = $1 AND period_type = $2 AND is_active = true`

	updateBudgetQuery = `
		UPDATE neurondb_agent.agent_budgets 
		SET budget_amount = $3, start_date = $4, end_date = $5, metadata = $6::jsonb, updated_at = NOW()
		WHERE agent_id = $1 AND period_type = $2 AND is_active = true
		RETURNING id, agent_id, budget_amount, period_type, start_date, end_date, is_active, metadata, created_at, updated_at`

	deactivateBudgetQuery = `
		UPDATE neurondb_agent.agent_budgets 
		SET is_active = false, updated_at = NOW()
		WHERE agent_id = $1 AND period_type = $2`
)

/* AgentBudget represents a budget for an agent */
type AgentBudget struct {
	ID           uuid.UUID  `db:"id"`
	AgentID      uuid.UUID  `db:"agent_id"`
	BudgetAmount float64    `db:"budget_amount"`
	PeriodType   string     `db:"period_type"`
	StartDate    time.Time  `db:"start_date"`
	EndDate      *time.Time `db:"end_date"`
	IsActive     bool       `db:"is_active"`
	Metadata     JSONBMap   `db:"metadata"`
	CreatedAt    time.Time  `db:"created_at"`
	UpdatedAt    time.Time  `db:"updated_at"`
}

/* CreateBudget creates or updates a budget for an agent */
func (q *Queries) CreateBudget(ctx context.Context, budget *AgentBudget) error {
	/* First, deactivate any existing active budget for this agent and period type */
	_, _ = q.DB.ExecContext(ctx, deactivateBudgetQuery, budget.AgentID, budget.PeriodType)

	params := []interface{}{
		budget.AgentID, budget.BudgetAmount, budget.PeriodType,
		budget.StartDate, budget.EndDate, budget.IsActive, budget.Metadata,
	}
	err := q.DB.GetContext(ctx, budget, createBudgetQuery, params...)
	if err != nil {
		return q.formatQueryError("INSERT", createBudgetQuery, len(params), "neurondb_agent.agent_budgets", err)
	}
	return nil
}

/* GetBudget gets the active budget for an agent and period type */
func (q *Queries) GetBudget(ctx context.Context, agentID uuid.UUID, periodType string) (*AgentBudget, error) {
	var budget AgentBudget
	err := q.DB.GetContext(ctx, &budget, getBudgetQuery, agentID, periodType)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("budget not found on %s: query='%s', agent_id='%s', period_type='%s', table='neurondb_agent.agent_budgets', error=%w",
			q.getConnInfoString(), getBudgetQuery, agentID.String(), periodType, err)
	}
	if err != nil {
		return nil, q.formatQueryError("SELECT", getBudgetQuery, 2, "neurondb_agent.agent_budgets", err)
	}
	return &budget, nil
}

/* UpdateBudget updates an existing budget */
func (q *Queries) UpdateBudget(ctx context.Context, budget *AgentBudget) error {
	params := []interface{}{
		budget.AgentID, budget.PeriodType, budget.BudgetAmount,
		budget.StartDate, budget.EndDate, budget.Metadata,
	}
	err := q.DB.GetContext(ctx, budget, updateBudgetQuery, params...)
	if err == sql.ErrNoRows {
		return fmt.Errorf("budget not found on %s: query='%s', agent_id='%s', period_type='%s', table='neurondb_agent.agent_budgets', error=%w",
			q.getConnInfoString(), updateBudgetQuery, budget.AgentID.String(), budget.PeriodType, err)
	}
	if err != nil {
		return q.formatQueryError("UPDATE", updateBudgetQuery, len(params), "neurondb_agent.agent_budgets", err)
	}
	return nil
}
