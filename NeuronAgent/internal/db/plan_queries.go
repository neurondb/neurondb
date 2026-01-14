/*-------------------------------------------------------------------------
 *
 * plan_queries.go
 *    Database queries for plans
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/db/plan_queries.go
 *
 *-------------------------------------------------------------------------
 */

package db

import (
	"context"
	"database/sql"
	"fmt"

	"github.com/google/uuid"
)

/* Plan queries */
const (
	createPlanQuery = `
		INSERT INTO neurondb_agent.plans 
		(agent_id, session_id, task_description, steps, status, result)
		VALUES ($1, $2, $3, $4::jsonb, $5, $6::jsonb)
		RETURNING id, created_at, updated_at, completed_at`

	getPlanQuery = `SELECT * FROM neurondb_agent.plans WHERE id = $1`

	listPlansQuery = `
		SELECT * FROM neurondb_agent.plans 
		WHERE ($1::uuid IS NULL OR agent_id = $1)
		AND ($2::uuid IS NULL OR session_id = $2)
		ORDER BY created_at DESC 
		LIMIT $3 OFFSET $4`

	updatePlanStatusQuery = `
		UPDATE neurondb_agent.plans 
		SET status = $2, result = $3::jsonb, updated_at = NOW(), completed_at = CASE WHEN $2 IN ('completed', 'failed', 'cancelled') THEN NOW() ELSE completed_at END
		WHERE id = $1
		RETURNING id, agent_id, session_id, task_description, steps, status, result, created_at, updated_at, completed_at`
)

/* Plan represents a stored plan */
type Plan struct {
	ID              uuid.UUID  `db:"id"`
	AgentID         *uuid.UUID `db:"agent_id"`
	SessionID       *uuid.UUID `db:"session_id"`
	TaskDescription string     `db:"task_description"`
	Steps           JSONBMap   `db:"steps"`
	Status          string     `db:"status"`
	Result          JSONBMap   `db:"result"`
	CreatedAt       string     `db:"created_at"`
	UpdatedAt       string     `db:"updated_at"`
	CompletedAt     *string    `db:"completed_at"`
}

/* CreatePlan creates a new plan */
func (q *Queries) CreatePlan(ctx context.Context, plan *Plan) error {
	params := []interface{}{
		plan.AgentID, plan.SessionID, plan.TaskDescription, plan.Steps, plan.Status, plan.Result,
	}
	err := q.DB.GetContext(ctx, plan, createPlanQuery, params...)
	if err != nil {
		return q.formatQueryError("INSERT", createPlanQuery, len(params), "neurondb_agent.plans", err)
	}
	return nil
}

/* GetPlan gets a plan by ID */
func (q *Queries) GetPlan(ctx context.Context, id uuid.UUID) (*Plan, error) {
	var plan Plan
	err := q.DB.GetContext(ctx, &plan, getPlanQuery, id)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("plan not found on %s: query='%s', plan_id='%s', table='neurondb_agent.plans', error=%w",
			q.getConnInfoString(), getPlanQuery, id.String(), err)
	}
	if err != nil {
		return nil, q.formatQueryError("SELECT", getPlanQuery, 1, "neurondb_agent.plans", err)
	}
	return &plan, nil
}

/* ListPlans lists plans with optional filters */
func (q *Queries) ListPlans(ctx context.Context, agentID, sessionID *uuid.UUID, limit, offset int) ([]Plan, error) {
	var plans []Plan
	params := []interface{}{agentID, sessionID, limit, offset}
	err := q.DB.SelectContext(ctx, &plans, listPlansQuery, params...)
	if err != nil {
		return nil, q.formatQueryError("SELECT", listPlansQuery, len(params), "neurondb_agent.plans", err)
	}
	return plans, nil
}

/* UpdatePlanStatus updates a plan's status */
func (q *Queries) UpdatePlanStatus(ctx context.Context, id uuid.UUID, status string, result JSONBMap) (*Plan, error) {
	var plan Plan
	params := []interface{}{id, status, result}
	err := q.DB.GetContext(ctx, &plan, updatePlanStatusQuery, params...)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("plan not found on %s: query='%s', plan_id='%s', table='neurondb_agent.plans', error=%w",
			q.getConnInfoString(), updatePlanStatusQuery, id.String(), err)
	}
	if err != nil {
		return nil, q.formatQueryError("UPDATE", updatePlanStatusQuery, len(params), "neurondb_agent.plans", err)
	}
	return &plan, nil
}
