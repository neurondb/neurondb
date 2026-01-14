/*-------------------------------------------------------------------------
 *
 * replay_queries.go
 *    Database queries for execution snapshots and replay
 *
 * Provides database query functions for storing and retrieving execution snapshots.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/db/replay_queries.go
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

/* Execution snapshot queries */
const (
	createExecutionSnapshotQuery = `
		INSERT INTO neurondb_agent.execution_snapshots 
		(session_id, agent_id, user_message, execution_state, deterministic_mode)
		VALUES ($1, $2, $3, $4::jsonb, $5)
		RETURNING id, created_at`

	getExecutionSnapshotByIDQuery = `SELECT * FROM neurondb_agent.execution_snapshots WHERE id = $1`

	listExecutionSnapshotsBySessionQuery = `
		SELECT * FROM neurondb_agent.execution_snapshots 
		WHERE session_id = $1 
		ORDER BY created_at DESC
		LIMIT $2 OFFSET $3`

	listExecutionSnapshotsByAgentQuery = `
		SELECT * FROM neurondb_agent.execution_snapshots 
		WHERE agent_id = $1 
		ORDER BY created_at DESC
		LIMIT $2 OFFSET $3`

	deleteExecutionSnapshotQuery = `DELETE FROM neurondb_agent.execution_snapshots WHERE id = $1`
)

/* Execution snapshot methods */
func (q *Queries) CreateExecutionSnapshot(ctx context.Context, snapshot *ExecutionSnapshot) error {
	executionStateValue, err := snapshot.ExecutionState.Value()
	if err != nil {
		return fmt.Errorf("failed to convert execution_state: %w", err)
	}

	params := []interface{}{snapshot.SessionID, snapshot.AgentID, snapshot.UserMessage, executionStateValue, snapshot.DeterministicMode}
	err = q.DB.GetContext(ctx, snapshot, createExecutionSnapshotQuery, params...)
	if err != nil {
		return fmt.Errorf("execution snapshot creation failed on %s: query='%s', params_count=%d, session_id='%s', agent_id='%s', table='neurondb_agent.execution_snapshots', error=%w",
			q.getConnInfoString(), createExecutionSnapshotQuery, len(params), snapshot.SessionID.String(), snapshot.AgentID.String(), err)
	}
	return nil
}

func (q *Queries) GetExecutionSnapshotByID(ctx context.Context, id uuid.UUID) (*ExecutionSnapshot, error) {
	var snapshot ExecutionSnapshot
	err := q.DB.GetContext(ctx, &snapshot, getExecutionSnapshotByIDQuery, id)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("execution snapshot not found on %s: query='%s', snapshot_id='%s', table='neurondb_agent.execution_snapshots', error=%w",
			q.getConnInfoString(), getExecutionSnapshotByIDQuery, id.String(), err)
	}
	if err != nil {
		return nil, q.formatQueryError("SELECT", getExecutionSnapshotByIDQuery, 1, "neurondb_agent.execution_snapshots", err)
	}
	return &snapshot, nil
}

func (q *Queries) ListExecutionSnapshotsBySession(ctx context.Context, sessionID uuid.UUID, limit, offset int) ([]ExecutionSnapshot, error) {
	var snapshots []ExecutionSnapshot
	params := []interface{}{sessionID, limit, offset}
	err := q.DB.SelectContext(ctx, &snapshots, listExecutionSnapshotsBySessionQuery, params...)
	if err != nil {
		return nil, q.formatQueryError("SELECT", listExecutionSnapshotsBySessionQuery, len(params), "neurondb_agent.execution_snapshots", err)
	}
	return snapshots, nil
}

func (q *Queries) ListExecutionSnapshotsByAgent(ctx context.Context, agentID uuid.UUID, limit, offset int) ([]ExecutionSnapshot, error) {
	var snapshots []ExecutionSnapshot
	params := []interface{}{agentID, limit, offset}
	err := q.DB.SelectContext(ctx, &snapshots, listExecutionSnapshotsByAgentQuery, params...)
	if err != nil {
		return nil, q.formatQueryError("SELECT", listExecutionSnapshotsByAgentQuery, len(params), "neurondb_agent.execution_snapshots", err)
	}
	return snapshots, nil
}

func (q *Queries) DeleteExecutionSnapshot(ctx context.Context, id uuid.UUID) error {
	_, err := q.DB.ExecContext(ctx, deleteExecutionSnapshotQuery, id)
	if err != nil {
		return fmt.Errorf("execution snapshot deletion failed: %w", err)
	}
	return nil
}
