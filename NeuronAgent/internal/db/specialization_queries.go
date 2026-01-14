/*-------------------------------------------------------------------------
 *
 * specialization_queries.go
 *    Database queries for agent specializations
 *
 * Provides database query functions for agent specialization management.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/db/specialization_queries.go
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

/* Agent specialization queries */
const (
	createAgentSpecializationQuery = `
		INSERT INTO neurondb_agent.agent_specializations 
		(agent_id, specialization_type, capabilities, config)
		VALUES ($1, $2, $3, $4::jsonb)
		ON CONFLICT (agent_id) 
		DO UPDATE SET specialization_type = EXCLUDED.specialization_type, 
		              capabilities = EXCLUDED.capabilities, 
		              config = EXCLUDED.config, 
		              updated_at = NOW()
		RETURNING id, created_at, updated_at`

	getAgentSpecializationByAgentIDQuery = `SELECT * FROM neurondb_agent.agent_specializations WHERE agent_id = $1`

	getAgentSpecializationByIDQuery = `SELECT * FROM neurondb_agent.agent_specializations WHERE id = $1`

	updateAgentSpecializationQuery = `
		UPDATE neurondb_agent.agent_specializations 
		SET specialization_type = $2, capabilities = $3, config = $4::jsonb, updated_at = NOW()
		WHERE id = $1
		RETURNING updated_at`

	deleteAgentSpecializationQuery = `DELETE FROM neurondb_agent.agent_specializations WHERE id = $1`

	deleteAgentSpecializationByAgentIDQuery = `DELETE FROM neurondb_agent.agent_specializations WHERE agent_id = $1`

	listAgentSpecializationsQuery = `
		SELECT * FROM neurondb_agent.agent_specializations 
		WHERE ($1::text IS NULL OR specialization_type = $1)
		ORDER BY created_at DESC`
)

/* Agent specialization methods */
func (q *Queries) CreateAgentSpecialization(ctx context.Context, specialization *AgentSpecialization) error {
	configValue, err := specialization.Config.Value()
	if err != nil {
		return fmt.Errorf("failed to convert config: %w", err)
	}

	params := []interface{}{specialization.AgentID, specialization.SpecializationType, specialization.Capabilities, configValue}
	err = q.DB.GetContext(ctx, specialization, createAgentSpecializationQuery, params...)
	if err != nil {
		return fmt.Errorf("agent specialization creation failed: %w", err)
	}
	return nil
}

func (q *Queries) GetAgentSpecializationByAgentID(ctx context.Context, agentID uuid.UUID) (*AgentSpecialization, error) {
	var specialization AgentSpecialization
	err := q.DB.GetContext(ctx, &specialization, getAgentSpecializationByAgentIDQuery, agentID)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("agent specialization not found: %w", err)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get agent specialization: %w", err)
	}
	return &specialization, nil
}

func (q *Queries) GetAgentSpecializationByID(ctx context.Context, id uuid.UUID) (*AgentSpecialization, error) {
	var specialization AgentSpecialization
	err := q.DB.GetContext(ctx, &specialization, getAgentSpecializationByIDQuery, id)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("agent specialization not found: %w", err)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get agent specialization: %w", err)
	}
	return &specialization, nil
}

func (q *Queries) UpdateAgentSpecialization(ctx context.Context, specialization *AgentSpecialization) error {
	configValue, err := specialization.Config.Value()
	if err != nil {
		return fmt.Errorf("failed to convert config: %w", err)
	}

	params := []interface{}{specialization.ID, specialization.SpecializationType, specialization.Capabilities, configValue}
	err = q.DB.GetContext(ctx, specialization, updateAgentSpecializationQuery, params...)
	if err != nil {
		return fmt.Errorf("agent specialization update failed: %w", err)
	}
	return nil
}

func (q *Queries) DeleteAgentSpecialization(ctx context.Context, id uuid.UUID) error {
	result, err := q.DB.ExecContext(ctx, deleteAgentSpecializationQuery, id)
	if err != nil {
		return fmt.Errorf("agent specialization deletion failed: %w", err)
	}
	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}
	if rowsAffected == 0 {
		return fmt.Errorf("agent specialization not found")
	}
	return nil
}

func (q *Queries) DeleteAgentSpecializationByAgentID(ctx context.Context, agentID uuid.UUID) error {
	result, err := q.DB.ExecContext(ctx, deleteAgentSpecializationByAgentIDQuery, agentID)
	if err != nil {
		return fmt.Errorf("agent specialization deletion failed: %w", err)
	}
	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}
	if rowsAffected == 0 {
		return fmt.Errorf("agent specialization not found")
	}
	return nil
}

func (q *Queries) ListAgentSpecializations(ctx context.Context, specializationType *string) ([]AgentSpecialization, error) {
	var specializations []AgentSpecialization
	err := q.DB.SelectContext(ctx, &specializations, listAgentSpecializationsQuery, specializationType)
	if err != nil {
		return nil, fmt.Errorf("failed to list agent specializations: %w", err)
	}
	return specializations, nil
}







