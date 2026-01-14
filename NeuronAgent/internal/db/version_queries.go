/*-------------------------------------------------------------------------
 *
 * version_queries.go
 *    Database queries for agent versioning
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/db/version_queries.go
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

/* Agent version queries */
const (
	createAgentVersionQuery = `
		INSERT INTO neurondb_agent.agent_versions 
		(agent_id, version_number, name, description, system_prompt, model_name, enabled_tools, config)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
		RETURNING id, is_active, created_at`

	getAgentVersionQuery = `
		SELECT * FROM neurondb_agent.agent_versions 
		WHERE agent_id = $1 AND version_number = $2`

	listAgentVersionsQuery = `
		SELECT * FROM neurondb_agent.agent_versions 
		WHERE agent_id = $1 
		ORDER BY version_number DESC`

	getActiveAgentVersionQuery = `
		SELECT * FROM neurondb_agent.agent_versions 
		WHERE agent_id = $1 AND is_active = true`

	activateAgentVersionQuery = `
		UPDATE neurondb_agent.agent_versions 
		SET is_active = CASE WHEN version_number = $2 THEN true ELSE false END
		WHERE agent_id = $1`
)

/* AgentVersion represents an agent version */
type AgentVersion struct {
	ID            uuid.UUID `db:"id"`
	AgentID       uuid.UUID `db:"agent_id"`
	VersionNumber int       `db:"version_number"`
	Name          *string   `db:"name"`
	Description   *string   `db:"description"`
	SystemPrompt  string    `db:"system_prompt"`
	ModelName     string    `db:"model_name"`
	EnabledTools  []string  `db:"enabled_tools"`
	Config        JSONBMap  `db:"config"`
	IsActive      bool      `db:"is_active"`
	CreatedAt     string    `db:"created_at"`
}

/* CreateAgentVersion creates a new agent version */
func (q *Queries) CreateAgentVersion(ctx context.Context, version *AgentVersion) error {
	params := []interface{}{
		version.AgentID, version.VersionNumber, version.Name, version.Description,
		version.SystemPrompt, version.ModelName, version.EnabledTools, version.Config,
	}
	err := q.DB.GetContext(ctx, version, createAgentVersionQuery, params...)
	if err != nil {
		return q.formatQueryError("INSERT", createAgentVersionQuery, len(params), "neurondb_agent.agent_versions", err)
	}
	return nil
}

/* GetAgentVersion gets a specific agent version */
func (q *Queries) GetAgentVersion(ctx context.Context, agentID uuid.UUID, versionNumber int) (*AgentVersion, error) {
	var version AgentVersion
	err := q.DB.GetContext(ctx, &version, getAgentVersionQuery, agentID, versionNumber)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("agent version not found on %s: query='%s', agent_id='%s', version_number=%d, table='neurondb_agent.agent_versions', error=%w",
			q.getConnInfoString(), getAgentVersionQuery, agentID.String(), versionNumber, err)
	}
	if err != nil {
		return nil, q.formatQueryError("SELECT", getAgentVersionQuery, 2, "neurondb_agent.agent_versions", err)
	}
	return &version, nil
}

/* ListAgentVersions lists all versions for an agent */
func (q *Queries) ListAgentVersions(ctx context.Context, agentID uuid.UUID) ([]AgentVersion, error) {
	var versions []AgentVersion
	err := q.DB.SelectContext(ctx, &versions, listAgentVersionsQuery, agentID)
	if err != nil {
		return nil, q.formatQueryError("SELECT", listAgentVersionsQuery, 1, "neurondb_agent.agent_versions", err)
	}
	return versions, nil
}

/* GetActiveAgentVersion gets the active version for an agent */
func (q *Queries) GetActiveAgentVersion(ctx context.Context, agentID uuid.UUID) (*AgentVersion, error) {
	var version AgentVersion
	err := q.DB.GetContext(ctx, &version, getActiveAgentVersionQuery, agentID)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("active agent version not found on %s: query='%s', agent_id='%s', table='neurondb_agent.agent_versions', error=%w",
			q.getConnInfoString(), getActiveAgentVersionQuery, agentID.String(), err)
	}
	if err != nil {
		return nil, q.formatQueryError("SELECT", getActiveAgentVersionQuery, 1, "neurondb_agent.agent_versions", err)
	}
	return &version, nil
}

/* ActivateAgentVersion activates a specific version and deactivates others */
func (q *Queries) ActivateAgentVersion(ctx context.Context, agentID uuid.UUID, versionNumber int) error {
	_, err := q.DB.ExecContext(ctx, activateAgentVersionQuery, agentID, versionNumber)
	if err != nil {
		return q.formatQueryError("UPDATE", activateAgentVersionQuery, 2, "neurondb_agent.agent_versions", err)
	}
	return nil
}
