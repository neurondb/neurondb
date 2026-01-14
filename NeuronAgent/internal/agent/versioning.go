/*-------------------------------------------------------------------------
 *
 * versioning.go
 *    Agent versioning and cloning
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/versioning.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"context"
	"fmt"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
)

type VersionManager struct {
	queries *db.Queries
}

/* NewVersionManager creates a new version manager */
func NewVersionManager(queries *db.Queries) *VersionManager {
	return &VersionManager{queries: queries}
}

/* CreateVersion creates a new version of an agent */
func (v *VersionManager) CreateVersion(ctx context.Context, agentID uuid.UUID) (*AgentVersion, error) {
	/* Get current agent */
	agent, err := v.queries.GetAgentByID(ctx, agentID)
	if err != nil {
		return nil, fmt.Errorf("agent versioning failed: agent_id='%s', agent_not_found=true, error=%w",
			agentID.String(), err)
	}

	/* Get next version number */
	query := `SELECT COALESCE(MAX(version_number), 0) + 1 AS next_version
		FROM neurondb_agent.agent_versions
		WHERE agent_id = $1`

	var nextVersion int
	err = v.queries.DB.GetContext(ctx, &nextVersion, query, agentID)
	if err != nil {
		return nil, fmt.Errorf("agent versioning failed: agent_id='%s', version_number_retrieval_failed=true, error=%w",
			agentID.String(), err)
	}

	/* Create version */
	versionQuery := `INSERT INTO neurondb_agent.agent_versions
		(agent_id, version_number, name, description, system_prompt, model_name, enabled_tools, config, is_active)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, false)
		RETURNING id, created_at`

	var version AgentVersion
	err = v.queries.DB.GetContext(ctx, &version, versionQuery,
		agentID, nextVersion, agent.Name, agent.Description, agent.SystemPrompt,
		agent.ModelName, agent.EnabledTools, agent.Config)
	if err != nil {
		return nil, fmt.Errorf("agent versioning failed: agent_id='%s', version_number=%d, version_creation_failed=true, error=%w",
			agentID.String(), nextVersion, err)
	}

	version.AgentID = agentID
	version.VersionNumber = nextVersion
	version.Name = agent.Name
	version.Description = agent.Description
	version.SystemPrompt = agent.SystemPrompt
	version.ModelName = agent.ModelName
	version.EnabledTools = agent.EnabledTools
	version.Config = agent.Config
	version.IsActive = false

	return &version, nil
}

/* ActivateVersion activates a specific version */
func (v *VersionManager) ActivateVersion(ctx context.Context, agentID uuid.UUID, versionNumber int) error {
	/* Deactivate all versions */
	deactivateQuery := `UPDATE neurondb_agent.agent_versions
		SET is_active = false
		WHERE agent_id = $1`

	_, err := v.queries.DB.ExecContext(ctx, deactivateQuery, agentID)
	if err != nil {
		return fmt.Errorf("agent version activation failed: agent_id='%s', version_number=%d, deactivation_failed=true, error=%w",
			agentID.String(), versionNumber, err)
	}

	/* Activate specified version */
	activateQuery := `UPDATE neurondb_agent.agent_versions
		SET is_active = true
		WHERE agent_id = $1 AND version_number = $2`

	result, err := v.queries.DB.ExecContext(ctx, activateQuery, agentID, versionNumber)
	if err != nil {
		return fmt.Errorf("agent version activation failed: agent_id='%s', version_number=%d, activation_failed=true, error=%w",
			agentID.String(), versionNumber, err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("agent version activation failed: agent_id='%s', version_number=%d, rows_affected_check_failed=true, error=%w",
			agentID.String(), versionNumber, err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("agent version activation failed: agent_id='%s', version_number=%d, version_not_found=true",
			agentID.String(), versionNumber)
	}

	return nil
}

/* ListVersions lists all versions of an agent */
func (v *VersionManager) ListVersions(ctx context.Context, agentID uuid.UUID) ([]AgentVersion, error) {
	query := `SELECT id, agent_id, version_number, name, description, system_prompt, model_name,
		enabled_tools, config, is_active, created_at
		FROM neurondb_agent.agent_versions
		WHERE agent_id = $1
		ORDER BY version_number DESC`

	var versions []AgentVersion
	err := v.queries.DB.SelectContext(ctx, &versions, query, agentID)
	if err != nil {
		return nil, fmt.Errorf("agent version listing failed: agent_id='%s', error=%w", agentID.String(), err)
	}

	return versions, nil
}

/* GetVersion gets a specific version */
func (v *VersionManager) GetVersion(ctx context.Context, agentID uuid.UUID, versionNumber int) (*AgentVersion, error) {
	query := `SELECT id, agent_id, version_number, name, description, system_prompt, model_name,
		enabled_tools, config, is_active, created_at
		FROM neurondb_agent.agent_versions
		WHERE agent_id = $1 AND version_number = $2`

	var version AgentVersion
	err := v.queries.DB.GetContext(ctx, &version, query, agentID, versionNumber)
	if err != nil {
		return nil, fmt.Errorf("agent version retrieval failed: agent_id='%s', version_number=%d, error=%w",
			agentID.String(), versionNumber, err)
	}

	return &version, nil
}

/* AgentVersion represents an agent version */
type AgentVersion struct {
	ID            uuid.UUID              `db:"id"`
	AgentID       uuid.UUID              `db:"agent_id"`
	VersionNumber int                    `db:"version_number"`
	Name          string                 `db:"name"`
	Description   *string                `db:"description"`
	SystemPrompt  string                 `db:"system_prompt"`
	ModelName     string                 `db:"model_name"`
	EnabledTools  []string               `db:"enabled_tools"`
	Config        map[string]interface{} `db:"config"`
	IsActive      bool                   `db:"is_active"`
	CreatedAt     string                 `db:"created_at"`
}
