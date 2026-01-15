/*-------------------------------------------------------------------------
 *
 * versioning.go
 *    Tool versioning system
 *
 * Provides multiple versions of same tool with migration support
 * and backward compatibility.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/tools/versioning.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"fmt"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
)

/* ToolVersionManager manages tool versions */
type ToolVersionManager struct {
	queries *db.Queries
}

/* ToolVersion represents a tool version */
type ToolVersion struct {
	ID          uuid.UUID
	ToolName    string
	Version     string
	Schema      map[string]interface{}
	Code        string
	IsDefault   bool
	Deprecated  bool
	Migration   string // Migration script for upgrading
	CreatedAt   string
}

/* NewToolVersionManager creates a new tool version manager */
func NewToolVersionManager(queries *db.Queries) *ToolVersionManager {
	return &ToolVersionManager{
		queries: queries,
	}
}

/* CreateVersion creates a new version of a tool */
func (tvm *ToolVersionManager) CreateVersion(ctx context.Context, version *ToolVersion) (uuid.UUID, error) {
	if version.ID == uuid.Nil {
		version.ID = uuid.New()
	}

	/* If this is the default version, unset other defaults */
	if version.IsDefault {
		unsetQuery := `UPDATE neurondb_agent.tool_versions
			SET is_default = false
			WHERE tool_name = $1 AND is_default = true`
		_, err := tvm.queries.DB.ExecContext(ctx, unsetQuery, version.ToolName)
		if err != nil {
			return uuid.Nil, fmt.Errorf("tool version creation failed: unset_default_error=true, error=%w", err)
		}
	}

	query := `INSERT INTO neurondb_agent.tool_versions
		(id, tool_name, version, schema, code, is_default, deprecated, migration, created_at)
		VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7, $8, NOW())`

	_, err := tvm.queries.DB.ExecContext(ctx, query,
		version.ID,
		version.ToolName,
		version.Version,
		version.Schema,
		version.Code,
		version.IsDefault,
		version.Deprecated,
		version.Migration,
	)

	return version.ID, err
}

/* GetVersion gets a specific tool version */
func (tvm *ToolVersionManager) GetVersion(ctx context.Context, toolName, version string) (*ToolVersion, error) {
	query := `SELECT id, tool_name, version, schema, code, is_default, deprecated, migration, created_at
		FROM neurondb_agent.tool_versions
		WHERE tool_name = $1 AND version = $2`

	type VersionRow struct {
		ID         uuid.UUID              `db:"id"`
		ToolName   string                 `db:"tool_name"`
		Version    string                 `db:"version"`
		Schema     map[string]interface{} `db:"schema"`
		Code       string                 `db:"code"`
		IsDefault  bool                   `db:"is_default"`
		Deprecated bool                   `db:"deprecated"`
		Migration  string                 `db:"migration"`
		CreatedAt  string                 `db:"created_at"`
	}

	var row VersionRow
	err := tvm.queries.DB.GetContext(ctx, &row, query, toolName, version)
	if err != nil {
		return nil, err
	}

	return &ToolVersion{
		ID:         row.ID,
		ToolName:   row.ToolName,
		Version:    row.Version,
		Schema:     row.Schema,
		Code:       row.Code,
		IsDefault:  row.IsDefault,
		Deprecated: row.Deprecated,
		Migration:  row.Migration,
		CreatedAt:  row.CreatedAt,
	}, nil
}

/* GetDefaultVersion gets the default version of a tool */
func (tvm *ToolVersionManager) GetDefaultVersion(ctx context.Context, toolName string) (*ToolVersion, error) {
	query := `SELECT id, tool_name, version, schema, code, is_default, deprecated, migration, created_at
		FROM neurondb_agent.tool_versions
		WHERE tool_name = $1 AND is_default = true
		ORDER BY created_at DESC
		LIMIT 1`

	type VersionRow struct {
		ID         uuid.UUID              `db:"id"`
		ToolName   string                 `db:"tool_name"`
		Version    string                 `db:"version"`
		Schema     map[string]interface{} `db:"schema"`
		Code       string                 `db:"code"`
		IsDefault  bool                   `db:"is_default"`
		Deprecated bool                   `db:"deprecated"`
		Migration  string                 `db:"migration"`
		CreatedAt  string                 `db:"created_at"`
	}

	var row VersionRow
	err := tvm.queries.DB.GetContext(ctx, &row, query, toolName)
	if err != nil {
		return nil, err
	}

	return &ToolVersion{
		ID:         row.ID,
		ToolName:   row.ToolName,
		Version:    row.Version,
		Schema:     row.Schema,
		Code:       row.Code,
		IsDefault:  row.IsDefault,
		Deprecated: row.Deprecated,
		Migration:  row.Migration,
		CreatedAt:  row.CreatedAt,
	}, nil
}

/* ListVersions lists all versions of a tool */
func (tvm *ToolVersionManager) ListVersions(ctx context.Context, toolName string) ([]*ToolVersion, error) {
	query := `SELECT id, tool_name, version, schema, code, is_default, deprecated, migration, created_at
		FROM neurondb_agent.tool_versions
		WHERE tool_name = $1
		ORDER BY created_at DESC`

	type VersionRow struct {
		ID         uuid.UUID              `db:"id"`
		ToolName   string                 `db:"tool_name"`
		Version    string                 `db:"version"`
		Schema     map[string]interface{} `db:"schema"`
		Code       string                 `db:"code"`
		IsDefault  bool                   `db:"is_default"`
		Deprecated bool                   `db:"deprecated"`
		Migration  string                 `db:"migration"`
		CreatedAt  string                 `db:"created_at"`
	}

	var rows []VersionRow
	err := tvm.queries.DB.SelectContext(ctx, &rows, query, toolName)
	if err != nil {
		return nil, err
	}

	versions := make([]*ToolVersion, len(rows))
	for i, row := range rows {
		versions[i] = &ToolVersion{
			ID:         row.ID,
			ToolName:   row.ToolName,
			Version:    row.Version,
			Schema:     row.Schema,
			Code:       row.Code,
			IsDefault:  row.IsDefault,
			Deprecated: row.Deprecated,
			Migration:  row.Migration,
			CreatedAt:  row.CreatedAt,
		}
	}

	return versions, nil
}

/* DeprecateVersion deprecates a tool version */
func (tvm *ToolVersionManager) DeprecateVersion(ctx context.Context, toolName, version string) error {
	query := `UPDATE neurondb_agent.tool_versions
		SET deprecated = true, updated_at = NOW()
		WHERE tool_name = $1 AND version = $2`

	_, err := tvm.queries.DB.ExecContext(ctx, query, toolName, version)
	return err
}

