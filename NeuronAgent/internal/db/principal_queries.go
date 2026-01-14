/*-------------------------------------------------------------------------
 *
 * principal_queries.go
 *    Database queries for principals and permissions
 *
 * Provides database query functions for principals, policies, tool permissions,
 * data permissions, and audit logging.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/db/principal_queries.go
 *
 *-------------------------------------------------------------------------
 */

package db

import (
	"context"
	"database/sql"
	"fmt"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/utils"
)

/* Principal queries */
const (
	createPrincipalQuery = `
		INSERT INTO neurondb_agent.principals (type, name, metadata)
		VALUES ($1, $2, $3::jsonb)
		RETURNING id, created_at, updated_at`

	getPrincipalByIDQuery = `SELECT * FROM neurondb_agent.principals WHERE id = $1`

	getPrincipalByTypeAndNameQuery = `
		SELECT * FROM neurondb_agent.principals 
		WHERE type = $1 AND name = $2`

	listPrincipalsByTypeQuery = `
		SELECT * FROM neurondb_agent.principals 
		WHERE type = $1 
		ORDER BY created_at DESC`

	updatePrincipalQuery = `
		UPDATE neurondb_agent.principals 
		SET name = $2, metadata = $3::jsonb, updated_at = NOW()
		WHERE id = $1
		RETURNING updated_at`

	deletePrincipalQuery = `DELETE FROM neurondb_agent.principals WHERE id = $1`
)

/* Policy queries */
const (
	createPolicyQuery = `
		INSERT INTO neurondb_agent.policies 
		(principal_id, resource_type, resource_id, permissions, conditions)
		VALUES ($1, $2, $3, $4, $5::jsonb)
		RETURNING id, created_at, updated_at`

	getPolicyByIDQuery = `SELECT * FROM neurondb_agent.policies WHERE id = $1`

	listPoliciesByPrincipalQuery = `
		SELECT * FROM neurondb_agent.policies 
		WHERE principal_id = $1 
		ORDER BY created_at DESC`

	listPoliciesByResourceQuery = `
		SELECT * FROM neurondb_agent.policies 
		WHERE resource_type = $1 AND ($2::text IS NULL OR resource_id = $2)
		ORDER BY created_at DESC`

	updatePolicyQuery = `
		UPDATE neurondb_agent.policies 
		SET resource_type = $2, resource_id = $3, permissions = $4, conditions = $5::jsonb, updated_at = NOW()
		WHERE id = $1
		RETURNING updated_at`

	deletePolicyQuery = `DELETE FROM neurondb_agent.policies WHERE id = $1`
)

/* Tool permission queries */
const (
	createToolPermissionQuery = `
		INSERT INTO neurondb_agent.tool_permissions 
		(agent_id, tool_name, allowed, conditions)
		VALUES ($1, $2, $3, $4::jsonb)
		RETURNING id, created_at, updated_at`

	getToolPermissionQuery = `
		SELECT * FROM neurondb_agent.tool_permissions 
		WHERE agent_id = $1 AND tool_name = $2`

	listToolPermissionsByAgentQuery = `
		SELECT * FROM neurondb_agent.tool_permissions 
		WHERE agent_id = $1 
		ORDER BY tool_name`

	updateToolPermissionQuery = `
		UPDATE neurondb_agent.tool_permissions 
		SET allowed = $3, conditions = $4::jsonb, updated_at = NOW()
		WHERE agent_id = $1 AND tool_name = $2
		RETURNING updated_at`

	deleteToolPermissionQuery = `
		DELETE FROM neurondb_agent.tool_permissions 
		WHERE agent_id = $1 AND tool_name = $2`
)

/* Session tool permission queries */
const (
	createSessionToolPermissionQuery = `
		INSERT INTO neurondb_agent.session_tool_permissions 
		(session_id, tool_name, allowed, conditions)
		VALUES ($1, $2, $3, $4::jsonb)
		RETURNING id, created_at`

	getSessionToolPermissionQuery = `
		SELECT * FROM neurondb_agent.session_tool_permissions 
		WHERE session_id = $1 AND tool_name = $2`

	listSessionToolPermissionsQuery = `
		SELECT * FROM neurondb_agent.session_tool_permissions 
		WHERE session_id = $1 
		ORDER BY tool_name`

	deleteSessionToolPermissionQuery = `
		DELETE FROM neurondb_agent.session_tool_permissions 
		WHERE session_id = $1 AND tool_name = $2`
)

/* Data permission queries */
const (
	createDataPermissionQuery = `
		INSERT INTO neurondb_agent.data_permissions 
		(principal_id, schema_name, table_name, row_filter, column_mask, permissions)
		VALUES ($1, $2, $3, $4, $5::jsonb, $6)
		RETURNING id, created_at, updated_at`

	getDataPermissionByIDQuery = `SELECT * FROM neurondb_agent.data_permissions WHERE id = $1`

	listDataPermissionsByPrincipalQuery = `
		SELECT * FROM neurondb_agent.data_permissions 
		WHERE principal_id = $1 
		ORDER BY schema_name, table_name`

	listDataPermissionsByResourceQuery = `
		SELECT * FROM neurondb_agent.data_permissions 
		WHERE ($1::text IS NULL OR schema_name = $1)
		AND ($2::text IS NULL OR table_name = $2)
		ORDER BY schema_name, table_name`

	updateDataPermissionQuery = `
		UPDATE neurondb_agent.data_permissions 
		SET schema_name = $2, table_name = $3, row_filter = $4, column_mask = $5::jsonb, permissions = $6, updated_at = NOW()
		WHERE id = $1
		RETURNING updated_at`

	deleteDataPermissionQuery = `DELETE FROM neurondb_agent.data_permissions WHERE id = $1`
)

/* Audit log queries */
const (
	createAuditLogQuery = `
		INSERT INTO neurondb_agent.audit_log 
		(timestamp, principal_id, api_key_id, agent_id, session_id, action, resource_type, resource_id, inputs_hash, outputs_hash, inputs, outputs, metadata)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb, $12::jsonb, $13::jsonb)
		RETURNING id, created_at`

	getAuditLogByIDQuery = `SELECT * FROM neurondb_agent.audit_log WHERE id = $1`

	listAuditLogsQuery = `
		SELECT * FROM neurondb_agent.audit_log 
		WHERE ($1::uuid IS NULL OR principal_id = $1)
		AND ($2::uuid IS NULL OR api_key_id = $2)
		AND ($3::uuid IS NULL OR agent_id = $3)
		AND ($4::uuid IS NULL OR session_id = $4)
		AND ($5::text IS NULL OR action = $5)
		AND ($6::text IS NULL OR resource_type = $6)
		AND timestamp >= $7
		AND timestamp <= $8
		ORDER BY timestamp DESC
		LIMIT $9 OFFSET $10`
)

/* Principal methods */
func (q *Queries) CreatePrincipal(ctx context.Context, principal *Principal) error {
	metadataValue, err := principal.Metadata.Value()
	if err != nil {
		return fmt.Errorf("failed to convert metadata: %w", err)
	}

	params := []interface{}{principal.Type, principal.Name, metadataValue}
	err = q.DB.GetContext(ctx, principal, createPrincipalQuery, params...)
	if err != nil {
		return fmt.Errorf("principal creation failed on %s: query='%s', params_count=%d, type='%s', name='%s', table='neurondb_agent.principals', error=%w",
			q.getConnInfoString(), createPrincipalQuery, len(params), principal.Type, principal.Name, err)
	}
	return nil
}

func (q *Queries) GetPrincipalByID(ctx context.Context, id uuid.UUID) (*Principal, error) {
	var principal Principal
	err := q.DB.GetContext(ctx, &principal, getPrincipalByIDQuery, id)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("principal not found on %s: query='%s', principal_id='%s', table='neurondb_agent.principals', error=%w",
			q.getConnInfoString(), getPrincipalByIDQuery, id.String(), err)
	}
	if err != nil {
		return nil, q.formatQueryError("SELECT", getPrincipalByIDQuery, 1, "neurondb_agent.principals", err)
	}
	return &principal, nil
}

func (q *Queries) GetPrincipalByTypeAndName(ctx context.Context, principalType, name string) (*Principal, error) {
	var principal Principal
	err := q.DB.GetContext(ctx, &principal, getPrincipalByTypeAndNameQuery, principalType, name)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("principal not found on %s: query='%s', type='%s', name='%s', table='neurondb_agent.principals', error=%w",
			q.getConnInfoString(), getPrincipalByTypeAndNameQuery, principalType, name, err)
	}
	if err != nil {
		return nil, q.formatQueryError("SELECT", getPrincipalByTypeAndNameQuery, 2, "neurondb_agent.principals", err)
	}
	return &principal, nil
}

func (q *Queries) ListPrincipalsByType(ctx context.Context, principalType string) ([]Principal, error) {
	var principals []Principal
	err := q.DB.SelectContext(ctx, &principals, listPrincipalsByTypeQuery, principalType)
	if err != nil {
		return nil, q.formatQueryError("SELECT", listPrincipalsByTypeQuery, 1, "neurondb_agent.principals", err)
	}
	return principals, nil
}

func (q *Queries) UpdatePrincipal(ctx context.Context, principal *Principal) error {
	metadataValue, err := principal.Metadata.Value()
	if err != nil {
		return fmt.Errorf("failed to convert metadata: %w", err)
	}

	params := []interface{}{principal.ID, principal.Name, metadataValue}
	err = q.DB.GetContext(ctx, principal, updatePrincipalQuery, params...)
	if err != nil {
		return fmt.Errorf("principal update failed on %s: query='%s', params_count=%d, principal_id='%s', name='%s', table='neurondb_agent.principals', error=%w",
			q.getConnInfoString(), updatePrincipalQuery, len(params), principal.ID.String(), principal.Name, err)
	}
	return nil
}

func (q *Queries) DeletePrincipal(ctx context.Context, id uuid.UUID) error {
	_, err := q.DB.ExecContext(ctx, deletePrincipalQuery, id)
	if err != nil {
		return fmt.Errorf("principal deletion failed on %s: query='%s', principal_id='%s', table='neurondb_agent.principals', error=%w",
			q.getConnInfoString(), deletePrincipalQuery, id.String(), err)
	}
	return nil
}

/* Policy methods */
func (q *Queries) CreatePolicy(ctx context.Context, policy *Policy) error {
	conditionsValue, err := policy.Conditions.Value()
	if err != nil {
		return fmt.Errorf("failed to convert conditions: %w", err)
	}

	params := []interface{}{policy.PrincipalID, policy.ResourceType, policy.ResourceID, policy.Permissions, conditionsValue}
	err = q.DB.GetContext(ctx, policy, createPolicyQuery, params...)
	if err != nil {
		return fmt.Errorf("policy creation failed on %s: query='%s', params_count=%d, principal_id='%s', resource_type='%s', table='neurondb_agent.policies', error=%w",
			q.getConnInfoString(), createPolicyQuery, len(params), policy.PrincipalID.String(), policy.ResourceType, err)
	}
	return nil
}

func (q *Queries) GetPolicyByID(ctx context.Context, id uuid.UUID) (*Policy, error) {
	var policy Policy
	err := q.DB.GetContext(ctx, &policy, getPolicyByIDQuery, id)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("policy not found on %s: query='%s', policy_id='%s', table='neurondb_agent.policies', error=%w",
			q.getConnInfoString(), getPolicyByIDQuery, id.String(), err)
	}
	if err != nil {
		return nil, q.formatQueryError("SELECT", getPolicyByIDQuery, 1, "neurondb_agent.policies", err)
	}
	return &policy, nil
}

func (q *Queries) ListPoliciesByPrincipal(ctx context.Context, principalID uuid.UUID) ([]Policy, error) {
	var policies []Policy
	err := q.DB.SelectContext(ctx, &policies, listPoliciesByPrincipalQuery, principalID)
	if err != nil {
		return nil, q.formatQueryError("SELECT", listPoliciesByPrincipalQuery, 1, "neurondb_agent.policies", err)
	}
	return policies, nil
}

func (q *Queries) ListPoliciesByResource(ctx context.Context, resourceType string, resourceID *string) ([]Policy, error) {
	var policies []Policy
	err := q.DB.SelectContext(ctx, &policies, listPoliciesByResourceQuery, resourceType, resourceID)
	if err != nil {
		return nil, q.formatQueryError("SELECT", listPoliciesByResourceQuery, 2, "neurondb_agent.policies", err)
	}
	return policies, nil
}

func (q *Queries) UpdatePolicy(ctx context.Context, policy *Policy) error {
	conditionsValue, err := policy.Conditions.Value()
	if err != nil {
		return fmt.Errorf("failed to convert conditions: %w", err)
	}

	params := []interface{}{policy.ID, policy.ResourceType, policy.ResourceID, policy.Permissions, conditionsValue}
	err = q.DB.GetContext(ctx, policy, updatePolicyQuery, params...)
	if err != nil {
		return fmt.Errorf("policy update failed on %s: query='%s', params_count=%d, policy_id='%s', resource_type='%s', table='neurondb_agent.policies', error=%w",
			q.getConnInfoString(), updatePolicyQuery, len(params), policy.ID.String(), policy.ResourceType, err)
	}
	return nil
}

func (q *Queries) DeletePolicy(ctx context.Context, id uuid.UUID) error {
	_, err := q.DB.ExecContext(ctx, deletePolicyQuery, id)
	if err != nil {
		return fmt.Errorf("policy deletion failed on %s: query='%s', policy_id='%s', table='neurondb_agent.policies', error=%w",
			q.getConnInfoString(), deletePolicyQuery, id.String(), err)
	}
	return nil
}

/* Tool permission methods */
func (q *Queries) CreateToolPermission(ctx context.Context, toolPerm *ToolPermission) error {
	conditionsValue, err := toolPerm.Conditions.Value()
	if err != nil {
		return fmt.Errorf("failed to convert conditions: %w", err)
	}

	params := []interface{}{toolPerm.AgentID, toolPerm.ToolName, toolPerm.Allowed, conditionsValue}
	err = q.DB.GetContext(ctx, toolPerm, createToolPermissionQuery, params...)
	if err != nil {
		return fmt.Errorf("tool permission creation failed on %s: query='%s', params_count=%d, agent_id='%s', tool_name='%s', table='neurondb_agent.tool_permissions', error=%w",
			q.getConnInfoString(), createToolPermissionQuery, len(params), toolPerm.AgentID.String(), toolPerm.ToolName, err)
	}
	return nil
}

func (q *Queries) GetToolPermission(ctx context.Context, agentID uuid.UUID, toolName string) (*ToolPermission, error) {
	var toolPerm ToolPermission
	err := q.DB.GetContext(ctx, &toolPerm, getToolPermissionQuery, agentID, toolName)
	if err == sql.ErrNoRows {
		return nil, nil // Not found is not an error for permissions
	}
	if err != nil {
		return nil, q.formatQueryError("SELECT", getToolPermissionQuery, 2, "neurondb_agent.tool_permissions", err)
	}
	return &toolPerm, nil
}

func (q *Queries) ListToolPermissionsByAgent(ctx context.Context, agentID uuid.UUID) ([]ToolPermission, error) {
	var toolPerms []ToolPermission
	err := q.DB.SelectContext(ctx, &toolPerms, listToolPermissionsByAgentQuery, agentID)
	if err != nil {
		return nil, q.formatQueryError("SELECT", listToolPermissionsByAgentQuery, 1, "neurondb_agent.tool_permissions", err)
	}
	return toolPerms, nil
}

func (q *Queries) UpdateToolPermission(ctx context.Context, toolPerm *ToolPermission) error {
	conditionsValue, err := toolPerm.Conditions.Value()
	if err != nil {
		return fmt.Errorf("failed to convert conditions: %w", err)
	}

	params := []interface{}{toolPerm.AgentID, toolPerm.ToolName, toolPerm.Allowed, conditionsValue}
	err = q.DB.GetContext(ctx, toolPerm, updateToolPermissionQuery, params...)
	if err != nil {
		return fmt.Errorf("tool permission update failed on %s: query='%s', params_count=%d, agent_id='%s', tool_name='%s', table='neurondb_agent.tool_permissions', error=%w",
			q.getConnInfoString(), updateToolPermissionQuery, len(params), toolPerm.AgentID.String(), toolPerm.ToolName, err)
	}
	return nil
}

func (q *Queries) DeleteToolPermission(ctx context.Context, agentID uuid.UUID, toolName string) error {
	_, err := q.DB.ExecContext(ctx, deleteToolPermissionQuery, agentID, toolName)
	if err != nil {
		return fmt.Errorf("tool permission deletion failed on %s: query='%s', agent_id='%s', tool_name='%s', table='neurondb_agent.tool_permissions', error=%w",
			q.getConnInfoString(), deleteToolPermissionQuery, agentID.String(), toolName, err)
	}
	return nil
}

/* Session tool permission methods */
func (q *Queries) CreateSessionToolPermission(ctx context.Context, sessionToolPerm *SessionToolPermission) error {
	conditionsValue, err := sessionToolPerm.Conditions.Value()
	if err != nil {
		return fmt.Errorf("failed to convert conditions: %w", err)
	}

	params := []interface{}{sessionToolPerm.SessionID, sessionToolPerm.ToolName, sessionToolPerm.Allowed, conditionsValue}
	err = q.DB.GetContext(ctx, sessionToolPerm, createSessionToolPermissionQuery, params...)
	if err != nil {
		return fmt.Errorf("session tool permission creation failed on %s: query='%s', params_count=%d, session_id='%s', tool_name='%s', table='neurondb_agent.session_tool_permissions', error=%w",
			q.getConnInfoString(), createSessionToolPermissionQuery, len(params), sessionToolPerm.SessionID.String(), sessionToolPerm.ToolName, err)
	}
	return nil
}

func (q *Queries) GetSessionToolPermission(ctx context.Context, sessionID uuid.UUID, toolName string) (*SessionToolPermission, error) {
	var sessionToolPerm SessionToolPermission
	err := q.DB.GetContext(ctx, &sessionToolPerm, getSessionToolPermissionQuery, sessionID, toolName)
	if err == sql.ErrNoRows {
		return nil, nil // Not found is not an error for permissions
	}
	if err != nil {
		return nil, q.formatQueryError("SELECT", getSessionToolPermissionQuery, 2, "neurondb_agent.session_tool_permissions", err)
	}
	return &sessionToolPerm, nil
}

func (q *Queries) ListSessionToolPermissions(ctx context.Context, sessionID uuid.UUID) ([]SessionToolPermission, error) {
	var sessionToolPerms []SessionToolPermission
	err := q.DB.SelectContext(ctx, &sessionToolPerms, listSessionToolPermissionsQuery, sessionID)
	if err != nil {
		return nil, q.formatQueryError("SELECT", listSessionToolPermissionsQuery, 1, "neurondb_agent.session_tool_permissions", err)
	}
	return sessionToolPerms, nil
}

func (q *Queries) DeleteSessionToolPermission(ctx context.Context, sessionID uuid.UUID, toolName string) error {
	_, err := q.DB.ExecContext(ctx, deleteSessionToolPermissionQuery, sessionID, toolName)
	if err != nil {
		return fmt.Errorf("session tool permission deletion failed on %s: query='%s', session_id='%s', tool_name='%s', table='neurondb_agent.session_tool_permissions', error=%w",
			q.getConnInfoString(), deleteSessionToolPermissionQuery, sessionID.String(), toolName, err)
	}
	return nil
}

/* Data permission methods */
func (q *Queries) CreateDataPermission(ctx context.Context, dataPerm *DataPermission) error {
	columnMaskValue, err := dataPerm.ColumnMask.Value()
	if err != nil {
		return fmt.Errorf("failed to convert column_mask: %w", err)
	}

	params := []interface{}{dataPerm.PrincipalID, dataPerm.SchemaName, dataPerm.TableName, dataPerm.RowFilter, columnMaskValue, dataPerm.Permissions}
	err = q.DB.GetContext(ctx, dataPerm, createDataPermissionQuery, params...)
	if err != nil {
		return fmt.Errorf("data permission creation failed on %s: query='%s', params_count=%d, principal_id='%s', schema_name=%s, table_name=%s, table='neurondb_agent.data_permissions', error=%w",
			q.getConnInfoString(), createDataPermissionQuery, len(params), dataPerm.PrincipalID.String(),
			utils.SanitizeValue(dataPerm.SchemaName), utils.SanitizeValue(dataPerm.TableName), err)
	}
	return nil
}

func (q *Queries) GetDataPermissionByID(ctx context.Context, id uuid.UUID) (*DataPermission, error) {
	var dataPerm DataPermission
	err := q.DB.GetContext(ctx, &dataPerm, getDataPermissionByIDQuery, id)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("data permission not found on %s: query='%s', data_permission_id='%s', table='neurondb_agent.data_permissions', error=%w",
			q.getConnInfoString(), getDataPermissionByIDQuery, id.String(), err)
	}
	if err != nil {
		return nil, q.formatQueryError("SELECT", getDataPermissionByIDQuery, 1, "neurondb_agent.data_permissions", err)
	}
	return &dataPerm, nil
}

func (q *Queries) ListDataPermissionsByPrincipal(ctx context.Context, principalID uuid.UUID) ([]DataPermission, error) {
	var dataPerms []DataPermission
	err := q.DB.SelectContext(ctx, &dataPerms, listDataPermissionsByPrincipalQuery, principalID)
	if err != nil {
		return nil, q.formatQueryError("SELECT", listDataPermissionsByPrincipalQuery, 1, "neurondb_agent.data_permissions", err)
	}
	return dataPerms, nil
}

func (q *Queries) ListDataPermissionsByResource(ctx context.Context, schemaName, tableName *string) ([]DataPermission, error) {
	var dataPerms []DataPermission
	err := q.DB.SelectContext(ctx, &dataPerms, listDataPermissionsByResourceQuery, schemaName, tableName)
	if err != nil {
		return nil, q.formatQueryError("SELECT", listDataPermissionsByResourceQuery, 2, "neurondb_agent.data_permissions", err)
	}
	return dataPerms, nil
}

func (q *Queries) UpdateDataPermission(ctx context.Context, dataPerm *DataPermission) error {
	columnMaskValue, err := dataPerm.ColumnMask.Value()
	if err != nil {
		return fmt.Errorf("failed to convert column_mask: %w", err)
	}

	params := []interface{}{dataPerm.ID, dataPerm.SchemaName, dataPerm.TableName, dataPerm.RowFilter, columnMaskValue, dataPerm.Permissions}
	err = q.DB.GetContext(ctx, dataPerm, updateDataPermissionQuery, params...)
	if err != nil {
		return fmt.Errorf("data permission update failed on %s: query='%s', params_count=%d, data_permission_id='%s', table='neurondb_agent.data_permissions', error=%w",
			q.getConnInfoString(), updateDataPermissionQuery, len(params), dataPerm.ID.String(), err)
	}
	return nil
}

func (q *Queries) DeleteDataPermission(ctx context.Context, id uuid.UUID) error {
	_, err := q.DB.ExecContext(ctx, deleteDataPermissionQuery, id)
	if err != nil {
		return fmt.Errorf("data permission deletion failed on %s: query='%s', data_permission_id='%s', table='neurondb_agent.data_permissions', error=%w",
			q.getConnInfoString(), deleteDataPermissionQuery, id.String(), err)
	}
	return nil
}

/* Audit log methods */
func (q *Queries) CreateAuditLog(ctx context.Context, auditLog *AuditLog) error {
	var inputsValue, outputsValue, metadataValue interface{}
	var err error

	if auditLog.Inputs != nil {
		inputsValue, err = auditLog.Inputs.Value()
		if err != nil {
			return fmt.Errorf("failed to convert inputs: %w", err)
		}
	}
	if auditLog.Outputs != nil {
		outputsValue, err = auditLog.Outputs.Value()
		if err != nil {
			return fmt.Errorf("failed to convert outputs: %w", err)
		}
	}
	if auditLog.Metadata != nil {
		metadataValue, err = auditLog.Metadata.Value()
		if err != nil {
			return fmt.Errorf("failed to convert metadata: %w", err)
		}
	}

	params := []interface{}{
		auditLog.Timestamp, auditLog.PrincipalID, auditLog.APIKeyID, auditLog.AgentID, auditLog.SessionID,
		auditLog.Action, auditLog.ResourceType, auditLog.ResourceID,
		auditLog.InputsHash, auditLog.OutputsHash,
		inputsValue, outputsValue, metadataValue,
	}
	err = q.DB.GetContext(ctx, auditLog, createAuditLogQuery, params...)
	if err != nil {
		return fmt.Errorf("audit log creation failed on %s: query='%s', params_count=%d, action='%s', resource_type='%s', table='neurondb_agent.audit_log', error=%w",
			q.getConnInfoString(), createAuditLogQuery, len(params), auditLog.Action, auditLog.ResourceType, err)
	}
	return nil
}

func (q *Queries) GetAuditLogByID(ctx context.Context, id int64) (*AuditLog, error) {
	var auditLog AuditLog
	err := q.DB.GetContext(ctx, &auditLog, getAuditLogByIDQuery, id)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("audit log not found on %s: query='%s', audit_log_id=%d, table='neurondb_agent.audit_log', error=%w",
			q.getConnInfoString(), getAuditLogByIDQuery, id, err)
	}
	if err != nil {
		return nil, q.formatQueryError("SELECT", getAuditLogByIDQuery, 1, "neurondb_agent.audit_log", err)
	}
	return &auditLog, nil
}

func (q *Queries) ListAuditLogs(ctx context.Context, principalID, apiKeyID, agentID, sessionID *uuid.UUID, action, resourceType *string, startTime, endTime string, limit, offset int) ([]AuditLog, error) {
	var auditLogs []AuditLog
	params := []interface{}{principalID, apiKeyID, agentID, sessionID, action, resourceType, startTime, endTime, limit, offset}
	err := q.DB.SelectContext(ctx, &auditLogs, listAuditLogsQuery, params...)
	if err != nil {
		return nil, q.formatQueryError("SELECT", listAuditLogsQuery, len(params), "neurondb_agent.audit_log", err)
	}
	return auditLogs, nil
}
