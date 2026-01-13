/*-------------------------------------------------------------------------
 *
 * audit.go
 *    Unified audit logging for NeuronDB ecosystem
 *
 * Provides standardized audit logging across all components.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    pkg/identity/audit.go
 *
 *-------------------------------------------------------------------------
 */

package identity

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"time"

	"github.com/google/uuid"
)

/* AuditAction represents standard audit action types */
type AuditAction string

const (
	ActionToolCall     AuditAction = "tool_call"
	ActionSQLExecute   AuditAction = "sql_execute"
	ActionAgentExecute AuditAction = "agent_execute"
	ActionLogin        AuditAction = "login"
	ActionLogout       AuditAction = "logout"
	ActionCreate        AuditAction = "create"
	ActionUpdate        AuditAction = "update"
	ActionDelete        AuditAction = "delete"
	ActionRead          AuditAction = "read"
	ActionPermissionGrant AuditAction = "permission_grant"
	ActionPermissionRevoke AuditAction = "permission_revoke"
)

/* AuditLogger provides methods for audit logging */
type AuditLogger interface {
	Log(ctx interface{}, entry *AuditLogEntry) error
	LogAction(ctx interface{}, action AuditAction, resourceType ResourceType, resourceID *string, metadata map[string]interface{}) error
}

/* HashInputs creates a SHA-256 hash of inputs */
func HashInputs(inputs map[string]interface{}) (string, error) {
	data, err := json.Marshal(inputs)
	if err != nil {
		return "", err
	}
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:]), nil
}

/* HashOutputs creates a SHA-256 hash of outputs */
func HashOutputs(outputs map[string]interface{}) (string, error) {
	data, err := json.Marshal(outputs)
	if err != nil {
		return "", err
	}
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:]), nil
}

/* NewAuditLogEntry creates a new audit log entry */
func NewAuditLogEntry(
	action AuditAction,
	resourceType ResourceType,
	resourceID *string,
	principalID *uuid.UUID,
	apiKeyID *uuid.UUID,
	inputs map[string]interface{},
	outputs map[string]interface{},
) (*AuditLogEntry, error) {
	entry := &AuditLogEntry{
		Timestamp:    time.Now(),
		PrincipalID:  principalID,
		APIKeyID:     apiKeyID,
		Action:       string(action),
		ResourceType: string(resourceType),
		ResourceID:   resourceID,
		Inputs:       inputs,
		Outputs:      outputs,
		CreatedAt:    time.Now(),
	}

	// Hash inputs if provided
	if inputs != nil {
		hash, err := HashInputs(inputs)
		if err == nil {
			entry.InputsHash = &hash
		}
	}

	// Hash outputs if provided
	if outputs != nil {
		hash, err := HashOutputs(outputs)
		if err == nil {
			entry.OutputsHash = &hash
		}
	}

	return entry, nil
}

/* StandardAuditEventTypes defines standard event types for audit logging */
var StandardAuditEventTypes = map[string]string{
	"tool_call":      "Tool execution",
	"sql_execute":    "SQL query execution",
	"agent_execute":  "Agent execution",
	"login":          "User login",
	"logout":         "User logout",
	"create":         "Resource creation",
	"update":         "Resource update",
	"delete":         "Resource deletion",
	"read":           "Resource read",
	"permission_grant": "Permission granted",
	"permission_revoke": "Permission revoked",
}







