/*-------------------------------------------------------------------------
 *
 * audit.go
 *    Audit logging for NeuronAgent
 *
 * Provides audit logging for tool calls and SQL statements with input/output hashes.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/auth/audit.go
 *
 *-------------------------------------------------------------------------
 */

package auth

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
)

type AuditLogger struct {
	queries *db.Queries
}

func NewAuditLogger(queries *db.Queries) *AuditLogger {
	return &AuditLogger{queries: queries}
}

/* LogToolCall logs a tool call to the audit log */
func (a *AuditLogger) LogToolCall(ctx context.Context, principalID, apiKeyID, agentID, sessionID *uuid.UUID, toolName string, inputs, outputs map[string]interface{}) error {
	inputsHash, err := a.hashMap(inputs)
	if err != nil {
		return fmt.Errorf("failed to hash inputs: %w", err)
	}

	outputsHash, err := a.hashMap(outputs)
	if err != nil {
		return fmt.Errorf("failed to hash outputs: %w", err)
	}

	auditLog := &db.AuditLog{
		Timestamp:    time.Now(),
		PrincipalID:  principalID,
		APIKeyID:     apiKeyID,
		AgentID:      agentID,
		SessionID:    sessionID,
		Action:       "tool_call",
		ResourceType: "tool",
		ResourceID:   &toolName,
		InputsHash:   &inputsHash,
		OutputsHash:  &outputsHash,
		Inputs:       inputs,
		Outputs:      outputs,
		Metadata:     make(db.JSONBMap),
	}

	if err := a.queries.CreateAuditLog(ctx, auditLog); err != nil {
		return fmt.Errorf("failed to create audit log: %w", err)
	}

	return nil
}

/* LogSQLStatement logs a SQL statement to the audit log */
func (a *AuditLogger) LogSQLStatement(ctx context.Context, principalID, apiKeyID, agentID, sessionID *uuid.UUID, sqlQuery string, inputs, outputs map[string]interface{}) error {
	inputsHash, err := a.hashMap(inputs)
	if err != nil {
		return fmt.Errorf("failed to hash inputs: %w", err)
	}

	outputsHash, err := a.hashMap(outputs)
	if err != nil {
		return fmt.Errorf("failed to hash outputs: %w", err)
	}

	queryHash := a.hashString(sqlQuery)

	auditLog := &db.AuditLog{
		Timestamp:    time.Now(),
		PrincipalID:  principalID,
		APIKeyID:     apiKeyID,
		AgentID:      agentID,
		SessionID:    sessionID,
		Action:       "sql_execute",
		ResourceType: "sql",
		ResourceID:   &queryHash,
		InputsHash:   &inputsHash,
		OutputsHash:  &outputsHash,
		Inputs:       inputs,
		Outputs:      outputs,
		Metadata: db.JSONBMap{
			"query": sqlQuery,
		},
	}

	if err := a.queries.CreateAuditLog(ctx, auditLog); err != nil {
		return fmt.Errorf("failed to create audit log: %w", err)
	}

	return nil
}

/* LogAgentExecution logs an agent execution to the audit log */
func (a *AuditLogger) LogAgentExecution(ctx context.Context, principalID, apiKeyID, agentID, sessionID *uuid.UUID, action string, metadata map[string]interface{}) error {
	auditLog := &db.AuditLog{
		Timestamp:    time.Now(),
		PrincipalID:  principalID,
		APIKeyID:     apiKeyID,
		AgentID:      agentID,
		SessionID:    sessionID,
		Action:       action,
		ResourceType: "agent",
		ResourceID:   stringPtr(agentID.String()),
		Metadata:     metadata,
	}

	if err := a.queries.CreateAuditLog(ctx, auditLog); err != nil {
		return fmt.Errorf("failed to create audit log: %w", err)
	}

	return nil
}

/* hashMap computes SHA-256 hash of a map (JSON-encoded) */
func (a *AuditLogger) hashMap(m map[string]interface{}) (string, error) {
	if m == nil || len(m) == 0 {
		return "", nil
	}

	jsonBytes, err := json.Marshal(m)
	if err != nil {
		return "", err
	}

	hash := sha256.Sum256(jsonBytes)
	return hex.EncodeToString(hash[:]), nil
}

/* hashString computes SHA-256 hash of a string */
func (a *AuditLogger) hashString(s string) string {
	hash := sha256.Sum256([]byte(s))
	return hex.EncodeToString(hash[:])
}

/* stringPtr returns a pointer to a string */
func stringPtr(s string) *string {
	return &s
}
