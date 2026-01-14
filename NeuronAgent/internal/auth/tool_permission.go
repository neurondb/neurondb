/*-------------------------------------------------------------------------
 *
 * tool_permission.go
 *    Tool permission checking for NeuronAgent
 *
 * Provides permission checking for tool execution at agent and session levels.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/auth/tool_permission.go
 *
 *-------------------------------------------------------------------------
 */

package auth

import (
	"context"
	"fmt"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
)

type ToolPermissionChecker struct {
	queries *db.Queries
}

func NewToolPermissionChecker(queries *db.Queries) *ToolPermissionChecker {
	return &ToolPermissionChecker{queries: queries}
}

/* CheckToolPermission checks if a tool can be executed for an agent and session */
func (c *ToolPermissionChecker) CheckToolPermission(ctx context.Context, agentID, sessionID uuid.UUID, toolName string) (bool, error) {
	/* First check session-level permission (takes precedence) */
	sessionPerm, err := c.queries.GetSessionToolPermission(ctx, sessionID, toolName)
	if err != nil {
		return false, fmt.Errorf("failed to check session tool permission: %w", err)
	}

	if sessionPerm != nil {
		return sessionPerm.Allowed, nil
	}

	/* Then check agent-level permission */
	agentPerm, err := c.queries.GetToolPermission(ctx, agentID, toolName)
	if err != nil {
		return false, fmt.Errorf("failed to check agent tool permission: %w", err)
	}

	if agentPerm != nil {
		return agentPerm.Allowed, nil
	}

	/* Default: allow if no explicit permission is set */
	return true, nil
}

/* GetAllowedTools returns list of tools allowed for an agent and session */
func (c *ToolPermissionChecker) GetAllowedTools(ctx context.Context, agentID, sessionID uuid.UUID) (map[string]bool, error) {
	allowedTools := make(map[string]bool)

	/* Get all agent-level permissions */
	agentPerms, err := c.queries.ListToolPermissionsByAgent(ctx, agentID)
	if err != nil {
		return nil, fmt.Errorf("failed to list agent tool permissions: %w", err)
	}

	for _, perm := range agentPerms {
		allowedTools[perm.ToolName] = perm.Allowed
	}

	/* Override with session-level permissions */
	sessionPerms, err := c.queries.ListSessionToolPermissions(ctx, sessionID)
	if err != nil {
		return nil, fmt.Errorf("failed to list session tool permissions: %w", err)
	}

	for _, perm := range sessionPerms {
		allowedTools[perm.ToolName] = perm.Allowed
	}

	return allowedTools, nil
}
