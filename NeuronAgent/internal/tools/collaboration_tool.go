/*-------------------------------------------------------------------------
 *
 * collaboration_tool.go
 *    Collaboration tool for workspace operations
 *
 * Provides agent access to collaboration workspace features for
 * multi-user and multi-agent collaboration.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/tools/collaboration_tool.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/collaboration"
	"github.com/neurondb/NeuronAgent/internal/db"
)

/* CollaborationTool provides workspace operations for agents */
type CollaborationTool struct {
	workspace *collaboration.WorkspaceManager
}

/* NewCollaborationTool creates a new collaboration tool */
func NewCollaborationTool(workspace *collaboration.WorkspaceManager) *CollaborationTool {
	return &CollaborationTool{
		workspace: workspace,
	}
}

/* Execute executes a collaboration operation */
func (t *CollaborationTool) Execute(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	action, ok := args["action"].(string)
	if !ok {
		return "", fmt.Errorf("collaboration tool requires action parameter")
	}

	switch action {
	case "create_workspace":
		return t.createWorkspace(ctx, args)
	case "add_participant":
		return t.addParticipant(ctx, args)
	case "get_workspace":
		return t.getWorkspace(ctx, args)
	case "broadcast_update":
		return t.broadcastUpdate(ctx, args)
	default:
		return "", fmt.Errorf("unknown collaboration action: %s", action)
	}
}

/* createWorkspace creates a new workspace */
func (t *CollaborationTool) createWorkspace(ctx context.Context, args map[string]interface{}) (string, error) {
	name, ok := args["name"].(string)
	if !ok {
		return "", fmt.Errorf("create_workspace requires name parameter")
	}

	var ownerID *uuid.UUID
	if ownerIDStr, ok := args["owner_id"].(string); ok && ownerIDStr != "" {
		parsed, err := uuid.Parse(ownerIDStr)
		if err == nil {
			ownerID = &parsed
		}
	}

	workspaceID, err := t.workspace.CreateWorkspace(ctx, name, ownerID)
	if err != nil {
		return "", fmt.Errorf("workspace creation failed: %w", err)
	}

	result := map[string]interface{}{
		"action":       "create_workspace",
		"workspace_id": workspaceID.String(),
		"name":         name,
		"status":       "success",
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON), nil
}

/* addParticipant adds a participant to a workspace */
func (t *CollaborationTool) addParticipant(ctx context.Context, args map[string]interface{}) (string, error) {
	workspaceIDStr, ok := args["workspace_id"].(string)
	if !ok {
		return "", fmt.Errorf("add_participant requires workspace_id parameter")
	}

	workspaceID, err := uuid.Parse(workspaceIDStr)
	if err != nil {
		return "", fmt.Errorf("invalid workspace_id: %w", err)
	}

	role := "member"
	if r, ok := args["role"].(string); ok {
		role = r
	}

	var userID, agentID *uuid.UUID
	if userIDStr, ok := args["user_id"].(string); ok && userIDStr != "" {
		parsed, err := uuid.Parse(userIDStr)
		if err == nil {
			userID = &parsed
		}
	}

	if agentIDStr, ok := args["agent_id"].(string); ok && agentIDStr != "" {
		parsed, err := uuid.Parse(agentIDStr)
		if err == nil {
			agentID = &parsed
		}
	}

	err = t.workspace.AddParticipant(ctx, workspaceID, userID, agentID, role)
	if err != nil {
		return "", fmt.Errorf("participant addition failed: %w", err)
	}

	result := map[string]interface{}{
		"action":       "add_participant",
		"workspace_id": workspaceID.String(),
		"role":         role,
		"status":       "success",
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON), nil
}

/* getWorkspace retrieves workspace state */
func (t *CollaborationTool) getWorkspace(ctx context.Context, args map[string]interface{}) (string, error) {
	workspaceIDStr, ok := args["workspace_id"].(string)
	if !ok {
		return "", fmt.Errorf("get_workspace requires workspace_id parameter")
	}

	workspaceID, err := uuid.Parse(workspaceIDStr)
	if err != nil {
		return "", fmt.Errorf("invalid workspace_id: %w", err)
	}

	workspace, participants, err := t.workspace.GetWorkspaceState(ctx, workspaceID)
	if err != nil {
		return "", fmt.Errorf("workspace retrieval failed: %w", err)
	}

	result := map[string]interface{}{
		"action":       "get_workspace",
		"workspace":    workspace,
		"participants": participants,
		"status":       "success",
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON), nil
}

/* broadcastUpdate broadcasts an update to workspace */
func (t *CollaborationTool) broadcastUpdate(ctx context.Context, args map[string]interface{}) (string, error) {
	workspaceIDStr, ok := args["workspace_id"].(string)
	if !ok {
		return "", fmt.Errorf("broadcast_update requires workspace_id parameter")
	}

	workspaceID, err := uuid.Parse(workspaceIDStr)
	if err != nil {
		return "", fmt.Errorf("invalid workspace_id: %w", err)
	}

	updateType, ok := args["update_type"].(string)
	if !ok {
		updateType = "message"
	}

	content, ok := args["content"].(string)
	if !ok {
		return "", fmt.Errorf("broadcast_update requires content parameter")
	}

	var userID, agentID *uuid.UUID
	if userIDStr, ok := args["user_id"].(string); ok && userIDStr != "" {
		parsed, err := uuid.Parse(userIDStr)
		if err == nil {
			userID = &parsed
		}
	}

	if agentIDStr, ok := args["agent_id"].(string); ok && agentIDStr != "" {
		parsed, err := uuid.Parse(agentIDStr)
		if err == nil {
			agentID = &parsed
		}
	}

	metadata := make(map[string]interface{})
	if m, ok := args["metadata"].(map[string]interface{}); ok {
		metadata = m
	}

	err = t.workspace.BroadcastUpdate(ctx, workspaceID, userID, agentID, updateType, content, metadata)
	if err != nil {
		return "", fmt.Errorf("update broadcast failed: %w", err)
	}

	result := map[string]interface{}{
		"action":       "broadcast_update",
		"workspace_id": workspaceID.String(),
		"update_type":  updateType,
		"status":       "success",
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON), nil
}

/* Validate validates tool arguments */
func (t *CollaborationTool) Validate(args map[string]interface{}, schema map[string]interface{}) error {
	action, ok := args["action"].(string)
	if !ok {
		return fmt.Errorf("action parameter required")
	}

	validActions := map[string]bool{
		"create_workspace": true,
		"add_participant":  true,
		"get_workspace":    true,
		"broadcast_update": true,
	}

	if !validActions[action] {
		return fmt.Errorf("invalid action: %s", action)
	}

	return nil
}
