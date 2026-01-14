/*-------------------------------------------------------------------------
 *
 * workspace.go
 *    Real-time collaboration workspace management
 *
 * Provides workspace creation, participant management, and real-time
 * update broadcasting for collaborative agent tasks.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/collaboration/workspace.go
 *
 *-------------------------------------------------------------------------
 */

package collaboration

import (
	"context"
	"fmt"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
)

/* WorkspaceManager manages collaboration workspaces */
type WorkspaceManager struct {
	queries *db.Queries
	pubsub  *PubSub
}

/* Workspace represents a collaboration workspace */
type Workspace struct {
	ID            uuid.UUID
	Name          string
	OwnerID       *uuid.UUID
	Description   *string
	SharedContext map[string]interface{}
	CreatedAt     string
	UpdatedAt     string
}

/* Participant represents a workspace participant */
type Participant struct {
	ID          uuid.UUID
	WorkspaceID uuid.UUID
	UserID      *uuid.UUID
	AgentID     *uuid.UUID
	Role        string
	JoinedAt    string
}

/* WorkspaceUpdate represents a real-time update */
type WorkspaceUpdate struct {
	ID          uuid.UUID
	WorkspaceID uuid.UUID
	UserID      *uuid.UUID
	AgentID     *uuid.UUID
	UpdateType  string
	Content     string
	Metadata    map[string]interface{}
	CreatedAt   string
}

/* NewWorkspaceManager creates a new workspace manager */
func NewWorkspaceManager(queries *db.Queries, pubsub *PubSub) *WorkspaceManager {
	return &WorkspaceManager{
		queries: queries,
		pubsub:  pubsub,
	}
}

/* CreateWorkspace creates a new collaboration workspace */
func (w *WorkspaceManager) CreateWorkspace(ctx context.Context, name string, ownerID *uuid.UUID) (uuid.UUID, error) {
	query := `INSERT INTO neurondb_agent.collaboration_workspaces
		(name, owner_id)
		VALUES ($1, $2)
		RETURNING id`

	var workspaceID uuid.UUID
	err := w.queries.GetDB().GetContext(ctx, &workspaceID, query, name, ownerID)
	if err != nil {
		return uuid.Nil, fmt.Errorf("workspace creation failed: error=%w", err)
	}

	/* Add owner as participant */
	if ownerID != nil {
		err = w.AddParticipant(ctx, workspaceID, ownerID, nil, "owner")
		if err != nil {
			return workspaceID, err /* Return workspace ID even if participant add fails */
		}
	}

	return workspaceID, nil
}

/* AddParticipant adds a participant to a workspace */
func (w *WorkspaceManager) AddParticipant(ctx context.Context, workspaceID uuid.UUID, userID, agentID *uuid.UUID, role string) error {
	if role == "" {
		role = "member"
	}

	query := `INSERT INTO neurondb_agent.workspace_participants
		(workspace_id, user_id, agent_id, role)
		VALUES ($1, $2, $3, $4)
		ON CONFLICT (workspace_id, COALESCE(user_id::text, ''), COALESCE(agent_id::text, ''))
		DO UPDATE SET role = EXCLUDED.role`

	_, err := w.queries.GetDB().ExecContext(ctx, query, workspaceID, userID, agentID, role)
	if err != nil {
		return fmt.Errorf("participant addition failed: error=%w", err)
	}

	/* Broadcast update */
	w.BroadcastUpdate(ctx, workspaceID, userID, agentID, "state_change", "Participant added", map[string]interface{}{
		"user_id":  userID,
		"agent_id": agentID,
		"role":     role,
	})

	return nil
}

/* BroadcastUpdate broadcasts an update to all workspace participants */
func (w *WorkspaceManager) BroadcastUpdate(ctx context.Context, workspaceID uuid.UUID, userID, agentID *uuid.UUID, updateType, content string, metadata map[string]interface{}) error {
	/* Store update */
	query := `INSERT INTO neurondb_agent.workspace_updates
		(workspace_id, user_id, agent_id, update_type, content, metadata)
		VALUES ($1, $2, $3, $4, $5, $6::jsonb)
		RETURNING id`

	var updateID uuid.UUID
	err := w.queries.GetDB().GetContext(ctx, &updateID, query, workspaceID, userID, agentID, updateType, content, metadata)
	if err != nil {
		return fmt.Errorf("update storage failed: error=%w", err)
	}

	/* Broadcast via pubsub */
	if w.pubsub != nil {
		w.pubsub.Publish(ctx, fmt.Sprintf("workspace:%s", workspaceID.String()), map[string]interface{}{
			"id":          updateID.String(),
			"update_type": updateType,
			"content":     content,
			"metadata":    metadata,
		})
	}

	return nil
}

/* GetWorkspaceState retrieves current workspace state */
func (w *WorkspaceManager) GetWorkspaceState(ctx context.Context, workspaceID uuid.UUID) (*Workspace, []Participant, error) {
	/* Get workspace */
	query := `SELECT id, name, owner_id, description, shared_context, created_at, updated_at
		FROM neurondb_agent.collaboration_workspaces
		WHERE id = $1`

	type WorkspaceRow struct {
		ID            uuid.UUID              `db:"id"`
		Name          string                 `db:"name"`
		OwnerID       *uuid.UUID             `db:"owner_id"`
		Description   *string                `db:"description"`
		SharedContext map[string]interface{} `db:"shared_context"`
		CreatedAt     string                 `db:"created_at"`
		UpdatedAt     string                 `db:"updated_at"`
	}

	var wsRow WorkspaceRow
	err := w.queries.GetDB().GetContext(ctx, &wsRow, query, workspaceID)
	if err != nil {
		return nil, nil, fmt.Errorf("workspace retrieval failed: error=%w", err)
	}

	workspace := &Workspace{
		ID:            wsRow.ID,
		Name:          wsRow.Name,
		OwnerID:       wsRow.OwnerID,
		Description:   wsRow.Description,
		SharedContext: wsRow.SharedContext,
		CreatedAt:     wsRow.CreatedAt,
		UpdatedAt:     wsRow.UpdatedAt,
	}

	/* Get participants */
	participantsQuery := `SELECT id, workspace_id, user_id, agent_id, role, joined_at
		FROM neurondb_agent.workspace_participants
		WHERE workspace_id = $1`

	type ParticipantRow struct {
		ID          uuid.UUID  `db:"id"`
		WorkspaceID uuid.UUID  `db:"workspace_id"`
		UserID      *uuid.UUID `db:"user_id"`
		AgentID     *uuid.UUID `db:"agent_id"`
		Role        string     `db:"role"`
		JoinedAt    string     `db:"joined_at"`
	}

	var participantRows []ParticipantRow
	err = w.queries.GetDB().SelectContext(ctx, &participantRows, participantsQuery, workspaceID)
	if err != nil {
		return workspace, nil, nil /* Return workspace even if participants fail */
	}

	participants := make([]Participant, len(participantRows))
	for i, row := range participantRows {
		participants[i] = Participant{
			ID:          row.ID,
			WorkspaceID: row.WorkspaceID,
			UserID:      row.UserID,
			AgentID:     row.AgentID,
			Role:        row.Role,
			JoinedAt:    row.JoinedAt,
		}
	}

	return workspace, participants, nil
}

/* SyncContext syncs session context with workspace */
func (w *WorkspaceManager) SyncContext(ctx context.Context, workspaceID, sessionID uuid.UUID) error {
	query := `INSERT INTO neurondb_agent.workspace_sessions (workspace_id, session_id)
		VALUES ($1, $2)
		ON CONFLICT (workspace_id, session_id) DO NOTHING`

	_, err := w.queries.GetDB().ExecContext(ctx, query, workspaceID, sessionID)
	if err != nil {
		return fmt.Errorf("context sync failed: error=%w", err)
	}

	return nil
}
