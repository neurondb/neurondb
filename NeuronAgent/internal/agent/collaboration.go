/*-------------------------------------------------------------------------
 *
 * collaboration.go
 *    Multi-agent collaboration system
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/collaboration.go
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

type CollaborationManager struct {
	queries *db.Queries
	runtime *Runtime
}

/* NewCollaborationManager creates a new collaboration manager */
func NewCollaborationManager(queries *db.Queries, runtime *Runtime) *CollaborationManager {
	return &CollaborationManager{
		queries: queries,
		runtime: runtime,
	}
}

/* DelegateTask delegates a task to another agent */
func (c *CollaborationManager) DelegateTask(ctx context.Context, fromAgentID, toAgentID uuid.UUID, task string, sessionID uuid.UUID) (string, error) {
	/* Validate inputs */
	if task == "" {
		return "", fmt.Errorf("agent delegation failed: from_agent_id='%s', to_agent_id='%s', task_empty=true",
			fromAgentID.String(), toAgentID.String())
	}
	if fromAgentID == toAgentID {
		return "", fmt.Errorf("agent delegation failed: from_agent_id='%s', to_agent_id='%s', cannot_delegate_to_self=true",
			fromAgentID.String(), toAgentID.String())
	}

	/* Get target agent */
	targetAgent, err := c.queries.GetAgentByID(ctx, toAgentID)
	if err != nil {
		return "", fmt.Errorf("agent delegation failed: from_agent_id='%s', to_agent_id='%s', task_length=%d, target_agent_not_found=true, error=%w",
			fromAgentID.String(), toAgentID.String(), len(task), err)
	}

	/* Get or create session for target agent */
	targetSession, err := c.queries.GetSession(ctx, sessionID)
	if err != nil {
		/* Create new session for target agent */
		targetSession = &db.Session{
			AgentID: toAgentID,
		}
		if err := c.queries.CreateSession(ctx, targetSession); err != nil {
			return "", fmt.Errorf("agent delegation failed: from_agent_id='%s', to_agent_id='%s', session_creation_failed=true, error=%w",
				fromAgentID.String(), toAgentID.String(), err)
		}
	} else {
		/* Verify session belongs to target agent */
		if targetSession.AgentID != toAgentID {
			/* Create new session for target agent */
			targetSession = &db.Session{
				AgentID: toAgentID,
			}
			if err := c.queries.CreateSession(ctx, targetSession); err != nil {
				return "", fmt.Errorf("agent delegation failed: from_agent_id='%s', to_agent_id='%s', session_creation_failed=true, error=%w",
					fromAgentID.String(), toAgentID.String(), err)
			}
		}
	}

	/* Execute task with target agent */
	state, err := c.runtime.Execute(ctx, targetSession.ID, task)
	if err != nil {
		return "", fmt.Errorf("agent delegation execution failed: from_agent_id='%s', to_agent_id='%s', target_agent_name='%s', task_length=%d, error=%w",
			fromAgentID.String(), toAgentID.String(), targetAgent.Name, len(task), err)
	}

	return state.FinalAnswer, nil
}

/* SendMessage sends a message from one agent to another */
func (c *CollaborationManager) SendMessage(ctx context.Context, fromAgentID, toAgentID uuid.UUID, message string, sessionID uuid.UUID) error {
	/* Store inter-agent message */
	interAgentMessage := fmt.Sprintf("[From Agent %s] %s", fromAgentID.String()[:8], message)

	_, err := c.queries.CreateMessage(ctx, &db.Message{
		SessionID: sessionID,
		Role:      "system",
		Content:   interAgentMessage,
		Metadata: map[string]interface{}{
			"from_agent_id": fromAgentID.String(),
			"to_agent_id":   toAgentID.String(),
			"type":          "inter_agent",
		},
	})
	if err != nil {
		return fmt.Errorf("inter-agent message failed: from_agent_id='%s', to_agent_id='%s', message_length=%d, error=%w",
			fromAgentID.String(), toAgentID.String(), len(message), err)
	}

	return nil
}

/* GetAgentRelationships gets relationships for an agent */
func (c *CollaborationManager) GetAgentRelationships(ctx context.Context, agentID uuid.UUID) ([]AgentRelationship, error) {
	/* Query agent relationships table */
	query := `SELECT id, from_agent_id, to_agent_id, relationship_type, metadata, created_at
		FROM neurondb_agent.agent_relationships
		WHERE from_agent_id = $1 OR to_agent_id = $1
		ORDER BY created_at DESC`

	var relationships []AgentRelationship
	/* Access DB directly - Queries struct has db field */
	err := c.queries.GetDB().SelectContext(ctx, &relationships, query, agentID)
	if err != nil {
		return nil, fmt.Errorf("agent relationships retrieval failed: agent_id='%s', error=%w", agentID.String(), err)
	}

	return relationships, nil
}

/* CreateRelationship creates a relationship between agents */
func (c *CollaborationManager) CreateRelationship(ctx context.Context, fromAgentID, toAgentID uuid.UUID, relationshipType string, metadata map[string]interface{}) error {
	query := `INSERT INTO neurondb_agent.agent_relationships
		(from_agent_id, to_agent_id, relationship_type, metadata)
		VALUES ($1, $2, $3, $4::jsonb)`

	_, err := c.queries.DB.ExecContext(ctx, query, fromAgentID, toAgentID, relationshipType, metadata)
	if err != nil {
		return fmt.Errorf("agent relationship creation failed: from_agent_id='%s', to_agent_id='%s', relationship_type='%s', error=%w",
			fromAgentID.String(), toAgentID.String(), relationshipType, err)
	}

	return nil
}

/* AgentRelationship represents a relationship between agents */
type AgentRelationship struct {
	ID               uuid.UUID              `db:"id"`
	FromAgentID      uuid.UUID              `db:"from_agent_id"`
	ToAgentID        uuid.UUID              `db:"to_agent_id"`
	RelationshipType string                 `db:"relationship_type"`
	Metadata         map[string]interface{} `db:"metadata"`
	CreatedAt        string                 `db:"created_at"`
}
