/*-------------------------------------------------------------------------
 *
 * manager.go
 *    Session management for NeuronAgent
 *
 * Provides session lifecycle management including creation, retrieval,
 * and caching of agent conversation sessions.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/session/manager.go
 *
 *-------------------------------------------------------------------------
 */

package session

import (
	"context"
	"fmt"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/metrics"
)

type Manager struct {
	queries *db.Queries
	cache   *Cache
}

func NewManager(queries *db.Queries, cache *Cache) *Manager {
	return &Manager{
		queries: queries,
		cache:   cache,
	}
}

/* Create creates a new session */
func (m *Manager) Create(ctx context.Context, agentID uuid.UUID, externalUserID *string, metadata map[string]interface{}) (*db.Session, error) {
	/* Validate input */
	if agentID == uuid.Nil {
		return nil, fmt.Errorf("session creation failed: agent_id_empty=true")
	}

	/* Check context cancellation */
	if ctx.Err() != nil {
		return nil, fmt.Errorf("session creation cancelled: context_error=%w", ctx.Err())
	}

	session := &db.Session{
		AgentID:        agentID,
		ExternalUserID: externalUserID,
		Metadata:       metadata,
	}

	if err := m.queries.CreateSession(ctx, session); err != nil {
		return nil, fmt.Errorf("session creation failed: agent_id='%s', error=%w", agentID.String(), err)
	}

	/* Cache the session */
	if m.cache != nil {
		m.cache.Set(session.ID, session)
	}

	return session, nil
}

/* Get retrieves a session by ID */
func (m *Manager) Get(ctx context.Context, id uuid.UUID) (*db.Session, error) {
	/* Validate input */
	if id == uuid.Nil {
		return nil, fmt.Errorf("session retrieval failed: session_id_empty=true")
	}

	/* Check context cancellation */
	if ctx.Err() != nil {
		return nil, fmt.Errorf("session retrieval cancelled: session_id='%s', context_error=%w", id.String(), ctx.Err())
	}

	/* Try cache first */
	if m.cache != nil {
		if session := m.cache.Get(id); session != nil {
			return session, nil
		}
	}

	/* Get from database */
	session, err := m.queries.GetSession(ctx, id)
	if err != nil {
		return nil, fmt.Errorf("session retrieval failed: session_id='%s', error=%w", id.String(), err)
	}

	/* Cache it */
	if m.cache != nil {
		m.cache.Set(id, session)
	}

	return session, nil
}

/* List lists sessions for an agent */
func (m *Manager) List(ctx context.Context, agentID uuid.UUID, limit, offset int) ([]db.Session, error) {
	return m.queries.ListSessions(ctx, agentID, limit, offset)
}

/* Delete deletes a session */
func (m *Manager) Delete(ctx context.Context, id uuid.UUID) error {
	/* Validate input */
	if id == uuid.Nil {
		return fmt.Errorf("session deletion failed: session_id_empty=true")
	}

	/* Check context cancellation */
	if ctx.Err() != nil {
		return fmt.Errorf("session deletion cancelled: session_id='%s', context_error=%w", id.String(), ctx.Err())
	}

	if err := m.queries.DeleteSession(ctx, id); err != nil {
		return fmt.Errorf("session deletion failed: session_id='%s', error=%w", id.String(), err)
	}

	/* Remove from cache */
	if m.cache != nil {
		m.cache.Delete(id)
	}

	return nil
}

/* UpdateActivity updates the last activity time for a session */
func (m *Manager) UpdateActivity(ctx context.Context, id uuid.UUID) error {
	/* Validate input */
	if id == uuid.Nil {
		return fmt.Errorf("session activity update failed: session_id_empty=true")
	}

	/* Check context cancellation */
	if ctx.Err() != nil {
		return fmt.Errorf("session activity update cancelled: session_id='%s', context_error=%w", id.String(), ctx.Err())
	}

	/* This is handled by the database trigger, but we can refresh cache */
	if m.cache != nil {
		if session, err := m.queries.GetSession(ctx, id); err == nil {
			m.cache.Set(id, session)
		} else {
			/* Log error but don't fail - cache refresh is non-critical */
			metrics.WarnWithContext(ctx, "Failed to refresh session cache during activity update", map[string]interface{}{
				"session_id": id.String(),
				"error":      err.Error(),
			})
		}
	}
	return nil
}
