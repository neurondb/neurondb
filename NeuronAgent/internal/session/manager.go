package session

import (
	"context"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
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

// Create creates a new session
func (m *Manager) Create(ctx context.Context, agentID uuid.UUID, externalUserID *string, metadata map[string]interface{}) (*db.Session, error) {
	session := &db.Session{
		AgentID:       agentID,
		ExternalUserID: externalUserID,
		Metadata:      metadata,
	}

	if err := m.queries.CreateSession(ctx, session); err != nil {
		return nil, err
	}

	// Cache the session
	if m.cache != nil {
		m.cache.Set(session.ID, session)
	}

	return session, nil
}

// Get retrieves a session by ID
func (m *Manager) Get(ctx context.Context, id uuid.UUID) (*db.Session, error) {
	// Try cache first
	if m.cache != nil {
		if session := m.cache.Get(id); session != nil {
			return session, nil
		}
	}

	// Get from database
	session, err := m.queries.GetSession(ctx, id)
	if err != nil {
		return nil, err
	}

	// Cache it
	if m.cache != nil {
		m.cache.Set(id, session)
	}

	return session, nil
}

// List lists sessions for an agent
func (m *Manager) List(ctx context.Context, agentID uuid.UUID, limit, offset int) ([]db.Session, error) {
	return m.queries.ListSessions(ctx, agentID, limit, offset)
}

// Delete deletes a session
func (m *Manager) Delete(ctx context.Context, id uuid.UUID) error {
	if err := m.queries.DeleteSession(ctx, id); err != nil {
		return err
	}

	// Remove from cache
	if m.cache != nil {
		m.cache.Delete(id)
	}

	return nil
}

// UpdateActivity updates the last activity time for a session
func (m *Manager) UpdateActivity(ctx context.Context, id uuid.UUID) error {
	// This is handled by the database trigger, but we can refresh cache
	if m.cache != nil {
		if session, err := m.queries.GetSession(ctx, id); err == nil {
			m.cache.Set(id, session)
		}
	}
	return nil
}

