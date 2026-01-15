/*-------------------------------------------------------------------------
 *
 * episodic_memory.go
 *    Episodic memory system
 *
 * Provides episodic memory for storing specific events with temporal context.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/episodic_memory.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/pkg/neurondb"
)

/* EpisodicMemoryManager manages episodic memory */
type EpisodicMemoryManager struct {
	queries *db.Queries
	embed   *neurondb.EmbeddingClient
}

/* EpisodicMemory represents an episodic memory chunk */
type EpisodicMemory struct {
	ID          uuid.UUID
	AgentID     uuid.UUID
	SessionID   uuid.UUID
	Event       string
	Context     string
	Timestamp   time.Time
	Importance  float64
	EmotionalValence float64
	Embedding   []float32
	Metadata    map[string]interface{}
}

/* NewEpisodicMemoryManager creates a new episodic memory manager */
func NewEpisodicMemoryManager(queries *db.Queries, embedClient *neurondb.EmbeddingClient) *EpisodicMemoryManager {
	return &EpisodicMemoryManager{
		queries: queries,
		embed:   embedClient,
	}
}

/* Store stores an episodic memory */
func (emm *EpisodicMemoryManager) Store(ctx context.Context, agentID, sessionID uuid.UUID, event, context string, importance, emotionalValence float64, metadata map[string]interface{}) (uuid.UUID, error) {
	/* Generate embedding for the event */
	embedding, err := emm.embed.Embed(ctx, fmt.Sprintf("%s %s", event, context), "default")
	if err != nil {
		return uuid.Nil, fmt.Errorf("episodic memory storage failed: embedding_error=true, error=%w", err)
	}

	memoryID := uuid.New()

	/* Store in database */
	query := `INSERT INTO neurondb_agent.episodic_memory
		(id, agent_id, session_id, event, context, timestamp, importance, emotional_valence, embedding, metadata, created_at)
		VALUES ($1, $2, $3, $4, $5, NOW(), $6, $7, $8::vector, $9::jsonb, NOW())`

	_, err = emm.queries.DB.ExecContext(ctx, query,
		memoryID,
		agentID,
		sessionID,
		event,
		context,
		importance,
		emotionalValence,
		embedding,
		metadata,
	)

	if err != nil {
		return uuid.Nil, fmt.Errorf("episodic memory storage failed: database_error=true, error=%w", err)
	}

	return memoryID, nil
}

/* Retrieve retrieves episodic memories by similarity */
func (emm *EpisodicMemoryManager) Retrieve(ctx context.Context, agentID uuid.UUID, query string, limit int) ([]*EpisodicMemory, error) {
	if limit <= 0 {
		limit = 10
	}

	/* Generate query embedding */
	queryEmbedding, err := emm.embed.Embed(ctx, query, "default")
	if err != nil {
		return nil, fmt.Errorf("episodic memory retrieval failed: embedding_error=true, error=%w", err)
	}

	/* Search using vector similarity */
	searchQuery := `SELECT id, agent_id, session_id, event, context, timestamp, importance, emotional_valence, metadata
		FROM neurondb_agent.episodic_memory
		WHERE agent_id = $1
		ORDER BY embedding <=> $2::vector
		LIMIT $3`

	type MemoryRow struct {
		ID              uuid.UUID              `db:"id"`
		AgentID         uuid.UUID              `db:"agent_id"`
		SessionID       uuid.UUID              `db:"session_id"`
		Event           string                 `db:"event"`
		Context         string                 `db:"context"`
		Timestamp       time.Time              `db:"timestamp"`
		Importance      float64                `db:"importance"`
		EmotionalValence float64               `db:"emotional_valence"`
		Metadata        map[string]interface{} `db:"metadata"`
	}

	var rows []MemoryRow
	err = emm.queries.DB.SelectContext(ctx, &rows, searchQuery, agentID, queryEmbedding, limit)
	if err != nil {
		return nil, fmt.Errorf("episodic memory retrieval failed: database_error=true, error=%w", err)
	}

	memories := make([]*EpisodicMemory, len(rows))
	for i, row := range rows {
		memories[i] = &EpisodicMemory{
			ID:              row.ID,
			AgentID:         row.AgentID,
			SessionID:       row.SessionID,
			Event:           row.Event,
			Context:         row.Context,
			Timestamp:       row.Timestamp,
			Importance:      row.Importance,
			EmotionalValence: row.EmotionalValence,
			Metadata:        row.Metadata,
		}
	}

	return memories, nil
}

/* RetrieveByTime retrieves episodic memories within a time range */
func (emm *EpisodicMemoryManager) RetrieveByTime(ctx context.Context, agentID uuid.UUID, startTime, endTime time.Time, limit int) ([]*EpisodicMemory, error) {
	if limit <= 0 {
		limit = 10
	}

	query := `SELECT id, agent_id, session_id, event, context, timestamp, importance, emotional_valence, metadata
		FROM neurondb_agent.episodic_memory
		WHERE agent_id = $1 AND timestamp >= $2 AND timestamp <= $3
		ORDER BY timestamp DESC
		LIMIT $4`

	type MemoryRow struct {
		ID              uuid.UUID              `db:"id"`
		AgentID         uuid.UUID              `db:"agent_id"`
		SessionID       uuid.UUID              `db:"session_id"`
		Event           string                 `db:"event"`
		Context         string                 `db:"context"`
		Timestamp       time.Time              `db:"timestamp"`
		Importance      float64                `db:"importance"`
		EmotionalValence float64               `db:"emotional_valence"`
		Metadata        map[string]interface{} `db:"metadata"`
	}

	var rows []MemoryRow
	err := emm.queries.DB.SelectContext(ctx, &rows, query, agentID, startTime, endTime, limit)
	if err != nil {
		return nil, fmt.Errorf("episodic memory retrieval by time failed: database_error=true, error=%w", err)
	}

	memories := make([]*EpisodicMemory, len(rows))
	for i, row := range rows {
		memories[i] = &EpisodicMemory{
			ID:              row.ID,
			AgentID:         row.AgentID,
			SessionID:       row.SessionID,
			Event:           row.Event,
			Context:         row.Context,
			Timestamp:       row.Timestamp,
			Importance:      row.Importance,
			EmotionalValence: row.EmotionalValence,
			Metadata:        row.Metadata,
		}
	}

	return memories, nil
}

