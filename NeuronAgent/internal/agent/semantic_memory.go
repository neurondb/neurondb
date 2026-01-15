/*-------------------------------------------------------------------------
 *
 * semantic_memory.go
 *    Semantic memory system
 *
 * Provides semantic memory for storing factual knowledge and concepts.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/semantic_memory.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"context"
	"fmt"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/pkg/neurondb"
)

/* SemanticMemoryManager manages semantic memory */
type SemanticMemoryManager struct {
	queries *db.Queries
	embed   *neurondb.EmbeddingClient
}

/* SemanticMemory represents a semantic memory chunk */
type SemanticMemory struct {
	ID          uuid.UUID
	AgentID     uuid.UUID
	Concept     string
	Fact        string
	Category    string
	Confidence  float64
	Embedding   []float32
	Relations   []string
	Metadata    map[string]interface{}
}

/* NewSemanticMemoryManager creates a new semantic memory manager */
func NewSemanticMemoryManager(queries *db.Queries, embedClient *neurondb.EmbeddingClient) *SemanticMemoryManager {
	return &SemanticMemoryManager{
		queries: queries,
		embed:   embedClient,
	}
}

/* Store stores a semantic memory */
func (smm *SemanticMemoryManager) Store(ctx context.Context, agentID uuid.UUID, concept, fact, category string, confidence float64, relations []string, metadata map[string]interface{}) (uuid.UUID, error) {
	/* Generate embedding for the concept and fact */
	embedding, err := smm.embed.Embed(ctx, fmt.Sprintf("%s: %s", concept, fact), "default")
	if err != nil {
		return uuid.Nil, fmt.Errorf("semantic memory storage failed: embedding_error=true, error=%w", err)
	}

	memoryID := uuid.New()

	/* Store in database */
	query := `INSERT INTO neurondb_agent.semantic_memory
		(id, agent_id, concept, fact, category, confidence, embedding, relations, metadata, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7::vector, $8::text[], $9::jsonb, NOW())`

	_, err = smm.queries.DB.ExecContext(ctx, query,
		memoryID,
		agentID,
		concept,
		fact,
		category,
		confidence,
		embedding,
		relations,
		metadata,
	)

	if err != nil {
		return uuid.Nil, fmt.Errorf("semantic memory storage failed: database_error=true, error=%w", err)
	}

	return memoryID, nil
}

/* Retrieve retrieves semantic memories by concept or similarity */
func (smm *SemanticMemoryManager) Retrieve(ctx context.Context, agentID uuid.UUID, query string, limit int) ([]*SemanticMemory, error) {
	if limit <= 0 {
		limit = 10
	}

	/* Generate query embedding */
	queryEmbedding, err := smm.embed.Embed(ctx, query, "default")
	if err != nil {
		return nil, fmt.Errorf("semantic memory retrieval failed: embedding_error=true, error=%w", err)
	}

	/* Search using vector similarity */
	searchQuery := `SELECT id, agent_id, concept, fact, category, confidence, relations, metadata
		FROM neurondb_agent.semantic_memory
		WHERE agent_id = $1
		ORDER BY embedding <=> $2::vector
		LIMIT $3`

	type MemoryRow struct {
		ID         uuid.UUID              `db:"id"`
		AgentID    uuid.UUID              `db:"agent_id"`
		Concept    string                 `db:"concept"`
		Fact       string                 `db:"fact"`
		Category   string                 `db:"category"`
		Confidence float64                `db:"confidence"`
		Relations  []string               `db:"relations"`
		Metadata   map[string]interface{} `db:"metadata"`
	}

	var rows []MemoryRow
	err = smm.queries.DB.SelectContext(ctx, &rows, searchQuery, agentID, queryEmbedding, limit)
	if err != nil {
		return nil, fmt.Errorf("semantic memory retrieval failed: database_error=true, error=%w", err)
	}

	memories := make([]*SemanticMemory, len(rows))
	for i, row := range rows {
		memories[i] = &SemanticMemory{
			ID:         row.ID,
			AgentID:    row.AgentID,
			Concept:    row.Concept,
			Fact:       row.Fact,
			Category:   row.Category,
			Confidence: row.Confidence,
			Relations:  row.Relations,
			Metadata:   row.Metadata,
		}
	}

	return memories, nil
}

/* RetrieveByCategory retrieves semantic memories by category */
func (smm *SemanticMemoryManager) RetrieveByCategory(ctx context.Context, agentID uuid.UUID, category string, limit int) ([]*SemanticMemory, error) {
	if limit <= 0 {
		limit = 10
	}

	query := `SELECT id, agent_id, concept, fact, category, confidence, relations, metadata
		FROM neurondb_agent.semantic_memory
		WHERE agent_id = $1 AND category = $2
		ORDER BY confidence DESC, created_at DESC
		LIMIT $3`

	type MemoryRow struct {
		ID         uuid.UUID              `db:"id"`
		AgentID    uuid.UUID              `db:"agent_id"`
		Concept    string                 `db:"concept"`
		Fact       string                 `db:"fact"`
		Category   string                 `db:"category"`
		Confidence float64                `db:"confidence"`
		Relations  []string               `db:"relations"`
		Metadata   map[string]interface{} `db:"metadata"`
	}

	var rows []MemoryRow
	err := smm.queries.DB.SelectContext(ctx, &rows, query, agentID, category, limit)
	if err != nil {
		return nil, fmt.Errorf("semantic memory retrieval by category failed: database_error=true, error=%w", err)
	}

	memories := make([]*SemanticMemory, len(rows))
	for i, row := range rows {
		memories[i] = &SemanticMemory{
			ID:         row.ID,
			AgentID:    row.AgentID,
			Concept:    row.Concept,
			Fact:       row.Fact,
			Category:   row.Category,
			Confidence: row.Confidence,
			Relations:  row.Relations,
			Metadata:   row.Metadata,
		}
	}

	return memories, nil
}

