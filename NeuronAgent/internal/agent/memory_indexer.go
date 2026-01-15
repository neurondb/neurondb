/*-------------------------------------------------------------------------
 *
 * memory_indexer.go
 *    Multi-dimensional memory indexing
 *
 * Provides indexing by time, importance, recency, relevance, and category
 * using NeuronDB vector search for multi-aspect retrieval.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/memory_indexer.go
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

/* MemoryIndexer provides multi-dimensional memory indexing */
type MemoryIndexer struct {
	queries *db.Queries
	embed   *neurondb.EmbeddingClient
}

/* MemoryQuery represents a multi-dimensional memory query */
type MemoryQuery struct {
	AgentID      uuid.UUID
	TextQuery    string
	Categories   []string
	StartTime    *time.Time
	EndTime      *time.Time
	MinImportance float64
	MaxResults   int
	SortBy       string // "relevance", "time", "importance", "recency"
}

/* NewMemoryIndexer creates a new memory indexer */
func NewMemoryIndexer(queries *db.Queries, embedClient *neurondb.EmbeddingClient) *MemoryIndexer {
	return &MemoryIndexer{
		queries: queries,
		embed:   embedClient,
	}
}

/* Search performs multi-dimensional memory search */
func (mi *MemoryIndexer) Search(ctx context.Context, query *MemoryQuery) ([]*IndexedMemory, error) {
	if query.MaxResults <= 0 {
		query.MaxResults = 10
	}

	/* Generate query embedding if text query provided */
	var queryEmbedding []float32
	var err error
	if query.TextQuery != "" {
		queryEmbedding, err = mi.embed.Embed(ctx, query.TextQuery, "default")
		if err != nil {
			return nil, fmt.Errorf("memory indexing search failed: embedding_error=true, error=%w", err)
		}
	}

	/* Build SQL query based on dimensions */
	sqlQuery := `SELECT 
		id, agent_id, session_id, content, memory_type, importance, 
		created_at, last_accessed, category, metadata
		FROM neurondb_agent.memory_chunks
		WHERE agent_id = $1`

	args := []interface{}{query.AgentID}
	argIndex := 2

	/* Add category filter */
	if len(query.Categories) > 0 {
		sqlQuery += fmt.Sprintf(" AND category = ANY($%d::text[])", argIndex)
		args = append(args, query.Categories)
		argIndex++
	}

	/* Add time range filter */
	if query.StartTime != nil {
		sqlQuery += fmt.Sprintf(" AND created_at >= $%d", argIndex)
		args = append(args, *query.StartTime)
		argIndex++
	}
	if query.EndTime != nil {
		sqlQuery += fmt.Sprintf(" AND created_at <= $%d", argIndex)
		args = append(args, *query.EndTime)
		argIndex++
	}

	/* Add importance filter */
	if query.MinImportance > 0 {
		sqlQuery += fmt.Sprintf(" AND importance >= $%d", argIndex)
		args = append(args, query.MinImportance)
		argIndex++
	}

	/* Add sorting */
	switch query.SortBy {
	case "time":
		sqlQuery += " ORDER BY created_at DESC"
	case "importance":
		sqlQuery += " ORDER BY importance DESC"
	case "recency":
		sqlQuery += " ORDER BY last_accessed DESC NULLS LAST"
	case "relevance":
		if len(queryEmbedding) > 0 {
			sqlQuery += fmt.Sprintf(" ORDER BY embedding <=> $%d::vector", argIndex)
			args = append(args, queryEmbedding)
			argIndex++
		} else {
			sqlQuery += " ORDER BY importance DESC"
		}
	default:
		if len(queryEmbedding) > 0 {
			sqlQuery += fmt.Sprintf(" ORDER BY embedding <=> $%d::vector", argIndex)
			args = append(args, queryEmbedding)
			argIndex++
		} else {
			sqlQuery += " ORDER BY created_at DESC"
		}
	}

	sqlQuery += fmt.Sprintf(" LIMIT $%d", argIndex)
	args = append(args, query.MaxResults)

	type MemoryRow struct {
		ID           uuid.UUID              `db:"id"`
		AgentID      uuid.UUID              `db:"agent_id"`
		SessionID    uuid.UUID              `db:"session_id"`
		Content      string                 `db:"content"`
		MemoryType   string                 `db:"memory_type"`
		Importance   float64                `db:"importance"`
		CreatedAt    time.Time              `db:"created_at"`
		LastAccessed *time.Time             `db:"last_accessed"`
		Category     string                 `db:"category"`
		Metadata     map[string]interface{} `db:"metadata"`
	}

	var rows []MemoryRow
	err = mi.queries.DB.SelectContext(ctx, &rows, sqlQuery, args...)
	if err != nil {
		return nil, fmt.Errorf("memory indexing search failed: database_error=true, error=%w", err)
	}

	memories := make([]*IndexedMemory, len(rows))
	for i, row := range rows {
		memories[i] = &IndexedMemory{
			ID:           row.ID,
			AgentID:      row.AgentID,
			SessionID:    row.SessionID,
			Content:      row.Content,
			MemoryType:   row.MemoryType,
			Importance:   row.Importance,
			CreatedAt:    row.CreatedAt,
			LastAccessed: row.LastAccessed,
			Category:     row.Category,
			Metadata:     row.Metadata,
		}
	}

	return memories, nil
}

/* IndexedMemory represents an indexed memory chunk */
type IndexedMemory struct {
	ID           uuid.UUID
	AgentID      uuid.UUID
	SessionID    uuid.UUID
	Content      string
	MemoryType   string
	Importance   float64
	CreatedAt    time.Time
	LastAccessed *time.Time
	Category     string
	Metadata     map[string]interface{}
}

