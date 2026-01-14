/*-------------------------------------------------------------------------
 *
 * hierarchical_memory.go
 *    Hierarchical memory system with STM/MTM/LPM tiers
 *
 * Implements a three-tier memory system: Short-Term Memory (STM) for
 * real-time conversation data, Mid-Term Memory (MTM) for topic summaries,
 * and Long-Term Personal Memory (LPM) for permanent preferences and knowledge.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/hierarchical_memory.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"context"
	"fmt"
	"strings"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/pkg/neurondb"
)

/* HierarchicalMemoryManager manages hierarchical memory system */
type HierarchicalMemoryManager struct {
	db      *db.DB
	queries *db.Queries
	embed   *neurondb.EmbeddingClient
	stm     *ShortTermMemory
	mtm     *MidTermMemory
	lpm     *LongTermPersonalMemory
}

/* ShortTermMemory manages short-term memory (1 hour TTL) */
type ShortTermMemory struct {
	db      *db.DB
	queries *db.Queries
	embed   *neurondb.EmbeddingClient
}

/* MidTermMemory manages mid-term memory (7 days TTL) */
type MidTermMemory struct {
	db      *db.DB
	queries *db.Queries
	embed   *neurondb.EmbeddingClient
}

/* LongTermPersonalMemory manages long-term personal memory (permanent) */
type LongTermPersonalMemory struct {
	db      *db.DB
	queries *db.Queries
	embed   *neurondb.EmbeddingClient
}

/* NewHierarchicalMemoryManager creates a new hierarchical memory manager */
func NewHierarchicalMemoryManager(database *db.DB, queries *db.Queries, embedClient *neurondb.EmbeddingClient) *HierarchicalMemoryManager {
	return &HierarchicalMemoryManager{
		db:      database,
		queries: queries,
		embed:   embedClient,
		stm:     &ShortTermMemory{db: database, queries: queries, embed: embedClient},
		mtm:     &MidTermMemory{db: database, queries: queries, embed: embedClient},
		lpm:     &LongTermPersonalMemory{db: database, queries: queries, embed: embedClient},
	}
}

/* StoreSTM stores content in short-term memory */
func (h *HierarchicalMemoryManager) StoreSTM(ctx context.Context, agentID, sessionID uuid.UUID, content string, importance float64) (uuid.UUID, error) {
	/* Compute embedding */
	embedding, err := h.embed.Embed(ctx, content, "all-MiniLM-L6-v2")
	if err != nil {
		return uuid.Nil, fmt.Errorf("STM embedding failed: error=%w", err)
	}

	/* Store in STM table */
	query := `INSERT INTO neurondb_agent.memory_stm
		(agent_id, session_id, content, embedding, importance_score)
		VALUES ($1, $2, $3, $4, $5)
		RETURNING id`

	var id uuid.UUID
	err = h.db.DB.GetContext(ctx, &id, query, agentID, sessionID, content, embedding, importance)
	if err != nil {
		return uuid.Nil, fmt.Errorf("STM storage failed: error=%w", err)
	}

	return id, nil
}

/* PromoteToMTM promotes STM entries to MTM */
func (h *HierarchicalMemoryManager) PromoteToMTM(ctx context.Context, agentID uuid.UUID, stmIDs []uuid.UUID, topic string) (uuid.UUID, error) {
	/* Retrieve STM entries */
	query := `SELECT content, importance_score FROM neurondb_agent.memory_stm
		WHERE id = ANY($1) AND agent_id = $2`

	type STMRow struct {
		Content         string  `db:"content"`
		ImportanceScore float64 `db:"importance_score"`
	}

	var rows []STMRow
	err := h.db.DB.SelectContext(ctx, &rows, query, stmIDs, agentID)
	if err != nil {
		return uuid.Nil, fmt.Errorf("STM retrieval for promotion failed: error=%w", err)
	}

	if len(rows) == 0 {
		return uuid.Nil, fmt.Errorf("no STM entries found for promotion")
	}

	/* Combine content */
	var combined strings.Builder
	var avgImportance float64
	for _, row := range rows {
		combined.WriteString(row.Content)
		combined.WriteString("\n\n")
		avgImportance += row.ImportanceScore
	}
	avgImportance /= float64(len(rows))

	/* Compute embedding for combined content */
	embedding, err := h.embed.Embed(ctx, combined.String(), "all-MiniLM-L6-v2")
	if err != nil {
		return uuid.Nil, fmt.Errorf("MTM embedding failed: error=%w", err)
	}

	/* Store in MTM table */
	insertQuery := `INSERT INTO neurondb_agent.memory_mtm
		(agent_id, topic, content, embedding, importance_score, source_stm_ids)
		VALUES ($1, $2, $3, $4, $5, $6)
		RETURNING id`

	var mtmID uuid.UUID
	err = h.db.DB.GetContext(ctx, &mtmID, insertQuery, agentID, topic, combined.String(), embedding, avgImportance, stmIDs)
	if err != nil {
		return uuid.Nil, fmt.Errorf("MTM storage failed: error=%w", err)
	}

	/* Record transition */
	for _, stmID := range stmIDs {
		transitionQuery := `INSERT INTO neurondb_agent.memory_transitions
			(agent_id, from_tier, to_tier, source_id, target_id, reason)
			VALUES ($1, 'stm', 'mtm', $2, $3, $4)`

		_, err = h.db.DB.ExecContext(ctx, transitionQuery, agentID, stmID, mtmID, "pattern_detected")
		/* Continue even if transition logging fails */
	}

	return mtmID, nil
}

/* PromoteToLPM promotes MTM entries to LPM */
func (h *HierarchicalMemoryManager) PromoteToLPM(ctx context.Context, agentID uuid.UUID, mtmIDs []uuid.UUID, category string, userID *uuid.UUID) (uuid.UUID, error) {
	/* Retrieve MTM entries */
	query := `SELECT content, importance_score, pattern_count FROM neurondb_agent.memory_mtm
		WHERE id = ANY($1) AND agent_id = $2`

	type MTMRow struct {
		Content         string  `db:"content"`
		ImportanceScore float64 `db:"importance_score"`
		PatternCount    int     `db:"pattern_count"`
	}

	var rows []MTMRow
	err := h.db.DB.SelectContext(ctx, &rows, query, mtmIDs, agentID)
	if err != nil {
		return uuid.Nil, fmt.Errorf("MTM retrieval for promotion failed: error=%w", err)
	}

	if len(rows) == 0 {
		return uuid.Nil, fmt.Errorf("no MTM entries found for promotion")
	}

	/* Combine content and compute confidence */
	var combined strings.Builder
	var avgImportance float64
	var totalPatternCount int

	for _, row := range rows {
		combined.WriteString(row.Content)
		combined.WriteString("\n\n")
		avgImportance += row.ImportanceScore
		totalPatternCount += row.PatternCount
	}
	avgImportance /= float64(len(rows))

	/* Confidence based on pattern count */
	confidence := float64(totalPatternCount) / (float64(totalPatternCount) + 5.0)
	if confidence > 0.95 {
		confidence = 0.95
	}

	/* Compute embedding */
	embedding, err := h.embed.Embed(ctx, combined.String(), "all-MiniLM-L6-v2")
	if err != nil {
		return uuid.Nil, fmt.Errorf("LPM embedding failed: error=%w", err)
	}

	/* Store in LPM table */
	insertQuery := `INSERT INTO neurondb_agent.memory_lpm
		(agent_id, user_id, category, content, embedding, importance_score, source_mtm_ids, confidence)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
		RETURNING id`

	var lpmID uuid.UUID
	err = h.db.DB.GetContext(ctx, &lpmID, insertQuery, agentID, userID, category, combined.String(), embedding, avgImportance, mtmIDs, confidence)
	if err != nil {
		return uuid.Nil, fmt.Errorf("LPM storage failed: error=%w", err)
	}

	/* Record transitions */
	for _, mtmID := range mtmIDs {
		transitionQuery := `INSERT INTO neurondb_agent.memory_transitions
			(agent_id, from_tier, to_tier, source_id, target_id, reason)
			VALUES ($1, 'mtm', 'lpm', $2, $3, $4)`

		_, err = h.db.DB.ExecContext(ctx, transitionQuery, agentID, mtmID, lpmID, "high_confidence_pattern")
		/* Continue even if transition logging fails */
	}

	return lpmID, nil
}

/* RetrieveHierarchical queries across memory tiers */
func (h *HierarchicalMemoryManager) RetrieveHierarchical(ctx context.Context, agentID uuid.UUID, query string, tiers []string, topK int) (map[string][]MemoryChunk, error) {
	/* Compute query embedding */
	queryEmbedding, err := h.embed.Embed(ctx, query, "all-MiniLM-L6-v2")
	if err != nil {
		return nil, fmt.Errorf("query embedding failed: error=%w", err)
	}

	results := make(map[string][]MemoryChunk)

	/* Query each tier */
	for _, tier := range tiers {
		var tableName string
		switch tier {
		case "stm":
			tableName = "memory_stm"
		case "mtm":
			tableName = "memory_mtm"
		case "lpm":
			tableName = "memory_lpm"
		default:
			continue
		}

		sqlQuery := fmt.Sprintf(`SELECT id::text, content, importance_score,
			1 - (embedding <=> $1::neurondb_vector) AS similarity
			FROM neurondb_agent.%s
			WHERE agent_id = $2
			ORDER BY embedding <=> $1::neurondb_vector
			LIMIT $3`, tableName)

		type ResultRow struct {
			ID              string  `db:"id"`
			Content         string  `db:"content"`
			ImportanceScore float64 `db:"importance_score"`
			Similarity      float64 `db:"similarity"`
		}

		var rows []ResultRow
		err := h.db.DB.SelectContext(ctx, &rows, sqlQuery, queryEmbedding, agentID, topK)
		if err != nil {
			/* Continue to next tier on error */
			continue
		}

		chunks := make([]MemoryChunk, len(rows))
		for i, row := range rows {
			chunks[i] = MemoryChunk{
				Content:         row.Content,
				ImportanceScore: row.ImportanceScore,
				Similarity:      row.Similarity,
			}
		}

		results[tier] = chunks
	}

	return results, nil
}

/* CleanupExpired removes expired STM and MTM entries */
func (h *HierarchicalMemoryManager) CleanupExpired(ctx context.Context) (int, error) {
	/* Delete expired STM entries */
	stmQuery := `DELETE FROM neurondb_agent.memory_stm
		WHERE expires_at < NOW()
		RETURNING id`

	var deletedSTM []uuid.UUID
	err := h.db.DB.SelectContext(ctx, &deletedSTM, stmQuery)
	if err != nil {
		return 0, fmt.Errorf("STM cleanup failed: error=%w", err)
	}

	/* Delete expired MTM entries */
	mtmQuery := `DELETE FROM neurondb_agent.memory_mtm
		WHERE expires_at < NOW()
		RETURNING id`

	var deletedMTM []uuid.UUID
	err = h.db.DB.SelectContext(ctx, &deletedMTM, mtmQuery)
	if err != nil {
		return len(deletedSTM), fmt.Errorf("MTM cleanup failed: error=%w", err)
	}

	return len(deletedSTM) + len(deletedMTM), nil
}
