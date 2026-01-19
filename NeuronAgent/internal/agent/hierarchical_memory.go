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
func (h *HierarchicalMemoryManager) RetrieveHierarchical(ctx context.Context, agentID uuid.UUID, query string, tiers []string, topK int) ([]map[string]interface{}, error) {
	/* Compute query embedding */
	queryEmbedding, err := h.embed.Embed(ctx, query, "all-MiniLM-L6-v2")
	if err != nil {
		return nil, fmt.Errorf("query embedding failed: error=%w", err)
	}

	var results []map[string]interface{}

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
			1 - (embedding <=> $1::neurondb_vector) AS similarity,
			COALESCE(metadata, '{}'::jsonb) AS metadata
			FROM neurondb_agent.%s
			WHERE agent_id = $2
			ORDER BY embedding <=> $1::neurondb_vector
			LIMIT $3`, tableName)

		type ResultRow struct {
			ID              string                 `db:"id"`
			Content         string                 `db:"content"`
			ImportanceScore float64                `db:"importance_score"`
			Similarity      float64                `db:"similarity"`
			Metadata        map[string]interface{} `db:"metadata"`
		}

		var rows []ResultRow
		err := h.db.DB.SelectContext(ctx, &rows, sqlQuery, queryEmbedding, agentID, topK)
		if err != nil {
			/* Continue to next tier on error */
			continue
		}

		/* Convert rows to map format */
		for _, row := range rows {
			result := map[string]interface{}{
				"id":              row.ID,
				"tier":            tier,
				"content":         row.Content,
				"importance_score": row.ImportanceScore,
				"similarity":      row.Similarity,
				"metadata":        row.Metadata,
			}
			results = append(results, result)
		}
	}

	return results, nil
}

/* UpdateMemory updates existing memory in any tier */
func (h *HierarchicalMemoryManager) UpdateMemory(ctx context.Context, agentID uuid.UUID, memoryID uuid.UUID, tier string, content *string, importance *float64, topic *string, category *string) error {
	/* Validate tier */
	if tier != "stm" && tier != "mtm" && tier != "lpm" {
		return fmt.Errorf("invalid tier: %s (must be stm, mtm, or lpm)", tier)
	}

	/* Check if memory exists and retrieve current content/embedding */
	type MemoryRow struct {
		Content  string    `db:"content"`
		Embedding []float32 `db:"embedding"`
	}

	var currentRow MemoryRow
	var checkQuery string

	switch tier {
	case "stm":
		checkQuery = `SELECT content, embedding FROM neurondb_agent.memory_stm WHERE id = $1 AND agent_id = $2`
	case "mtm":
		checkQuery = `SELECT content, embedding FROM neurondb_agent.memory_mtm WHERE id = $1 AND agent_id = $2`
	case "lpm":
		checkQuery = `SELECT content, embedding FROM neurondb_agent.memory_lpm WHERE id = $1 AND agent_id = $2`
	}

	err := h.db.DB.GetContext(ctx, &currentRow, checkQuery, memoryID, agentID)
	if err != nil {
		return fmt.Errorf("%s memory not found: error=%w", strings.ToUpper(tier), err)
	}

	/* Determine if content changed and needs re-embedding */
	needsReembedding := content != nil && *content != currentRow.Content
	finalContent := currentRow.Content
	if content != nil {
		finalContent = *content
	}

	/* Recompute embedding if content changed */
	var finalEmbedding interface{} = nil
	if needsReembedding {
		newEmbedding, err := h.embed.Embed(ctx, finalContent, "all-MiniLM-L6-v2")
		if err != nil {
			return fmt.Errorf("embedding recalculation failed: error=%w", err)
		}
		finalEmbedding = newEmbedding
	}

	/* Build update query based on tier */
	switch tier {
	case "stm":
		if finalEmbedding != nil {
			updateQuery := `UPDATE neurondb_agent.memory_stm 
				SET content = COALESCE($3, content),
					embedding = $4::neurondb_vector,
					importance_score = COALESCE($5, importance_score),
					updated_at = NOW()
				WHERE id = $1 AND agent_id = $2`
			_, err := h.db.DB.ExecContext(ctx, updateQuery, memoryID, agentID, content, finalEmbedding, importance)
			if err != nil {
				return fmt.Errorf("STM update failed: error=%w", err)
			}
		} else {
			updateQuery := `UPDATE neurondb_agent.memory_stm 
				SET content = COALESCE($3, content),
					importance_score = COALESCE($4, importance_score),
					updated_at = NOW()
				WHERE id = $1 AND agent_id = $2`
			_, err := h.db.DB.ExecContext(ctx, updateQuery, memoryID, agentID, content, importance)
			if err != nil {
				return fmt.Errorf("STM update failed: error=%w", err)
			}
		}

	case "mtm":
		if finalEmbedding != nil {
			updateQuery := `UPDATE neurondb_agent.memory_mtm 
				SET content = COALESCE($3, content),
					embedding = $4::neurondb_vector,
					importance_score = COALESCE($5, importance_score),
					topic = COALESCE($6, topic),
					updated_at = NOW()
				WHERE id = $1 AND agent_id = $2`
			_, err := h.db.DB.ExecContext(ctx, updateQuery, memoryID, agentID, content, finalEmbedding, importance, topic)
			if err != nil {
				return fmt.Errorf("MTM update failed: error=%w", err)
			}
		} else {
			updateQuery := `UPDATE neurondb_agent.memory_mtm 
				SET content = COALESCE($3, content),
					importance_score = COALESCE($4, importance_score),
					topic = COALESCE($5, topic),
					updated_at = NOW()
				WHERE id = $1 AND agent_id = $2`
			_, err := h.db.DB.ExecContext(ctx, updateQuery, memoryID, agentID, content, importance, topic)
			if err != nil {
				return fmt.Errorf("MTM update failed: error=%w", err)
			}
		}

	case "lpm":
		if finalEmbedding != nil {
			updateQuery := `UPDATE neurondb_agent.memory_lpm 
				SET content = COALESCE($3, content),
					embedding = $4::neurondb_vector,
					importance_score = COALESCE($5, importance_score),
					category = COALESCE($6, category),
					updated_at = NOW()
				WHERE id = $1 AND agent_id = $2`
			_, err := h.db.DB.ExecContext(ctx, updateQuery, memoryID, agentID, content, finalEmbedding, importance, category)
			if err != nil {
				return fmt.Errorf("LPM update failed: error=%w", err)
			}
		} else {
			updateQuery := `UPDATE neurondb_agent.memory_lpm 
				SET content = COALESCE($3, content),
					importance_score = COALESCE($4, importance_score),
					category = COALESCE($5, category),
					updated_at = NOW()
				WHERE id = $1 AND agent_id = $2`
			_, err := h.db.DB.ExecContext(ctx, updateQuery, memoryID, agentID, content, importance, category)
			if err != nil {
				return fmt.Errorf("LPM update failed: error=%w", err)
			}
		}
	}

	return nil
}

/* DeleteMemory deletes memory from any tier */
func (h *HierarchicalMemoryManager) DeleteMemory(ctx context.Context, agentID uuid.UUID, memoryID uuid.UUID, tier string) error {
	/* Validate tier */
	if tier != "stm" && tier != "mtm" && tier != "lpm" {
		return fmt.Errorf("invalid tier: %s (must be stm, mtm, or lpm)", tier)
	}

	/* Record deletion in transitions table for audit */
	transitionQuery := `INSERT INTO neurondb_agent.memory_transitions
		(agent_id, from_tier, to_tier, source_id, target_id, reason)
		VALUES ($1, $2, 'deleted', $3, $3, 'agent_deletion')`
	_, _ = h.db.DB.ExecContext(ctx, transitionQuery, agentID, tier, memoryID)
	/* Continue even if transition logging fails */

	/* Delete from appropriate table */
	switch tier {
	case "stm":
		deleteQuery := `DELETE FROM neurondb_agent.memory_stm WHERE id = $1 AND agent_id = $2`
		result, err := h.db.DB.ExecContext(ctx, deleteQuery, memoryID, agentID)
		if err != nil {
			return fmt.Errorf("STM deletion failed: error=%w", err)
		}
		rowsAffected, _ := result.RowsAffected()
		if rowsAffected == 0 {
			return fmt.Errorf("STM memory not found")
		}

	case "mtm":
		deleteQuery := `DELETE FROM neurondb_agent.memory_mtm WHERE id = $1 AND agent_id = $2`
		result, err := h.db.DB.ExecContext(ctx, deleteQuery, memoryID, agentID)
		if err != nil {
			return fmt.Errorf("MTM deletion failed: error=%w", err)
		}
		rowsAffected, _ := result.RowsAffected()
		if rowsAffected == 0 {
			return fmt.Errorf("MTM memory not found")
		}

	case "lpm":
		deleteQuery := `DELETE FROM neurondb_agent.memory_lpm WHERE id = $1 AND agent_id = $2`
		result, err := h.db.DB.ExecContext(ctx, deleteQuery, memoryID, agentID)
		if err != nil {
			return fmt.Errorf("LPM deletion failed: error=%w", err)
		}
		rowsAffected, _ := result.RowsAffected()
		if rowsAffected == 0 {
			return fmt.Errorf("LPM memory not found")
		}
	}

	return nil
}

/* StoreMTM stores content directly in mid-term memory */
func (h *HierarchicalMemoryManager) StoreMTM(ctx context.Context, agentID uuid.UUID, topic string, content string, importance float64) (uuid.UUID, error) {
	/* Compute embedding */
	embedding, err := h.embed.Embed(ctx, content, "all-MiniLM-L6-v2")
	if err != nil {
		return uuid.Nil, fmt.Errorf("MTM embedding failed: error=%w", err)
	}

	/* Store in MTM table */
	query := `INSERT INTO neurondb_agent.memory_mtm
		(agent_id, topic, content, embedding, importance_score)
		VALUES ($1, $2, $3, $4, $5)
		RETURNING id`

	var id uuid.UUID
	err = h.db.DB.GetContext(ctx, &id, query, agentID, topic, content, embedding, importance)
	if err != nil {
		return uuid.Nil, fmt.Errorf("MTM storage failed: error=%w", err)
	}

	return id, nil
}

/* StoreLPM stores content directly in long-term personal memory */
func (h *HierarchicalMemoryManager) StoreLPM(ctx context.Context, agentID uuid.UUID, category string, content string, importance float64, userID *uuid.UUID) (uuid.UUID, error) {
	/* Compute embedding */
	embedding, err := h.embed.Embed(ctx, content, "all-MiniLM-L6-v2")
	if err != nil {
		return uuid.Nil, fmt.Errorf("LPM embedding failed: error=%w", err)
	}

	/* Store in LPM table */
	query := `INSERT INTO neurondb_agent.memory_lpm
		(agent_id, user_id, category, content, embedding, importance_score, confidence)
		VALUES ($1, $2, $3, $4, $5, $6, 0.8)
		RETURNING id`

	var id uuid.UUID
	err = h.db.DB.GetContext(ctx, &id, query, agentID, userID, category, content, embedding, importance)
	if err != nil {
		return uuid.Nil, fmt.Errorf("LPM storage failed: error=%w", err)
	}

	return id, nil
}

/* PromoteMemory promotes memory from one tier to another (agent-controlled) */
func (h *HierarchicalMemoryManager) PromoteMemory(ctx context.Context, agentID uuid.UUID, memoryID uuid.UUID, fromTier string, toTier string, topic *string, category *string, userID *uuid.UUID, reason string) (uuid.UUID, error) {
	/* Validate tier transition */
	if fromTier == "stm" && toTier != "mtm" {
		return uuid.Nil, fmt.Errorf("invalid promotion: STM can only be promoted to MTM")
	}
	if fromTier == "mtm" && toTier != "lpm" {
		return uuid.Nil, fmt.Errorf("invalid promotion: MTM can only be promoted to LPM")
	}
	if fromTier == "lpm" {
		return uuid.Nil, fmt.Errorf("invalid promotion: LPM cannot be promoted further")
	}

	/* STM to MTM promotion */
	if fromTier == "stm" && toTier == "mtm" {
		if topic == nil || *topic == "" {
			return uuid.Nil, fmt.Errorf("topic required for STM to MTM promotion")
		}
		return h.PromoteToMTM(ctx, agentID, []uuid.UUID{memoryID}, *topic)
	}

	/* MTM to LPM promotion */
	if fromTier == "mtm" && toTier == "lpm" {
		if category == nil || *category == "" {
			return uuid.Nil, fmt.Errorf("category required for MTM to LPM promotion")
		}
		/* Update transition reason if provided */
		lpmID, err := h.PromoteToLPM(ctx, agentID, []uuid.UUID{memoryID}, *category, userID)
		if err != nil {
			return uuid.Nil, err
		}
		/* Update transition reason if custom reason provided */
		if reason != "" {
			updateQuery := `UPDATE neurondb_agent.memory_transitions
				SET reason = $1
				WHERE agent_id = $2 AND from_tier = 'mtm' AND to_tier = 'lpm' AND source_id = $3 AND target_id = $4`
			_, _ = h.db.DB.ExecContext(ctx, updateQuery, reason, agentID, memoryID, lpmID)
		}
		return lpmID, nil
	}

	return uuid.Nil, fmt.Errorf("unsupported promotion: %s to %s", fromTier, toTier)
}

/* DemoteMemory demotes memory from one tier to another (agent-controlled) */
func (h *HierarchicalMemoryManager) DemoteMemory(ctx context.Context, agentID uuid.UUID, memoryID uuid.UUID, fromTier string, toTier string, reason string) (uuid.UUID, error) {
	/* Validate tier transition */
	if fromTier == "lpm" && toTier != "mtm" {
		return uuid.Nil, fmt.Errorf("invalid demotion: LPM can only be demoted to MTM")
	}
	if fromTier == "mtm" && toTier != "stm" {
		return uuid.Nil, fmt.Errorf("invalid demotion: MTM can only be demoted to STM")
	}
	if fromTier == "stm" {
		return uuid.Nil, fmt.Errorf("invalid demotion: STM cannot be demoted further")
	}

	/* LPM to MTM demotion */
	if fromTier == "lpm" && toTier == "mtm" {
		/* Retrieve LPM entry */
		query := `SELECT content, importance_score, category FROM neurondb_agent.memory_lpm
			WHERE id = $1 AND agent_id = $2`

		type LPMRow struct {
			Content         string  `db:"content"`
			ImportanceScore float64 `db:"importance_score"`
			Category        string  `db:"category"`
		}

		var row LPMRow
		err := h.db.DB.GetContext(ctx, &row, query, memoryID, agentID)
		if err != nil {
			return uuid.Nil, fmt.Errorf("LPM retrieval for demotion failed: error=%w", err)
		}

		/* Compute embedding */
		embedding, err := h.embed.Embed(ctx, row.Content, "all-MiniLM-L6-v2")
		if err != nil {
			return uuid.Nil, fmt.Errorf("MTM embedding failed: error=%w", err)
		}

		/* Store in MTM table */
		insertQuery := `INSERT INTO neurondb_agent.memory_mtm
			(agent_id, topic, content, embedding, importance_score)
			VALUES ($1, $2, $3, $4, $5)
			RETURNING id`

		var mtmID uuid.UUID
		err = h.db.DB.GetContext(ctx, &mtmID, insertQuery, agentID, row.Category, row.Content, embedding, row.ImportanceScore)
		if err != nil {
			return uuid.Nil, fmt.Errorf("MTM storage failed: error=%w", err)
		}

		/* Record transition */
		transitionQuery := `INSERT INTO neurondb_agent.memory_transitions
			(agent_id, from_tier, to_tier, source_id, target_id, reason)
			VALUES ($1, 'lpm', 'mtm', $2, $3, $4)`
		transitionReason := reason
		if transitionReason == "" {
			transitionReason = "agent_demotion"
		}
		_, _ = h.db.DB.ExecContext(ctx, transitionQuery, agentID, memoryID, mtmID, transitionReason)

		return mtmID, nil
	}

	/* MTM to STM demotion */
	if fromTier == "mtm" && toTier == "stm" {
		/* Retrieve MTM entry */
		query := `SELECT content, importance_score FROM neurondb_agent.memory_mtm
			WHERE id = $1 AND agent_id = $2`

		type MTMRow struct {
			Content         string  `db:"content"`
			ImportanceScore float64 `db:"importance_score"`
		}

		var row MTMRow
		err := h.db.DB.GetContext(ctx, &row, query, memoryID, agentID)
		if err != nil {
			return uuid.Nil, fmt.Errorf("MTM retrieval for demotion failed: error=%w", err)
		}

		/* Compute embedding */
		embedding, err := h.embed.Embed(ctx, row.Content, "all-MiniLM-L6-v2")
		if err != nil {
			return uuid.Nil, fmt.Errorf("STM embedding failed: error=%w", err)
		}

		/* Store in STM table (session_id is NULL for demoted entries) */
		insertQuery := `INSERT INTO neurondb_agent.memory_stm
			(agent_id, session_id, content, embedding, importance_score)
			VALUES ($1, NULL, $2, $3, $4)
			RETURNING id`

		var stmID uuid.UUID
		err = h.db.DB.GetContext(ctx, &stmID, insertQuery, agentID, nil, row.Content, embedding, row.ImportanceScore)
		if err != nil {
			return uuid.Nil, fmt.Errorf("STM storage failed: error=%w", err)
		}

		/* Record transition */
		transitionQuery := `INSERT INTO neurondb_agent.memory_transitions
			(agent_id, from_tier, to_tier, source_id, target_id, reason)
			VALUES ($1, 'mtm', 'stm', $2, $3, $4)`
		transitionReason := reason
		if transitionReason == "" {
			transitionReason = "agent_demotion"
		}
		_, _ = h.db.DB.ExecContext(ctx, transitionQuery, agentID, memoryID, stmID, transitionReason)

		return stmID, nil
	}

	return uuid.Nil, fmt.Errorf("unsupported demotion: %s to %s", fromTier, toTier)
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
