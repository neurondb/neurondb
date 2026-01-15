/*-------------------------------------------------------------------------
 *
 * memory_consolidation.go
 *    Memory consolidation and forgetting system
 *
 * Provides automatic memory promotion (Working → Episodic → Semantic)
 * and intelligent forgetting strategies.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/memory_consolidation.go
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
	"github.com/neurondb/NeuronAgent/internal/metrics"
)

/* MemoryConsolidator manages memory consolidation */
type MemoryConsolidator struct {
	queries          *db.Queries
	episodicMemory   *EpisodicMemoryManager
	semanticMemory   *SemanticMemoryManager
	llmClient        *LLMClient
	consolidationThreshold float64
	forgettingThreshold    float64
}

/* NewMemoryConsolidator creates a new memory consolidator */
func NewMemoryConsolidator(queries *db.Queries, episodicMemory *EpisodicMemoryManager, semanticMemory *SemanticMemoryManager, llmClient *LLMClient) *MemoryConsolidator {
	return &MemoryConsolidator{
		queries:                queries,
		episodicMemory:         episodicMemory,
		semanticMemory:         semanticMemory,
		llmClient:              llmClient,
		consolidationThreshold: 0.7, // 70% importance threshold
		forgettingThreshold:    0.1, // 10% importance threshold
	}
}

/* Consolidate consolidates working memory to episodic memory */
func (mc *MemoryConsolidator) Consolidate(ctx context.Context, agentID uuid.UUID) error {
	/* Get working memory chunks that meet consolidation threshold */
	query := `SELECT id, agent_id, session_id, content, importance, created_at, metadata
		FROM neurondb_agent.memory_chunks
		WHERE agent_id = $1
		  AND memory_type = 'working'
		  AND importance >= $2
		  AND created_at < NOW() - INTERVAL '1 hour'
		ORDER BY importance DESC, created_at DESC
		LIMIT 100`

	type WorkingMemoryRow struct {
		ID         uuid.UUID              `db:"id"`
		AgentID    uuid.UUID              `db:"agent_id"`
		SessionID  uuid.UUID              `db:"session_id"`
		Content    string                 `db:"content"`
		Importance float64                `db:"importance"`
		CreatedAt  time.Time              `db:"created_at"`
		Metadata   map[string]interface{} `db:"metadata"`
	}

	var rows []WorkingMemoryRow
	err := mc.queries.DB.SelectContext(ctx, &rows, query, agentID, mc.consolidationThreshold)
	if err != nil {
		return fmt.Errorf("memory consolidation failed: query_error=true, error=%w", err)
	}

	consolidated := 0
	for _, row := range rows {
		/* Extract event and context from content */
		event := row.Content
		context := ""
		if ctxVal, ok := row.Metadata["context"].(string); ok {
			context = ctxVal
		}

		/* Calculate emotional valence from metadata */
		emotionalValence := 0.0
		if valence, ok := row.Metadata["emotional_valence"].(float64); ok {
			emotionalValence = valence
		}

		/* Store as episodic memory */
		_, err := mc.episodicMemory.Store(ctx, row.AgentID, row.SessionID, event, context, row.Importance, emotionalValence, row.Metadata)
		if err != nil {
			metrics.WarnWithContext(ctx, "Failed to consolidate memory to episodic", map[string]interface{}{
				"memory_id": row.ID.String(),
				"error":     err.Error(),
			})
			continue
		}

		/* Update memory type to episodic */
		updateQuery := `UPDATE neurondb_agent.memory_chunks
			SET memory_type = 'episodic', updated_at = NOW()
			WHERE id = $1`
		_, err = mc.queries.DB.ExecContext(ctx, updateQuery, row.ID)
		if err != nil {
			metrics.WarnWithContext(ctx, "Failed to update memory type", map[string]interface{}{
				"memory_id": row.ID.String(),
				"error":     err.Error(),
			})
		}

		consolidated++
	}

	metrics.InfoWithContext(ctx, "Memory consolidation completed", map[string]interface{}{
		"agent_id":    agentID.String(),
		"consolidated": consolidated,
	})

	return nil
}

/* PromoteToSemantic promotes episodic memories to semantic memories */
func (mc *MemoryConsolidator) PromoteToSemantic(ctx context.Context, agentID uuid.UUID) error {
	/* Get episodic memories that are highly important and frequently accessed */
	query := `SELECT id, agent_id, event, context, importance, metadata
		FROM neurondb_agent.episodic_memory
		WHERE agent_id = $1
		  AND importance >= $2
		  AND access_count >= 5
		ORDER BY importance DESC, access_count DESC
		LIMIT 50`

	type EpisodicRow struct {
		ID         uuid.UUID              `db:"id"`
		AgentID    uuid.UUID              `db:"agent_id"`
		Event      string                 `db:"event"`
		Context    string                 `db:"context"`
		Importance float64                `db:"importance"`
		Metadata   map[string]interface{} `db:"metadata"`
	}

	var rows []EpisodicRow
	err := mc.queries.DB.SelectContext(ctx, &rows, query, agentID, 0.8) // 80% importance threshold
	if err != nil {
		return fmt.Errorf("semantic promotion failed: query_error=true, error=%w", err)
	}

	promoted := 0
	for _, row := range rows {
		/* Extract concept and fact from event */
		concept := row.Event
		fact := row.Context
		if fact == "" {
			fact = row.Event
		}

		/* Extract category from metadata */
		category := "general"
		if cat, ok := row.Metadata["category"].(string); ok {
			category = cat
		}

		/* Extract relations from metadata */
		relations := []string{}
		if rels, ok := row.Metadata["relations"].([]string); ok {
			relations = rels
		}

		/* Store as semantic memory */
		_, err := mc.semanticMemory.Store(ctx, row.AgentID, concept, fact, category, row.Importance, relations, row.Metadata)
		if err != nil {
			metrics.WarnWithContext(ctx, "Failed to promote memory to semantic", map[string]interface{}{
				"memory_id": row.ID.String(),
				"error":     err.Error(),
			})
			continue
		}

		promoted++
	}

	metrics.InfoWithContext(ctx, "Semantic promotion completed", map[string]interface{}{
		"agent_id": agentID.String(),
		"promoted": promoted,
	})

	return nil
}

/* Forget forgets memories below the forgetting threshold */
func (mc *MemoryConsolidator) Forget(ctx context.Context, agentID uuid.UUID) error {
	/* Delete working memories below threshold and older than 24 hours */
	query := `DELETE FROM neurondb_agent.memory_chunks
		WHERE agent_id = $1
		  AND memory_type = 'working'
		  AND importance < $2
		  AND created_at < NOW() - INTERVAL '24 hours'`

	result, err := mc.queries.DB.ExecContext(ctx, query, agentID, mc.forgettingThreshold)
	if err != nil {
		return fmt.Errorf("memory forgetting failed: database_error=true, error=%w", err)
	}

	deleted, _ := result.RowsAffected()

	/* Delete episodic memories below threshold and older than 30 days */
	episodicQuery := `DELETE FROM neurondb_agent.episodic_memory
		WHERE agent_id = $1
		  AND importance < $2
		  AND timestamp < NOW() - INTERVAL '30 days'`

	result, err = mc.queries.DB.ExecContext(ctx, episodicQuery, agentID, mc.forgettingThreshold)
	if err != nil {
		return fmt.Errorf("episodic memory forgetting failed: database_error=true, error=%w", err)
	}

	episodicDeleted, _ := result.RowsAffected()

	metrics.InfoWithContext(ctx, "Memory forgetting completed", map[string]interface{}{
		"agent_id":        agentID.String(),
		"working_deleted": deleted,
		"episodic_deleted": episodicDeleted,
	})

	return nil
}

/* CompressMemory compresses old memories by summarization */
func (mc *MemoryConsolidator) CompressMemory(ctx context.Context, agentID uuid.UUID) error {
	/* Get old episodic memories that haven't been accessed recently */
	query := `SELECT id, event, context, importance
		FROM neurondb_agent.episodic_memory
		WHERE agent_id = $1
		  AND timestamp < NOW() - INTERVAL '90 days'
		  AND last_accessed < NOW() - INTERVAL '30 days'
		ORDER BY importance ASC
		LIMIT 100`

	type MemoryRow struct {
		ID         uuid.UUID `db:"id"`
		Event      string    `db:"event"`
		Context    string    `db:"context"`
		Importance float64   `db:"importance"`
	}

	var rows []MemoryRow
	err := mc.queries.DB.SelectContext(ctx, &rows, query, agentID)
	if err != nil {
		return fmt.Errorf("memory compression failed: query_error=true, error=%w", err)
	}

	if len(rows) == 0 {
		return nil
	}

	/* Group memories into batches for summarization (max 10 memories per batch) */
	batchSize := 10
	compressed := 0

	for i := 0; i < len(rows); i += batchSize {
		end := i + batchSize
		if end > len(rows) {
			end = len(rows)
		}
		batch := rows[i:end]

		/* Build summary prompt from batch of memories */
		memoryTexts := []string{}
		memoryIDs := []uuid.UUID{}
		for _, row := range batch {
			memoryText := row.Event
			if row.Context != "" {
				memoryText += " (Context: " + row.Context + ")"
			}
			memoryTexts = append(memoryTexts, memoryText)
			memoryIDs = append(memoryIDs, row.ID)
		}

		/* Use LLM to summarize the batch */
		if mc.llmClient != nil {
			prompt := fmt.Sprintf(`Summarize the following memories into a concise summary that preserves key information:

%s

Provide a single summary that captures the essential information from all memories.`, 
				fmt.Sprintf("%d. %s", 1, memoryTexts[0]))
			
			/* Build numbered list for better summarization */
			for idx, text := range memoryTexts[1:] {
				prompt += fmt.Sprintf("\n%d. %s", idx+2, text)
			}

			llmConfig := map[string]interface{}{
				"temperature": 0.3, /* Lower temperature for more focused summarization */
				"max_tokens":  500,
			}

			/* Get default model from agent config or use default */
			modelName := "default"
			agent, err := mc.queries.GetAgentByID(ctx, agentID)
			if err == nil && agent.ModelName != "" {
				modelName = agent.ModelName
			}

			llmResponse, err := mc.llmClient.Generate(ctx, modelName, prompt, llmConfig)
			if err != nil {
				metrics.WarnWithContext(ctx, "LLM summarization failed, marking as compressed", map[string]interface{}{
					"agent_id":    agentID.String(),
					"batch_size":  len(batch),
					"error":       err.Error(),
				})
				/* Fall back to marking as compressed without summarization */
				for _, row := range batch {
					updateQuery := `UPDATE neurondb_agent.episodic_memory
						SET metadata = jsonb_set(COALESCE(metadata, '{}'::jsonb), '{compressed}', 'true'::jsonb),
						    updated_at = NOW()
						WHERE id = $1`
					mc.queries.DB.ExecContext(ctx, updateQuery, row.ID)
					compressed++
				}
				continue
			}

			/* Store summary in the first memory's context and mark all as compressed */
			summary := llmResponse.Content
			for idx, memoryID := range memoryIDs {
				updateQuery := `UPDATE neurondb_agent.episodic_memory
					SET metadata = jsonb_set(
						jsonb_set(COALESCE(metadata, '{}'::jsonb), '{compressed}', 'true'::jsonb),
						'{summary}',
						$2::jsonb
					),
					updated_at = NOW()`
				
				/* Store summary in first memory, reference in others */
				if idx == 0 {
					mc.queries.DB.ExecContext(ctx, updateQuery, memoryID, fmt.Sprintf(`"%s"`, summary))
				} else {
					/* Reference the first memory's summary */
					mc.queries.DB.ExecContext(ctx, updateQuery, memoryID, fmt.Sprintf(`"See memory %s"`, memoryIDs[0].String()[:8]))
				}
				compressed++
			}
		} else {
			/* No LLM client available, just mark as compressed */
			for _, row := range batch {
				updateQuery := `UPDATE neurondb_agent.episodic_memory
					SET metadata = jsonb_set(COALESCE(metadata, '{}'::jsonb), '{compressed}', 'true'::jsonb),
					    updated_at = NOW()
					WHERE id = $1`
				mc.queries.DB.ExecContext(ctx, updateQuery, row.ID)
				compressed++
			}
		}
	}

	metrics.InfoWithContext(ctx, "Memory compression completed", map[string]interface{}{
		"agent_id":   agentID.String(),
		"compressed": compressed,
	})

	return nil
}

