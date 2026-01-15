/*-------------------------------------------------------------------------
 *
 * memory_processor.go
 *    Background job processor for memory consolidation
 *
 * Provides background processing for memory consolidation, promotion,
 * and forgetting.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/jobs/memory_processor.go
 *
 *-------------------------------------------------------------------------
 */

package jobs

import (
	"context"
	"time"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/agent"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/metrics"
)

/* MemoryProcessor processes memory consolidation jobs */
type MemoryProcessor struct {
	queries          *db.Queries
	consolidator     *agent.MemoryConsolidator
	processingInterval time.Duration
	stopChan         chan struct{}
}

/* NewMemoryProcessor creates a new memory processor */
func NewMemoryProcessor(queries *db.Queries, consolidator *agent.MemoryConsolidator) *MemoryProcessor {
	return &MemoryProcessor{
		queries:            queries,
		consolidator:       consolidator,
		processingInterval: 1 * time.Hour,
		stopChan:           make(chan struct{}),
	}
}

/* Start starts the memory processor */
func (mp *MemoryProcessor) Start(ctx context.Context) {
	ticker := time.NewTicker(mp.processingInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-mp.stopChan:
			return
		case <-ticker.C:
			mp.processAllAgents(ctx)
		}
	}
}

/* Stop stops the memory processor */
func (mp *MemoryProcessor) Stop() {
	close(mp.stopChan)
}

/* processAllAgents processes memory for all agents */
func (mp *MemoryProcessor) processAllAgents(ctx context.Context) {
	/* Get all active agents */
	query := `SELECT id FROM neurondb_agent.agents WHERE id IS NOT NULL`

	type AgentRow struct {
		ID uuid.UUID `db:"id"`
	}

	var agents []AgentRow
	err := mp.queries.DB.SelectContext(ctx, &agents, query)
	if err != nil {
		metrics.WarnWithContext(ctx, "Failed to get agents for memory processing", map[string]interface{}{
			"error": err.Error(),
		})
		return
	}

	for _, agentRow := range agents {
		/* Consolidate working memory to episodic */
		if err := mp.consolidator.Consolidate(ctx, agentRow.ID); err != nil {
			metrics.WarnWithContext(ctx, "Memory consolidation failed", map[string]interface{}{
				"agent_id": agentRow.ID.String(),
				"error":    err.Error(),
			})
		}

		/* Promote episodic to semantic */
		if err := mp.consolidator.PromoteToSemantic(ctx, agentRow.ID); err != nil {
			metrics.WarnWithContext(ctx, "Semantic promotion failed", map[string]interface{}{
				"agent_id": agentRow.ID.String(),
				"error":    err.Error(),
			})
		}

		/* Forget low-importance memories */
		if err := mp.consolidator.Forget(ctx, agentRow.ID); err != nil {
			metrics.WarnWithContext(ctx, "Memory forgetting failed", map[string]interface{}{
				"agent_id": agentRow.ID.String(),
				"error":    err.Error(),
			})
		}

		/* Compress old memories */
		if err := mp.consolidator.CompressMemory(ctx, agentRow.ID); err != nil {
			metrics.WarnWithContext(ctx, "Memory compression failed", map[string]interface{}{
				"agent_id": agentRow.ID.String(),
				"error":    err.Error(),
			})
		}
	}

	metrics.InfoWithContext(ctx, "Memory processing completed", map[string]interface{}{
		"agents_processed": len(agents),
	})
}

