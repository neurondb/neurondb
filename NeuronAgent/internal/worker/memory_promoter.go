/*-------------------------------------------------------------------------
 *
 * memory_promoter.go
 *    Background worker for memory promotion between tiers
 *
 * Automatically promotes memories from STM to MTM and MTM to LPM based on
 * patterns, access frequency, and importance scoring.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/worker/memory_promoter.go
 *
 *-------------------------------------------------------------------------
 */

package worker

import (
	"context"
	"time"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/agent"
	"github.com/neurondb/NeuronAgent/internal/db"
)

/* MemoryPromoter handles automatic memory promotion */
type MemoryPromoter struct {
	hierMemory *agent.HierarchicalMemoryManager
	queries    *db.Queries
	interval   time.Duration
}

/* NewMemoryPromoter creates a new memory promoter worker */
func NewMemoryPromoter(hierMemory *agent.HierarchicalMemoryManager, queries *db.Queries, interval time.Duration) *MemoryPromoter {
	return &MemoryPromoter{
		hierMemory: hierMemory,
		queries:    queries,
		interval:   interval,
	}
}

/* Start starts the memory promoter worker */
func (m *MemoryPromoter) Start(ctx context.Context) error {
	ticker := time.NewTicker(m.interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			/* Run promotion cycle */
			err := m.runPromotionCycle(ctx)
			if err != nil {
				/* Log error but continue - silently handle */
				_ = err
			}

			/* Run cleanup */
			_, err = m.hierMemory.CleanupExpired(ctx)
			if err != nil {
				/* Silently handle cleanup errors */
				_ = err
			}
		}
	}
}

/* runPromotionCycle runs a single promotion cycle */
func (m *MemoryPromoter) runPromotionCycle(ctx context.Context) error {
	/* Promote STM to MTM */
	err := m.promoteSTMToMTM(ctx)
	if err != nil {
		return err
	}

	/* Promote MTM to LPM */
	err = m.promoteMTMToLPM(ctx)
	if err != nil {
		return err
	}

	return nil
}

/* promoteSTMToMTM promotes high-value STM entries to MTM */
func (m *MemoryPromoter) promoteSTMToMTM(ctx context.Context) error {
	/* Find STM entries ready for promotion */
	query := `SELECT agent_id, id, content, importance_score, access_count
		FROM neurondb_agent.memory_stm
		WHERE importance_score > 0.6
		AND access_count > 2
		AND created_at < NOW() - INTERVAL '10 minutes'
		AND expires_at > NOW()
		ORDER BY importance_score DESC, access_count DESC
		LIMIT 100`

	type STMCandidate struct {
		AgentID         uuid.UUID `db:"agent_id"`
		ID              uuid.UUID `db:"id"`
		Content         string    `db:"content"`
		ImportanceScore float64   `db:"importance_score"`
		AccessCount     int       `db:"access_count"`
	}

	var candidates []STMCandidate
	err := m.queries.GetDB().SelectContext(ctx, &candidates, query)
	if err != nil {
		return err
	}

	/* Group by agent and topic */
	agentGroups := make(map[uuid.UUID][]uuid.UUID)
	for _, candidate := range candidates {
		agentGroups[candidate.AgentID] = append(agentGroups[candidate.AgentID], candidate.ID)
	}

	/* Promote each group */
	for agentID, stmIDs := range agentGroups {
		if len(stmIDs) == 0 {
			continue
		}

		/* Extract topic from content (simplified) */
		topic := "general"

		_, err := m.hierMemory.PromoteToMTM(ctx, agentID, stmIDs, topic)
		if err != nil {
			/* Log error but continue with other agents - silently handle */
			_ = err
			continue
		}
	}

	return nil
}

/* promoteMTMToLPM promotes high-confidence MTM entries to LPM */
func (m *MemoryPromoter) promoteMTMToLPM(ctx context.Context) error {
	/* Find MTM entries ready for promotion */
	query := `SELECT agent_id, id, topic, importance_score, pattern_count
		FROM neurondb_agent.memory_mtm
		WHERE importance_score > 0.75
		AND pattern_count >= 3
		AND last_reinforced_at < NOW() - INTERVAL '1 day'
		AND expires_at > NOW()
		ORDER BY importance_score DESC, pattern_count DESC
		LIMIT 50`

	type MTMCandidate struct {
		AgentID         uuid.UUID `db:"agent_id"`
		ID              uuid.UUID `db:"id"`
		Topic           string    `db:"topic"`
		ImportanceScore float64   `db:"importance_score"`
		PatternCount    int       `db:"pattern_count"`
	}

	var candidates []MTMCandidate
	err := m.queries.GetDB().SelectContext(ctx, &candidates, query)
	if err != nil {
		return err
	}

	/* Group by agent */
	agentGroups := make(map[uuid.UUID][]uuid.UUID)
	agentCategories := make(map[uuid.UUID]string)
	for _, candidate := range candidates {
		agentGroups[candidate.AgentID] = append(agentGroups[candidate.AgentID], candidate.ID)
		agentCategories[candidate.AgentID] = candidate.Topic
	}

	/* Promote each group */
	for agentID, mtmIDs := range agentGroups {
		if len(mtmIDs) == 0 {
			continue
		}

		category := agentCategories[agentID]
		_, err := m.hierMemory.PromoteToLPM(ctx, agentID, mtmIDs, category, nil)
		if err != nil {
			/* Log error but continue with other agents - silently handle */
			_ = err
			continue
		}
	}

	return nil
}
