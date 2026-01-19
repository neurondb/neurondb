/*-------------------------------------------------------------------------
 *
 * memory_learning.go
 *    Memory learning from user feedback
 *
 * Tracks user feedback on memory retrievals and updates memory quality
 * scores and importance based on feedback.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/memory_learning.go
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

/* MemoryLearningManager manages learning from user feedback */
type MemoryLearningManager struct {
	db      *db.DB
	queries *db.Queries
}

/* NewMemoryLearningManager creates a new memory learning manager */
func NewMemoryLearningManager(db *db.DB, queries *db.Queries) *MemoryLearningManager {
	return &MemoryLearningManager{
		db:      db,
		queries: queries,
	}
}

/* MemoryFeedback represents user feedback on a memory */
type MemoryFeedback struct {
	ID            uuid.UUID
	AgentID       uuid.UUID
	SessionID     *uuid.UUID
	MemoryID      uuid.UUID
	MemoryTier    string
	FeedbackType  string /* positive, negative, neutral, correction */
	FeedbackText  string
	Query         string
	RelevanceScore *float64
	Metadata      map[string]interface{}
	CreatedAt     time.Time
}

/* RecordFeedback records user feedback on a memory */
func (m *MemoryLearningManager) RecordFeedback(ctx context.Context, feedback *MemoryFeedback) (uuid.UUID, error) {
	feedbackID := uuid.New()
	if feedback.ID != uuid.Nil {
		feedbackID = feedback.ID
	}

	query := `INSERT INTO neurondb_agent.memory_feedback
		(id, agent_id, session_id, memory_id, memory_tier, feedback_type, feedback_text, query, relevance_score, metadata, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, NOW())
		RETURNING id`

	err := m.db.DB.GetContext(ctx, &feedbackID, query,
		feedbackID,
		feedback.AgentID,
		feedback.SessionID,
		feedback.MemoryID,
		feedback.MemoryTier,
		feedback.FeedbackType,
		feedback.FeedbackText,
		feedback.Query,
		feedback.RelevanceScore,
		feedback.Metadata,
	)

	if err != nil {
		return uuid.Nil, fmt.Errorf("failed to record memory feedback: %w", err)
	}

	/* Update memory quality metrics */
	if err := m.updateQualityMetrics(ctx, feedback); err != nil {
		/* Log error but don't fail feedback recording */
		metrics.WarnWithContext(ctx, "Failed to update quality metrics", map[string]interface{}{
			"feedback_id": feedbackID.String(),
			"memory_id":   feedback.MemoryID.String(),
			"error":       err.Error(),
		})
	}

	/* Update memory importance based on feedback */
	if err := m.updateMemoryImportance(ctx, feedback); err != nil {
		/* Log error but don't fail feedback recording */
		metrics.WarnWithContext(ctx, "Failed to update memory importance", map[string]interface{}{
			"feedback_id": feedbackID.String(),
			"memory_id":   feedback.MemoryID.String(),
			"error":       err.Error(),
		})
	}

	/* Record metrics */
	metrics.InfoWithContext(ctx, "Memory feedback recorded", map[string]interface{}{
		"feedback_id":   feedbackID.String(),
		"agent_id":      feedback.AgentID.String(),
		"memory_id":     feedback.MemoryID.String(),
		"feedback_type": feedback.FeedbackType,
	})

	return feedbackID, nil
}

/* updateQualityMetrics updates quality metrics for a memory */
func (m *MemoryLearningManager) updateQualityMetrics(ctx context.Context, feedback *MemoryFeedback) error {
	/* Get or create quality metrics record */
	var metricsID uuid.UUID
	var retrievalCount, positiveCount, negativeCount int
	var avgRelevance, qualityScore float64

	query := `SELECT id, retrieval_count, positive_feedback_count, negative_feedback_count, average_relevance_score, quality_score
		FROM neurondb_agent.memory_quality_metrics
		WHERE agent_id = $1 AND memory_id = $2 AND memory_tier = $3`

	err := m.db.DB.GetContext(ctx, &struct {
		ID              uuid.UUID `db:"id"`
		RetrievalCount  int       `db:"retrieval_count"`
		PositiveCount   int       `db:"positive_feedback_count"`
		NegativeCount   int       `db:"negative_feedback_count"`
		AvgRelevance    *float64  `db:"average_relevance_score"`
		QualityScore    *float64  `db:"quality_score"`
	}{
		ID:             metricsID,
		RetrievalCount: retrievalCount,
		PositiveCount:  positiveCount,
		NegativeCount:  negativeCount,
		AvgRelevance:   &avgRelevance,
		QualityScore:   &qualityScore,
	}, query, feedback.AgentID, feedback.MemoryID, feedback.MemoryTier)

	/* Update counts based on feedback */
	if feedback.FeedbackType == "positive" {
		positiveCount++
	} else if feedback.FeedbackType == "negative" {
		negativeCount++
	}

	/* Update average relevance score */
	if feedback.RelevanceScore != nil {
		if avgRelevance == 0 {
			avgRelevance = *feedback.RelevanceScore
		} else {
			/* Weighted average */
			totalFeedback := float64(positiveCount + negativeCount)
			avgRelevance = (avgRelevance*totalFeedback + *feedback.RelevanceScore) / (totalFeedback + 1)
		}
	}

	/* Calculate quality score */
	qualityScore = m.calculateQualityScore(positiveCount, negativeCount, avgRelevance)

	if err != nil {
		/* Create new metrics record */
		metricsID = uuid.New()
		insertQuery := `INSERT INTO neurondb_agent.memory_quality_metrics
			(id, agent_id, memory_id, memory_tier, retrieval_count, positive_feedback_count, negative_feedback_count, average_relevance_score, quality_score, updated_at)
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())`

		_, err = m.db.DB.ExecContext(ctx, insertQuery,
			metricsID,
			feedback.AgentID,
			feedback.MemoryID,
			feedback.MemoryTier,
			retrievalCount,
			positiveCount,
			negativeCount,
			avgRelevance,
			qualityScore,
		)
	} else {
		/* Update existing metrics record */
		updateQuery := `UPDATE neurondb_agent.memory_quality_metrics
			SET positive_feedback_count = $1,
			    negative_feedback_count = $2,
			    average_relevance_score = $3,
			    quality_score = $4,
			    updated_at = NOW()
			WHERE id = $5`

		_, err = m.db.DB.ExecContext(ctx, updateQuery,
			positiveCount,
			negativeCount,
			avgRelevance,
			qualityScore,
			metricsID,
		)
	}

	return err
}

/* updateMemoryImportance updates memory importance based on feedback */
func (m *MemoryLearningManager) updateMemoryImportance(ctx context.Context, feedback *MemoryFeedback) error {
	/* Determine importance adjustment based on feedback */
	importanceAdjustment := 0.0
	switch feedback.FeedbackType {
	case "positive":
		importanceAdjustment = 0.1
	case "negative":
		importanceAdjustment = -0.1
	case "correction":
		/* Corrections might increase or decrease importance depending on context */
		importanceAdjustment = 0.05
	}

	if importanceAdjustment == 0 {
		return nil /* No adjustment needed */
	}

	/* Update importance in the appropriate memory table */
	var tableName string
	switch feedback.MemoryTier {
	case "chunk":
		tableName = "memory_chunks"
	case "stm":
		tableName = "memory_stm"
	case "mtm":
		tableName = "memory_mtm"
	case "lpm":
		tableName = "memory_lpm"
	default:
		return fmt.Errorf("invalid memory tier: %s", feedback.MemoryTier)
	}

	/* Update importance score, clamped to [0, 1] */
	updateQuery := fmt.Sprintf(`UPDATE neurondb_agent.%s
		SET importance_score = GREATEST(0.0, LEAST(1.0, importance_score + $1)),
		    updated_at = NOW()
		WHERE id = $2 AND agent_id = $3`, tableName)

	_, err := m.db.DB.ExecContext(ctx, updateQuery, importanceAdjustment, feedback.MemoryID, feedback.AgentID)
	return err
}

/* calculateQualityScore calculates quality score from feedback */
func (m *MemoryLearningManager) calculateQualityScore(positiveCount, negativeCount int, avgRelevance float64) float64 {
	totalFeedback := positiveCount + negativeCount
	if totalFeedback == 0 {
		return 0.5 /* Default neutral score */
	}

	/* Base score from feedback ratio */
	feedbackScore := float64(positiveCount) / float64(totalFeedback)

	/* Combine with relevance score */
	qualityScore := (feedbackScore*0.7 + avgRelevance*0.3)

	/* Boost for high feedback count (more reliable) */
	if totalFeedback > 5 {
		qualityScore = qualityScore * 1.1
		if qualityScore > 1.0 {
			qualityScore = 1.0
		}
	}

	return qualityScore
}

/* GetMemoryQuality returns quality metrics for a memory */
func (m *MemoryLearningManager) GetMemoryQuality(ctx context.Context, agentID uuid.UUID, memoryID uuid.UUID, tier string) (map[string]interface{}, error) {
	query := `SELECT retrieval_count, positive_feedback_count, negative_feedback_count, average_relevance_score, quality_score, last_retrieved_at
		FROM neurondb_agent.memory_quality_metrics
		WHERE agent_id = $1 AND memory_id = $2 AND memory_tier = $3`

	type QualityRow struct {
		RetrievalCount  int        `db:"retrieval_count"`
		PositiveCount   int        `db:"positive_feedback_count"`
		NegativeCount   int        `db:"negative_feedback_count"`
		AvgRelevance    *float64   `db:"average_relevance_score"`
		QualityScore    *float64   `db:"quality_score"`
		LastRetrieved   *time.Time `db:"last_retrieved_at"`
	}

	var row QualityRow
	err := m.db.DB.GetContext(ctx, &row, query, agentID, memoryID, tier)
	if err != nil {
		/* Return default values if no metrics found */
		return map[string]interface{}{
			"retrieval_count":        0,
			"positive_feedback_count": 0,
			"negative_feedback_count": 0,
			"average_relevance_score": 0.0,
			"quality_score":          0.5,
			"last_retrieved_at":      nil,
		}, nil
	}

	result := map[string]interface{}{
		"retrieval_count":        row.RetrievalCount,
		"positive_feedback_count": row.PositiveCount,
		"negative_feedback_count": row.NegativeCount,
		"last_retrieved_at":       row.LastRetrieved,
	}

	if row.AvgRelevance != nil {
		result["average_relevance_score"] = *row.AvgRelevance
	} else {
		result["average_relevance_score"] = 0.0
	}

	if row.QualityScore != nil {
		result["quality_score"] = *row.QualityScore
	} else {
		result["quality_score"] = 0.5
	}

	return result, nil
}

/* RecordRetrieval records a memory retrieval for quality tracking */
func (m *MemoryLearningManager) RecordRetrieval(ctx context.Context, agentID uuid.UUID, memoryID uuid.UUID, tier string) error {
	/* Update or create quality metrics with retrieval count */
	query := `INSERT INTO neurondb_agent.memory_quality_metrics
		(agent_id, memory_id, memory_tier, retrieval_count, last_retrieved_at, updated_at)
		VALUES ($1, $2, $3, 1, NOW(), NOW())
		ON CONFLICT (agent_id, memory_id, memory_tier) DO UPDATE
		SET retrieval_count = memory_quality_metrics.retrieval_count + 1,
		    last_retrieved_at = NOW(),
		    updated_at = NOW()`

	_, err := m.db.DB.ExecContext(ctx, query, agentID, memoryID, tier)
	return err
}
