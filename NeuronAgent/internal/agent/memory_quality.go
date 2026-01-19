/*-------------------------------------------------------------------------
 *
 * memory_quality.go
 *    Memory quality scoring with multi-dimensional metrics
 *
 * Provides enhanced quality metrics: completeness, accuracy, relevance,
 * freshness, and consistency.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/memory_quality.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"context"
	"fmt"
	"math"
	"time"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/metrics"
)

/* MemoryQualityScorer scores memory quality */
type MemoryQualityScorer struct {
	db      *db.DB
	queries *db.Queries
}

/* NewMemoryQualityScorer creates a new quality scorer */
func NewMemoryQualityScorer(db *db.DB, queries *db.Queries) *MemoryQualityScorer {
	return &MemoryQualityScorer{
		db:      db,
		queries: queries,
	}
}

/* MemoryQualityScore represents multi-dimensional quality metrics for memories */
type MemoryQualityScore struct {
	Completeness float64 /* How complete is the memory? */
	Accuracy     float64 /* Confidence in correctness */
	Relevance    float64 /* How often is it retrieved? */
	Freshness    float64 /* How recent is the information? */
	Consistency  float64 /* Does it conflict with other memories? */
	OverallScore float64 /* Weighted average of all metrics */
}

/* ScoreMemory calculates quality scores for a memory */
func (s *MemoryQualityScorer) ScoreMemory(ctx context.Context, agentID, memoryID uuid.UUID, tier string) (*MemoryQualityScore, error) {
	score := &MemoryQualityScore{}

	/* Calculate completeness */
	completeness, err := s.calculateCompleteness(ctx, memoryID, tier)
	if err != nil {
		return nil, err
	}
	score.Completeness = completeness

	/* Calculate accuracy */
	accuracy, err := s.calculateAccuracy(ctx, memoryID, tier)
	if err != nil {
		return nil, err
	}
	score.Accuracy = accuracy

	/* Calculate relevance */
	relevance, err := s.calculateRelevance(ctx, memoryID)
	if err != nil {
		return nil, err
	}
	score.Relevance = relevance

	/* Calculate freshness */
	freshness, err := s.calculateFreshness(ctx, memoryID, tier)
	if err != nil {
		return nil, err
	}
	score.Freshness = freshness

	/* Calculate consistency */
	consistency, err := s.calculateConsistency(ctx, agentID, memoryID, tier)
	if err != nil {
		return nil, err
	}
	score.Consistency = consistency

	/* Calculate overall score (weighted average) */
	score.OverallScore = (score.Completeness*0.15 +
		score.Accuracy*0.30 +
		score.Relevance*0.25 +
		score.Freshness*0.15 +
		score.Consistency*0.15)

	return score, nil
}

/* calculateCompleteness measures how complete the memory is */
func (s *MemoryQualityScorer) calculateCompleteness(ctx context.Context, memoryID uuid.UUID, tier string) (float64, error) {
	var tableName string
	switch tier {
	case "stm":
		tableName = "memory_stm"
	case "mtm":
		tableName = "memory_mtm"
	case "lpm":
		tableName = "memory_lpm"
	default:
		return 0, fmt.Errorf("invalid tier: %s", tier)
	}

	query := fmt.Sprintf(`SELECT 
		CASE WHEN content IS NULL OR content = '' THEN 0 ELSE 1 END as has_content,
		CASE WHEN embedding IS NULL OR array_length(embedding, 1) = 0 THEN 0 ELSE 1 END as has_embedding,
		CASE WHEN metadata IS NULL THEN 0 ELSE 1 END as has_metadata,
		LENGTH(content) as content_length
		FROM neurondb_agent.%s WHERE id = $1`, tableName)

	type CompletenessRow struct {
		HasContent   bool   `db:"has_content"`
		HasEmbedding bool   `db:"has_embedding"`
		HasMetadata  bool   `db:"has_metadata"`
		ContentLength int   `db:"content_length"`
	}

	var row CompletenessRow
	err := s.db.DB.GetContext(ctx, &row, query, memoryID)
	if err != nil {
		return 0, err
	}

	/* Base score from required fields */
	score := 0.0
	if row.HasContent {
		score += 0.5
	}
	if row.HasEmbedding {
		score += 0.3
	}
	if row.HasMetadata {
		score += 0.2
	}

	/* Bonus for content length (more complete) */
	if row.ContentLength > 100 {
		score += 0.1
	}
	if row.ContentLength > 500 {
		score += 0.1
	}

	return math.Min(1.0, score), nil
}

/* calculateAccuracy measures confidence in correctness */
func (s *MemoryQualityScorer) calculateAccuracy(ctx context.Context, memoryID uuid.UUID, tier string) (float64, error) {
	var tableName string
	switch tier {
	case "stm":
		tableName = "memory_stm"
	case "mtm":
		tableName = "memory_mtm"
	case "lpm":
		tableName = "memory_lpm"
	default:
		return 0, fmt.Errorf("invalid tier: %s", tier)
	}

	query := fmt.Sprintf(`SELECT 
		importance_score,
		metadata->>'confidence' as confidence,
		metadata->>'verified' as verified
		FROM neurondb_agent.%s WHERE id = $1`, tableName)

	type AccuracyRow struct {
		ImportanceScore float64  `db:"importance_score"`
		Confidence      *float64 `db:"confidence"`
		Verified        *bool    `db:"verified"`
	}

	var row AccuracyRow
	err := s.db.DB.GetContext(ctx, &row, query, memoryID)
	if err != nil {
		return 0, err
	}

	/* Start with importance score */
	score := row.ImportanceScore

	/* Adjust based on explicit confidence */
	if row.Confidence != nil {
		score = (score + *row.Confidence) / 2.0
	}

	/* Boost if verified */
	if row.Verified != nil && *row.Verified {
		score = math.Min(1.0, score+0.2)
	}

	/* Check if memory has been corrected (lower accuracy) */
	hasCorrections, _ := s.memoryHasCorrections(ctx, memoryID)
	if hasCorrections {
		score *= 0.8
	}

	return math.Min(1.0, math.Max(0.0, score)), nil
}

/* calculateRelevance measures how often memory is retrieved */
func (s *MemoryQualityScorer) calculateRelevance(ctx context.Context, memoryID uuid.UUID) (float64, error) {
	/* Get access count and recency with more detailed statistics */
	query := `SELECT 
		COUNT(*) as access_count,
		MAX(accessed_at) as last_access,
		MIN(accessed_at) as first_access,
		COUNT(DISTINCT session_id) as unique_sessions
		FROM neurondb_agent.memory_access_log
		WHERE memory_id = $1`

	type RelevanceRow struct {
		AccessCount    int        `db:"access_count"`
		LastAccess     *time.Time `db:"last_access"`
		FirstAccess    *time.Time `db:"first_access"`
		UniqueSessions int        `db:"unique_sessions"`
	}

	var row RelevanceRow
	err := s.db.DB.GetContext(ctx, &row, query, memoryID)
	if err != nil {
		/* If no access log, return low relevance */
		return 0.1, nil
	}

	/* Base score from access count (logarithmic scale with diminishing returns) */
	accessScore := math.Min(1.0, math.Log10(float64(row.AccessCount+1))/3.0)

	/* Recency score (exponential decay) */
	recencyScore := 0.0
	if row.LastAccess != nil {
		daysSinceAccess := time.Since(*row.LastAccess).Hours() / 24.0
		/* Exponential decay: half-life of 7 days */
		recencyScore = math.Exp(-daysSinceAccess / 7.0)
	}

	/* Diversity score (accessed from multiple sessions = more relevant) */
	diversityScore := 0.0
	if row.UniqueSessions > 1 {
		/* More unique sessions = higher diversity score */
		diversityScore = math.Min(0.3, float64(row.UniqueSessions-1)*0.1)
	}

	/* Frequency score (accesses per day since first access) */
	frequencyScore := 0.0
	if row.FirstAccess != nil && row.AccessCount > 0 {
		daysSinceFirst := time.Since(*row.FirstAccess).Hours() / 24.0
		if daysSinceFirst > 0 {
			accessesPerDay := float64(row.AccessCount) / daysSinceFirst
			/* Normalize: 1 access per day = 0.5 score, 5+ per day = 1.0 */
			frequencyScore = math.Min(1.0, accessesPerDay/5.0) * 0.2
		}
	}

	/* Weighted combination */
	score := accessScore*0.4 + recencyScore*0.3 + diversityScore*0.2 + frequencyScore*0.1

	return math.Min(1.0, math.Max(0.0, score)), nil
}

/* calculateFreshness measures how recent the information is */
func (s *MemoryQualityScorer) calculateFreshness(ctx context.Context, memoryID uuid.UUID, tier string) (float64, error) {
	var tableName string
	switch tier {
	case "stm":
		tableName = "memory_stm"
	case "mtm":
		tableName = "memory_mtm"
	case "lpm":
		tableName = "memory_lpm"
	default:
		return 0, fmt.Errorf("invalid tier: %s", tier)
	}

	query := fmt.Sprintf(`SELECT created_at, updated_at, last_accessed_at
		FROM neurondb_agent.%s WHERE id = $1`, tableName)
	
	/* Note: last_accessed_at may not exist in all memory tables, handle gracefully */

	type FreshnessRow struct {
		CreatedAt      time.Time  `db:"created_at"`
		UpdatedAt      *time.Time `db:"updated_at"`
		LastAccessedAt *time.Time `db:"last_accessed_at"`
	}

	var row FreshnessRow
	err := s.db.DB.GetContext(ctx, &row, query, memoryID)
	if err != nil {
		/* Try without last_accessed_at if column doesn't exist */
		queryFallback := fmt.Sprintf(`SELECT created_at, updated_at
			FROM neurondb_agent.%s WHERE id = $1`, tableName)
		type FreshnessRowFallback struct {
			CreatedAt time.Time  `db:"created_at"`
			UpdatedAt *time.Time `db:"updated_at"`
		}
		var rowFallback FreshnessRowFallback
		err = s.db.DB.GetContext(ctx, &rowFallback, queryFallback, memoryID)
		if err != nil {
			return 0, err
		}
		row.CreatedAt = rowFallback.CreatedAt
		row.UpdatedAt = rowFallback.UpdatedAt
		row.LastAccessedAt = nil
	}

	/* Use most recent timestamp: last_accessed > updated_at > created_at */
	referenceTime := row.CreatedAt
	if row.UpdatedAt != nil && row.UpdatedAt.After(referenceTime) {
		referenceTime = *row.UpdatedAt
	}
	if row.LastAccessedAt != nil && row.LastAccessedAt.After(referenceTime) {
		referenceTime = *row.LastAccessedAt
	}

	/* Calculate age in days */
	ageDays := time.Since(referenceTime).Hours() / 24.0

	/* Tier-specific half-lives for exponential decay */
	halfLifeDays := 30.0
	switch tier {
	case "stm":
		halfLifeDays = 0.5 /* STM: half-life of 12 hours */
	case "mtm":
		halfLifeDays = 7.0 /* MTM: half-life of 7 days */
	case "lpm":
		halfLifeDays = 90.0 /* LPM: half-life of 90 days (freshness less critical) */
	}

	/* Score decreases with age (exponential decay) */
	score := math.Exp(-ageDays / halfLifeDays)

	/* Boost for recently updated memories (indicates active maintenance) */
	if row.UpdatedAt != nil && row.CreatedAt.Before(*row.UpdatedAt) {
		updateAgeDays := time.Since(*row.UpdatedAt).Hours() / 24.0
		if updateAgeDays < 7 {
			/* Recently updated = more fresh */
			score = math.Min(1.0, score+0.2)
		}
	}

	/* Boost for recently accessed memories */
	if row.LastAccessedAt != nil {
		accessAgeDays := time.Since(*row.LastAccessedAt).Hours() / 24.0
		if accessAgeDays < 1 {
			score = math.Min(1.0, score+0.1)
		}
	}

	return math.Min(1.0, math.Max(0.0, score)), nil
}

/* calculateConsistency measures if memory conflicts with others */
func (s *MemoryQualityScorer) calculateConsistency(ctx context.Context, agentID, memoryID uuid.UUID, tier string) (float64, error) {
	/* Check for conflicts */
	query := `SELECT COUNT(*) > 0 as has_conflict
		FROM neurondb_agent.memory_conflicts
		WHERE $1 = ANY(memory_ids) AND tier = $2 AND resolved = false`

	var hasConflict bool
	err := s.db.DB.GetContext(ctx, &hasConflict, query, memoryID, tier)
	if err != nil {
		return 0.5, nil /* Default to neutral if error */
	}

	if hasConflict {
		return 0.3, nil /* Low consistency if has unresolved conflicts */
	}

	/* Check if memory has been corrected */
	hasCorrections, _ := s.memoryHasCorrections(ctx, memoryID)
	if hasCorrections {
		return 0.6, nil /* Medium consistency if corrected */
	}

	return 1.0, nil /* High consistency if no conflicts */
}

/* UpdateMemoryQuality updates quality scores for a memory */
func (s *MemoryQualityScorer) UpdateMemoryQuality(ctx context.Context, agentID, memoryID uuid.UUID, tier string) error {
	score, err := s.ScoreMemory(ctx, agentID, memoryID, tier)
	if err != nil {
		return err
	}

	/* Store quality scores in metadata or separate table */
	var tableName string
	switch tier {
	case "stm":
		tableName = "memory_stm"
	case "mtm":
		tableName = "memory_mtm"
	case "lpm":
		tableName = "memory_lpm"
	default:
		return fmt.Errorf("invalid tier: %s", tier)
	}

	/* Update metadata with quality scores */
	query := fmt.Sprintf(`UPDATE neurondb_agent.%s
		SET metadata = COALESCE(metadata, '{}'::jsonb) || jsonb_build_object(
			'quality_completeness', $1,
			'quality_accuracy', $2,
			'quality_relevance', $3,
			'quality_freshness', $4,
			'quality_consistency', $5,
			'quality_overall', $6,
			'quality_updated_at', NOW()
		),
		updated_at = NOW()
		WHERE id = $7`, tableName)

	_, err = s.db.DB.ExecContext(ctx, query,
		score.Completeness,
		score.Accuracy,
		score.Relevance,
		score.Freshness,
		score.Consistency,
		score.OverallScore,
		memoryID)

	if err != nil {
		metrics.WarnWithContext(ctx, "Failed to update memory quality", map[string]interface{}{
			"memory_id": memoryID.String(),
			"tier":      tier,
			"error":     err.Error(),
		})
		return err
	}

	return nil
}

/* UpdateAllMemoryQuality updates quality scores for all memories of an agent */
func (s *MemoryQualityScorer) UpdateAllMemoryQuality(ctx context.Context, agentID uuid.UUID) (int, error) {
	updated := 0

	/* Update each tier */
	for _, tier := range []string{"stm", "mtm", "lpm"} {
		memoryIDs, err := s.getMemoryIDsForTier(ctx, agentID, tier)
		if err != nil {
			continue
		}

		for _, memoryID := range memoryIDs {
			if err := s.UpdateMemoryQuality(ctx, agentID, memoryID, tier); err == nil {
				updated++
			}
		}
	}

	return updated, nil
}

/* Helper methods */

func (s *MemoryQualityScorer) memoryHasCorrections(ctx context.Context, memoryID uuid.UUID) (bool, error) {
	query := `SELECT COUNT(*) > 0
		FROM neurondb_agent.memory_corruption_log
		WHERE memory_id = $1 AND action = 'repaired'`

	var hasCorrections bool
	err := s.db.DB.GetContext(ctx, &hasCorrections, query, memoryID)
	return hasCorrections, err
}

func (s *MemoryQualityScorer) getMemoryIDsForTier(ctx context.Context, agentID uuid.UUID, tier string) ([]uuid.UUID, error) {
	var tableName string
	switch tier {
	case "stm":
		tableName = "memory_stm"
	case "mtm":
		tableName = "memory_mtm"
	case "lpm":
		tableName = "memory_lpm"
	default:
		return nil, fmt.Errorf("invalid tier: %s", tier)
	}

	query := fmt.Sprintf(`SELECT id FROM neurondb_agent.%s WHERE agent_id = $1`, tableName)
	var memoryIDs []uuid.UUID
	err := s.db.DB.SelectContext(ctx, &memoryIDs, query, agentID)
	return memoryIDs, err
}
