/*-------------------------------------------------------------------------
 *
 * memory_forgetting.go
 *    Intelligent memory forgetting mechanisms
 *
 * Implements various forgetting strategies: time-based, importance-based,
 * relevance-based, and conflict-based forgetting.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/memory_forgetting.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"context"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/metrics"
	"github.com/neurondb/NeuronAgent/pkg/neurondb"
)

/* MemoryForgettingManager manages intelligent forgetting */
type MemoryForgettingManager struct {
	db      *db.DB
	queries *db.Queries
	embed   *neurondb.EmbeddingClient
}

/* NewMemoryForgettingManager creates a new forgetting manager */
func NewMemoryForgettingManager(db *db.DB, queries *db.Queries, embed *neurondb.EmbeddingClient) *MemoryForgettingManager {
	return &MemoryForgettingManager{
		db:      db,
		queries: queries,
		embed:   embed,
	}
}

/* ForgettingStrategy defines forgetting strategy */
type ForgettingStrategy string

const (
	ForgettingStrategyTime      ForgettingStrategy = "time"      /* Time-based */
	ForgettingStrategyImportance ForgettingStrategy = "importance" /* Importance-based */
	ForgettingStrategyRelevance  ForgettingStrategy = "relevance"  /* Relevance-based */
	ForgettingStrategyConflict   ForgettingStrategy = "conflict"   /* Conflict-based */
	ForgettingStrategyHybrid     ForgettingStrategy = "hybrid"     /* Multiple strategies */
)

/* ForgettingConfig configures forgetting behavior */
type ForgettingConfig struct {
	Strategy              ForgettingStrategy
	TimeThreshold         time.Duration /* For time-based */
	ImportanceThreshold   float64       /* For importance-based */
	RelevanceThreshold    float64       /* For relevance-based */
	MaxMemorySize         int           /* Maximum memories before forgetting */
	ArchiveBeforeDelete   bool          /* Archive before deleting */
	MinRetentionPeriod    time.Duration /* Minimum time to keep memories */
}

/* DefaultForgettingConfig returns default forgetting configuration */
func DefaultForgettingConfig() *ForgettingConfig {
	return &ForgettingConfig{
		Strategy:            ForgettingStrategyHybrid,
		TimeThreshold:       30 * 24 * time.Hour, /* 30 days */
		ImportanceThreshold: 0.3,
		RelevanceThreshold:  0.2,
		MaxMemorySize:       10000,
		ArchiveBeforeDelete: true,
		MinRetentionPeriod:  7 * 24 * time.Hour, /* 7 days */
	}
}

/* ShouldForget determines if a memory should be forgotten */
func (m *MemoryForgettingManager) ShouldForget(ctx context.Context, agentID uuid.UUID, memoryID uuid.UUID, tier string, config *ForgettingConfig) (bool, string, error) {
	if config == nil {
		config = DefaultForgettingConfig()
	}

	/* Check minimum retention period */
	createdAt, err := m.getMemoryCreatedAt(ctx, memoryID, tier)
	if err != nil {
		return false, "", err
	}

	if time.Since(createdAt) < config.MinRetentionPeriod {
		return false, "within_min_retention_period", nil
	}

	/* Apply strategy-specific checks */
	switch config.Strategy {
	case ForgettingStrategyTime:
		return m.shouldForgetTimeBased(ctx, memoryID, tier, config)
	case ForgettingStrategyImportance:
		return m.shouldForgetImportanceBased(ctx, memoryID, tier, config)
	case ForgettingStrategyRelevance:
		return m.shouldForgetRelevanceBased(ctx, agentID, memoryID, tier, config)
	case ForgettingStrategyConflict:
		return m.shouldForgetConflictBased(ctx, memoryID, tier, config)
	case ForgettingStrategyHybrid:
		return m.shouldForgetHybrid(ctx, agentID, memoryID, tier, config)
	default:
		return false, "unknown_strategy", nil
	}
}

/* shouldForgetTimeBased checks if memory should be forgotten based on age */
func (m *MemoryForgettingManager) shouldForgetTimeBased(ctx context.Context, memoryID uuid.UUID, tier string, config *ForgettingConfig) (bool, string, error) {
	createdAt, err := m.getMemoryCreatedAt(ctx, memoryID, tier)
	if err != nil {
		return false, "", err
	}

	age := time.Since(createdAt)
	if age > config.TimeThreshold {
		return true, fmt.Sprintf("age_exceeded_threshold: %v", age), nil
	}

	return false, "within_time_threshold", nil
}

/* shouldForgetImportanceBased checks if memory should be forgotten based on importance */
func (m *MemoryForgettingManager) shouldForgetImportanceBased(ctx context.Context, memoryID uuid.UUID, tier string, config *ForgettingConfig) (bool, string, error) {
	importance, err := m.getMemoryImportance(ctx, memoryID, tier)
	if err != nil {
		return false, "", err
	}

	if importance < config.ImportanceThreshold {
		return true, fmt.Sprintf("importance_below_threshold: %f", importance), nil
	}

	return false, "importance_above_threshold", nil
}

/* shouldForgetRelevanceBased checks if memory should be forgotten based on retrieval frequency */
func (m *MemoryForgettingManager) shouldForgetRelevanceBased(ctx context.Context, agentID uuid.UUID, memoryID uuid.UUID, tier string, config *ForgettingConfig) (bool, string, error) {
	/* Check access log for retrieval frequency */
	accessCount, lastAccess, err := m.getMemoryAccessInfo(ctx, memoryID)
	if err != nil {
		/* If no access log, assume never accessed */
		accessCount = 0
		lastAccess = time.Time{}
	}

	/* If never accessed and old enough, forget */
	if accessCount == 0 {
		createdAt, err := m.getMemoryCreatedAt(ctx, memoryID, tier)
		if err != nil {
			return false, "", err
		}
		if time.Since(createdAt) > config.TimeThreshold {
			return true, "never_accessed_and_old", nil
		}
	}

	/* If last access was long ago, consider forgetting */
	if !lastAccess.IsZero() && time.Since(lastAccess) > config.TimeThreshold*2 {
		return true, "not_accessed_recently", nil
	}

	return false, "recently_accessed", nil
}

/* shouldForgetConflictBased checks if memory should be forgotten due to conflicts */
func (m *MemoryForgettingManager) shouldForgetConflictBased(ctx context.Context, memoryID uuid.UUID, tier string, config *ForgettingConfig) (bool, string, error) {
	/* Check if memory is marked as conflicting/outdated */
	hasConflict, err := m.memoryHasConflict(ctx, memoryID, tier)
	if err != nil {
		return false, "", err
	}

	if hasConflict {
		/* Check if this is the older/less important version */
		isOutdated, err := m.isOutdatedMemory(ctx, memoryID, tier)
		if err != nil {
			return false, "", err
		}
		if isOutdated {
			return true, "conflicting_and_outdated", nil
		}
	}

	return false, "no_conflict", nil
}

/* shouldForgetHybrid applies multiple strategies with weighted scoring */
func (m *MemoryForgettingManager) shouldForgetHybrid(ctx context.Context, agentID uuid.UUID, memoryID uuid.UUID, tier string, config *ForgettingConfig) (bool, string, error) {
	/* Calculate forget score (0.0 = keep, 1.0 = forget) */
	forgetScore := 0.0
	reasons := make([]string, 0)

	/* Time-based score (weight: 0.3) */
	createdAt, err := m.getMemoryCreatedAt(ctx, memoryID, tier)
	if err == nil {
		age := time.Since(createdAt)
		if age > config.TimeThreshold {
			/* Score increases with age beyond threshold */
			timeScore := math.Min(1.0, float64(age-config.TimeThreshold)/float64(config.TimeThreshold))
			forgetScore += timeScore * 0.3
			reasons = append(reasons, fmt.Sprintf("age_exceeded: %v", age))
		}
	}

	/* Importance-based score (weight: 0.4) */
	importance, err := m.getMemoryImportance(ctx, memoryID, tier)
	if err == nil {
		if importance < config.ImportanceThreshold {
			/* Lower importance = higher forget score */
			importanceScore := (config.ImportanceThreshold - importance) / config.ImportanceThreshold
			forgetScore += importanceScore * 0.4
			reasons = append(reasons, fmt.Sprintf("low_importance: %.2f", importance))
		}
	}

	/* Relevance-based score (weight: 0.3) */
	accessCount, lastAccess, err := m.getMemoryAccessInfo(ctx, memoryID)
	if err == nil {
		if accessCount == 0 {
			/* Never accessed - higher forget score */
			createdAt, _ := m.getMemoryCreatedAt(ctx, memoryID, tier)
			if time.Since(createdAt) > config.TimeThreshold {
				forgetScore += 0.3
				reasons = append(reasons, "never_accessed")
			}
		} else if !lastAccess.IsZero() && time.Since(lastAccess) > config.TimeThreshold*2 {
			/* Not accessed recently - moderate forget score */
			relevanceScore := math.Min(0.3, float64(time.Since(lastAccess))/float64(config.TimeThreshold*4))
			forgetScore += relevanceScore
			reasons = append(reasons, fmt.Sprintf("not_accessed_recently: %v", time.Since(lastAccess)))
		}
	}

	/* Tier-specific adjustments */
	switch tier {
	case "stm":
		/* STM should be more aggressive in forgetting */
		forgetScore *= 1.2
	case "lpm":
		/* LPM should be more conservative */
		forgetScore *= 0.7
	}

	/* Decision threshold: forget if score > 0.5 */
	if forgetScore > 0.5 {
		reason := "hybrid_score: " + strings.Join(reasons, ", ")
		return true, reason, nil
	}

	return false, "no_forgetting_criteria_met", nil
}

/* ForgetMemories executes forgetting for memories that should be forgotten */
func (m *MemoryForgettingManager) ForgetMemories(ctx context.Context, agentID uuid.UUID, tier string, config *ForgettingConfig) (int, error) {
	if config == nil {
		config = DefaultForgettingConfig()
	}

	/* Validate tier */
	if tier != "stm" && tier != "mtm" && tier != "lpm" {
		return 0, fmt.Errorf("invalid tier: %s", tier)
	}

	/* Get all memories for this agent and tier */
	memories, err := m.getMemoriesForTier(ctx, agentID, tier)
	if err != nil {
		return 0, fmt.Errorf("failed to get memories: %w", err)
	}

	/* Limit processing for performance (process in batches if needed) */
	maxMemoriesToProcess := 10000
	if len(memories) > maxMemoriesToProcess {
		metrics.WarnWithContext(ctx, "Large memory set, processing in batches", map[string]interface{}{
			"agent_id": agentID.String(),
			"tier":     tier,
			"total":    len(memories),
			"limit":    maxMemoriesToProcess,
		})
		memories = memories[:maxMemoriesToProcess]
	}

	forgotten := 0
	failed := 0
	archived := 0

	/* Process memories with batching for large sets */
	batchSize := 100
	for i := 0; i < len(memories); i += batchSize {
		/* Check context cancellation */
		if ctx.Err() != nil {
			return forgotten, ctx.Err()
		}

		end := i + batchSize
		if end > len(memories) {
			end = len(memories)
		}

		batch := memories[i:end]
		for _, memoryID := range batch {
			shouldForget, reason, err := m.ShouldForget(ctx, agentID, memoryID, tier, config)
			if err != nil {
				failed++
				metrics.WarnWithContext(ctx, "Failed to check if memory should be forgotten", map[string]interface{}{
					"memory_id": memoryID.String(),
					"tier":      tier,
					"error":     err.Error(),
				})
				continue
			}

			if shouldForget {
				/* Archive before deleting if configured */
				if config.ArchiveBeforeDelete {
					if err := m.ArchiveBeforeForgetting(ctx, memoryID, tier, reason); err != nil {
						metrics.WarnWithContext(ctx, "Failed to archive memory before forgetting", map[string]interface{}{
							"memory_id": memoryID.String(),
							"tier":      tier,
							"error":     err.Error(),
						})
						/* Continue with deletion even if archiving fails */
					} else {
						archived++
					}
				}

				/* Delete memory */
				if err := m.deleteMemory(ctx, memoryID, tier); err != nil {
					failed++
					metrics.WarnWithContext(ctx, "Failed to delete memory", map[string]interface{}{
						"memory_id": memoryID.String(),
						"tier":      tier,
						"error":     err.Error(),
					})
					continue
				}

				/* Log forgetting event */
				m.logForgettingEvent(ctx, agentID, memoryID, tier, reason)

				forgotten++
			}
		}
	}

	if failed > 0 {
		metrics.WarnWithContext(ctx, "Some memories failed to forget", map[string]interface{}{
			"agent_id": agentID.String(),
			"tier":     tier,
			"forgotten": forgotten,
			"failed":   failed,
			"archived": archived,
		})
	}

	return forgotten, nil
}

/* ArchiveBeforeForgetting moves memory to archive before deletion */
func (m *MemoryForgettingManager) ArchiveBeforeForgetting(ctx context.Context, memoryID uuid.UUID, tier string, reason string) error {
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

	/* Copy to archive table */
	archiveQuery := fmt.Sprintf(`INSERT INTO neurondb_agent.memory_archive
		(memory_id, tier, content, embedding, metadata, importance_score, forgotten_at, forget_reason)
		SELECT id, '%s', content, embedding, metadata, importance_score, NOW(), $1
		FROM neurondb_agent.%s
		WHERE id = $2`, tier, tableName)

	_, err := m.db.DB.ExecContext(ctx, archiveQuery, reason, memoryID)
	return err
}

/* Helper methods */

func (m *MemoryForgettingManager) getMemoryCreatedAt(ctx context.Context, memoryID uuid.UUID, tier string) (time.Time, error) {
	var tableName string
	switch tier {
	case "stm":
		tableName = "memory_stm"
	case "mtm":
		tableName = "memory_mtm"
	case "lpm":
		tableName = "memory_lpm"
	default:
		return time.Time{}, fmt.Errorf("invalid tier: %s", tier)
	}

	query := fmt.Sprintf(`SELECT created_at FROM neurondb_agent.%s WHERE id = $1`, tableName)
	var createdAt time.Time
	err := m.db.DB.GetContext(ctx, &createdAt, query, memoryID)
	return createdAt, err
}

func (m *MemoryForgettingManager) getMemoryImportance(ctx context.Context, memoryID uuid.UUID, tier string) (float64, error) {
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

	query := fmt.Sprintf(`SELECT importance_score FROM neurondb_agent.%s WHERE id = $1`, tableName)
	var importance float64
	err := m.db.DB.GetContext(ctx, &importance, query, memoryID)
	return importance, err
}

func (m *MemoryForgettingManager) getMemoryAccessInfo(ctx context.Context, memoryID uuid.UUID) (int, time.Time, error) {
	query := `SELECT COUNT(*), MAX(accessed_at) 
		FROM neurondb_agent.memory_access_log 
		WHERE memory_id = $1`
	
	var count int
	var lastAccess time.Time
	err := m.db.DB.GetContext(ctx, &count, query, memoryID)
	if err != nil {
		return 0, time.Time{}, err
	}
	
	/* Get last access separately */
	_ = m.db.DB.GetContext(ctx, &lastAccess, `SELECT MAX(accessed_at) FROM neurondb_agent.memory_access_log WHERE memory_id = $1`, memoryID)
	
	return count, lastAccess, nil
}

func (m *MemoryForgettingManager) memoryHasConflict(ctx context.Context, memoryID uuid.UUID, tier string) (bool, error) {
	query := `SELECT COUNT(*) > 0 
		FROM neurondb_agent.memory_conflicts 
		WHERE memory_id = $1 AND tier = $2`
	
	var hasConflict bool
	err := m.db.DB.GetContext(ctx, &hasConflict, query, memoryID, tier)
	return hasConflict, err
}

func (m *MemoryForgettingManager) isOutdatedMemory(ctx context.Context, memoryID uuid.UUID, tier string) (bool, error) {
	/* Check if there's a newer conflicting memory */
	query := `SELECT COUNT(*) > 0
		FROM neurondb_agent.memory_conflicts mc1
		JOIN neurondb_agent.memory_conflicts mc2 ON mc1.conflict_group = mc2.conflict_group
		WHERE mc1.memory_id = $1 AND mc1.tier = $2
		AND mc2.created_at > mc1.created_at`
	
	var isOutdated bool
	err := m.db.DB.GetContext(ctx, &isOutdated, query, memoryID, tier)
	return isOutdated, err
}

func (m *MemoryForgettingManager) getMemoriesForTier(ctx context.Context, agentID uuid.UUID, tier string) ([]uuid.UUID, error) {
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
	err := m.db.DB.SelectContext(ctx, &memoryIDs, query, agentID)
	return memoryIDs, err
}

func (m *MemoryForgettingManager) deleteMemory(ctx context.Context, memoryID uuid.UUID, tier string) error {
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

	query := fmt.Sprintf(`DELETE FROM neurondb_agent.%s WHERE id = $1`, tableName)
	_, err := m.db.DB.ExecContext(ctx, query, memoryID)
	return err
}

func (m *MemoryForgettingManager) logForgettingEvent(ctx context.Context, agentID, memoryID uuid.UUID, tier, reason string) {
	query := `INSERT INTO neurondb_agent.memory_forgetting_log
		(agent_id, memory_id, tier, reason, forgotten_at)
		VALUES ($1, $2, $3, $4, NOW())
		ON CONFLICT DO NOTHING`

	_, _ = m.db.DB.ExecContext(ctx, query, agentID, memoryID, tier, reason)
}
