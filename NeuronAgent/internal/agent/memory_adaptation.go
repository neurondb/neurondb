/*-------------------------------------------------------------------------
 *
 * memory_adaptation.go
 *    Adaptive memory strategies
 *
 * Monitors memory usage patterns and adapts strategies for importance
 * scoring, forgetting, consolidation, and compression.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/memory_adaptation.go
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
)

/* MemoryAdaptationManager manages adaptive memory strategies */
type MemoryAdaptationManager struct {
	db      *db.DB
	queries *db.Queries
}

/* NewMemoryAdaptationManager creates a new memory adaptation manager */
func NewMemoryAdaptationManager(db *db.DB, queries *db.Queries) *MemoryAdaptationManager {
	return &MemoryAdaptationManager{
		db:      db,
		queries: queries,
	}
}

/* MemoryUsagePattern represents usage pattern for a memory */
type MemoryUsagePattern struct {
	MemoryID        uuid.UUID
	Tier            string
	RetrievalCount  int
	LastRetrieved   time.Time
	AverageInterval time.Duration /* Average time between retrievals */
	Trend           string        /* "increasing", "decreasing", "stable" */
}

/* AnalyzeUsagePatterns analyzes memory usage patterns */
func (m *MemoryAdaptationManager) AnalyzeUsagePatterns(ctx context.Context, agentID uuid.UUID, tier string, days int) ([]MemoryUsagePattern, error) {
	if days <= 0 {
		days = 30
	}

	var tableName string
	switch tier {
	case "chunk":
		tableName = "memory_chunks"
	case "stm":
		tableName = "memory_stm"
	case "mtm":
		tableName = "memory_mtm"
	case "lpm":
		tableName = "memory_lpm"
	default:
		return nil, fmt.Errorf("invalid tier: %s", tier)
	}

	/* Query memory access patterns */
	query := fmt.Sprintf(`SELECT 
		m.id as memory_id,
		COUNT(ma.id) as retrieval_count,
		MAX(ma.accessed_at) as last_retrieved,
		AVG(EXTRACT(EPOCH FROM (ma.accessed_at - LAG(ma.accessed_at) OVER (PARTITION BY ma.memory_id ORDER BY ma.accessed_at)))) as avg_interval_seconds
	FROM neurondb_agent.%s m
	LEFT JOIN neurondb_agent.memory_access_log ma ON ma.memory_id = m.id
	WHERE m.agent_id = $1
	AND (ma.accessed_at IS NULL OR ma.accessed_at > NOW() - INTERVAL '1 day' * $2)
	GROUP BY m.id
	ORDER BY retrieval_count DESC`, tableName)

	type UsageRow struct {
		MemoryID           uuid.UUID  `db:"memory_id"`
		RetrievalCount     int        `db:"retrieval_count"`
		LastRetrieved      *time.Time `db:"last_retrieved"`
		AvgIntervalSeconds *float64   `db:"avg_interval_seconds"`
	}

	var rows []UsageRow
	err := m.db.DB.SelectContext(ctx, &rows, query, agentID, days)
	if err != nil {
		return nil, fmt.Errorf("failed to analyze usage patterns: %w", err)
	}

	patterns := make([]MemoryUsagePattern, len(rows))
	for i, row := range rows {
		pattern := MemoryUsagePattern{
			MemoryID:       row.MemoryID,
			Tier:           tier,
			RetrievalCount: row.RetrievalCount,
		}

		if row.LastRetrieved != nil {
			pattern.LastRetrieved = *row.LastRetrieved
		}

		if row.AvgIntervalSeconds != nil {
			pattern.AverageInterval = time.Duration(*row.AvgIntervalSeconds) * time.Second
		}

		/* Determine trend based on recent vs older access patterns */
		pattern.Trend = m.determineTrend(ctx, row.MemoryID, tier)

		patterns[i] = pattern
	}

	return patterns, nil
}

/* determineTrend determines usage trend for a memory */
func (m *MemoryAdaptationManager) determineTrend(ctx context.Context, memoryID uuid.UUID, tier string) string {
	/* Compare recent (last 7 days) vs older (7-30 days) access counts */
	recentQuery := `SELECT COUNT(*) FROM neurondb_agent.memory_access_log
		WHERE memory_id = $1 AND accessed_at > NOW() - INTERVAL '7 days'`

	olderQuery := `SELECT COUNT(*) FROM neurondb_agent.memory_access_log
		WHERE memory_id = $1 AND accessed_at > NOW() - INTERVAL '30 days' AND accessed_at <= NOW() - INTERVAL '7 days'`

	var recentCount, olderCount int
	_ = m.db.DB.GetContext(ctx, &recentCount, recentQuery, memoryID)
	_ = m.db.DB.GetContext(ctx, &olderCount, olderQuery, memoryID)

	if recentCount > int(float64(olderCount)*1.2) {
		return "increasing"
	} else if recentCount < int(float64(olderCount)*0.8) {
		return "decreasing"
	}
	return "stable"
}

/* AdjustImportanceBasedOnUsage adjusts memory importance based on usage patterns */
func (m *MemoryAdaptationManager) AdjustImportanceBasedOnUsage(ctx context.Context, agentID uuid.UUID, patterns []MemoryUsagePattern) error {
	for _, pattern := range patterns {
		/* Calculate new importance based on usage */
		newImportance := m.calculateUsageBasedImportance(pattern)

		/* Update importance in appropriate table */
		var tableName string
		switch pattern.Tier {
		case "chunk":
			tableName = "memory_chunks"
		case "stm":
			tableName = "memory_stm"
		case "mtm":
			tableName = "memory_mtm"
		case "lpm":
			tableName = "memory_lpm"
		default:
			continue
		}

		/* Get current importance */
		currentQuery := fmt.Sprintf(`SELECT importance_score FROM neurondb_agent.%s WHERE id = $1 AND agent_id = $2`, tableName)
		var currentImportance float64
		err := m.db.DB.GetContext(ctx, &currentImportance, currentQuery, pattern.MemoryID, agentID)
		if err != nil {
			continue /* Skip if memory not found */
		}

		/* Only update if change is significant (> 0.05) */
		if math.Abs(newImportance-currentImportance) < 0.05 {
			continue
		}

		/* Update importance (weighted average: 70% current, 30% usage-based) */
		adjustedImportance := currentImportance*0.7 + newImportance*0.3
		adjustedImportance = math.Max(0.0, math.Min(1.0, adjustedImportance))

		updateQuery := fmt.Sprintf(`UPDATE neurondb_agent.%s
			SET importance_score = $1, updated_at = NOW()
			WHERE id = $2 AND agent_id = $3`, tableName)

		_, err = m.db.DB.ExecContext(ctx, updateQuery, adjustedImportance, pattern.MemoryID, agentID)
		if err != nil {
			metrics.WarnWithContext(ctx, "Failed to adjust memory importance", map[string]interface{}{
				"memory_id": pattern.MemoryID.String(),
				"tier":      pattern.Tier,
				"error":     err.Error(),
			})
			continue
		}
	}

	return nil
}

/* calculateUsageBasedImportance calculates importance based on usage pattern */
func (m *MemoryAdaptationManager) calculateUsageBasedImportance(pattern MemoryUsagePattern) float64 {
	baseImportance := 0.5

	/* Boost for high retrieval count */
	if pattern.RetrievalCount > 10 {
		baseImportance += 0.2
	} else if pattern.RetrievalCount > 5 {
		baseImportance += 0.1
	} else if pattern.RetrievalCount > 0 {
		baseImportance += 0.05
	}

	/* Boost for recent access */
	if !pattern.LastRetrieved.IsZero() {
		daysSinceAccess := time.Since(pattern.LastRetrieved).Hours() / 24.0
		if daysSinceAccess < 1 {
			baseImportance += 0.15
		} else if daysSinceAccess < 7 {
			baseImportance += 0.1
		} else if daysSinceAccess < 30 {
			baseImportance += 0.05
		}
	}

	/* Adjust for trend */
	switch pattern.Trend {
	case "increasing":
		baseImportance += 0.1
	case "decreasing":
		baseImportance -= 0.1
	case "stable":
		/* No adjustment for stable trend */
	}

	/* Clamp to [0, 1] */
	return math.Max(0.0, math.Min(1.0, baseImportance))
}

/* ConsolidateSimilarMemories consolidates similar memories */
func (m *MemoryAdaptationManager) ConsolidateSimilarMemories(ctx context.Context, agentID uuid.UUID, tier string, similarityThreshold float64) (int, error) {
	if similarityThreshold < 0.7 {
		similarityThreshold = 0.9 /* Default to high threshold for safety */
	}

	var tableName string
	switch tier {
	case "stm":
		tableName = "memory_stm"
	case "mtm":
		tableName = "memory_mtm"
	case "lpm":
		tableName = "memory_lpm"
	default:
		return 0, fmt.Errorf("invalid tier for consolidation: %s", tier)
	}

	/* Get all memories with embeddings */
	query := fmt.Sprintf(`SELECT id, content, embedding, importance_score, created_at
		FROM neurondb_agent.%s
		WHERE agent_id = $1 AND embedding IS NOT NULL
		ORDER BY created_at DESC
		LIMIT 1000`, tableName)

	type MemoryRow struct {
		ID              uuid.UUID `db:"id"`
		Content         string    `db:"content"`
		Embedding       []float32 `db:"embedding"`
		ImportanceScore float64   `db:"importance_score"`
		CreatedAt       time.Time `db:"created_at"`
	}

	var memories []MemoryRow
	err := m.db.DB.SelectContext(ctx, &memories, query, agentID)
	if err != nil {
		return 0, fmt.Errorf("failed to get memories for consolidation: %w", err)
	}

	consolidated := 0
	processed := make(map[uuid.UUID]bool)

	/* Find and merge similar memories */
	for i, mem1 := range memories {
		if processed[mem1.ID] {
			continue
		}

		similarMemories := []MemoryRow{mem1}
		bestImportance := mem1.ImportanceScore
		bestContent := mem1.Content
		bestID := mem1.ID

		/* Find similar memories */
		for j := i + 1; j < len(memories); j++ {
			mem2 := memories[j]
			if processed[mem2.ID] {
				continue
			}

			/* Calculate similarity */
			similarity := cosineSimilarityAdaptation(mem1.Embedding, mem2.Embedding)
			if similarity >= similarityThreshold {
				similarMemories = append(similarMemories, mem2)
				processed[mem2.ID] = true

				/* Keep memory with higher importance or newer content */
				if mem2.ImportanceScore > bestImportance || (mem2.ImportanceScore == bestImportance && mem2.CreatedAt.After(mem1.CreatedAt)) {
					bestImportance = mem2.ImportanceScore
					bestID = mem2.ID
					bestContent = mem2.Content
				} else if len(mem2.Content) > len(bestContent) {
					/* If same importance, prefer longer content */
					bestContent = mem2.Content
				}
			}
		}

		/* If similar memories found, consolidate */
		if len(similarMemories) > 1 {
			/* Update best memory with merged content if needed */
			if bestID != mem1.ID {
				/* Content is already best, just update importance if needed */
				updateQuery := fmt.Sprintf(`UPDATE neurondb_agent.%s
					SET importance_score = GREATEST(importance_score, $1), updated_at = NOW()
					WHERE id = $2`, tableName)
				_, _ = m.db.DB.ExecContext(ctx, updateQuery, bestImportance, bestID)
			}

			/* Delete other similar memories */
			for _, similar := range similarMemories {
				if similar.ID != bestID {
					deleteQuery := fmt.Sprintf(`DELETE FROM neurondb_agent.%s WHERE id = $1`, tableName)
					_, err := m.db.DB.ExecContext(ctx, deleteQuery, similar.ID)
					if err == nil {
						consolidated++
					}
				}
			}
		}
	}

	return consolidated, nil
}

/* CompressRarelyAccessedMemories compresses memories that are rarely accessed */
func (m *MemoryAdaptationManager) CompressRarelyAccessedMemories(ctx context.Context, agentID uuid.UUID, tier string, daysSinceAccess int, compressionRatio float64) (int, error) {
	if daysSinceAccess <= 0 {
		daysSinceAccess = 90
	}
	if compressionRatio <= 0 || compressionRatio >= 1 {
		compressionRatio = 0.5 /* Compress to 50% of original size */
	}

	var tableName string
	switch tier {
	case "stm":
		tableName = "memory_stm"
	case "mtm":
		tableName = "memory_mtm"
	case "lpm":
		tableName = "memory_lpm"
	default:
		return 0, fmt.Errorf("invalid tier for compression: %s", tier)
	}

	/* Find rarely accessed memories */
	query := fmt.Sprintf(`SELECT m.id, m.content
		FROM neurondb_agent.%s m
		LEFT JOIN neurondb_agent.memory_access_log ma ON ma.memory_id = m.id
		WHERE m.agent_id = $1
		AND (ma.accessed_at IS NULL OR ma.accessed_at < NOW() - INTERVAL '1 day' * $2)
		AND LENGTH(m.content) > 500
		LIMIT 100`, tableName)

	type MemoryRow struct {
		ID      uuid.UUID `db:"id"`
		Content string    `db:"content"`
	}

	var memories []MemoryRow
	err := m.db.DB.SelectContext(ctx, &memories, query, agentID, daysSinceAccess)
	if err != nil {
		return 0, fmt.Errorf("failed to get memories for compression: %w", err)
	}

	compressed := 0
	for _, mem := range memories {
		/* Simple compression: truncate to target length */
		targetLength := int(float64(len(mem.Content)) * compressionRatio)
		if targetLength < len(mem.Content) {
			/* Truncate at sentence boundary if possible */
			compressedContent := mem.Content[:targetLength]
			lastPeriod := strings.LastIndex(compressedContent, ".")
			if lastPeriod > int(float64(targetLength)*0.8) {
				compressedContent = compressedContent[:lastPeriod+1]
			} else {
				compressedContent = compressedContent + "..."
			}

			/* Update content and re-embed */
			updateQuery := fmt.Sprintf(`UPDATE neurondb_agent.%s
				SET content = $1, updated_at = NOW()
				WHERE id = $2`, tableName)
			_, err := m.db.DB.ExecContext(ctx, updateQuery, compressedContent, mem.ID)
			if err == nil {
				compressed++
			}
		}
	}

	return compressed, nil
}

/* AdjustForgettingThreshold adjusts forgetting threshold based on memory usage */
func (m *MemoryAdaptationManager) AdjustForgettingThreshold(ctx context.Context, agentID uuid.UUID, tier string, currentThreshold float64) (float64, error) {
	/* Analyze memory usage patterns */
	patterns, err := m.AnalyzeUsagePatterns(ctx, agentID, tier, 30)
	if err != nil {
		return currentThreshold, err
	}

	if len(patterns) == 0 {
		return currentThreshold, nil
	}

	/* Calculate average importance of accessed memories */
	totalImportance := 0.0
	accessedCount := 0
	for _, pattern := range patterns {
		if pattern.RetrievalCount > 0 {
			/* Get importance from database */
			var importance float64
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

			query := fmt.Sprintf(`SELECT importance_score FROM neurondb_agent.%s WHERE id = $1`, tableName)
			if err := m.db.DB.GetContext(ctx, &importance, query, pattern.MemoryID); err == nil {
				totalImportance += importance
				accessedCount++
			}
		}
	}

	if accessedCount == 0 {
		return currentThreshold, nil
	}

	avgImportance := totalImportance / float64(accessedCount)

	/* Adjust threshold: if average importance is high, be more conservative */
	/* If average importance is low, be more aggressive */
	if avgImportance > 0.7 {
		/* High importance memories - be more conservative (raise threshold) */
		return math.Min(1.0, currentThreshold+0.1), nil
	} else if avgImportance < 0.4 {
		/* Low importance memories - be more aggressive (lower threshold) */
		return math.Max(0.1, currentThreshold-0.1), nil
	}
	/* Default: no change */
	return currentThreshold, nil
}

/* cosineSimilarityAdaptation calculates cosine similarity for adaptation */
func cosineSimilarityAdaptation(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0.0
	}

	var dotProduct, normA, normB float64
	for i := 0; i < len(a); i++ {
		dotProduct += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	if normA == 0 || normB == 0 {
		return 0.0
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}
