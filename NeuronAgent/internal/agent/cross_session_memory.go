/*-------------------------------------------------------------------------
 *
 * cross_session_memory.go
 *    Cross-session memory management
 *
 * Manages memory across different sessions with deduplication,
 * session isolation, and cross-session sharing.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/cross_session_memory.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/metrics"
	"github.com/neurondb/NeuronAgent/pkg/neurondb"
)

/* CrossSessionMemoryManager manages cross-session memory */
type CrossSessionMemoryManager struct {
	db      *db.DB
	queries *db.Queries
	embed   *neurondb.EmbeddingClient
}

/* NewCrossSessionMemoryManager creates a new cross-session memory manager */
func NewCrossSessionMemoryManager(db *db.DB, queries *db.Queries, embed *neurondb.EmbeddingClient) *CrossSessionMemoryManager {
	return &CrossSessionMemoryManager{
		db:      db,
		queries: queries,
		embed:   embed,
	}
}

/* ShouldShareMemory determines if a memory should be shared across sessions */
func (m *CrossSessionMemoryManager) ShouldShareMemory(ctx context.Context, agentID uuid.UUID, memoryID uuid.UUID, tier string, currentSessionID uuid.UUID, enabled bool) (bool, []uuid.UUID, error) {
	if !enabled {
		return false, nil, nil /* Cross-session sharing disabled */
	}

	/* Privacy check: don't share sensitive memories */
	if m.isSensitiveMemory(ctx, memoryID, tier) {
		return false, nil, nil
	}

	/* Get memory content and importance */
	var content string
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
		return false, nil, fmt.Errorf("invalid tier: %s", tier)
	}

	query := fmt.Sprintf(`SELECT content, importance_score FROM neurondb_agent.%s WHERE id = $1 AND agent_id = $2`, tableName)
	err := m.db.DB.GetContext(ctx, &struct {
		Content         string  `db:"content"`
		ImportanceScore float64 `db:"importance_score"`
	}{
		Content:         content,
		ImportanceScore: importance,
	}, query, memoryID, agentID)

	if err != nil {
		return false, nil, err
	}

	/* Only share high-importance memories (threshold: 0.6) */
	if importance < 0.6 {
		return false, nil, nil
	}

	/* Find related sessions (same user, similar topics) */
	relatedSessions, err := m.findRelatedSessions(ctx, agentID, currentSessionID, content)
	if err != nil {
		return false, nil, err
	}

	if len(relatedSessions) == 0 {
		return false, nil, nil
	}

	return true, relatedSessions, nil
}

/* isSensitiveMemory checks if memory contains sensitive information */
func (m *CrossSessionMemoryManager) isSensitiveMemory(ctx context.Context, memoryID uuid.UUID, tier string) bool {
	/* Check metadata for sensitivity flags */
	var tableName string
	switch tier {
	case "stm":
		tableName = "memory_stm"
	case "mtm":
		tableName = "memory_mtm"
	case "lpm":
		tableName = "memory_lpm"
	default:
		return true /* Default to sensitive if tier unknown */
	}

	query := fmt.Sprintf(`SELECT metadata FROM neurondb_agent.%s WHERE id = $1`, tableName)
	var metadataJSON []byte
	err := m.db.DB.GetContext(ctx, &metadataJSON, query, memoryID)
	if err != nil {
		return true /* Default to sensitive on error */
	}

	if len(metadataJSON) > 0 {
		/* Parse JSON and check for sensitive flag */
		var metadata map[string]interface{}
		if err := json.Unmarshal(metadataJSON, &metadata); err == nil {
			if sensitive, ok := metadata["sensitive"].(bool); ok && sensitive {
				return true
			}
		}
		/* Fallback: simple string check */
		if strings.Contains(strings.ToLower(string(metadataJSON)), "sensitive") {
			return true
		}
	}

	return false
}

/* findRelatedSessions finds sessions that should share this memory */
func (m *CrossSessionMemoryManager) findRelatedSessions(ctx context.Context, agentID uuid.UUID, currentSessionID uuid.UUID, content string) ([]uuid.UUID, error) {
	/* Get current session info */
	currentSession, err := m.queries.GetSession(ctx, currentSessionID)
	if err != nil {
		return nil, err
	}

	/* Find sessions with same external_user_id (same user) */
	if currentSession.ExternalUserID != nil && *currentSession.ExternalUserID != "" {
		query := `SELECT id FROM neurondb_agent.sessions
			WHERE agent_id = $1
			AND external_user_id = $2
			AND id != $3
			AND last_activity_at > NOW() - INTERVAL '90 days'
			ORDER BY last_activity_at DESC
			LIMIT 10`

		var sessionIDs []uuid.UUID
		err := m.db.DB.SelectContext(ctx, &sessionIDs, query, agentID, *currentSession.ExternalUserID, currentSessionID)
		if err == nil && len(sessionIDs) > 0 {
			return sessionIDs, nil
		}
	}

	/* Fallback: find sessions with similar topics (using content similarity) */
	/* Generate embedding for content */
	embedding, err := m.embed.Embed(ctx, content, "all-MiniLM-L6-v2")
	if err != nil {
		return []uuid.UUID{}, nil /* Return empty on error */
	}

	/* Find sessions with similar memories */
	similarSessionsQuery := `SELECT DISTINCT s.id
		FROM neurondb_agent.sessions s
		JOIN neurondb_agent.memory_stm m ON m.session_id = s.id
		WHERE s.agent_id = $1
		AND s.id != $2
		AND m.embedding <=> $3::neurondb_vector < 0.3
		AND s.last_activity_at > NOW() - INTERVAL '30 days'
		ORDER BY s.last_activity_at DESC
		LIMIT 5`

	var sessionIDs []uuid.UUID
	err = m.db.DB.SelectContext(ctx, &sessionIDs, similarSessionsQuery, agentID, currentSessionID, embedding)
	if err != nil {
		return []uuid.UUID{}, nil
	}

	return sessionIDs, nil
}

/* AutoShareRelevantMemories automatically shares relevant memories across sessions */
func (m *CrossSessionMemoryManager) AutoShareRelevantMemories(ctx context.Context, agentID uuid.UUID, sessionID uuid.UUID, enabled bool) error {
	if !enabled {
		return nil /* Auto-sharing disabled */
	}

	/* Check each memory tier */
	for _, tier := range []string{"stm", "mtm", "lpm"} {
		/* Get memories from current session */
		memories, err := m.getSessionMemories(ctx, agentID, sessionID, tier)
		if err != nil {
			continue
		}

		/* Check each memory for sharing */
		for _, memoryID := range memories {
			shouldShare, relatedSessions, err := m.ShouldShareMemory(ctx, agentID, memoryID, tier, sessionID, enabled)
			if err != nil || !shouldShare {
				continue
			}

			/* Share with related sessions */
			if len(relatedSessions) > 0 {
				/* Include current session in the list */
				allSessions := append([]uuid.UUID{sessionID}, relatedSessions...)
				if err := m.ShareMemoryAcrossSessions(ctx, agentID, memoryID, tier, allSessions); err != nil {
					metrics.WarnWithContext(ctx, "Failed to auto-share memory", map[string]interface{}{
						"memory_id": memoryID.String(),
						"tier":      tier,
						"error":     err.Error(),
					})
				}
			}
		}
	}

	return nil
}

/* getSessionMemories gets memory IDs for a session */
func (m *CrossSessionMemoryManager) getSessionMemories(ctx context.Context, agentID uuid.UUID, sessionID uuid.UUID, tier string) ([]uuid.UUID, error) {
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

	query := fmt.Sprintf(`SELECT id FROM neurondb_agent.%s
		WHERE agent_id = $1 AND (session_id = $2 OR $2 = ANY(session_ids))
		LIMIT 100`, tableName)

	var memoryIDs []uuid.UUID
	err := m.db.DB.SelectContext(ctx, &memoryIDs, query, agentID, sessionID)
	return memoryIDs, err
}

/* ShareMemoryAcrossSessions shares memory across multiple sessions */
func (m *CrossSessionMemoryManager) ShareMemoryAcrossSessions(ctx context.Context, agentID uuid.UUID, memoryID uuid.UUID, tier string, sessionIDs []uuid.UUID) error {
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

	/* Update session_ids array */
	query := fmt.Sprintf(`UPDATE neurondb_agent.%s 
		SET session_ids = $1, updated_at = NOW()
		WHERE id = $2 AND agent_id = $3`, tableName)

	_, err := m.db.DB.ExecContext(ctx, query, pq.Array(sessionIDs), memoryID, agentID)
	if err != nil {
		return err
	}

	/* Log access for each session */
	for _, sessionID := range sessionIDs {
		m.logMemoryAccess(ctx, memoryID, sessionID, "shared")
	}

	return nil
}

/* GetCrossSessionMemories retrieves memories shared across sessions */
func (m *CrossSessionMemoryManager) GetCrossSessionMemories(ctx context.Context, agentID uuid.UUID, sessionID uuid.UUID, tier string) ([]uuid.UUID, error) {
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

	/* Get memories where session_id is in session_ids array or session_ids is NULL (shared with all) */
	query := fmt.Sprintf(`SELECT id FROM neurondb_agent.%s
		WHERE agent_id = $1
		AND (session_ids IS NULL OR $2 = ANY(session_ids))`, tableName)

	var memoryIDs []uuid.UUID
	err := m.db.DB.SelectContext(ctx, &memoryIDs, query, agentID, sessionID)
	if err != nil {
		return nil, err
	}

	/* Log access */
	for _, memoryID := range memoryIDs {
		m.logMemoryAccess(ctx, memoryID, sessionID, "retrieved")
	}

	return memoryIDs, nil
}

/* DeduplicateMemories detects and merges duplicate memories across sessions */
func (m *CrossSessionMemoryManager) DeduplicateMemories(ctx context.Context, agentID uuid.UUID, tier string, similarityThreshold float64) (int, error) {
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

	/* Validate similarity threshold */
	if similarityThreshold < 0.5 || similarityThreshold > 1.0 {
		similarityThreshold = 0.9 /* Default to high threshold */
	}

	/* Get all memories for this agent and tier with embeddings */
	type MemoryRow struct {
		ID         uuid.UUID      `db:"id"`
		Content    string         `db:"content"`
		Embedding  []float32      `db:"embedding"`
		SessionIDs pq.StringArray `db:"session_ids"`
		CreatedAt  time.Time      `db:"created_at"`
		Importance float64        `db:"importance_score"`
	}

	query := fmt.Sprintf(`SELECT id, content, embedding, session_ids, created_at, importance_score
		FROM neurondb_agent.%s
		WHERE agent_id = $1 AND embedding IS NOT NULL AND array_length(embedding, 1) > 0
		ORDER BY created_at DESC`, tableName)

	var memories []MemoryRow
	err := m.db.DB.SelectContext(ctx, &memories, query, agentID)
	if err != nil {
		return 0, fmt.Errorf("failed to get memories: %w", err)
	}

	if len(memories) == 0 {
		return 0, nil
	}

	merged := 0
	processed := make(map[uuid.UUID]bool)

	/* Limit comparisons for performance (O(n²) complexity) */
	maxComparisons := 5000
	comparisonCount := 0

	for i, mem1 := range memories {
		if processed[mem1.ID] {
			continue
		}

		/* Find similar memories */
		duplicates := []MemoryRow{mem1}
		allSessionIDs := make(map[string]bool)
		if mem1.SessionIDs != nil {
			for _, sid := range mem1.SessionIDs {
				allSessionIDs[sid] = true
			}
		}

		/* Only compare with memories that haven't been processed */
		for j := i + 1; j < len(memories); j++ {
			if comparisonCount >= maxComparisons {
				break
			}

			mem2 := memories[j]
			if processed[mem2.ID] {
				continue
			}

			comparisonCount++

			/* Check similarity using cosine similarity */
			if len(mem1.Embedding) == 0 || len(mem2.Embedding) == 0 {
				continue
			}

			similarity := cosineSimilarityCrossSession(mem1.Embedding, mem2.Embedding)
			if similarity >= similarityThreshold {
				duplicates = append(duplicates, mem2)
				processed[mem2.ID] = true

				/* Collect session IDs */
				if mem2.SessionIDs != nil {
					for _, sid := range mem2.SessionIDs {
						allSessionIDs[sid] = true
					}
				}
			}
		}

		/* If duplicates found, merge them */
		if len(duplicates) > 1 {
			/* Select best memory to keep (newest, highest importance, or most sessions) */
			keepIndex := 0
			bestScore := 0.0
			for idx, dup := range duplicates {
				score := dup.Importance
				/* Boost score for newer memories */
				ageDays := time.Since(dup.CreatedAt).Hours() / 24.0
				score += math.Exp(-ageDays/30.0) * 0.3
				/* Boost score for memories with more sessions */
				if dup.SessionIDs != nil {
					score += float64(len(dup.SessionIDs)) * 0.1
				}
				if score > bestScore {
					bestScore = score
					keepIndex = idx
				}
			}

			keepID := duplicates[keepIndex].ID
			sessionIDList := make([]string, 0, len(allSessionIDs))
			for sid := range allSessionIDs {
				sessionIDList = append(sessionIDList, sid)
			}

			/* Merge content if similar but not identical (take longest/most complete) */
			mergedContent := duplicates[keepIndex].Content
			for _, dup := range duplicates {
				if len(dup.Content) > len(mergedContent) {
					mergedContent = dup.Content
				}
			}

			/* Update kept memory with merged session IDs and potentially merged content */
			updateQuery := fmt.Sprintf(`UPDATE neurondb_agent.%s
				SET session_ids = $1, content = $2, updated_at = NOW()
				WHERE id = $3`, tableName)
			_, err := m.db.DB.ExecContext(ctx, updateQuery, pq.Array(sessionIDList), mergedContent, keepID)
			if err != nil {
				metrics.WarnWithContext(ctx, "Failed to update merged memory", map[string]interface{}{
					"memory_id": keepID.String(),
					"error":     err.Error(),
				})
				continue
			}

			/* Delete duplicate memories */
			for idx, dup := range duplicates {
				if idx == keepIndex {
					continue
				}
				deleteQuery := fmt.Sprintf(`DELETE FROM neurondb_agent.%s WHERE id = $1`, tableName)
				_, err := m.db.DB.ExecContext(ctx, deleteQuery, dup.ID)
				if err != nil {
					metrics.WarnWithContext(ctx, "Failed to delete duplicate memory", map[string]interface{}{
						"memory_id": dup.ID.String(),
						"error":     err.Error(),
					})
					continue
				}
			}

			merged += len(duplicates) - 1
		}
	}

	return merged, nil
}

/* IsolateSessionMemories makes memories session-specific */
func (m *CrossSessionMemoryManager) IsolateSessionMemories(ctx context.Context, agentID uuid.UUID, sessionID uuid.UUID, tier string) error {
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

	/* Update memories to only include this session */
	query := fmt.Sprintf(`UPDATE neurondb_agent.%s
		SET session_ids = ARRAY[$1]::uuid[], updated_at = NOW()
		WHERE agent_id = $2
		AND ($1 = ANY(session_ids) OR session_ids IS NULL)`, tableName)

	_, err := m.db.DB.ExecContext(ctx, query, sessionID, agentID)
	return err
}

/* GetSessionMemoryStats returns statistics about session memory usage */
func (m *CrossSessionMemoryManager) GetSessionMemoryStats(ctx context.Context, agentID uuid.UUID, sessionID uuid.UUID) (map[string]interface{}, error) {
	stats := make(map[string]interface{})

	/* Count memories per tier for this session */
	for _, tier := range []string{"stm", "mtm", "lpm"} {
		var tableName string
		switch tier {
		case "stm":
			tableName = "memory_stm"
		case "mtm":
			tableName = "memory_mtm"
		case "lpm":
			tableName = "memory_lpm"
		}

		query := fmt.Sprintf(`SELECT COUNT(*) FROM neurondb_agent.%s
			WHERE agent_id = $1
			AND (session_ids IS NULL OR $2 = ANY(session_ids))`, tableName)

		var count int
		err := m.db.DB.GetContext(ctx, &count, query, agentID, sessionID)
		if err != nil {
			continue
		}

		stats[tier+"_count"] = count
	}

	/* Count shared memories */
	sharedQuery := `SELECT COUNT(*) FROM (
		SELECT id FROM neurondb_agent.memory_stm WHERE agent_id = $1 AND (session_ids IS NULL OR array_length(session_ids, 1) > 1)
		UNION ALL
		SELECT id FROM neurondb_agent.memory_mtm WHERE agent_id = $1 AND (session_ids IS NULL OR array_length(session_ids, 1) > 1)
		UNION ALL
		SELECT id FROM neurondb_agent.memory_lpm WHERE agent_id = $1 AND (session_ids IS NULL OR array_length(session_ids, 1) > 1)
	) shared`

	var sharedCount int
	_ = m.db.DB.GetContext(ctx, &sharedCount, sharedQuery, agentID)
	stats["shared_count"] = sharedCount

	return stats, nil
}

/* logMemoryAccess logs memory access for tracking */
func (m *CrossSessionMemoryManager) logMemoryAccess(ctx context.Context, memoryID, sessionID uuid.UUID, action string) {
	query := `INSERT INTO neurondb_agent.memory_access_log
		(memory_id, session_id, action, accessed_at)
		VALUES ($1, $2, $3, NOW())
		ON CONFLICT DO NOTHING`

	_, _ = m.db.DB.ExecContext(ctx, query, memoryID, sessionID, action)
}

/* cosineSimilarityCrossSession calculates cosine similarity for cross-session memory */
func cosineSimilarityCrossSession(a, b []float32) float64 {
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
