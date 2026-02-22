/*-------------------------------------------------------------------------
 *
 * memory_conflict_resolver.go
 *    Memory conflict detection and resolution
 *
 * Detects and resolves conflicts when the same information has different
 * values using various resolution strategies.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/memory_conflict_resolver.go
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
	"github.com/neurondb/NeuronAgent/pkg/neurondb"
)

/* MemoryConflictResolver resolves memory conflicts */
type MemoryConflictResolver struct {
	db      *db.DB
	queries *db.Queries
	llm     *LLMClient
	embed   *neurondb.EmbeddingClient
}

/* NewMemoryConflictResolver creates a new conflict resolver */
func NewMemoryConflictResolver(db *db.DB, queries *db.Queries, llm *LLMClient, embed *neurondb.EmbeddingClient) *MemoryConflictResolver {
	return &MemoryConflictResolver{
		db:      db,
		queries: queries,
		llm:     llm,
		embed:   embed,
	}
}

/* ConflictResolutionStrategy defines resolution strategy */
type ConflictResolutionStrategy string

const (
	ResolutionStrategyTimestamp  ConflictResolutionStrategy = "timestamp"  /* Prefer newer */
	ResolutionStrategyConfidence ConflictResolutionStrategy = "confidence" /* Prefer higher confidence */
	ResolutionStrategySource     ConflictResolutionStrategy = "source"     /* Prefer verified sources */
	ResolutionStrategyLLM        ConflictResolutionStrategy = "llm"        /* Use LLM to decide */
	ResolutionStrategyMerge      ConflictResolutionStrategy = "merge"      /* Merge complementary info */
)

/* Conflict represents a detected conflict */
type Conflict struct {
	ConflictID   uuid.UUID
	MemoryIDs    []uuid.UUID
	Tier         string
	ConflictType string
	Description  string
	Resolved     bool
	Resolution   string
}

/* DetectConflicts finds conflicting memories */
func (r *MemoryConflictResolver) DetectConflicts(ctx context.Context, agentID uuid.UUID) ([]Conflict, error) {
	conflicts := make([]Conflict, 0)

	/* Check for semantic conflicts (similar content, different information) */
	semanticConflicts, err := r.detectSemanticConflicts(ctx, agentID)
	if err != nil {
		metrics.WarnWithContext(ctx, "Failed to detect semantic conflicts", map[string]interface{}{
			"agent_id": agentID.String(),
			"error":    err.Error(),
		})
	} else {
		conflicts = append(conflicts, semanticConflicts...)
	}

	/* Check for factual conflicts (same entity, different facts) */
	factualConflicts, err := r.detectFactualConflicts(ctx, agentID)
	if err != nil {
		metrics.WarnWithContext(ctx, "Failed to detect factual conflicts", map[string]interface{}{
			"agent_id": agentID.String(),
			"error":    err.Error(),
		})
	} else {
		conflicts = append(conflicts, factualConflicts...)
	}

	/* Store conflicts in database */
	for _, conflict := range conflicts {
		r.storeConflict(ctx, agentID, conflict)
	}

	return conflicts, nil
}

/* detectSemanticConflicts detects conflicts based on semantic similarity */
func (r *MemoryConflictResolver) detectSemanticConflicts(ctx context.Context, agentID uuid.UUID) ([]Conflict, error) {
	conflicts := make([]Conflict, 0)

	/* Get all memories with embeddings */
	query := `SELECT id, content, embedding, tier, created_at, importance_score
		FROM (
			SELECT id, content, embedding, 'stm' as tier, created_at, importance_score FROM neurondb_agent.memory_stm WHERE agent_id = $1
			UNION ALL
			SELECT id, content, embedding, 'mtm' as tier, created_at, importance_score FROM neurondb_agent.memory_mtm WHERE agent_id = $1
			UNION ALL
			SELECT id, content, embedding, 'lpm' as tier, created_at, importance_score FROM neurondb_agent.memory_lpm WHERE agent_id = $1
		) all_memories
		WHERE embedding IS NOT NULL AND array_length(embedding, 1) > 0`

	type MemoryRow struct {
		ID              uuid.UUID `db:"id"`
		Content         string    `db:"content"`
		Embedding       []float32 `db:"embedding"`
		Tier            string    `db:"tier"`
		CreatedAt       time.Time `db:"created_at"`
		ImportanceScore float64   `db:"importance_score"`
	}

	var memories []MemoryRow
	err := r.db.DB.SelectContext(ctx, &memories, query, agentID)
	if err != nil {
		return conflicts, err
	}

	/* Use more efficient approach: group by similarity clusters first */
	/* Compare memories for high similarity but different content */
	/* Limit comparisons to avoid O(n²) complexity on large datasets */
	maxComparisons := 10000
	comparisonCount := 0

	for i, mem1 := range memories {
		if comparisonCount >= maxComparisons {
			/* Log that we're limiting comparisons */
			metrics.WarnWithContext(ctx, "Conflict detection limited comparisons", map[string]interface{}{
				"agent_id":        agentID.String(),
				"total_memories":  len(memories),
				"max_comparisons": maxComparisons,
			})
			break
		}

		/* Only compare with memories in same tier for efficiency */
		for j := i + 1; j < len(memories); j++ {
			mem2 := memories[j]
			comparisonCount++

			/* Skip if different tiers (less likely to conflict) */
			if mem1.Tier != mem2.Tier {
				continue
			}

			/* Calculate similarity using helper function */
			similarity := calculateCosineSimilarity(mem1.Embedding, mem2.Embedding)

			/* Adaptive threshold based on tier */
			similarityThreshold := 0.8
			if mem1.Tier == "lpm" {
				/* LPM memories should have higher threshold (more important) */
				similarityThreshold = 0.85
			} else if mem1.Tier == "stm" {
				/* STM can have lower threshold (more transient) */
				similarityThreshold = 0.75
			}

			/* If highly similar but content differs significantly, potential conflict */
			if similarity > similarityThreshold && mem1.Content != mem2.Content {
				/* Check content length similarity (very different lengths might not be conflicts) */
				lengthRatio := float64(len(mem1.Content)) / float64(len(mem2.Content))
				if lengthRatio > 0.3 && lengthRatio < 3.0 { /* Reasonable length ratio */
					/* Check if content is actually conflicting (not just different phrasing) */
					if r.isConflictingContent(mem1.Content, mem2.Content) {
						/* Check if this conflict was already detected */
						alreadyDetected := false
						for _, existingConflict := range conflicts {
							for _, existingID := range existingConflict.MemoryIDs {
								if existingID == mem1.ID || existingID == mem2.ID {
									alreadyDetected = true
									break
								}
							}
							if alreadyDetected {
								break
							}
						}

						if !alreadyDetected {
							conflicts = append(conflicts, Conflict{
								ConflictID:   uuid.New(),
								MemoryIDs:    []uuid.UUID{mem1.ID, mem2.ID},
								Tier:         mem1.Tier,
								ConflictType: "semantic",
								Description:  fmt.Sprintf("High similarity (%.2f) but conflicting content. Age difference: %v", similarity, mem2.CreatedAt.Sub(mem1.CreatedAt)),
								Resolved:     false,
							})
						}
					}
				}
			}
		}
	}

	return conflicts, nil
}

/* detectFactualConflicts detects conflicts in factual information */
func (r *MemoryConflictResolver) detectFactualConflicts(ctx context.Context, agentID uuid.UUID) ([]Conflict, error) {
	conflicts := make([]Conflict, 0)

	/* Get memories with embeddings for better topic matching */
	query := `SELECT id, content, tier, created_at, embedding
		FROM (
			SELECT id, content, 'stm' as tier, created_at, embedding FROM neurondb_agent.memory_stm WHERE agent_id = $1 AND embedding IS NOT NULL
			UNION ALL
			SELECT id, content, 'mtm' as tier, created_at, embedding FROM neurondb_agent.memory_mtm WHERE agent_id = $1 AND embedding IS NOT NULL
			UNION ALL
			SELECT id, content, 'lpm' as tier, created_at, embedding FROM neurondb_agent.memory_lpm WHERE agent_id = $1 AND embedding IS NOT NULL
		) all_memories`

	type MemoryRow struct {
		ID        uuid.UUID `db:"id"`
		Content   string    `db:"content"`
		Tier      string    `db:"tier"`
		CreatedAt time.Time `db:"created_at"`
		Embedding []float32 `db:"embedding"`
	}

	var memories []MemoryRow
	err := r.db.DB.SelectContext(ctx, &memories, query, agentID)
	if err != nil {
		return conflicts, err
	}

	/* Enhanced contradictory patterns with context */
	contradictoryPatterns := []struct {
		positive []string
		negative []string
		weight   float64
	}{
		{[]string{"yes", "true", "correct", "right", "agree"}, []string{"no", "false", "incorrect", "wrong", "disagree"}, 1.0},
		{[]string{"like", "love", "prefer", "enjoy", "favorite"}, []string{"dislike", "hate", "avoid", "don't like", "least favorite"}, 0.8},
		{[]string{"always", "never fails", "consistent"}, []string{"never", "doesn't work", "inconsistent"}, 0.7},
		{[]string{"good", "great", "excellent", "positive"}, []string{"bad", "terrible", "poor", "negative"}, 0.6},
	}

	/* Limit comparisons for performance */
	maxComparisons := 5000
	comparisonCount := 0

	for i, mem1 := range memories {
		if comparisonCount >= maxComparisons {
			break
		}

		for j := i + 1; j < len(memories); j++ {
			mem2 := memories[j]
			comparisonCount++

			/* Use embedding similarity to check if about same topic */
			similarity := calculateCosineSimilarity(mem1.Embedding, mem2.Embedding)
			if similarity < 0.6 {
				/* Not similar enough to be about same topic */
				continue
			}

			/* Check for contradictory patterns */
			for _, pattern := range contradictoryPatterns {
				hasPositive1 := false
				hasNegative1 := false
				hasPositive2 := false
				hasNegative2 := false

				for _, word := range pattern.positive {
					if containsWord(mem1.Content, word) {
						hasPositive1 = true
					}
					if containsWord(mem2.Content, word) {
						hasPositive2 = true
					}
				}
				for _, word := range pattern.negative {
					if containsWord(mem1.Content, word) {
						hasNegative1 = true
					}
					if containsWord(mem2.Content, word) {
						hasNegative2 = true
					}
				}

				/* Check for contradiction */
				if (hasPositive1 && hasNegative2) || (hasNegative1 && hasPositive2) {
					/* Potential conflict - verify it's about same entity */
					if r.isSameTopic(mem1.Content, mem2.Content) {
						/* Check if already detected */
						alreadyDetected := false
						for _, existingConflict := range conflicts {
							for _, existingID := range existingConflict.MemoryIDs {
								if existingID == mem1.ID || existingID == mem2.ID {
									alreadyDetected = true
									break
								}
							}
							if alreadyDetected {
								break
							}
						}

						if !alreadyDetected {
							conflicts = append(conflicts, Conflict{
								ConflictID:   uuid.New(),
								MemoryIDs:    []uuid.UUID{mem1.ID, mem2.ID},
								Tier:         mem1.Tier,
								ConflictType: "factual",
								Description:  fmt.Sprintf("Contradictory statements detected (similarity: %.2f, pattern weight: %.1f)", similarity, pattern.weight),
								Resolved:     false,
							})
							break /* Only one conflict per pair */
						}
					}
				}
			}
		}
	}

	return conflicts, nil
}

/* ResolveConflict resolves a conflict using specified strategy */
func (r *MemoryConflictResolver) ResolveConflict(ctx context.Context, conflict Conflict, strategy ConflictResolutionStrategy) error {
	switch strategy {
	case ResolutionStrategyTimestamp:
		return r.resolveByTimestamp(ctx, conflict)
	case ResolutionStrategyConfidence:
		return r.resolveByConfidence(ctx, conflict)
	case ResolutionStrategySource:
		return r.resolveBySource(ctx, conflict)
	case ResolutionStrategyLLM:
		return r.resolveByLLM(ctx, conflict)
	case ResolutionStrategyMerge:
		return r.resolveByMerge(ctx, conflict)
	default:
		return fmt.Errorf("unknown resolution strategy: %s", strategy)
	}
}

/* resolveByTimestamp prefers newer information */
func (r *MemoryConflictResolver) resolveByTimestamp(ctx context.Context, conflict Conflict) error {
	if len(conflict.MemoryIDs) < 2 {
		return fmt.Errorf("conflict must have at least 2 memories")
	}

	/* Get timestamps */
	var keepID uuid.UUID
	var latestTime time.Time

	for _, memID := range conflict.MemoryIDs {
		createdAt, err := r.getMemoryCreatedAt(ctx, memID, conflict.Tier)
		if err != nil {
			continue
		}
		if createdAt.After(latestTime) {
			latestTime = createdAt
			keepID = memID
		}
	}

	/* Delete older memories */
	for _, memID := range conflict.MemoryIDs {
		if memID != keepID {
			if err := r.deleteMemory(ctx, memID, conflict.Tier); err != nil {
				return err
			}
		}
	}

	return r.markConflictResolved(ctx, conflict.ConflictID, "timestamp", keepID.String())
}

/* resolveByConfidence prefers higher confidence */
func (r *MemoryConflictResolver) resolveByConfidence(ctx context.Context, conflict Conflict) error {
	if len(conflict.MemoryIDs) < 2 {
		return fmt.Errorf("conflict must have at least 2 memories")
	}

	/* Get confidence scores (using importance_score as proxy) */
	var keepID uuid.UUID
	var highestConfidence float64

	for _, memID := range conflict.MemoryIDs {
		importance, err := r.getMemoryImportance(ctx, memID, conflict.Tier)
		if err != nil {
			continue
		}
		if importance > highestConfidence {
			highestConfidence = importance
			keepID = memID
		}
	}

	/* Delete lower confidence memories */
	for _, memID := range conflict.MemoryIDs {
		if memID != keepID {
			if err := r.deleteMemory(ctx, memID, conflict.Tier); err != nil {
				return err
			}
		}
	}

	return r.markConflictResolved(ctx, conflict.ConflictID, "confidence", keepID.String())
}

/* resolveBySource prefers verified sources */
func (r *MemoryConflictResolver) resolveBySource(ctx context.Context, conflict Conflict) error {
	/* Check metadata for source verification */
	var keepID uuid.UUID
	var bestSource string

	for _, memID := range conflict.MemoryIDs {
		source, verified := r.getMemorySource(ctx, memID, conflict.Tier)
		if verified && (bestSource == "" || source == "verified") {
			bestSource = source
			keepID = memID
		}
	}

	/* If no verified source, fall back to timestamp */
	if keepID == uuid.Nil {
		return r.resolveByTimestamp(ctx, conflict)
	}

	/* Delete other memories */
	for _, memID := range conflict.MemoryIDs {
		if memID != keepID {
			if err := r.deleteMemory(ctx, memID, conflict.Tier); err != nil {
				return err
			}
		}
	}

	return r.markConflictResolved(ctx, conflict.ConflictID, "source", keepID.String())
}

/* resolveByLLM uses LLM to determine which is correct */
func (r *MemoryConflictResolver) resolveByLLM(ctx context.Context, conflict Conflict) error {
	if len(conflict.MemoryIDs) < 2 {
		return fmt.Errorf("conflict must have at least 2 memories")
	}

	/* Get memory contents */
	memories := make([]string, 0, len(conflict.MemoryIDs))
	for _, memID := range conflict.MemoryIDs {
		content, err := r.getMemoryContent(ctx, memID, conflict.Tier)
		if err != nil {
			continue
		}
		memories = append(memories, content)
	}

	if len(memories) < 2 {
		return fmt.Errorf("could not retrieve memory contents")
	}

	/* Ask LLM which is correct */
	prompt := fmt.Sprintf(`Two conflicting pieces of information:
1. %s
2. %s

Which one is more accurate and should be kept? Respond with "1" or "2".`, memories[0], memories[1])

	llmConfig := map[string]interface{}{
		"temperature": 0.1,
		"max_tokens":  10,
	}

	response, err := r.llm.Generate(ctx, "gpt-4", prompt, llmConfig)
	if err != nil {
		return fmt.Errorf("llm conflict resolution: %w", err)
	}

	/* Parse response */
	var keepIndex int
	if response.Content == "1" {
		keepIndex = 0
	} else if response.Content == "2" {
		keepIndex = 1
	} else {
		/* Default to first if unclear */
		keepIndex = 0
	}

	keepID := conflict.MemoryIDs[keepIndex]

	/* Delete other memories */
	for i, memID := range conflict.MemoryIDs {
		if i != keepIndex {
			if err := r.deleteMemory(ctx, memID, conflict.Tier); err != nil {
				return err
			}
		}
	}

	return r.markConflictResolved(ctx, conflict.ConflictID, "llm", keepID.String())
}

/* resolveByMerge merges complementary information */
func (r *MemoryConflictResolver) resolveByMerge(ctx context.Context, conflict Conflict) error {
	if len(conflict.MemoryIDs) < 2 {
		return fmt.Errorf("conflict must have at least 2 memories")
	}

	/* Get all memory contents */
	contents := make([]string, 0, len(conflict.MemoryIDs))
	for _, memID := range conflict.MemoryIDs {
		content, err := r.getMemoryContent(ctx, memID, conflict.Tier)
		if err != nil {
			continue
		}
		contents = append(contents, content)
	}

	if len(contents) < 2 {
		return fmt.Errorf("could not retrieve memory contents")
	}

	/* Merge contents */
	mergedContent := ""
	for i, content := range contents {
		if i > 0 {
			mergedContent += "\n\n"
		}
		mergedContent += content
	}

	/* Keep first memory, update with merged content */
	keepID := conflict.MemoryIDs[0]
	if err := r.updateMemoryContent(ctx, keepID, conflict.Tier, mergedContent); err != nil {
		return fmt.Errorf("resolveByMerge update memory: %w", err)
	}

	/* Delete other memories */
	for i := 1; i < len(conflict.MemoryIDs); i++ {
		if err := r.deleteMemory(ctx, conflict.MemoryIDs[i], conflict.Tier); err != nil {
			return err
		}
	}

	return r.markConflictResolved(ctx, conflict.ConflictID, "merge", keepID.String())
}

/* Helper methods */

func (r *MemoryConflictResolver) isConflictingContent(content1, content2 string) bool {
	/* Simple heuristic: check for contradictory keywords */
	contradictoryPairs := [][]string{
		{"yes", "no"}, {"true", "false"}, {"like", "dislike"},
	}
	for _, pair := range contradictoryPairs {
		if (containsWord(content1, pair[0]) && containsWord(content2, pair[1])) ||
			(containsWord(content1, pair[1]) && containsWord(content2, pair[0])) {
			return true
		}
	}
	return false
}

func (r *MemoryConflictResolver) isSameTopic(content1, content2 string) bool {
	/* Simple heuristic: check for common words (excluding stop words) */
	words1 := extractWords(content1)
	words2 := extractWords(content2)

	common := 0
	for w1 := range words1 {
		if words2[w1] {
			common++
		}
	}

	/* If more than 2 common words, likely same topic */
	return common > 2
}

func containsWord(text, word string) bool {
	/* Simple word boundary check */
	return len(text) >= len(word) &&
		(text == word ||
			text[:len(word)] == word && (len(text) == len(word) || text[len(word)] == ' ') ||
			text[len(text)-len(word):] == word && (len(text) == len(word) || text[len(text)-len(word)-1] == ' '))
}

func extractWords(text string) map[string]bool {
	words := make(map[string]bool)
	/* Simplified word extraction */
	current := ""
	for _, char := range text {
		if (char >= 'a' && char <= 'z') || (char >= 'A' && char <= 'Z') || (char >= '0' && char <= '9') {
			current += string(char)
		} else {
			if len(current) > 2 {
				words[current] = true
			}
			current = ""
		}
	}
	if len(current) > 2 {
		words[current] = true
	}
	return words
}

/* calculateCosineSimilarity calculates cosine similarity between two embeddings */
func calculateCosineSimilarity(a, b []float32) float64 {
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

func (r *MemoryConflictResolver) getMemoryCreatedAt(ctx context.Context, memoryID uuid.UUID, tier string) (time.Time, error) {
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
	err := r.db.DB.GetContext(ctx, &createdAt, query, memoryID)
	return createdAt, err
}

func (r *MemoryConflictResolver) getMemoryImportance(ctx context.Context, memoryID uuid.UUID, tier string) (float64, error) {
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
	err := r.db.DB.GetContext(ctx, &importance, query, memoryID)
	return importance, err
}

func (r *MemoryConflictResolver) getMemorySource(ctx context.Context, memoryID uuid.UUID, tier string) (string, bool) {
	var tableName string
	switch tier {
	case "stm":
		tableName = "memory_stm"
	case "mtm":
		tableName = "memory_mtm"
	case "lpm":
		tableName = "memory_lpm"
	default:
		return "", false
	}

	query := fmt.Sprintf(`SELECT metadata FROM neurondb_agent.%s WHERE id = $1`, tableName)
	var metadata map[string]interface{}
	err := r.db.DB.GetContext(ctx, &metadata, query, memoryID)
	if err != nil {
		return "", false
	}

	if source, ok := metadata["source"].(string); ok {
		verified := false
		if v, ok := metadata["verified"].(bool); ok {
			verified = v
		}
		return source, verified
	}

	return "", false
}

func (r *MemoryConflictResolver) getMemoryContent(ctx context.Context, memoryID uuid.UUID, tier string) (string, error) {
	var tableName string
	switch tier {
	case "stm":
		tableName = "memory_stm"
	case "mtm":
		tableName = "memory_mtm"
	case "lpm":
		tableName = "memory_lpm"
	default:
		return "", fmt.Errorf("invalid tier: %s", tier)
	}

	query := fmt.Sprintf(`SELECT content FROM neurondb_agent.%s WHERE id = $1`, tableName)
	var content string
	err := r.db.DB.GetContext(ctx, &content, query, memoryID)
	return content, err
}

func (r *MemoryConflictResolver) updateMemoryContent(ctx context.Context, memoryID uuid.UUID, tier string, content string) error {
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

	/* Regenerate embedding */
	embedding, err := r.embed.Embed(ctx, content, "all-MiniLM-L6-v2")
	if err != nil {
		return err
	}

	query := fmt.Sprintf(`UPDATE neurondb_agent.%s 
		SET content = $1, embedding = $2::neurondb_vector, updated_at = NOW()
		WHERE id = $3`, tableName)
	_, err = r.db.DB.ExecContext(ctx, query, content, embedding, memoryID)
	if err != nil {
		return fmt.Errorf("update memory content: %w", err)
	}
	return nil
}

func (r *MemoryConflictResolver) deleteMemory(ctx context.Context, memoryID uuid.UUID, tier string) error {
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
	_, err := r.db.DB.ExecContext(ctx, query, memoryID)
	if err != nil {
		return fmt.Errorf("delete memory %s: %w", memoryID, err)
	}
	return nil
}

func (r *MemoryConflictResolver) storeConflict(ctx context.Context, agentID uuid.UUID, conflict Conflict) {
	query := `INSERT INTO neurondb_agent.memory_conflicts
		(conflict_id, agent_id, memory_ids, tier, conflict_type, description, resolved, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
		ON CONFLICT (conflict_id) DO UPDATE SET
			resolved = EXCLUDED.resolved`

	_, _ = r.db.DB.ExecContext(ctx, query, conflict.ConflictID, agentID, conflict.MemoryIDs, conflict.Tier, conflict.ConflictType, conflict.Description, conflict.Resolved)
}

func (r *MemoryConflictResolver) markConflictResolved(ctx context.Context, conflictID uuid.UUID, resolution string, keptMemoryID string) error {
	query := `UPDATE neurondb_agent.memory_conflicts
		SET resolved = true, resolution = $1, resolved_at = NOW(), kept_memory_id = $2
		WHERE conflict_id = $3`

	_, err := r.db.DB.ExecContext(ctx, query, resolution, keptMemoryID, conflictID)
	if err != nil {
		return fmt.Errorf("mark conflict resolved: %w", err)
	}
	return nil
}
