/*-------------------------------------------------------------------------
 *
 * memory_corruption.go
 *    Memory corruption detection and repair
 *
 * Detects and repairs corrupted or invalid memories including embedding
 * dimension mismatches, invalid JSON, orphaned references, and conflicts.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/memory_corruption.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/metrics"
	"github.com/neurondb/NeuronAgent/pkg/neurondb"
)

/* MemoryCorruptionDetector detects and repairs memory corruption */
type MemoryCorruptionDetector struct {
	db      *db.DB
	queries *db.Queries
	embed   *neurondb.EmbeddingClient
}

/* NewMemoryCorruptionDetector creates a new corruption detector */
func NewMemoryCorruptionDetector(db *db.DB, queries *db.Queries, embed *neurondb.EmbeddingClient) *MemoryCorruptionDetector {
	return &MemoryCorruptionDetector{
		db:      db,
		queries: queries,
		embed:   embed,
	}
}

/* CorruptionIssue represents a detected corruption issue */
type CorruptionIssue struct {
	MemoryID    uuid.UUID
	Tier        string
	IssueType   string
	Description string
	Severity    string /* "low", "medium", "high", "critical" */
	Repairable  bool
}

/* ValidateMemoryIntegrity checks all memories for corruption */
func (d *MemoryCorruptionDetector) ValidateMemoryIntegrity(ctx context.Context, agentID uuid.UUID) ([]CorruptionIssue, error) {
	issues := make([]CorruptionIssue, 0)

	/* Check all memory tiers */
	tiers := []string{"stm", "mtm", "lpm"}
	for _, tier := range tiers {
		tierIssues, err := d.validateTier(ctx, agentID, tier)
		if err != nil {
			metrics.WarnWithContext(ctx, "Failed to validate memory tier", map[string]interface{}{
				"agent_id": agentID.String(),
				"tier":     tier,
				"error":    err.Error(),
			})
			continue
		}
		issues = append(issues, tierIssues...)
	}

	return issues, nil
}

/* validateTier validates memories in a specific tier */
func (d *MemoryCorruptionDetector) validateTier(ctx context.Context, agentID uuid.UUID, tier string) ([]CorruptionIssue, error) {
	issues := make([]CorruptionIssue, 0)

	var tableName string
	switch tier {
	case "stm":
		tableName = "memory_stm"
	case "mtm":
		tableName = "memory_mtm"
	case "lpm":
		tableName = "memory_lpm"
	default:
		return issues, fmt.Errorf("invalid tier: %s", tier)
	}

	/* Query all memories for this agent and tier */
	query := fmt.Sprintf(`SELECT id, content, embedding, metadata, importance_score
		FROM neurondb_agent.%s
		WHERE agent_id = $1`, tableName)

	type MemoryRow struct {
		ID              uuid.UUID       `db:"id"`
		Content         string          `db:"content"`
		Embedding       []float32       `db:"embedding"`
		Metadata        json.RawMessage `db:"metadata"`
		ImportanceScore float64         `db:"importance_score"`
	}

	var rows []MemoryRow
	err := d.db.DB.SelectContext(ctx, &rows, query, agentID)
	if err != nil {
		return issues, err
	}

	/* Expected embedding dimension (typically 768 for all-MiniLM-L6-v2) */
	expectedEmbeddingDim := 768

	for _, row := range rows {
		/* Check embedding dimensions and validity */
		if len(row.Embedding) == 0 {
			issues = append(issues, CorruptionIssue{
				MemoryID:    row.ID,
				Tier:        tier,
				IssueType:   "empty_embedding",
				Description: "Memory has empty embedding vector",
				Severity:    "high",
				Repairable:  true,
			})
		} else if len(row.Embedding) != expectedEmbeddingDim {
			/* Check for dimension mismatch */
			issues = append(issues, CorruptionIssue{
				MemoryID:    row.ID,
				Tier:        tier,
				IssueType:   "embedding_dimension_mismatch",
				Description: fmt.Sprintf("Embedding dimension mismatch: expected %d, got %d", expectedEmbeddingDim, len(row.Embedding)),
				Severity:    "high",
				Repairable:  true,
			})
		} else {
			/* Check for NaN or Inf values in embedding */
			hasInvalidValues := false
			for _, val := range row.Embedding {
				if val != val || val > 1e10 || val < -1e10 { /* NaN check: val != val, and reasonable range */
					hasInvalidValues = true
					break
				}
			}
			if hasInvalidValues {
				issues = append(issues, CorruptionIssue{
					MemoryID:    row.ID,
					Tier:        tier,
					IssueType:   "invalid_embedding_values",
					Description: "Embedding contains NaN or infinite values",
					Severity:    "high",
					Repairable:  true,
				})
			}
		}

		/* Check metadata JSON validity and structure */
		if len(row.Metadata) > 0 {
			var metadata map[string]interface{}
			if err := json.Unmarshal(row.Metadata, &metadata); err != nil {
				issues = append(issues, CorruptionIssue{
					MemoryID:    row.ID,
					Tier:        tier,
					IssueType:   "invalid_metadata_json",
					Description: fmt.Sprintf("Invalid JSON in metadata: %v", err),
					Severity:    "medium",
					Repairable:  true,
				})
			} else {
				/* Check for suspiciously large metadata (potential corruption) */
				if len(row.Metadata) > 100000 { /* 100KB limit */
					issues = append(issues, CorruptionIssue{
						MemoryID:    row.ID,
						Tier:        tier,
						IssueType:   "metadata_too_large",
						Description: fmt.Sprintf("Metadata size suspiciously large: %d bytes", len(row.Metadata)),
						Severity:    "medium",
						Repairable:  false,
					})
				}
			}
		}

		/* Check for empty content */
		if row.Content == "" {
			issues = append(issues, CorruptionIssue{
				MemoryID:    row.ID,
				Tier:        tier,
				IssueType:   "empty_content",
				Description: "Memory has empty content",
				Severity:    "high",
				Repairable:  false,
			})
		} else if len(row.Content) > 1000000 { /* 1MB limit */
			/* Check for suspiciously large content */
			issues = append(issues, CorruptionIssue{
				MemoryID:    row.ID,
				Tier:        tier,
				IssueType:   "content_too_large",
				Description: fmt.Sprintf("Content size suspiciously large: %d bytes", len(row.Content)),
				Severity:    "medium",
				Repairable:  false,
			})
		}

		/* Check importance score range and validity */
		if row.ImportanceScore < 0 || row.ImportanceScore > 1 {
			issues = append(issues, CorruptionIssue{
				MemoryID:    row.ID,
				Tier:        tier,
				IssueType:   "invalid_importance_score",
				Description: fmt.Sprintf("Importance score out of range: %f (expected 0.0-1.0)", row.ImportanceScore),
				Severity:    "low",
				Repairable:  true,
			})
		} else if row.ImportanceScore != row.ImportanceScore { /* NaN check */
			issues = append(issues, CorruptionIssue{
				MemoryID:    row.ID,
				Tier:        tier,
				IssueType:   "invalid_importance_score",
				Description: "Importance score is NaN",
				Severity:    "medium",
				Repairable:  true,
			})
		}
	}

	return issues, nil
}

/* DetectConflicts finds conflicting memories */
func (d *MemoryCorruptionDetector) DetectConflicts(ctx context.Context, agentID uuid.UUID) ([]CorruptionIssue, error) {
	issues := make([]CorruptionIssue, 0)

	/* Check for duplicate content with different information */
	query := `SELECT id, content, embedding, tier
		FROM (
			SELECT id, content, embedding, 'stm' as tier FROM neurondb_agent.memory_stm WHERE agent_id = $1
			UNION ALL
			SELECT id, content, embedding, 'mtm' as tier FROM neurondb_agent.memory_mtm WHERE agent_id = $1
			UNION ALL
			SELECT id, content, embedding, 'lpm' as tier FROM neurondb_agent.memory_lpm WHERE agent_id = $1
		) all_memories
		ORDER BY content`

	type MemoryRow struct {
		ID        uuid.UUID `db:"id"`
		Content   string    `db:"content"`
		Embedding []float32 `db:"embedding"`
		Tier      string    `db:"tier"`
	}

	var rows []MemoryRow
	err := d.db.DB.SelectContext(ctx, &rows, query, agentID)
	if err != nil {
		return issues, err
	}

	/* Group by similar content (simplified: exact match for now) */
	contentMap := make(map[string][]MemoryRow)
	for _, row := range rows {
		contentMap[row.Content] = append(contentMap[row.Content], row)
	}

	/* Find duplicates */
	for _, memories := range contentMap {
		if len(memories) > 1 {
			for i, mem := range memories {
				if i > 0 {
					issues = append(issues, CorruptionIssue{
						MemoryID:    mem.ID,
						Tier:        mem.Tier,
						IssueType:   "duplicate_content",
						Description: fmt.Sprintf("Duplicate content found: %d other memories with same content", len(memories)-1),
						Severity:    "medium",
						Repairable:  true,
					})
				}
			}
		}
	}

	return issues, nil
}

/* RepairCorruptedMemory attempts to repair corrupted memory */
func (d *MemoryCorruptionDetector) RepairCorruptedMemory(ctx context.Context, issue CorruptionIssue) error {
	switch issue.IssueType {
	case "empty_embedding":
		return d.repairEmptyEmbedding(ctx, issue)
	case "invalid_metadata_json":
		return d.repairInvalidMetadata(ctx, issue)
	case "invalid_importance_score":
		return d.repairImportanceScore(ctx, issue)
	case "duplicate_content":
		/* Mark for deletion or merging (handled separately) */
		return nil
	default:
		return fmt.Errorf("cannot repair issue type: %s", issue.IssueType)
	}
}

// allowedMemoryTables is the allowlist for table names used in raw SQL (injection-safe).
var allowedMemoryTables = map[string]string{"stm": "memory_stm", "mtm": "memory_mtm", "lpm": "memory_lpm"}

/* repairEmptyEmbedding regenerates embedding for memory */

func (d *MemoryCorruptionDetector) repairEmptyEmbedding(ctx context.Context, issue CorruptionIssue) error {
	tableName, ok := allowedMemoryTables[issue.Tier]
	if !ok {
		return fmt.Errorf("invalid tier: %s", issue.Tier)
	}

	/* Get content */
	query := fmt.Sprintf(`SELECT content FROM neurondb_agent.%s WHERE id = $1`, tableName)
	var content string
	err := d.db.DB.GetContext(ctx, &content, query, issue.MemoryID)
	if err != nil {
		return fmt.Errorf("get memory content for repair: %w", err)
	}

	embedding, err := d.embed.Embed(ctx, content, "all-MiniLM-L6-v2")
	if err != nil {
		return fmt.Errorf("embed for repair: %w", err)
	}

	updateQuery := fmt.Sprintf(`UPDATE neurondb_agent.%s SET embedding = $1::neurondb_vector, updated_at = NOW() WHERE id = $2`, tableName)
	_, err = d.db.DB.ExecContext(ctx, updateQuery, embedding, issue.MemoryID)
	if err != nil {
		return fmt.Errorf("update embedding repair: %w", err)
	}

	/* Log repair */
	d.logCorruptionEvent(ctx, issue.MemoryID, issue.Tier, "repaired", issue.IssueType)

	return nil
}

/* repairInvalidMetadata fixes invalid JSON metadata */
func (d *MemoryCorruptionDetector) repairInvalidMetadata(ctx context.Context, issue CorruptionIssue) error {
	var tableName string
	switch issue.Tier {
	case "stm":
		tableName = "memory_stm"
	case "mtm":
		tableName = "memory_mtm"
	case "lpm":
		tableName = "memory_lpm"
	default:
		return fmt.Errorf("invalid tier: %s", issue.Tier)
	}

	/* Set metadata to empty JSON object */
	updateQuery := fmt.Sprintf(`UPDATE neurondb_agent.%s SET metadata = '{}'::jsonb, updated_at = NOW() WHERE id = $1`, tableName)
	_, err := d.db.DB.ExecContext(ctx, updateQuery, issue.MemoryID)
	if err != nil {
		return fmt.Errorf("repair invalid metadata: %w", err)
	}

	d.logCorruptionEvent(ctx, issue.MemoryID, issue.Tier, "repaired", issue.IssueType)
	return nil
}

/* repairImportanceScore fixes out-of-range importance scores */
func (d *MemoryCorruptionDetector) repairImportanceScore(ctx context.Context, issue CorruptionIssue) error {
	var tableName string
	switch issue.Tier {
	case "stm":
		tableName = "memory_stm"
	case "mtm":
		tableName = "memory_mtm"
	case "lpm":
		tableName = "memory_lpm"
	default:
		return fmt.Errorf("invalid tier: %s", issue.Tier)
	}

	/* Clamp importance score to [0, 1] */
	updateQuery := fmt.Sprintf(`UPDATE neurondb_agent.%s 
		SET importance_score = GREATEST(0.0, LEAST(1.0, importance_score)), updated_at = NOW() 
		WHERE id = $1`, tableName)
	_, err := d.db.DB.ExecContext(ctx, updateQuery, issue.MemoryID)
	if err != nil {
		return fmt.Errorf("repair importance score: %w", err)
	}

	d.logCorruptionEvent(ctx, issue.MemoryID, issue.Tier, "repaired", issue.IssueType)
	return nil
}

/* logCorruptionEvent logs corruption detection/repair event */
func (d *MemoryCorruptionDetector) logCorruptionEvent(ctx context.Context, memoryID uuid.UUID, tier, action, issueType string) {
	query := `INSERT INTO neurondb_agent.memory_corruption_log
		(memory_id, tier, action, issue_type, created_at)
		VALUES ($1, $2, $3, $4, NOW())
		ON CONFLICT DO NOTHING`

	_, _ = d.db.DB.ExecContext(ctx, query, memoryID, tier, action, issueType)
}

/* ValidateAndRepair validates and repairs all memories for an agent */
func (d *MemoryCorruptionDetector) ValidateAndRepair(ctx context.Context, agentID uuid.UUID) (int, int, error) {
	/* Detect issues */
	issues, err := d.ValidateMemoryIntegrity(ctx, agentID)
	if err != nil {
		return 0, 0, err
	}

	conflicts, err := d.DetectConflicts(ctx, agentID)
	if err != nil {
		return 0, 0, err
	}

	allIssues := append(issues, conflicts...)

	/* Repair repairable issues */
	repaired := 0
	for _, issue := range allIssues {
		if issue.Repairable {
			if err := d.RepairCorruptedMemory(ctx, issue); err == nil {
				repaired++
			}
		}
	}

	return len(allIssues), repaired, nil
}
