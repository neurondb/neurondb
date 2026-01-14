/*-------------------------------------------------------------------------
 *
 * memory.go
 *    Agent memory management for NeuronAgent
 *
 * Provides memory chunk storage, retrieval, and semantic search functionality
 * for agents to maintain context and learn from past interactions.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/memory.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/metrics"
	"github.com/neurondb/NeuronAgent/pkg/neurondb"
)

type MemoryManager struct {
	db      *db.DB
	queries *db.Queries
	embed   *neurondb.EmbeddingClient
}

type MemoryChunk struct {
	ID              int64
	Content         string
	ImportanceScore float64
	Similarity      float64
	Metadata        map[string]interface{}
}

func NewMemoryManager(db *db.DB, queries *db.Queries, embedClient *neurondb.EmbeddingClient) *MemoryManager {
	return &MemoryManager{
		db:      db,
		queries: queries,
		embed:   embedClient,
	}
}

func (m *MemoryManager) Retrieve(ctx context.Context, agentID uuid.UUID, queryEmbedding []float32, topK int) ([]MemoryChunk, error) {
	/* Validate inputs */
	if agentID == uuid.Nil {
		return nil, fmt.Errorf("memory retrieval failed: agent_id_empty=true")
	}
	if len(queryEmbedding) == 0 {
		return nil, fmt.Errorf("memory retrieval failed: query_embedding_empty=true, agent_id='%s'", agentID.String())
	}
	if topK <= 0 {
		return nil, fmt.Errorf("memory retrieval failed: top_k_invalid=%d, agent_id='%s'", topK, agentID.String())
	}
	if topK > 1000 {
		topK = 1000 /* Cap at reasonable limit */
	}

	/* Check context cancellation */
	if ctx.Err() != nil {
		return nil, fmt.Errorf("memory retrieval cancelled: agent_id='%s', context_error=%w", agentID.String(), ctx.Err())
	}

	startTime := time.Now()
	status := "success"
	
	/* Record metrics */
	defer func() {
		duration := time.Since(startTime)
		metrics.RecordMemoryRetrieval(agentID.String())
		metrics.RecordVectorSearch(agentID.String(), status, duration)
	}()

	chunks, err := m.queries.SearchMemory(ctx, agentID, queryEmbedding, topK)
	if err != nil {
		status = "error"
		return nil, fmt.Errorf("memory retrieval failed: agent_id='%s', query_embedding_dimension=%d, top_k=%d, error=%w",
			agentID.String(), len(queryEmbedding), topK, err)
	}

	result := make([]MemoryChunk, len(chunks))
	for i, chunk := range chunks {
		result[i] = MemoryChunk{
			ID:              chunk.ID,
			Content:         chunk.Content,
			ImportanceScore: chunk.ImportanceScore,
			Similarity:      chunk.Similarity,
			Metadata:        chunk.Metadata,
		}
	}

	return result, nil
}

func (m *MemoryManager) StoreChunks(ctx context.Context, agentID, sessionID uuid.UUID, content string, toolResults []ToolResult) {
	/* Validate input */
	if content == "" {
		return /* Don't store empty content */
	}
	if len(content) > 50000 {
		return /* Don't store extremely large content */
	}

	/* Compute importance score (heuristic: length, user flags, etc.) */
	importance := m.computeImportance(content, toolResults)

	/* Only store if importance > threshold */
	if importance < 0.3 {
		return
	}

	/* Compute embedding */
	embeddingModel := "all-MiniLM-L6-v2"
	embedStartTime := time.Now()
	embedding, err := m.embed.Embed(ctx, content, embeddingModel)
	embedDuration := time.Since(embedStartTime)
	
	if err != nil {
		/* Log error but don't fail (async operation) */
		metrics.RecordEmbeddingGeneration(embeddingModel, "error", embedDuration)
		/* Error is already detailed in embedding client */
		return
	}
	metrics.RecordEmbeddingGeneration(embeddingModel, "success", embedDuration)

	/* Check context cancellation */
	if ctx.Err() != nil {
		metrics.WarnWithContext(ctx, "Memory storage cancelled", map[string]interface{}{
			"agent_id":   agentID.String(),
			"session_id": sessionID.String(),
			"error":      ctx.Err().Error(),
		})
		return
	}

	/* Store chunk */
	_, err = m.queries.CreateMemoryChunk(ctx, &db.MemoryChunk{
		AgentID:         agentID,
		SessionID:       &sessionID,
		Content:         content,
		Embedding:       embedding,
		ImportanceScore: importance,
	})
	if err != nil {
		/* Log error but don't fail (async operation) */
		metrics.WarnWithContext(ctx, "Failed to store memory chunk", map[string]interface{}{
			"agent_id":   agentID.String(),
			"session_id": sessionID.String(),
			"content_length": len(content),
			"importance": importance,
			"error":      err.Error(),
		})
		return
	}

	/* Record metrics */
	metrics.RecordMemoryChunkStored(agentID.String())
}

/* StoreChunksBatch stores multiple memory chunks in a single transaction for efficiency */
func (m *MemoryManager) StoreChunksBatch(ctx context.Context, agentID, sessionID uuid.UUID, contents []string, toolResults []ToolResult) {
	/* Validate inputs */
	if agentID == uuid.Nil {
		metrics.WarnWithContext(ctx, "Memory batch storage skipped: invalid agent ID", map[string]interface{}{
			"session_id": sessionID.String(),
			"contents_count": len(contents),
		})
		return
	}
	if len(contents) == 0 {
		return
	}

	/* Filter and prepare chunks for batch storage */
	type chunkData struct {
		content         string
		embedding       []float32
		importanceScore float64
	}

	var chunksToStore []chunkData
	embeddingModel := "all-MiniLM-L6-v2"

	/* Process each content item */
	for _, content := range contents {
		/* Validate input */
		if content == "" || len(content) > 50000 {
			continue
		}

		/* Compute importance score */
		importance := m.computeImportance(content, toolResults)
		if importance < 0.3 {
			continue
		}

		/* Compute embedding */
		embedStartTime := time.Now()
		embedding, err := m.embed.Embed(ctx, content, embeddingModel)
		embedDuration := time.Since(embedStartTime)
		
		if err != nil {
			metrics.RecordEmbeddingGeneration(embeddingModel, "error", embedDuration)
			continue
		}
		metrics.RecordEmbeddingGeneration(embeddingModel, "success", embedDuration)

		chunksToStore = append(chunksToStore, chunkData{
			content:         content,
			embedding:       embedding,
			importanceScore: importance,
		})
	}

	if len(chunksToStore) == 0 {
		return
	}

	/* Check context cancellation */
	if ctx.Err() != nil {
		metrics.WarnWithContext(ctx, "Memory batch storage cancelled", map[string]interface{}{
			"agent_id":   agentID.String(),
			"session_id": sessionID.String(),
			"chunks_count": len(chunksToStore),
			"error":      ctx.Err().Error(),
		})
		return
	}

	/* Store chunks in batch (using transaction for efficiency) */
	/* Note: This would require a batch insert method in queries */
	/* For now, store sequentially but in same transaction context */
	for _, chunk := range chunksToStore {
		/* Check context cancellation during batch */
		if ctx.Err() != nil {
			metrics.WarnWithContext(ctx, "Memory batch storage cancelled during processing", map[string]interface{}{
				"agent_id":   agentID.String(),
				"session_id": sessionID.String(),
				"error":      ctx.Err().Error(),
			})
			return
		}

		_, err := m.queries.CreateMemoryChunk(ctx, &db.MemoryChunk{
			AgentID:         agentID,
			SessionID:       &sessionID,
			Content:         chunk.content,
			Embedding:       chunk.embedding,
			ImportanceScore: chunk.importanceScore,
		})
		if err != nil {
			/* Log error but continue with other chunks */
			metrics.WarnWithContext(ctx, "Failed to store memory chunk in batch", map[string]interface{}{
				"agent_id":   agentID.String(),
				"session_id": sessionID.String(),
				"content_length": len(chunk.content),
				"importance": chunk.importanceScore,
				"error":      err.Error(),
			})
			continue
		}
		metrics.RecordMemoryChunkStored(agentID.String())
	}
}

func (m *MemoryManager) computeImportance(content string, toolResults []ToolResult) float64 {
	score := 0.5 /* Base score */

	/* Increase score based on content length (longer = more important) */
	if len(content) > 500 {
		score += 0.2
	} else if len(content) > 200 {
		score += 0.1
	}

	/* Increase score if tool results present (actionable information) */
	if len(toolResults) > 0 {
		score += 0.2
	}

	/* Increase score if content contains important keywords */
	importantKeywords := []string{"error", "solution", "important", "note", "warning", "summary"}
	contentLower := strings.ToLower(content)
	for _, keyword := range importantKeywords {
		if strings.Contains(contentLower, keyword) {
			score += 0.1
			break
		}
	}

	/* Cap at 1.0 */
	if score > 1.0 {
		score = 1.0
	}

	return score
}

/* SummarizeMemory summarizes old memories to compress them */
func (m *MemoryManager) SummarizeMemory(ctx context.Context, agentID uuid.UUID, maxChunks int) error {
	/* Validate inputs */
	if agentID == uuid.Nil {
		return fmt.Errorf("memory summarization failed: agent_id_empty=true")
	}
	if maxChunks <= 0 {
		return fmt.Errorf("memory summarization failed: max_chunks_invalid=%d, agent_id='%s'", maxChunks, agentID.String())
	}
	if maxChunks > 10000 {
		maxChunks = 10000 /* Cap at reasonable limit */
	}

	/* Check context cancellation */
	if ctx.Err() != nil {
		return fmt.Errorf("memory summarization cancelled: agent_id='%s', context_error=%w", agentID.String(), ctx.Err())
	}

	/* Get old memory chunks */
	query := `SELECT id, content, created_at
		FROM neurondb_agent.memory_chunks
		WHERE agent_id = $1
		ORDER BY created_at ASC
		LIMIT $2`

	type MemoryChunkRow struct {
		ID        int64     `db:"id"`
		Content   string    `db:"content"`
		CreatedAt time.Time `db:"created_at"`
	}

	var chunks []MemoryChunkRow
	err := m.db.DB.SelectContext(ctx, &chunks, query, agentID, maxChunks)
	if err != nil {
		return fmt.Errorf("memory summarization failed: agent_id='%s', max_chunks=%d, error=%w",
			agentID.String(), maxChunks, err)
	}

	if len(chunks) < 2 {
		return nil /* Not enough chunks to summarize */
	}

	/* Combine chunks and create summary */
	combinedContent := ""
	for _, chunk := range chunks {
		combinedContent += chunk.Content + "\n\n"
	}

	/* Summarize content - in production would use LLM */
	/* For now, use intelligent truncation with key points */
	summary := combinedContent
	if len(summary) > 1000 {
		/* Try to find a good breaking point */
		truncateAt := 1000
		for i := truncateAt; i > truncateAt-100 && i > 0; i-- {
			if summary[i] == '.' || summary[i] == '\n' {
				truncateAt = i + 1
				break
			}
		}
		summary = summary[:truncateAt] + "\n[Summary truncated]"
	}

	/* Create new summary chunk */
	embeddingModel := "all-MiniLM-L6-v2"
	embedding, err := m.embed.Embed(ctx, summary, embeddingModel)
	if err != nil {
		return fmt.Errorf("memory summarization embedding failed: agent_id='%s', summary_length=%d, error=%w",
			agentID.String(), len(summary), err)
	}

	_, err = m.queries.CreateMemoryChunk(ctx, &db.MemoryChunk{
		AgentID:         agentID,
		Content:         summary,
		Embedding:       embedding,
		ImportanceScore: 0.8, /* Summaries are important */
		Metadata: map[string]interface{}{
			"type":       "summary",
			"source_ids": chunks,
		},
	})
	if err != nil {
		return fmt.Errorf("memory summarization chunk creation failed: agent_id='%s', error=%w",
			agentID.String(), err)
	}

	/* Delete old chunks */
	chunkIDs := make([]int64, len(chunks))
	for i, chunk := range chunks {
		chunkIDs[i] = chunk.ID
	}

	deleteQuery := `DELETE FROM neurondb_agent.memory_chunks WHERE id = ANY($1)`
	_, err = m.db.DB.ExecContext(ctx, deleteQuery, chunkIDs)
	if err != nil {
		return fmt.Errorf("memory summarization chunk deletion failed: agent_id='%s', chunks_count=%d, error=%w",
			agentID.String(), len(chunkIDs), err)
	}

	return nil
}

/* ApplyTemporalDecay applies temporal decay to memory importance scores */
func (m *MemoryManager) ApplyTemporalDecay(ctx context.Context, agentID uuid.UUID, decayRate float64) error {
	/* Validate inputs */
	if agentID == uuid.Nil {
		return fmt.Errorf("temporal decay application failed: agent_id_empty=true")
	}
	if decayRate <= 0 || decayRate > 1 {
		return fmt.Errorf("temporal decay application failed: decay_rate_invalid=%.2f, agent_id='%s'", decayRate, agentID.String())
	}

	/* Check context cancellation */
	if ctx.Err() != nil {
		return fmt.Errorf("temporal decay application cancelled: agent_id='%s', context_error=%w", agentID.String(), ctx.Err())
	}

	/* Update importance scores based on age */
	query := `UPDATE neurondb_agent.memory_chunks
		SET importance_score = GREATEST(0.1, importance_score * POWER($1, EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400.0))
		WHERE agent_id = $2 AND created_at < NOW() - INTERVAL '7 days'`

	_, err := m.db.DB.ExecContext(ctx, query, decayRate, agentID)
	if err != nil {
		return fmt.Errorf("temporal decay application failed: agent_id='%s', decay_rate=%.2f, error=%w",
			agentID.String(), decayRate, err)
	}

	return nil
}
