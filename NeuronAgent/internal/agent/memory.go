package agent

import (
	"context"
	"fmt"
	"strings"

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
	// Record metrics
	defer func() {
		metrics.RecordMemoryRetrieval(agentID.String())
	}()

	chunks, err := m.queries.SearchMemory(ctx, agentID, queryEmbedding, topK)
	if err != nil {
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
	// Compute importance score (heuristic: length, user flags, etc.)
	importance := m.computeImportance(content, toolResults)

	// Only store if importance > threshold
	if importance < 0.3 {
		return
	}

	// Compute embedding
	embeddingModel := "all-MiniLM-L6-v2"
	embedding, err := m.embed.Embed(ctx, content, embeddingModel)
	if err != nil {
		// Log error but don't fail (async operation)
		// Error is already detailed in embedding client
		return
	}

	// Store chunk
	_, err = m.queries.CreateMemoryChunk(ctx, &db.MemoryChunk{
		AgentID:         agentID,
		SessionID:       &sessionID,
		Content:         content,
		Embedding:       embedding,
		ImportanceScore: importance,
	})
	if err != nil {
		// Log error but don't fail (async operation)
		// Error is already detailed in queries.CreateMemoryChunk
		return
	}

	// Record metrics
	metrics.RecordMemoryChunkStored(agentID.String())
}

func (m *MemoryManager) computeImportance(content string, toolResults []ToolResult) float64 {
	score := 0.5 // Base score

	// Increase score based on content length (longer = more important)
	if len(content) > 500 {
		score += 0.2
	} else if len(content) > 200 {
		score += 0.1
	}

	// Increase score if tool results present (actionable information)
	if len(toolResults) > 0 {
		score += 0.2
	}

	// Increase score if content contains important keywords
	importantKeywords := []string{"error", "solution", "important", "note", "warning", "summary"}
	contentLower := strings.ToLower(content)
	for _, keyword := range importantKeywords {
		if strings.Contains(contentLower, keyword) {
			score += 0.1
			break
		}
	}

	// Cap at 1.0
	if score > 1.0 {
		score = 1.0
	}

	return score
}
