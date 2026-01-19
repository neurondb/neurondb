/*-------------------------------------------------------------------------
 *
 * retrieval_adapter.go
 *    Adapter for retrieval tool interfaces
 *
 * Bridges agent components with tool interfaces to enable retrieval tool
 * functionality without circular dependencies.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/retrieval_adapter.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"context"
	"fmt"

	"github.com/google/uuid"
)

/* RetrievalAdapter adapts agent components to RetrievalInterface */
type RetrievalAdapter struct {
	memory           *MemoryManager
	hierMemory       *HierarchicalMemoryManager
	relevanceChecker *RelevanceChecker
}

/* NewRetrievalAdapter creates a new retrieval adapter */
func NewRetrievalAdapter(memory *MemoryManager, hierMemory *HierarchicalMemoryManager, relevanceChecker *RelevanceChecker) *RetrievalAdapter {
	return &RetrievalAdapter{
		memory:           memory,
		hierMemory:       hierMemory,
		relevanceChecker: relevanceChecker,
	}
}

/* RetrieveFromVectorDB retrieves from vector database */
func (a *RetrievalAdapter) RetrieveFromVectorDB(ctx context.Context, agentID uuid.UUID, query string, topK int) ([]map[string]interface{}, error) {
	if a.memory == nil {
		return nil, fmt.Errorf("memory manager not available")
	}

	/* Validate inputs */
	if query == "" {
		return nil, fmt.Errorf("query cannot be empty")
	}
	if topK <= 0 {
		topK = 5
	}
	if topK > 100 {
		topK = 100 /* Cap at reasonable limit */
	}

	/* Try hierarchical memory first if available (more comprehensive) */
	if a.hierMemory != nil {
		hierResults, err := a.hierMemory.RetrieveHierarchical(ctx, agentID, query, []string{"stm", "mtm", "lpm"}, topK)
		if err == nil && len(hierResults) > 0 {
			/* Convert hierarchical results to map format */
			results := make([]map[string]interface{}, 0, len(hierResults))
			for _, chunkMap := range hierResults {
				result := make(map[string]interface{})
				if id, ok := chunkMap["id"].(string); ok {
					result["id"] = id
				} else if id, ok := chunkMap["id"].(uuid.UUID); ok {
					result["id"] = id.String()
				}
				if content, ok := chunkMap["content"].(string); ok {
					result["content"] = content
				}
				if importance, ok := chunkMap["importance_score"].(float64); ok {
					result["importance_score"] = importance
				}
				if similarity, ok := chunkMap["similarity"].(float64); ok {
					result["similarity"] = similarity
				}
				if tier, ok := chunkMap["tier"].(string); ok {
					result["tier"] = tier
					result["source"] = "hierarchical_memory"
				}
				if metadata, ok := chunkMap["metadata"].(map[string]interface{}); ok {
					result["metadata"] = metadata
				}
				results = append(results, result)
			}
			return results, nil
		}
	}

	/* Fallback to standard memory manager */
	embedding, err := a.memory.embed.Embed(ctx, query, "all-MiniLM-L6-v2")
	if err != nil {
		return nil, fmt.Errorf("embedding generation failed: %w", err)
	}

	/* Retrieve using MemoryManager */
	chunks, err := a.memory.Retrieve(ctx, agentID, embedding, topK)
	if err != nil {
		return nil, fmt.Errorf("memory retrieval failed: %w", err)
	}

	/* Convert to map format */
	results := make([]map[string]interface{}, len(chunks))
	for i, chunk := range chunks {
		results[i] = map[string]interface{}{
			"id":              chunk.ID,
			"content":         chunk.Content,
			"importance_score": chunk.ImportanceScore,
			"similarity":      chunk.Similarity,
			"metadata":        chunk.Metadata,
			"source":          "vector_db",
		}
	}

	return results, nil
}

/* ShouldRetrieve determines if retrieval is needed */
func (a *RetrievalAdapter) ShouldRetrieve(ctx context.Context, query string, context string) (bool, float64, string, error) {
	if a.relevanceChecker == nil {
		/* Default: always retrieve if no checker available */
		return true, 0.8, "no_relevance_checker", nil
	}

	return a.relevanceChecker.ShouldRetrieve(ctx, query, context)
}

/* CheckRelevance checks if context is relevant */
func (a *RetrievalAdapter) CheckRelevance(ctx context.Context, query string, existingContext []string) (float64, bool, error) {
	if a.relevanceChecker == nil {
		/* Default: assume relevant if no checker */
		return 0.7, true, nil
	}

	return a.relevanceChecker.CheckRelevance(ctx, query, existingContext)
}
