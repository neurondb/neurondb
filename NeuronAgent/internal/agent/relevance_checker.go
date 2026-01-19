/*-------------------------------------------------------------------------
 *
 * relevance_checker.go
 *    Context relevance checker for retrieval decisions
 *
 * Evaluates whether retrieved context is likely to be relevant before
 * performing expensive retrieval operations.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/relevance_checker.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"context"
	"fmt"
	"math"
	"strings"

	"github.com/neurondb/NeuronAgent/internal/metrics"
	"github.com/neurondb/NeuronAgent/pkg/neurondb"
)

/* RelevanceChecker checks if context is likely to be relevant */
type RelevanceChecker struct {
	llm   *LLMClient
	embed *neurondb.EmbeddingClient
}

/* NewRelevanceChecker creates a new relevance checker */
func NewRelevanceChecker(llm *LLMClient, embed *neurondb.EmbeddingClient) *RelevanceChecker {
	return &RelevanceChecker{
		llm:   llm,
		embed: embed,
	}
}

/* CheckRelevance checks if retrieval is likely to be relevant */
func (r *RelevanceChecker) CheckRelevance(ctx context.Context, query string, existingContext []string) (float64, bool, error) {
	/* If no existing context, retrieval is likely relevant */
	if len(existingContext) == 0 {
		return 1.0, true, nil
	}

	/* Method 1: Embedding-based similarity check */
	embeddingScore, err := r.checkRelevanceEmbedding(ctx, query, existingContext)
	if err != nil {
		/* Fallback to LLM-based check */
		llmScore, llmErr := r.checkRelevanceLLM(ctx, query, existingContext)
		if llmErr != nil {
			return 0.5, true, llmErr
		}
		isRelevant := llmScore > 0.5
		return llmScore, isRelevant, nil
	}

	/* Method 2: LLM-based relevance check */
	llmScore, err := r.checkRelevanceLLM(ctx, query, existingContext)
	if err != nil {
		/* Use embedding score only */
		isRelevant := embeddingScore > 0.5
		return embeddingScore, isRelevant, nil
	}

	/* Combine scores (weighted average) */
	combinedScore := (embeddingScore*0.6 + llmScore*0.4)
	isRelevant := combinedScore > 0.5

	return combinedScore, isRelevant, nil
}

/* checkRelevanceEmbedding uses embedding similarity to check relevance */
func (r *RelevanceChecker) checkRelevanceEmbedding(ctx context.Context, query string, existingContext []string) (float64, error) {
	/* Generate query embedding */
	queryEmbedding, err := r.embed.Embed(ctx, query, "all-MiniLM-L6-v2")
	if err != nil {
		return 0.0, err
	}

	/* Check similarity with existing context */
	maxSimilarity := 0.0
	for _, ctxStr := range existingContext {
		if ctxStr == "" {
			continue
		}

		ctxEmbedding, err := r.embed.Embed(ctx, ctxStr, "all-MiniLM-L6-v2")
		if err != nil {
			continue
		}

		similarity := cosineSimilarity(queryEmbedding, ctxEmbedding)
		if similarity > maxSimilarity {
			maxSimilarity = similarity
		}
	}

	/* If max similarity is high, existing context might be sufficient */
	/* Lower similarity means retrieval is more likely to be relevant */
	relevanceScore := 1.0 - maxSimilarity

	return relevanceScore, nil
}

/* checkRelevanceLLM uses LLM to check if retrieval is needed */
func (r *RelevanceChecker) checkRelevanceLLM(ctx context.Context, query string, existingContext []string) (float64, error) {
	if r.llm == nil {
		/* Fallback to embedding-based if LLM not available */
		return 0.6, nil
	}

	/* Build context string with length limit */
	contextStr := ""
	maxContextLength := 2000 /* Limit context to avoid token limits */
	for i, ctx := range existingContext {
		if i > 0 {
			contextStr += "\n"
		}
		if len(contextStr)+len(ctx) > maxContextLength {
			/* Truncate if too long */
			remaining := maxContextLength - len(contextStr)
			if remaining > 100 {
				contextStr += ctx[:remaining] + "..."
			}
			break
		}
		contextStr += ctx
	}

	/* Enhanced prompt with examples */
	prompt := fmt.Sprintf(`Analyze whether the given query requires additional information retrieval beyond the existing context.

Query: "%s"

Existing Context:
%s

Determine if retrieval is needed by considering:
1. Does the context fully answer the query?
2. Is the information in context current/accurate enough?
3. Does the query ask for information not in context?

Respond with ONLY a number from 0.0 to 1.0:
- 1.0 = Retrieval definitely needed (query not covered, needs current info, or requires external knowledge)
- 0.5 = Partial coverage, retrieval might help
- 0.0 = Retrieval not needed (context fully covers query)

Examples:
- Query: "What did I tell you yesterday?" with context about yesterday → 0.0
- Query: "What's the weather today?" with no weather context → 1.0
- Query: "Explain quantum computing" with partial explanation → 0.6

Score:`, query, contextStr)

	llmConfig := map[string]interface{}{
		"temperature": 0.1,
		"max_tokens":  20,
	}

	response, err := r.llm.Generate(ctx, "gpt-4", prompt, llmConfig)
	if err != nil {
		/* Fallback to neutral on error */
		metrics.WarnWithContext(ctx, "LLM relevance check failed, using fallback", map[string]interface{}{
			"query": query,
			"error": err.Error(),
		})
		return 0.5, nil
	}

	/* Extract number from response (handle various formats) */
	responseContent := strings.TrimSpace(response.Content)
	
	/* Try to find first number in response */
	var score float64
	_, err = fmt.Sscanf(responseContent, "%f", &score)
	if err != nil {
		/* Try to extract number from text like "Score: 0.75" */
		parts := strings.Fields(responseContent)
		for _, part := range parts {
			if _, scanErr := fmt.Sscanf(part, "%f", &score); scanErr == nil {
				break
			}
		}
		if err != nil {
			/* Default to neutral if parsing fails */
			return 0.5, nil
		}
	}

	/* Clamp to [0, 1] */
	score = math.Max(0.0, math.Min(1.0, score))

	return score, nil
}

/* cosineSimilarity calculates cosine similarity between two embeddings */
func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0.0
	}

	var dotProduct float64
	var normA, normB float64

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

/* ShouldRetrieve determines if retrieval should be performed */
func (r *RelevanceChecker) ShouldRetrieve(ctx context.Context, query string, context string) (bool, float64, string, error) {
	existingContext := []string{}
	if context != "" {
		existingContext = []string{context}
	}

	relevanceScore, isRelevant, err := r.CheckRelevance(ctx, query, existingContext)
	if err != nil {
		return true, 0.5, "error_checking_relevance", err
	}

	reason := "context_sufficient"
	if isRelevant {
		reason = "context_insufficient"
	}

	/* Record metrics */
	metrics.InfoWithContext(ctx, "Relevance check completed", map[string]interface{}{
		"query":           query,
		"relevance_score": relevanceScore,
		"is_relevant":     isRelevant,
		"reason":          reason,
	})

	return isRelevant, relevanceScore, reason, nil
}
