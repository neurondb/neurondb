/*-------------------------------------------------------------------------
 *
 * context.go
 *    Context loading and management for NeuronAgent
 *
 * Provides context loading functionality that combines message history
 * and memory chunks to build comprehensive context for agent execution.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/context.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"context"
	"fmt"
	"strings"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/metrics"
)

type Context struct {
	Messages     []db.Message
	MemoryChunks []MemoryChunk
}

type ContextLoader struct {
	queries          *db.Queries
	memory           *MemoryManager
	llm              *LLMClient
	relevanceChecker *RelevanceChecker
	knowledgeRouter  *KnowledgeRouter
	retrievalAdapter *RetrievalAdapter
	retrievalLearning *RetrievalLearningManager
}

func NewContextLoader(queries *db.Queries, memory *MemoryManager, llm *LLMClient) *ContextLoader {
	return &ContextLoader{
		queries: queries,
		memory:  memory,
		llm:     llm,
	}
}

/* NewContextLoaderWithRetrieval creates a context loader with retrieval components */
func NewContextLoaderWithRetrieval(queries *db.Queries, memory *MemoryManager, llm *LLMClient, relevanceChecker *RelevanceChecker, knowledgeRouter *KnowledgeRouter, retrievalAdapter *RetrievalAdapter, retrievalLearning *RetrievalLearningManager) *ContextLoader {
	return &ContextLoader{
		queries:          queries,
		memory:           memory,
		llm:              llm,
		relevanceChecker: relevanceChecker,
		knowledgeRouter:  knowledgeRouter,
		retrievalAdapter: retrievalAdapter,
		retrievalLearning: retrievalLearning,
	}
}

func (l *ContextLoader) Load(ctx context.Context, sessionID uuid.UUID, agentID uuid.UUID, userMessage string, maxMessages int, maxMemoryChunks int) (*Context, error) {
	return l.LoadWithOptions(ctx, sessionID, agentID, userMessage, maxMessages, maxMemoryChunks, false)
}

/* LoadWithOptions loads context with configurable retrieval options */
func (l *ContextLoader) LoadWithOptions(ctx context.Context, sessionID uuid.UUID, agentID uuid.UUID, userMessage string, maxMessages int, maxMemoryChunks int, skipAutoRetrieval bool) (*Context, error) {
	/* Load recent messages */
	messages, err := l.queries.GetRecentMessages(ctx, sessionID, maxMessages)
	if err != nil {
		return nil, fmt.Errorf("context loading failed (load messages): session_id='%s', agent_id='%s', user_message_length=%d, max_messages=%d, error=%w",
			sessionID.String(), agentID.String(), len(userMessage), maxMessages, err)
	}

	/* Skip automatic retrieval if requested (for agentic retrieval mode) */
	var memoryChunks []MemoryChunk
	if !skipAutoRetrieval {
		/* Generate embedding for user message to search memory */
		embeddingModel := "all-MiniLM-L6-v2"
		embedding, err := l.llm.Embed(ctx, embeddingModel, userMessage)
		if err != nil {
			/* If embedding fails, continue without memory chunks but log the error */
			embedding = nil
			/* Note: We continue without memory chunks, but this is logged */
		}

		/* Retrieve relevant memory chunks */
		if embedding != nil {
			chunks, err := l.memory.Retrieve(ctx, agentID, embedding, maxMemoryChunks)
			if err != nil {
				return nil, fmt.Errorf("context loading failed (retrieve memory): session_id='%s', agent_id='%s', user_message_length=%d, embedding_model='%s', embedding_dimension=%d, max_memory_chunks=%d, message_count=%d, error=%w",
					sessionID.String(), agentID.String(), len(userMessage), embeddingModel, len(embedding), maxMemoryChunks, len(messages), err)
			}
			memoryChunks = chunks
		}
	} else if l.relevanceChecker != nil && l.knowledgeRouter != nil && l.retrievalAdapter != nil {
		/* Agentic retrieval mode: use intelligent retrieval decision */
		memoryChunks, err = l.loadWithAgenticRetrieval(ctx, agentID, userMessage, messages, maxMemoryChunks)
		if err != nil {
			/* Log error but continue with empty memory chunks */
			/* Error is logged but doesn't fail context loading */
		}
	}

	return &Context{
		Messages:     messages,
		MemoryChunks: memoryChunks,
	}, nil
}

/* loadWithAgenticRetrieval performs intelligent retrieval using relevance checker and knowledge router */
func (l *ContextLoader) loadWithAgenticRetrieval(ctx context.Context, agentID uuid.UUID, userMessage string, existingMessages []db.Message, maxMemoryChunks int) ([]MemoryChunk, error) {
	/* Build existing context string from messages */
	existingContext := make([]string, 0, len(existingMessages))
	for _, msg := range existingMessages {
		if msg.Content != "" {
			existingContext = append(existingContext, msg.Content)
		}
	}

	/* Step 1: Check if retrieval is needed */
	shouldRetrieve, confidence, reason, err := l.relevanceChecker.ShouldRetrieve(ctx, userMessage, strings.Join(existingContext, "\n"))
	if err != nil {
		/* If relevance check fails, default to retrieving */
		shouldRetrieve = true
		confidence = 0.5
		reason = "relevance_check_failed"
	}

	/* Record retrieval decision metrics */
	metrics.InfoWithContext(ctx, "Retrieval decision made", map[string]interface{}{
		"agent_id":        agentID.String(),
		"should_retrieve": shouldRetrieve,
		"confidence":      confidence,
		"reason":          reason,
		"query_length":    len(userMessage),
	})

	/* If retrieval not needed or confidence is very low, return empty */
	if !shouldRetrieve || confidence < 0.3 {
		return []MemoryChunk{}, nil
	}

	/* Step 2: Route query to determine best sources */
	sources, scores, err := l.knowledgeRouter.RouteQuery(ctx, userMessage)
	if err != nil {
		/* If routing fails, default to vector DB */
		sources = []string{"vector_db"}
		scores = map[string]float64{"vector_db": 0.7}
	}

	/* Record routing decision metrics */
	metrics.InfoWithContext(ctx, "Knowledge routing decision", map[string]interface{}{
		"agent_id":       agentID.String(),
		"sources":        sources,
		"source_scores":  scores,
		"query_length":   len(userMessage),
	})

	/* Record decision for learning if learning manager is available */
	if l.retrievalLearning != nil {
		decision := &RetrievalDecision{
			AgentID:       agentID,
			Query:         userMessage,
			ShouldRetrieve: shouldRetrieve,
			Confidence:    confidence,
			Reason:        reason,
			Sources:       sources,
			SourceScores:  scores,
		}
		decisionID, err := l.retrievalLearning.RecordDecision(ctx, decision)
		if err == nil {
			/* Store decision ID in context for later outcome recording */
			/* This will be used when we record outcomes */
		}
		_ = decisionID /* Use decisionID later for outcome tracking */
	}

	/* Step 3: Retrieve from recommended sources */
	allChunks := make([]MemoryChunk, 0)
	
	for _, source := range sources {
		if score, ok := scores[source]; !ok || score < 0.3 {
			continue /* Skip low-confidence sources */
		}

		switch source {
		case "vector_db":
			/* Retrieve from vector database using retrieval adapter */
			if l.retrievalAdapter != nil {
				results, err := l.retrievalAdapter.RetrieveFromVectorDB(ctx, agentID, userMessage, maxMemoryChunks)
				if err == nil {
					/* Convert results to MemoryChunks */
					for _, result := range results {
						content, _ := result["content"].(string)
						importance, _ := result["importance_score"].(float64)
						similarity, _ := result["similarity"].(float64)
						metadata, _ := result["metadata"].(map[string]interface{})
						if metadata == nil {
							metadata = make(map[string]interface{})
						}
						metadata["source"] = "vector_db"
						
						var id int64
						if idStr, ok := result["id"].(string); ok {
							/* Try to parse as int64 or UUID */
							if parsedUUID, err := uuid.Parse(idStr); err == nil {
								id = int64(parsedUUID[0]) << 32 | int64(parsedUUID[1])
							}
						}

						allChunks = append(allChunks, MemoryChunk{
							ID:              id,
							Content:         content,
							ImportanceScore: importance,
							Similarity:      similarity,
							Metadata:        metadata,
						})
					}
				}
			}
		case "web":
			/* Web retrieval would be handled by the agent calling the retrieval tool */
			/* For now, we skip automatic web retrieval in context loading */
			/* The agent can use the retrieval tool explicitly if needed */
		case "api":
			/* API retrieval would be handled by the agent calling the retrieval tool */
			/* For now, we skip automatic API retrieval in context loading */
		}
	}

	/* Step 4: Check relevance of retrieved chunks */
	if len(allChunks) > 0 && l.relevanceChecker != nil {
		contextStrings := make([]string, len(allChunks))
		for i, chunk := range allChunks {
			contextStrings[i] = chunk.Content
		}
		
		relevanceScore, isRelevant, err := l.relevanceChecker.CheckRelevance(ctx, userMessage, contextStrings)
		if err == nil && !isRelevant && relevanceScore < 0.4 {
			/* If retrieved content is not relevant, return empty */
			return []MemoryChunk{}, nil
		}
	}

	/* Limit to maxMemoryChunks, sorted by similarity/importance */
	if len(allChunks) > maxMemoryChunks {
		/* Sort by combined score (similarity * importance) */
		/* Simple selection: take top maxMemoryChunks */
		allChunks = allChunks[:maxMemoryChunks]
	}

	return allChunks, nil
}

/* LoadWithRetrievalDecision loads context and uses retrieval decision tool */
func (l *ContextLoader) LoadWithRetrievalDecision(ctx context.Context, sessionID uuid.UUID, agentID uuid.UUID, userMessage string, maxMessages int, retrievalDecision interface{}) (*Context, error) {
	/* Load recent messages */
	messages, err := l.queries.GetRecentMessages(ctx, sessionID, maxMessages)
	if err != nil {
		return nil, fmt.Errorf("context loading failed (load messages): session_id='%s', agent_id='%s', user_message_length=%d, max_messages=%d, error=%w",
			sessionID.String(), agentID.String(), len(userMessage), maxMessages, err)
	}

	/* Apply retrieval decision if provided */
	var memoryChunks []MemoryChunk
	if retrievalDecision != nil {
		/* This would be implemented based on retrieval decision structure */
		/* For now, skip automatic retrieval when using retrieval decision */
	}

	return &Context{
		Messages:     messages,
		MemoryChunks: memoryChunks,
	}, nil
}

/* CompressContext reduces context size by summarizing or removing less important messages */
func CompressContext(ctx *Context, maxTokens int) *Context {
	/* Count tokens in current context */
	totalTokens := 0
	for _, msg := range ctx.Messages {
		totalTokens += EstimateTokens(msg.Content)
	}

	/* If within limit, return as is */
	if totalTokens <= maxTokens {
		return ctx
	}

	/* Strategy: Keep system messages, recent messages, and important memory chunks */
	compressed := &Context{
		Messages:     []db.Message{},
		MemoryChunks: []MemoryChunk{},
	}

	/* Keep all memory chunks (they're already filtered) */
	compressed.MemoryChunks = ctx.MemoryChunks
	memoryTokens := 0
	for _, chunk := range ctx.MemoryChunks {
		memoryTokens += EstimateTokens(chunk.Content)
	}

	availableTokens := maxTokens - memoryTokens
	if availableTokens < 100 {
		/* Not enough space, return minimal context */
		return compressed
	}

	/* Keep messages from most recent, up to token limit */
	tokensUsed := 0
	for i := len(ctx.Messages) - 1; i >= 0; i-- {
		msg := ctx.Messages[i]
		msgTokens := EstimateTokens(msg.Content)

		if tokensUsed+msgTokens > availableTokens {
			break
		}

		/* Prepend to maintain order */
		compressed.Messages = append([]db.Message{msg}, compressed.Messages...)
		tokensUsed += msgTokens
	}

	return compressed
}
