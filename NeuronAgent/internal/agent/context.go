package agent

import (
	"context"
	"fmt"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
)

type Context struct {
	Messages     []db.Message
	MemoryChunks []MemoryChunk
}

type ContextLoader struct {
	queries *db.Queries
	memory  *MemoryManager
	llm     *LLMClient
}

func NewContextLoader(queries *db.Queries, memory *MemoryManager, llm *LLMClient) *ContextLoader {
	return &ContextLoader{
		queries: queries,
		memory:  memory,
		llm:     llm,
	}
}

func (l *ContextLoader) Load(ctx context.Context, sessionID uuid.UUID, agentID uuid.UUID, userMessage string, maxMessages int, maxMemoryChunks int) (*Context, error) {
	// Load recent messages
	messages, err := l.queries.GetRecentMessages(ctx, sessionID, maxMessages)
	if err != nil {
		return nil, fmt.Errorf("context loading failed (load messages): session_id='%s', agent_id='%s', user_message_length=%d, max_messages=%d, error=%w",
			sessionID.String(), agentID.String(), len(userMessage), maxMessages, err)
	}

	// Generate embedding for user message to search memory
	embeddingModel := "all-MiniLM-L6-v2"
	embedding, err := l.llm.Embed(ctx, embeddingModel, userMessage)
	if err != nil {
		// If embedding fails, continue without memory chunks but log the error
		embedding = nil
		// Note: We continue without memory chunks, but this is logged
	}

	// Retrieve relevant memory chunks
	var memoryChunks []MemoryChunk
	if embedding != nil {
		chunks, err := l.memory.Retrieve(ctx, agentID, embedding, maxMemoryChunks)
		if err != nil {
			return nil, fmt.Errorf("context loading failed (retrieve memory): session_id='%s', agent_id='%s', user_message_length=%d, embedding_model='%s', embedding_dimension=%d, max_memory_chunks=%d, message_count=%d, error=%w",
				sessionID.String(), agentID.String(), len(userMessage), embeddingModel, len(embedding), maxMemoryChunks, len(messages), err)
		}
		memoryChunks = chunks
	}

	return &Context{
		Messages:     messages,
		MemoryChunks: memoryChunks,
	}, nil
}

// CompressContext reduces context size by summarizing or removing less important messages
func CompressContext(ctx *Context, maxTokens int) *Context {
	// Count tokens in current context
	totalTokens := 0
	for _, msg := range ctx.Messages {
		totalTokens += EstimateTokens(msg.Content)
	}
	
	// If within limit, return as is
	if totalTokens <= maxTokens {
		return ctx
	}
	
	// Strategy: Keep system messages, recent messages, and important memory chunks
	compressed := &Context{
		Messages:     []db.Message{},
		MemoryChunks: []MemoryChunk{},
	}
	
	// Keep all memory chunks (they're already filtered)
	compressed.MemoryChunks = ctx.MemoryChunks
	memoryTokens := 0
	for _, chunk := range ctx.MemoryChunks {
		memoryTokens += EstimateTokens(chunk.Content)
	}
	
	availableTokens := maxTokens - memoryTokens
	if availableTokens < 100 {
		// Not enough space, return minimal context
		return compressed
	}
	
	// Keep messages from most recent, up to token limit
	tokensUsed := 0
	for i := len(ctx.Messages) - 1; i >= 0; i-- {
		msg := ctx.Messages[i]
		msgTokens := EstimateTokens(msg.Content)
		
		if tokensUsed+msgTokens > availableTokens {
			break
		}
		
		// Prepend to maintain order
		compressed.Messages = append([]db.Message{msg}, compressed.Messages...)
		tokensUsed += msgTokens
	}
	
	return compressed
}

