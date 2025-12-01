package agent

import (
	"context"

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
		return nil, err
	}

	// Generate embedding for user message to search memory
	embedding, err := l.llm.Embed(ctx, "all-MiniLM-L6-v2", userMessage)
	if err != nil {
		// If embedding fails, continue without memory chunks
		embedding = nil
	}

	// Retrieve relevant memory chunks
	var memoryChunks []MemoryChunk
	if embedding != nil {
		chunks, err := l.memory.Retrieve(ctx, agentID, embedding, maxMemoryChunks)
		if err == nil {
			memoryChunks = chunks
		}
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

