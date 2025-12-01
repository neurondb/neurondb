package agent

import (
	"fmt"
	"strings"

	"github.com/neurondb/NeuronAgent/internal/db"
)

type PromptBuilder struct {
	maxTokens int
}

func NewPromptBuilder() *PromptBuilder {
	return &PromptBuilder{
		maxTokens: 4000, // Default max tokens
	}
}

func (p *PromptBuilder) SetMaxTokens(maxTokens int) {
	p.maxTokens = maxTokens
}

func (p *PromptBuilder) Build(agent *db.Agent, context *Context, userMessage string) (string, error) {
	var parts []string

	// System prompt
	parts = append(parts, agent.SystemPrompt)

	// Memory chunks
	if len(context.MemoryChunks) > 0 {
		parts = append(parts, "\n\n## Relevant Context:")
		for i, chunk := range context.MemoryChunks {
			parts = append(parts, fmt.Sprintf("\n[Context %d] %s", i+1, chunk.Content))
		}
	}

	// Conversation history
	if len(context.Messages) > 0 {
		parts = append(parts, "\n\n## Conversation History:")
		for _, msg := range context.Messages {
			role := strings.Title(msg.Role)
			parts = append(parts, fmt.Sprintf("\n%s: %s", role, msg.Content))
		}
	}

	// Current user message
	parts = append(parts, fmt.Sprintf("\n\n## Current Request:\nUser: %s", userMessage))
	parts = append(parts, "\n\nAssistant:")

	return strings.Join(parts, ""), nil
}

func (p *PromptBuilder) BuildWithToolResults(agent *db.Agent, context *Context, userMessage string, llmResponse *LLMResponse, toolResults []ToolResult) (string, error) {
	var parts []string

	// System prompt
	parts = append(parts, agent.SystemPrompt)

	// Memory chunks
	if len(context.MemoryChunks) > 0 {
		parts = append(parts, "\n\n## Relevant Context:")
		for i, chunk := range context.MemoryChunks {
			parts = append(parts, fmt.Sprintf("\n[Context %d] %s", i+1, chunk.Content))
		}
	}

	// Conversation history
	if len(context.Messages) > 0 {
		parts = append(parts, "\n\n## Conversation History:")
		for _, msg := range context.Messages {
			role := strings.Title(msg.Role)
			parts = append(parts, fmt.Sprintf("\n%s: %s", role, msg.Content))
		}
	}

	// Current user message
	parts = append(parts, fmt.Sprintf("\n\n## Current Request:\nUser: %s", userMessage))

	// Tool calls and results
	if len(llmResponse.ToolCalls) > 0 {
		parts = append(parts, "\n\n## Tool Calls:")
		for _, call := range llmResponse.ToolCalls {
			parts = append(parts, fmt.Sprintf("\nCalled: %s with args: %v", call.Name, call.Arguments))
		}

		parts = append(parts, "\n\n## Tool Results:")
		for _, result := range toolResults {
			if result.Error != nil {
				parts = append(parts, fmt.Sprintf("\nTool %s error: %v", result.ToolCallID, result.Error))
			} else {
				parts = append(parts, fmt.Sprintf("\nTool %s result: %s", result.ToolCallID, result.Content))
			}
		}
	}

	parts = append(parts, "\n\nAssistant:")

	return strings.Join(parts, ""), nil
}
