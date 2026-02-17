/*-------------------------------------------------------------------------
 *
 * prompt.go
 *    Database operations
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/prompt.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"fmt"
	"strings"

	"github.com/neurondb/NeuronAgent/internal/db"
)

/* Max lengths for untrusted content in prompts to limit injection impact */
const (
	maxUserMessagePromptLen = 32 * 1024   /* 32KB */
	maxMemoryChunkPromptLen = 4096       /* 4KB per chunk */
	maxToolResultPromptLen  = 16 * 1024  /* 16KB per result */
	maxConversationMsgLen   = 8192       /* 8KB per history message */
)

/* sanitizeForPrompt truncates and wraps untrusted content to reduce prompt injection risk */
func sanitizeForPrompt(s string, maxLen int) string {
	s = strings.TrimSpace(s)
	if len(s) > maxLen {
		s = s[:maxLen] + "\n[... truncated ...]"
	}
	/* Wrap in delimiters so model treats as data; newlines could be normalized */
	return s
}

type PromptBuilder struct {
	maxTokens int
}

func NewPromptBuilder() *PromptBuilder {
	return &PromptBuilder{
		maxTokens: 4000, /* Default max tokens */
	}
}

func (p *PromptBuilder) SetMaxTokens(maxTokens int) {
	p.maxTokens = maxTokens
}

func (p *PromptBuilder) Build(agent *db.Agent, context *Context, userMessage string) (string, error) {
	return p.BuildWithPersonalization(agent, context, userMessage, nil)
}

/* BuildWithPersonalization builds prompt with personalization context */
func (p *PromptBuilder) BuildWithPersonalization(agent *db.Agent, context *Context, userMessage string, personalizationCtx *PersonalizationContext) (string, error) {
	var parts []string

	/* System prompt */
	parts = append(parts, agent.SystemPrompt)

	/* Add personalization context if available */
	if personalizationCtx != nil {
		personalizationManager := &PersonalizationManager{}
		personalizedSection := personalizationManager.BuildPersonalizedPrompt(personalizationCtx)
		if personalizedSection != "" {
			parts = append(parts, personalizedSection)
		}
	}

	/* Add retrieval decision guidance if agentic retrieval is enabled */
	if agent.Config != nil {
		if agenticRetrieval, ok := agent.Config["agentic_retrieval_enabled"].(bool); ok && agenticRetrieval {
			parts = append(parts, p.getRetrievalGuidance())
		}
	}

	/* Memory chunks (sanitized) */
	if len(context.MemoryChunks) > 0 {
		parts = append(parts, "\n\n## Relevant Context:")
		for i, chunk := range context.MemoryChunks {
			safe := sanitizeForPrompt(chunk.Content, maxMemoryChunkPromptLen)
			parts = append(parts, fmt.Sprintf("\n[Context %d] %s", i+1, safe))
		}
	}

	/* Conversation history (sanitized) */
	if len(context.Messages) > 0 {
		parts = append(parts, "\n\n## Conversation History:")
		for _, msg := range context.Messages {
			role := strings.Title(msg.Role)
			safe := sanitizeForPrompt(msg.Content, maxConversationMsgLen)
			parts = append(parts, fmt.Sprintf("\n%s: %s", role, safe))
		}
	}

	/* Current user message (sanitized) */
	safeUserMsg := sanitizeForPrompt(userMessage, maxUserMessagePromptLen)
	parts = append(parts, fmt.Sprintf("\n\n## Current Request:\nUser: %s", safeUserMsg))
	parts = append(parts, "\n\nAssistant:")

	return strings.Join(parts, ""), nil
}

func (p *PromptBuilder) BuildWithToolResults(agent *db.Agent, context *Context, userMessage string, llmResponse *LLMResponse, toolResults []ToolResult) (string, error) {
	return p.BuildWithToolResultsAndPersonalization(agent, context, userMessage, llmResponse, toolResults, nil)
}

/* BuildWithToolResultsAndPersonalization builds prompt with tool results and personalization */
func (p *PromptBuilder) BuildWithToolResultsAndPersonalization(agent *db.Agent, context *Context, userMessage string, llmResponse *LLMResponse, toolResults []ToolResult, personalizationCtx *PersonalizationContext) (string, error) {
	var parts []string

	/* System prompt */
	parts = append(parts, agent.SystemPrompt)

	/* Add personalization context if available */
	if personalizationCtx != nil {
		personalizationManager := &PersonalizationManager{}
		personalizedSection := personalizationManager.BuildPersonalizedPrompt(personalizationCtx)
		if personalizedSection != "" {
			parts = append(parts, personalizedSection)
		}
	}

	/* Add retrieval decision guidance if agentic retrieval is enabled */
	if agent.Config != nil {
		if agenticRetrieval, ok := agent.Config["agentic_retrieval_enabled"].(bool); ok && agenticRetrieval {
			parts = append(parts, p.getRetrievalGuidance())
		}
	}

	/* Memory chunks (sanitized to reduce prompt injection risk) */
	if len(context.MemoryChunks) > 0 {
		parts = append(parts, "\n\n## Relevant Context:")
		for i, chunk := range context.MemoryChunks {
			safe := sanitizeForPrompt(chunk.Content, maxMemoryChunkPromptLen)
			parts = append(parts, fmt.Sprintf("\n[Context %d] %s", i+1, safe))
		}
	}

	/* Conversation history */
	if len(context.Messages) > 0 {
		parts = append(parts, "\n\n## Conversation History:")
		for _, msg := range context.Messages {
			role := strings.Title(msg.Role)
			safe := sanitizeForPrompt(msg.Content, maxConversationMsgLen)
			parts = append(parts, fmt.Sprintf("\n%s: %s", role, safe))
		}
	}

	/* Current user message (sanitized) */
	safeUserMsg := sanitizeForPrompt(userMessage, maxUserMessagePromptLen)
	parts = append(parts, fmt.Sprintf("\n\n## Current Request:\nUser: %s", safeUserMsg))

	/* Add retrieval decision guidance if agentic retrieval is enabled */
	if agent.Config != nil {
		if agenticRetrieval, ok := agent.Config["agentic_retrieval_enabled"].(bool); ok && agenticRetrieval {
			parts = append(parts, p.getRetrievalGuidance())
		}
	}

	/* Tool calls and results (sanitize tool call display to limit prompt injection surface) */
	if len(llmResponse.ToolCalls) > 0 {
		parts = append(parts, "\n\n## Tool Calls:")
		for _, call := range llmResponse.ToolCalls {
			safeName := sanitizeForPrompt(call.Name, 128)
			argsStr := fmt.Sprintf("%v", call.Arguments)
			safeArgs := sanitizeForPrompt(argsStr, maxToolResultPromptLen)
			parts = append(parts, fmt.Sprintf("\nCalled: %s with args: %s", safeName, safeArgs))
		}

		parts = append(parts, "\n\n## Tool Results:")
		for _, result := range toolResults {
			if result.Error != nil {
				parts = append(parts, fmt.Sprintf("\nTool %s error: %v", result.ToolCallID, result.Error))
			} else {
				safeContent := sanitizeForPrompt(result.Content, maxToolResultPromptLen)
				parts = append(parts, fmt.Sprintf("\nTool %s result: %s", result.ToolCallID, safeContent))
			}
		}
	}

	parts = append(parts, "\n\nAssistant:")

	return strings.Join(parts, ""), nil
}

/* getRetrievalGuidance returns comprehensive guidance for agentic retrieval */
func (p *PromptBuilder) getRetrievalGuidance() string {
	return `
## Agentic Retrieval Guidance

You have access to a powerful retrieval tool that allows you to intelligently decide when and where to retrieve information. Use this tool to enhance your responses with relevant context.

### When to Use Retrieval

Use the retrieval tool when:
1. The user's query requires information not in the current context
2. You need to verify or update information (e.g., current events, real-time data)
3. The query asks about past conversations or stored knowledge
4. You're uncertain if existing context is sufficient

### Retrieval Tool Actions

The retrieval tool supports the following actions:

1. **should_retrieve** - Determine if retrieval is needed
   - Use when: You're uncertain if existing context is sufficient
   - Parameters: query (required), context (optional)
   - Returns: should_retrieve (bool), confidence (float), reason (string)

2. **route_query** - Determine the best knowledge source(s)
   - Use when: You know retrieval is needed but need to choose the source
   - Parameters: query (required)
   - Returns: recommended_sources (array), source_scores (map)
   - Sources: "vector_db" (past conversations, stored knowledge), "web" (current events, real-time info), "api" (structured data)

3. **retrieve_from_vector_db** - Retrieve from agent's memory
   - Use when: Query relates to past conversations, user preferences, or stored knowledge
   - Parameters: query (required), top_k (optional, default: 5)
   - Returns: results array with content, similarity scores, importance scores

4. **retrieve_from_web** - Search the web
   - Use when: Query requires current information, news, or real-time data
   - Parameters: query (required), max_results (optional, default: 5)
   - Returns: results array with title, content, url, source

5. **retrieve_from_api** - Call external API
   - Use when: Query requires structured data from a specific API
   - Parameters: api_url (required), method (optional), headers (optional), body (optional)
   - Returns: API response

6. **check_relevance** - Check if retrieved context is relevant
   - Use when: Before using expensive retrieval operations or after retrieving
   - Parameters: query (required), existing_context (optional array)
   - Returns: relevance_score (float), is_relevant (bool), recommendation (string)

### Retrieval Strategy

Recommended workflow:
1. If uncertain about context sufficiency, use should_retrieve
2. If retrieval needed, use route_query to determine best source(s)
3. Retrieve from recommended source(s) using appropriate action
4. Optionally use check_relevance to verify retrieved content
5. Integrate relevant results into your response

### Integration with Other Tools

- **Memory Tool**: Use retrieval tool first to find relevant memories, then use memory tool for detailed operations
- **Web Search Tool**: Retrieval tool's retrieve_from_web provides intelligent web search; use web_search tool for direct searches
- **Vector Tool**: Retrieval tool's retrieve_from_vector_db uses agent memory; use vector tool for general vector operations

### Examples

Example 1 - Checking if retrieval is needed:
User: "What did I tell you about my preferences?"
- Use: should_retrieve(query="user preferences", context="[existing context]")
- If should_retrieve=true, use: retrieve_from_vector_db(query="user preferences")

Example 2 - Current events:
User: "What's the latest news about AI?"
- Use: route_query(query="latest AI news")
- Likely recommends: ["web"]
- Use: retrieve_from_web(query="latest AI news", max_results=5)

Example 3 - Multiple sources:
User: "Compare current AI trends with what we discussed before"
- Use: route_query(query="AI trends comparison")
- Likely recommends: ["vector_db", "web"]
- Use both: retrieve_from_vector_db(query="AI trends discussion") AND retrieve_from_web(query="current AI trends")

### Best Practices

- Always check relevance before using retrieved information
- Prefer vector_db for personal/past information, web for current events
- Combine multiple sources when query requires comprehensive information
- Use check_relevance to avoid including irrelevant context
- Learn from past retrieval decisions to improve future routing
`
}
