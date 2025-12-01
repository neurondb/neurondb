package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/pkg/neurondb"
)

type Runtime struct {
	db        *db.DB
	queries   *db.Queries
	memory    *MemoryManager
	planner   *Planner
	prompt    *PromptBuilder
	llm       *LLMClient
	tools     ToolRegistry
	embed     *neurondb.EmbeddingClient
}

type ExecutionState struct {
	SessionID   uuid.UUID
	AgentID     uuid.UUID
	UserMessage string
	Context     *Context
	LLMResponse *LLMResponse
	ToolCalls   []ToolCall
	ToolResults []ToolResult
	FinalAnswer string
	TokensUsed  int
	Error       error
}

type LLMResponse struct {
	Content   string
	ToolCalls []ToolCall
	Usage     TokenUsage
}

type ToolCall struct {
	ID        string
	Name      string
	Arguments map[string]interface{}
}

type ToolResult struct {
	ToolCallID string
	Content    string
	Error      error
}

type TokenUsage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
}

// ToolRegistry interface for tool management
type ToolRegistry interface {
	Get(name string) (*db.Tool, error)
	Execute(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error)
}

func NewRuntime(db *db.DB, queries *db.Queries, tools ToolRegistry, embedClient *neurondb.EmbeddingClient) *Runtime {
	return &Runtime{
		db:      db,
		queries: queries,
		memory:  NewMemoryManager(db, queries, embedClient),
		planner: NewPlanner(),
		prompt:  NewPromptBuilder(),
		llm:     NewLLMClient(db),
		tools:   tools,
		embed:   embedClient,
	}
}

func (r *Runtime) Execute(ctx context.Context, sessionID uuid.UUID, userMessage string) (*ExecutionState, error) {
	state := &ExecutionState{
		SessionID:   sessionID,
		UserMessage: userMessage,
	}

	// Step 1: Load agent and session
	session, err := r.queries.GetSession(ctx, sessionID)
	if err != nil {
		return nil, fmt.Errorf("failed to get session: %w", err)
	}
	state.AgentID = session.AgentID

	agent, err := r.queries.GetAgentByID(ctx, session.AgentID)
	if err != nil {
		return nil, fmt.Errorf("failed to get agent: %w", err)
	}

	// Step 2: Load context (recent messages + memory)
	contextLoader := NewContextLoader(r.queries, r.memory, r.llm)
	agentContext, err := contextLoader.Load(ctx, sessionID, agent.ID, userMessage, 20, 5)
	if err != nil {
		return nil, fmt.Errorf("failed to load context: %w", err)
	}
	state.Context = agentContext

	// Step 3: Build prompt
	prompt, err := r.prompt.Build(agent, agentContext, userMessage)
	if err != nil {
		return nil, fmt.Errorf("failed to build prompt: %w", err)
	}

	// Step 4: Call LLM via NeuronDB
	llmResponse, err := r.llm.Generate(ctx, agent.ModelName, prompt, agent.Config)
	if err != nil {
		return nil, fmt.Errorf("LLM generation failed: %w", err)
	}
	
	// Update token count in response
	if llmResponse.Usage.TotalTokens == 0 {
		// Estimate if not provided
		llmResponse.Usage.PromptTokens = EstimateTokens(prompt)
		llmResponse.Usage.CompletionTokens = EstimateTokens(llmResponse.Content)
		llmResponse.Usage.TotalTokens = llmResponse.Usage.PromptTokens + llmResponse.Usage.CompletionTokens
	}

	// Step 5: Parse tool calls from response
	toolCalls, err := ParseToolCalls(llmResponse.Content)
	if err == nil && len(toolCalls) > 0 {
		llmResponse.ToolCalls = toolCalls
	}
	state.LLMResponse = llmResponse

	// Step 6: Execute tools if any
	if len(llmResponse.ToolCalls) > 0 {
		state.ToolCalls = llmResponse.ToolCalls

		// Execute tools
		toolResults, err := r.executeTools(ctx, agent, llmResponse.ToolCalls)
		if err != nil {
			return nil, fmt.Errorf("tool execution failed: %w", err)
		}
		state.ToolResults = toolResults

		// Step 7: Call LLM again with tool results
		finalPrompt, err := r.prompt.BuildWithToolResults(agent, agentContext, userMessage, llmResponse, toolResults)
		if err != nil {
			return nil, fmt.Errorf("failed to build final prompt: %w", err)
		}

		finalResponse, err := r.llm.Generate(ctx, agent.ModelName, finalPrompt, agent.Config)
		if err != nil {
			return nil, fmt.Errorf("final LLM generation failed: %w", err)
		}
		
		// Update token counts
		if finalResponse.Usage.TotalTokens == 0 {
			finalResponse.Usage.PromptTokens = EstimateTokens(finalPrompt)
			finalResponse.Usage.CompletionTokens = EstimateTokens(finalResponse.Content)
			finalResponse.Usage.TotalTokens = finalResponse.Usage.PromptTokens + finalResponse.Usage.CompletionTokens
		}
		
		state.FinalAnswer = finalResponse.Content
		state.TokensUsed = llmResponse.Usage.TotalTokens + finalResponse.Usage.TotalTokens
	} else {
		state.FinalAnswer = llmResponse.Content
		state.TokensUsed = llmResponse.Usage.TotalTokens
		if state.TokensUsed == 0 {
			// Estimate if not provided
			state.TokensUsed = EstimateTokens(prompt) + EstimateTokens(state.FinalAnswer)
		}
	}

	// Step 8: Store messages with token counts
	if err := r.storeMessages(ctx, sessionID, userMessage, state.FinalAnswer, state.ToolCalls, state.ToolResults, state.TokensUsed); err != nil {
		return nil, fmt.Errorf("failed to store messages: %w", err)
	}

	// Step 9: Store memory chunks (async, non-blocking)
	go func() {
		bgCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		r.memory.StoreChunks(bgCtx, agent.ID, sessionID, state.FinalAnswer, state.ToolResults)
	}()

	return state, nil
}

func (r *Runtime) executeTools(ctx context.Context, agent *db.Agent, toolCalls []ToolCall) ([]ToolResult, error) {
	results := make([]ToolResult, 0, len(toolCalls))

	for _, call := range toolCalls {
		// Get tool from registry
		tool, err := r.tools.Get(call.Name)
		if err != nil {
			results = append(results, ToolResult{
				ToolCallID: call.ID,
				Error:      err,
			})
			continue
		}

		// Check if tool is enabled for this agent
		if !contains(agent.EnabledTools, call.Name) {
			results = append(results, ToolResult{
				ToolCallID: call.ID,
				Error:      fmt.Errorf("tool %s not enabled for agent", call.Name),
			})
			continue
		}

		// Execute tool
		result, err := r.tools.Execute(ctx, tool, call.Arguments)
		results = append(results, ToolResult{
			ToolCallID: call.ID,
			Content:    result,
			Error:      err,
		})
	}

	return results, nil
}

func (r *Runtime) storeMessages(ctx context.Context, sessionID uuid.UUID, userMsg, assistantMsg string, toolCalls []ToolCall, toolResults []ToolResult, totalTokens int) error {
	// Store user message
	userTokens := EstimateTokens(userMsg)
	if _, err := r.queries.CreateMessage(ctx, &db.Message{
		SessionID:  sessionID,
		Role:       "user",
		Content:    userMsg,
		TokenCount: &userTokens,
	}); err != nil {
		return err
	}

	// Store tool calls as messages
	for _, call := range toolCalls {
		callJSON, _ := json.Marshal(call.Arguments)
		toolCallID := call.ID
		if _, err := r.queries.CreateMessage(ctx, &db.Message{
			SessionID:  sessionID,
			Role:       "assistant",
			Content:    fmt.Sprintf("Tool call: %s with args: %s", call.Name, string(callJSON)),
			ToolCallID: &toolCallID,
			Metadata:   map[string]interface{}{"tool_call": call},
		}); err != nil {
			return err
		}
	}

	// Store tool results
	for _, result := range toolResults {
		toolName := result.ToolCallID
		toolCallID := result.ToolCallID
		if _, err := r.queries.CreateMessage(ctx, &db.Message{
			SessionID:  sessionID,
			Role:       "tool",
			Content:    result.Content,
			ToolName:   &toolName,
			ToolCallID: &toolCallID,
		}); err != nil {
			return err
		}
	}

	// Store assistant message
	assistantTokens := EstimateTokens(assistantMsg)
	if _, err := r.queries.CreateMessage(ctx, &db.Message{
		SessionID:  sessionID,
		Role:       "assistant",
		Content:    assistantMsg,
		TokenCount: &assistantTokens,
	}); err != nil {
		return err
	}

	return nil
}

// Helper function to check if a string is in an array
func contains(arr pq.StringArray, s string) bool {
	for _, item := range arr {
		if item == s {
			return true
		}
	}
	return false
}

