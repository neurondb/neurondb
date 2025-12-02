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
		return nil, fmt.Errorf("agent execution failed at step 1 (load session): session_id='%s', user_message_length=%d, error=%w",
			sessionID.String(), len(userMessage), err)
	}
	state.AgentID = session.AgentID

	agent, err := r.queries.GetAgentByID(ctx, session.AgentID)
	if err != nil {
		return nil, fmt.Errorf("agent execution failed at step 1 (load agent): session_id='%s', agent_id='%s', user_message_length=%d, error=%w",
			sessionID.String(), session.AgentID.String(), len(userMessage), err)
	}

	// Step 2: Load context (recent messages + memory)
	contextLoader := NewContextLoader(r.queries, r.memory, r.llm)
	agentContext, err := contextLoader.Load(ctx, sessionID, agent.ID, userMessage, 20, 5)
	if err != nil {
		return nil, fmt.Errorf("agent execution failed at step 2 (load context): session_id='%s', agent_id='%s', agent_name='%s', user_message_length=%d, max_messages=20, max_memory_chunks=5, error=%w",
			sessionID.String(), agent.ID.String(), agent.Name, len(userMessage), err)
	}
	state.Context = agentContext

	// Step 3: Build prompt
	prompt, err := r.prompt.Build(agent, agentContext, userMessage)
	if err != nil {
		messageCount := len(agentContext.Messages)
		memoryChunkCount := len(agentContext.MemoryChunks)
		return nil, fmt.Errorf("agent execution failed at step 3 (build prompt): session_id='%s', agent_id='%s', agent_name='%s', user_message_length=%d, context_message_count=%d, context_memory_chunk_count=%d, error=%w",
			sessionID.String(), agent.ID.String(), agent.Name, len(userMessage), messageCount, memoryChunkCount, err)
	}

	// Step 4: Call LLM via NeuronDB
	llmResponse, err := r.llm.Generate(ctx, agent.ModelName, prompt, agent.Config)
	if err != nil {
		promptTokens := EstimateTokens(prompt)
		return nil, fmt.Errorf("agent execution failed at step 4 (LLM generation): session_id='%s', agent_id='%s', agent_name='%s', model_name='%s', prompt_length=%d, prompt_tokens=%d, user_message_length=%d, error=%w",
			sessionID.String(), agent.ID.String(), agent.Name, agent.ModelName, len(prompt), promptTokens, len(userMessage), err)
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
			toolNames := make([]string, len(llmResponse.ToolCalls))
			for i, call := range llmResponse.ToolCalls {
				toolNames[i] = call.Name
			}
			return nil, fmt.Errorf("agent execution failed at step 6 (tool execution): session_id='%s', agent_id='%s', agent_name='%s', tool_call_count=%d, tool_names=[%s], error=%w",
				sessionID.String(), agent.ID.String(), agent.Name, len(llmResponse.ToolCalls), fmt.Sprintf("%v", toolNames), err)
		}
		state.ToolResults = toolResults

		// Step 7: Call LLM again with tool results
		finalPrompt, err := r.prompt.BuildWithToolResults(agent, agentContext, userMessage, llmResponse, toolResults)
		if err != nil {
			return nil, fmt.Errorf("agent execution failed at step 7 (build final prompt): session_id='%s', agent_id='%s', agent_name='%s', tool_result_count=%d, error=%w",
				sessionID.String(), agent.ID.String(), agent.Name, len(toolResults), err)
		}

		finalResponse, err := r.llm.Generate(ctx, agent.ModelName, finalPrompt, agent.Config)
		if err != nil {
			finalPromptTokens := EstimateTokens(finalPrompt)
			return nil, fmt.Errorf("agent execution failed at step 7 (final LLM generation): session_id='%s', agent_id='%s', agent_name='%s', model_name='%s', final_prompt_length=%d, final_prompt_tokens=%d, tool_result_count=%d, error=%w",
				sessionID.String(), agent.ID.String(), agent.Name, agent.ModelName, len(finalPrompt), finalPromptTokens, len(toolResults), err)
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
		return nil, fmt.Errorf("agent execution failed at step 8 (store messages): session_id='%s', agent_id='%s', agent_name='%s', user_message_length=%d, final_answer_length=%d, tool_call_count=%d, tool_result_count=%d, total_tokens=%d, error=%w",
			sessionID.String(), agent.ID.String(), agent.Name, len(userMessage), len(state.FinalAnswer), len(state.ToolCalls), len(state.ToolResults), state.TokensUsed, err)
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
			argKeys := make([]string, 0, len(call.Arguments))
			for k := range call.Arguments {
				argKeys = append(argKeys, k)
			}
			results = append(results, ToolResult{
				ToolCallID: call.ID,
				Error:      fmt.Errorf("tool retrieval failed for tool call: tool_call_id='%s', tool_name='%s', agent_id='%s', agent_name='%s', args_count=%d, arg_keys=[%v], error=%w",
					call.ID, call.Name, agent.ID.String(), agent.Name, len(call.Arguments), argKeys, err),
			})
			continue
		}

		// Check if tool is enabled for this agent
		if !contains(agent.EnabledTools, call.Name) {
			results = append(results, ToolResult{
				ToolCallID: call.ID,
				Error:      fmt.Errorf("tool not enabled for agent: tool_call_id='%s', tool_name='%s', agent_id='%s', agent_name='%s', enabled_tools=[%v]",
					call.ID, call.Name, agent.ID.String(), agent.Name, agent.EnabledTools),
			})
			continue
		}

		// Execute tool
		result, err := r.tools.Execute(ctx, tool, call.Arguments)
		if err != nil {
			argKeys := make([]string, 0, len(call.Arguments))
			for k := range call.Arguments {
				argKeys = append(argKeys, k)
			}
			results = append(results, ToolResult{
				ToolCallID: call.ID,
				Content:    result,
				Error:      fmt.Errorf("tool execution failed: tool_call_id='%s', tool_name='%s', handler_type='%s', agent_id='%s', agent_name='%s', args_count=%d, arg_keys=[%v], error=%w",
					call.ID, call.Name, tool.HandlerType, agent.ID.String(), agent.Name, len(call.Arguments), argKeys, err),
			})
		} else {
			results = append(results, ToolResult{
				ToolCallID: call.ID,
				Content:    result,
				Error:      nil,
			})
		}
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
		return fmt.Errorf("failed to store user message: session_id='%s', message_length=%d, token_count=%d, error=%w",
			sessionID.String(), len(userMsg), userTokens, err)
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
			return fmt.Errorf("failed to store tool call message: session_id='%s', tool_call_id='%s', tool_name='%s', args_count=%d, error=%w",
				sessionID.String(), call.ID, call.Name, len(call.Arguments), err)
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
			hasError := result.Error != nil
			return fmt.Errorf("failed to store tool result message: session_id='%s', tool_call_id='%s', content_length=%d, has_error=%v, error=%w",
				sessionID.String(), result.ToolCallID, len(result.Content), hasError, err)
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
		return fmt.Errorf("failed to store assistant message: session_id='%s', message_length=%d, token_count=%d, error=%w",
			sessionID.String(), len(assistantMsg), assistantTokens, err)
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

