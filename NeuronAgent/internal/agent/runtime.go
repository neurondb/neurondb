/*-------------------------------------------------------------------------
 *
 * runtime.go
 *    Agent runtime and execution engine for NeuronAgent
 *
 * Provides the core agent runtime that orchestrates agent execution,
 * including planning, reflection, tool execution, and LLM interactions.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/runtime.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"
	"github.com/neurondb/NeuronAgent/internal/auth"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/metrics"
	"github.com/neurondb/NeuronAgent/pkg/neurondb"
)

type Runtime struct {
	db                  *db.DB
	queries             *db.Queries
	memory              *MemoryManager
	hierMemory          *HierarchicalMemoryManager
	eventStream         *EventStreamManager
	verifier            *VerificationAgent
	vfs                 *VirtualFileSystem
	workspace           interface{} /* WorkspaceManager interface for collaboration */
	asyncExecutor       *AsyncTaskExecutor
	subAgentManager     *SubAgentManager
	alertManager        *TaskNotifier
	multimodalProcessor interface{} /* EnhancedMultimodalProcessor interface */
	planner             *Planner
	reflector           *Reflector
	prompt              *PromptBuilder
	llm                 *LLMClient
	tools               ToolRegistry
	embed               *neurondb.EmbeddingClient
	toolPermChecker     *auth.ToolPermissionChecker
	deterministicMode   bool
	coordinator         interface{} /* Distributed coordinator interface */
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

/* ToolRegistry interface for tool management */
type ToolRegistry interface {
	Get(ctx context.Context, name string) (*db.Tool, error)
	Execute(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error)
}

func NewRuntime(db *db.DB, queries *db.Queries, tools ToolRegistry, embedClient *neurondb.EmbeddingClient) *Runtime {
	llm := NewLLMClient(db)
	return &Runtime{
		db:              db,
		queries:         queries,
		memory:          NewMemoryManager(db, queries, embedClient),
		hierMemory:      NewHierarchicalMemoryManager(db, queries, embedClient),
		eventStream:     NewEventStreamManager(queries, llm),
		planner:         NewPlannerWithLLM(llm),
		reflector:       NewReflector(llm),
		prompt:          NewPromptBuilder(),
		llm:             llm,
		tools:           tools,
		embed:           embedClient,
		toolPermChecker: auth.NewToolPermissionChecker(queries),
	}
}

/* NewRuntimeWithFeatures creates runtime with all advanced features */
func NewRuntimeWithFeatures(db *db.DB, queries *db.Queries, tools ToolRegistry, embedClient *neurondb.EmbeddingClient, vfs *VirtualFileSystem, workspace interface{}) *Runtime {
	runtime := NewRuntime(db, queries, tools, embedClient)

	/* Set VFS if provided */
	if vfs != nil {
		runtime.vfs = vfs
	}

	/* Set workspace if provided */
	if workspace != nil {
		runtime.workspace = workspace
	}

	return runtime
}

func (r *Runtime) Execute(ctx context.Context, sessionID uuid.UUID, userMessage string) (*ExecutionState, error) {
	/* Validate input */
	if userMessage == "" {
		return nil, fmt.Errorf("agent execution failed: session_id='%s', user_message_empty=true", sessionID.String())
	}
	if len(userMessage) > 100000 {
		return nil, fmt.Errorf("agent execution failed: session_id='%s', user_message_too_large=true, length=%d, max_length=100000",
			sessionID.String(), len(userMessage))
	}

	/* Check if distributed coordinator is enabled and route accordingly */
	if r.coordinator != nil {
		if coordinator, ok := r.coordinator.(interface {
			IsEnabled() bool
			ExecuteAgent(ctx context.Context, sessionID uuid.UUID, userMessage string) (*ExecutionState, error)
		}); ok && coordinator.IsEnabled() {
			return coordinator.ExecuteAgent(ctx, sessionID, userMessage)
		}
	}

	/* Check if task should be async (long-running tasks) */
	if r.asyncExecutor != nil && r.shouldExecuteAsync(userMessage) {
		return r.executeAsync(ctx, sessionID, userMessage)
	}

	state := &ExecutionState{
		SessionID:   sessionID,
		UserMessage: userMessage,
	}

	/* Log user message to event stream */
	if r.eventStream != nil {
		r.eventStream.LogEvent(ctx, sessionID, "user_message", "user", userMessage, map[string]interface{}{})
	}

	/* Step 1: Load agent and session */
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

	/* Route to sub-agent if needed */
	if r.subAgentManager != nil {
		subAgent, err := r.subAgentManager.GetAgentSpecialization(ctx, agent.ID)
		if err == nil && subAgent != nil {
			/* Agent has specialization - could be used for routing decisions in future */
			/* For now, we log that specialization exists but continue with current agent */
			metrics.DebugWithContext(ctx, "Agent has specialization but using current agent", map[string]interface{}{
				"session_id":      sessionID.String(),
				"agent_id":        agent.ID.String(),
				"specialization": subAgent.Specialization,
			})
		}
	}

	/* Step 2: Load context using hierarchical memory and event stream */
	contextLoader := NewContextLoader(r.queries, r.memory, r.llm)
	agentContext, err := contextLoader.Load(ctx, sessionID, agent.ID, userMessage, 20, 5)
	if err != nil {
		return nil, fmt.Errorf("agent execution failed at step 2 (load context): session_id='%s', agent_id='%s', agent_name='%s', user_message_length=%d, max_messages=20, max_memory_chunks=5, error=%w",
			sessionID.String(), agent.ID.String(), agent.Name, len(userMessage), err)
	}

	/* Enhance context with hierarchical memory if available */
	if r.hierMemory != nil {
		hierMemoryChunks, err := r.hierMemory.RetrieveHierarchical(ctx, agent.ID, userMessage, []string{"stm", "mtm", "lpm"}, 5)
		if err == nil && hierMemoryChunks != nil {
			/* Add hierarchical memory chunks to context */
			for tier, chunks := range hierMemoryChunks {
				for _, chunk := range chunks {
					agentContext.MemoryChunks = append(agentContext.MemoryChunks, MemoryChunk{
						ID:              chunk.ID,
						Content:         chunk.Content,
						ImportanceScore: chunk.ImportanceScore,
						Similarity:      chunk.Similarity,
						Metadata:        map[string]interface{}{"tier": tier},
					})
				}
			}
		} else if err != nil {
			/* Log error but continue - hierarchical memory is optional */
			metrics.WarnWithContext(ctx, "Failed to retrieve hierarchical memory", map[string]interface{}{
				"session_id": sessionID.String(),
				"agent_id":   agent.ID.String(),
				"error":      err.Error(),
			})
		}
	}

	/* Load context from event stream if available */
	if r.eventStream != nil {
		_, summaries, err := r.eventStream.GetContextWindow(ctx, sessionID, 50)
		if err == nil && len(summaries) > 0 {
			/* Add event summaries to context */
			for _, summary := range summaries {
				agentContext.Messages = append(agentContext.Messages, db.Message{
					Role:    "system",
					Content: summary.SummaryText,
				})
			}
		}
	}

	state.Context = agentContext

	/* Step 3: Build prompt */
	prompt, err := r.prompt.Build(agent, agentContext, userMessage)
	if err != nil {
		messageCount := len(agentContext.Messages)
		memoryChunkCount := len(agentContext.MemoryChunks)
		return nil, fmt.Errorf("agent execution failed at step 3 (build prompt): session_id='%s', agent_id='%s', agent_name='%s', user_message_length=%d, context_message_count=%d, context_memory_chunk_count=%d, error=%w",
			sessionID.String(), agent.ID.String(), agent.Name, len(userMessage), messageCount, memoryChunkCount, err)
	}

	/* Step 4: Call LLM via NeuronDB */
	llmResponse, err := r.llm.Generate(ctx, agent.ModelName, prompt, agent.Config)
	if err != nil {
		promptTokens := EstimateTokens(prompt)
		return nil, fmt.Errorf("agent execution failed at step 4 (LLM generation): session_id='%s', agent_id='%s', agent_name='%s', model_name='%s', prompt_length=%d, prompt_tokens=%d, user_message_length=%d, error=%w",
			sessionID.String(), agent.ID.String(), agent.Name, agent.ModelName, len(prompt), promptTokens, len(userMessage), err)
	}

	/* Update token count in response */
	if llmResponse.Usage.TotalTokens == 0 {
		/* Estimate if not provided */
		llmResponse.Usage.PromptTokens = EstimateTokens(prompt)
		llmResponse.Usage.CompletionTokens = EstimateTokens(llmResponse.Content)
		llmResponse.Usage.TotalTokens = llmResponse.Usage.PromptTokens + llmResponse.Usage.CompletionTokens
	}

	/* Step 5: Parse tool calls from response */
	toolCalls, err := ParseToolCalls(llmResponse.Content)
	if err == nil && len(toolCalls) > 0 {
		llmResponse.ToolCalls = toolCalls
	}
	state.LLMResponse = llmResponse

	/* Step 6: Execute tools if any (limit to prevent excessive tool calls) */
	maxToolCalls := 20
	if len(llmResponse.ToolCalls) > maxToolCalls {
		llmResponse.ToolCalls = llmResponse.ToolCalls[:maxToolCalls]
	}

	if len(llmResponse.ToolCalls) > 0 {
		state.ToolCalls = llmResponse.ToolCalls

		/* Log agent action to event stream */
		if r.eventStream != nil {
			r.eventStream.LogEvent(ctx, sessionID, "agent_action", agent.ID.String(), fmt.Sprintf("Executing %d tool calls", len(llmResponse.ToolCalls)), map[string]interface{}{
				"tool_count": len(llmResponse.ToolCalls),
			})
		}

		/* Execute tools - add sessionID to context for permission checking */
		toolCtx := WithSessionID(WithAgentID(ctx, agent.ID), sessionID)
		toolResults, err := r.executeTools(toolCtx, agent, llmResponse.ToolCalls, sessionID)
		if err != nil {
			toolNames := make([]string, len(llmResponse.ToolCalls))
			for i, call := range llmResponse.ToolCalls {
				toolNames[i] = call.Name
			}
			return nil, fmt.Errorf("agent execution failed at step 6 (tool execution): session_id='%s', agent_id='%s', agent_name='%s', tool_call_count=%d, tool_names=[%s], error=%w",
				sessionID.String(), agent.ID.String(), agent.Name, len(llmResponse.ToolCalls), fmt.Sprintf("%v", toolNames), err)
		}
		state.ToolResults = toolResults

		/* Step 7: Call LLM again with tool results */
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

		/* Update token counts */
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
			/* Estimate if not provided */
			state.TokensUsed = EstimateTokens(prompt) + EstimateTokens(state.FinalAnswer)
		}
	}

	/* Log agent response to event stream */
	if r.eventStream != nil {
		r.eventStream.LogEvent(ctx, sessionID, "agent_response", agent.ID.String(), state.FinalAnswer, map[string]interface{}{
			"tokens_used": state.TokensUsed,
		})
	}

	/* Store in short-term memory */
	if r.hierMemory != nil {
		importance := 0.5
		if len(state.FinalAnswer) > 200 {
			importance = 0.7
		}
		if _, err := r.hierMemory.StoreSTM(ctx, agent.ID, sessionID, state.FinalAnswer, importance); err != nil {
			/* Log error but continue - STM storage is non-critical */
			metrics.WarnWithContext(ctx, "Failed to store in short-term memory", map[string]interface{}{
				"session_id": sessionID.String(),
				"agent_id":   agent.ID.String(),
				"error":      err.Error(),
			})
		}
	}

	/* Queue output for verification */
	if r.verifier != nil {
		if _, err := r.verifier.QueueVerification(ctx, sessionID, nil, state.FinalAnswer, "medium"); err != nil {
			/* Log error but continue - verification queuing is non-critical */
			metrics.WarnWithContext(ctx, "Failed to queue output for verification", map[string]interface{}{
				"session_id": sessionID.String(),
				"agent_id":   agent.ID.String(),
				"error":      err.Error(),
			})
		}
	}

	/* Step 8: Store messages with token counts */
	if err := r.storeMessages(ctx, sessionID, userMessage, state.FinalAnswer, state.ToolCalls, state.ToolResults, state.TokensUsed); err != nil {
		return nil, fmt.Errorf("agent execution failed at step 8 (store messages): session_id='%s', agent_id='%s', agent_name='%s', user_message_length=%d, final_answer_length=%d, tool_call_count=%d, tool_result_count=%d, total_tokens=%d, error=%w",
			sessionID.String(), agent.ID.String(), agent.Name, len(userMessage), len(state.FinalAnswer), len(state.ToolCalls), len(state.ToolResults), state.TokensUsed, err)
	}

	/* Step 9: Store memory chunks (async, non-blocking) */
	go func() {
		bgCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		/* Recover from any panics in background goroutine */
		defer func() {
			if r := recover(); r != nil {
				metrics.ErrorWithContext(bgCtx, "Panic in background memory storage goroutine", fmt.Errorf("panic: %v", r), map[string]interface{}{
					"session_id": sessionID.String(),
					"agent_id":   agent.ID.String(),
				})
			}
		}()

		/* Store chunks - errors are handled internally by StoreChunks */
		r.memory.StoreChunks(bgCtx, agent.ID, sessionID, state.FinalAnswer, state.ToolResults)

		/* Check if context was cancelled or timed out */
		if bgCtx.Err() != nil {
			metrics.WarnWithContext(bgCtx, "Memory storage context cancelled or timed out", map[string]interface{}{
				"session_id": sessionID.String(),
				"agent_id":   agent.ID.String(),
				"error":      bgCtx.Err().Error(),
			})
		}
	}()

	/* Step 10: Send completion alert if configured */
	if r.alertManager != nil && state.FinalAnswer != "" {
		/* Completion alerts are handled by async task notifier */
		/* For synchronous execution, we could send immediate alerts here */
	}

	return state, nil
}

func (r *Runtime) executeTools(ctx context.Context, agent *db.Agent, toolCalls []ToolCall, sessionID uuid.UUID) ([]ToolResult, error) {
	/* Log tool execution start to event stream */
	if r.eventStream != nil {
		for _, call := range toolCalls {
			r.eventStream.LogEvent(ctx, sessionID, "tool_execution", call.Name, fmt.Sprintf("Executing tool: %s", call.Name), map[string]interface{}{
				"tool_name": call.Name,
				"tool_id":   call.ID,
			})
		}
	}

	/* Check if tools can be executed in parallel */
	if r.canExecuteParallel(toolCalls) {
		return r.executeToolsParallel(ctx, agent, toolCalls, sessionID)
	}

	/* Execute sequentially */
	return r.executeToolsSequential(ctx, agent, toolCalls, sessionID)
}

/* canExecuteParallel checks if tools can be executed in parallel */
func (r *Runtime) canExecuteParallel(toolCalls []ToolCall) bool {
	/* Simple heuristic: if multiple tools and none depend on others */
	if len(toolCalls) <= 1 {
		return false
	}

	/* Check for dependencies (simplified - in production would use dependency graph) */
	/* For now, allow parallel execution if tools are different */
	toolNames := make(map[string]bool)
	for _, call := range toolCalls {
		if toolNames[call.Name] {
			/* Same tool called multiple times - might have dependencies */
			return false
		}
		toolNames[call.Name] = true
	}

	return true
}

/* executeToolsParallel executes tools in parallel */
func (r *Runtime) executeToolsParallel(ctx context.Context, agent *db.Agent, toolCalls []ToolCall, sessionID uuid.UUID) ([]ToolResult, error) {
	type resultWithIndex struct {
		index  int
		result ToolResult
	}

	results := make([]ToolResult, len(toolCalls))
	resultChan := make(chan resultWithIndex, len(toolCalls))

	/* Execute all tools in parallel */
	for i, call := range toolCalls {
		go func(idx int, toolCall ToolCall) {
			result := r.executeSingleTool(ctx, agent, toolCall)
			resultChan <- resultWithIndex{index: idx, result: result}
		}(i, call)
	}

	/* Collect results with context cancellation support */
	for i := 0; i < len(toolCalls); i++ {
		select {
		case ri := <-resultChan:
			results[ri.index] = ri.result
		case <-ctx.Done():
			/* Context cancelled - return partial results with error */
			return results, fmt.Errorf("tool execution cancelled: context_error=%w", ctx.Err())
		}
	}

	return results, nil
}

/* executeToolsSequential executes tools sequentially */
func (r *Runtime) executeToolsSequential(ctx context.Context, agent *db.Agent, toolCalls []ToolCall, sessionID uuid.UUID) ([]ToolResult, error) {
	results := make([]ToolResult, 0, len(toolCalls))

	for _, call := range toolCalls {
		result := r.executeSingleTool(ctx, agent, call)
		results = append(results, result)

		/* If tool failed and it's critical, stop execution */
		if result.Error != nil {
			/* Continue for now - could add critical flag to tools */
		}
	}

	return results, nil
}

/* executeSingleTool executes a single tool */
func (r *Runtime) executeSingleTool(ctx context.Context, agent *db.Agent, call ToolCall) ToolResult {
	/* Add timeout context for tool execution (5 minutes max) */
	toolCtx, cancel := context.WithTimeout(ctx, 5*time.Minute)
	defer cancel()

	/* Check if context is already cancelled */
	if ctx.Err() != nil {
		return ToolResult{
			ToolCallID: call.ID,
			Error: fmt.Errorf("tool execution cancelled: tool_call_id='%s', tool_name='%s', context_error=%w",
				call.ID, call.Name, ctx.Err()),
		}
	}

	/* Get tool from registry */
	tool, err := r.tools.Get(toolCtx, call.Name)
	if err != nil {
		argKeys := make([]string, 0, len(call.Arguments))
		for k := range call.Arguments {
			argKeys = append(argKeys, k)
		}
		return ToolResult{
			ToolCallID: call.ID,
			Error: fmt.Errorf("tool retrieval failed for tool call: tool_call_id='%s', tool_name='%s', agent_id='%s', agent_name='%s', args_count=%d, arg_keys=[%v], error=%w",
				call.ID, call.Name, agent.ID.String(), agent.Name, len(call.Arguments), argKeys, err),
		}
	}

	/* Check if tool is enabled for this agent */
	if !contains(agent.EnabledTools, call.Name) {
		return ToolResult{
			ToolCallID: call.ID,
			Error: fmt.Errorf("tool not enabled for agent: tool_call_id='%s', tool_name='%s', agent_id='%s', agent_name='%s', enabled_tools=[%v]",
				call.ID, call.Name, agent.ID.String(), agent.Name, agent.EnabledTools),
		}
	}

	/* Check tool permissions */
	sessionID, hasSessionID := GetSessionIDFromContext(ctx)
	if hasSessionID {
		allowed, err := r.toolPermChecker.CheckToolPermission(ctx, agent.ID, sessionID, call.Name)
		if err != nil {
			return ToolResult{
				ToolCallID: call.ID,
				Error: fmt.Errorf("tool permission check failed: tool_call_id='%s', tool_name='%s', agent_id='%s', session_id='%s', error=%w",
					call.ID, call.Name, agent.ID.String(), sessionID.String(), err),
			}
		}
		if !allowed {
			return ToolResult{
				ToolCallID: call.ID,
				Error: fmt.Errorf("tool execution not allowed: tool_call_id='%s', tool_name='%s', agent_id='%s', session_id='%s'",
					call.ID, call.Name, agent.ID.String(), sessionID.String()),
			}
		}
	}

	/* Execute tool */
	result, err := r.tools.Execute(toolCtx, tool, call.Arguments)
	if err != nil {
		argKeys := make([]string, 0, len(call.Arguments))
		for k := range call.Arguments {
			argKeys = append(argKeys, k)
		}
		return ToolResult{
			ToolCallID: call.ID,
			Content:    result,
			Error: fmt.Errorf("tool execution failed: tool_call_id='%s', tool_name='%s', handler_type='%s', agent_id='%s', agent_name='%s', args_count=%d, arg_keys=[%v], error=%w",
				call.ID, call.Name, tool.HandlerType, agent.ID.String(), agent.Name, len(call.Arguments), argKeys, err),
		}
	}

	return ToolResult{
		ToolCallID: call.ID,
		Content:    result,
		Error:      nil,
	}
}

func (r *Runtime) storeMessages(ctx context.Context, sessionID uuid.UUID, userMsg, assistantMsg string, toolCalls []ToolCall, toolResults []ToolResult, totalTokens int) error {
	/* Store user message */
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

	/* Store tool calls as messages */
	for _, call := range toolCalls {
		var callJSONStr string
		callJSON, err := json.Marshal(call.Arguments)
		if err != nil {
			/* If JSON marshaling fails, use fallback string representation */
			metrics.WarnWithContext(ctx, "Failed to marshal tool call arguments to JSON", map[string]interface{}{
				"session_id": sessionID.String(),
				"tool_call_id": call.ID,
				"tool_name": call.Name,
				"error": err.Error(),
			})
			callJSONStr = fmt.Sprintf("%v", call.Arguments)
		} else {
			callJSONStr = string(callJSON)
		}
		toolCallID := call.ID
		if _, err := r.queries.CreateMessage(ctx, &db.Message{
			SessionID:  sessionID,
			Role:       "assistant",
			Content:    fmt.Sprintf("Tool call: %s with args: %s", call.Name, callJSONStr),
			ToolCallID: &toolCallID,
			Metadata:   map[string]interface{}{"tool_call": call},
		}); err != nil {
			return fmt.Errorf("failed to store tool call message: session_id='%s', tool_call_id='%s', tool_name='%s', args_count=%d, error=%w",
				sessionID.String(), call.ID, call.Name, len(call.Arguments), err)
		}
	}

	/* Store tool results */
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

	/* Store assistant message */
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

/* GetPlanner returns the planner */
func (r *Runtime) GetPlanner() *Planner {
	return r.planner
}

/* GetReflector returns the reflector */
func (r *Runtime) GetReflector() *Reflector {
	return r.reflector
}

/* GetMemoryManager returns the memory manager */
func (r *Runtime) GetMemoryManager() *MemoryManager {
	return r.memory
}

/* Helper function to check if a string is in an array */
func contains(arr pq.StringArray, s string) bool {
	for _, item := range arr {
		if item == s {
			return true
		}
	}
	return false
}

/* HierMemory returns the hierarchical memory manager */
func (r *Runtime) HierMemory() *HierarchicalMemoryManager {
	return r.hierMemory
}

/* EventStream returns the event stream manager */
func (r *Runtime) EventStream() *EventStreamManager {
	return r.eventStream
}

/* Verifier returns the verification agent */
func (r *Runtime) Verifier() *VerificationAgent {
	return r.verifier
}

/* VFS returns the virtual file system */
func (r *Runtime) VFS() *VirtualFileSystem {
	return r.vfs
}

/* Workspace returns the workspace manager */
func (r *Runtime) Workspace() interface{} {
	return r.workspace
}

/* SetAsyncExecutor sets the async task executor */
func (r *Runtime) SetAsyncExecutor(executor *AsyncTaskExecutor) {
	r.asyncExecutor = executor
}

/* SetSubAgentManager sets the sub-agent manager */
func (r *Runtime) SetSubAgentManager(manager *SubAgentManager) {
	r.subAgentManager = manager
}

/* SetAlertManager sets the alert manager */
func (r *Runtime) SetAlertManager(manager *TaskNotifier) {
	r.alertManager = manager
}

/* SetMultimodalProcessor sets the multimodal processor */
func (r *Runtime) SetMultimodalProcessor(processor interface{}) {
	r.multimodalProcessor = processor
}

/* shouldExecuteAsync determines if a task should be executed asynchronously */
func (r *Runtime) shouldExecuteAsync(userMessage string) bool {
	/* Check for async keywords */
	asyncKeywords := []string{"long-running", "background", "async", "process large", "analyze dataset"}
	messageLower := strings.ToLower(userMessage)

	for _, keyword := range asyncKeywords {
		if strings.Contains(messageLower, keyword) {
			return true
		}
	}

	/* Check message length (very long messages might benefit from async) */
	if len(userMessage) > 10000 {
		return true
	}

	return false
}

/* executeAsync executes a task asynchronously */
func (r *Runtime) executeAsync(ctx context.Context, sessionID uuid.UUID, userMessage string) (*ExecutionState, error) {
	/* Load session to get agent ID */
	session, err := r.queries.GetSession(ctx, sessionID)
	if err != nil {
		return nil, fmt.Errorf("async execution failed: session_load_error=true, session_id='%s', error=%w", sessionID.String(), err)
	}

	/* Create async task */
	input := map[string]interface{}{
		"user_message": userMessage,
		"async":        true,
	}

	task, err := r.asyncExecutor.ExecuteAsync(ctx, sessionID, session.AgentID, "agent_execution", input, 0)
	if err != nil {
		return nil, fmt.Errorf("async execution failed: task_creation_error=true, session_id='%s', agent_id='%s', error=%w",
			sessionID.String(), session.AgentID.String(), err)
	}

	/* Return state indicating async execution */
	return &ExecutionState{
		SessionID:   sessionID,
		AgentID:     session.AgentID,
		UserMessage: userMessage,
		FinalAnswer: fmt.Sprintf("Task queued for asynchronous execution. Task ID: %s. Use GET /api/v1/async-tasks/%s to check status.", task.ID.String(), task.ID.String()),
	}, nil
}

/* StreamCallback is called for each chunk of streamed output */
type StreamCallback func(chunk string, eventType string) error

/* streamWriter implements io.Writer and calls callback for each write */
type streamWriter struct {
	builder  *strings.Builder
	callback StreamCallback
}

func (w *streamWriter) Write(p []byte) (n int, err error) {
	chunk := string(p)
	w.builder.WriteString(chunk)
	if w.callback != nil {
		if err := w.callback(chunk, "chunk"); err != nil {
			return len(p), err
		}
	}
	return len(p), nil
}

/* ExecuteStream executes agent with streaming support */
func (r *Runtime) ExecuteStream(ctx context.Context, sessionID uuid.UUID, userMessage string, callback StreamCallback) (*ExecutionState, error) {
	/* Validate input */
	if userMessage == "" {
		return nil, fmt.Errorf("agent execution failed: session_id='%s', user_message_empty=true", sessionID.String())
	}
	if len(userMessage) > 100000 {
		return nil, fmt.Errorf("agent execution failed: session_id='%s', user_message_too_large=true, length=%d, max_length=100000",
			sessionID.String(), len(userMessage))
	}

	state := &ExecutionState{
		SessionID:   sessionID,
		UserMessage: userMessage,
	}

	/* Log user message to event stream */
	if r.eventStream != nil {
		r.eventStream.LogEvent(ctx, sessionID, "user_message", "user", userMessage, map[string]interface{}{})
	}

	/* Step 1: Load agent and session */
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

	/* Check context cancellation before proceeding */
	if ctx.Err() != nil {
		return nil, fmt.Errorf("agent execution cancelled (streaming): session_id='%s', context_error=%w", sessionID.String(), ctx.Err())
	}

	/* Step 2: Load context */
	contextLoader := NewContextLoader(r.queries, r.memory, r.llm)
	agentContext, err := contextLoader.Load(ctx, sessionID, agent.ID, userMessage, 20, 5)
	if err != nil {
		return nil, fmt.Errorf("agent execution failed at step 2 (load context): session_id='%s', agent_id='%s', agent_name='%s', user_message_length=%d, max_messages=20, max_memory_chunks=5, error=%w",
			sessionID.String(), agent.ID.String(), agent.Name, len(userMessage), err)
	}

	/* Enhance context with hierarchical memory if available */
	if r.hierMemory != nil {
		hierMemoryChunks, err := r.hierMemory.RetrieveHierarchical(ctx, agent.ID, userMessage, []string{"stm", "mtm", "lpm"}, 5)
		if err == nil && hierMemoryChunks != nil {
			/* Add hierarchical memory chunks to context */
			for tier, chunks := range hierMemoryChunks {
				for _, chunk := range chunks {
					agentContext.MemoryChunks = append(agentContext.MemoryChunks, MemoryChunk{
						ID:              chunk.ID,
						Content:         chunk.Content,
						ImportanceScore: chunk.ImportanceScore,
						Similarity:      chunk.Similarity,
						Metadata:        map[string]interface{}{"tier": tier},
					})
				}
			}
		} else if err != nil {
			/* Log error but continue - hierarchical memory is optional */
			metrics.WarnWithContext(ctx, "Failed to retrieve hierarchical memory in streaming", map[string]interface{}{
				"session_id": sessionID.String(),
				"agent_id":   agent.ID.String(),
				"error":      err.Error(),
			})
		}
	}

	state.Context = agentContext

	/* Step 3: Build prompt */
	prompt, err := r.prompt.Build(agent, agentContext, userMessage)
	if err != nil {
		messageCount := len(agentContext.Messages)
		memoryChunkCount := len(agentContext.MemoryChunks)
		return nil, fmt.Errorf("agent execution failed at step 3 (build prompt): session_id='%s', agent_id='%s', agent_name='%s', user_message_length=%d, context_message_count=%d, context_memory_chunk_count=%d, error=%w",
			sessionID.String(), agent.ID.String(), agent.Name, len(userMessage), messageCount, memoryChunkCount, err)
	}

	/* Step 4: Stream LLM response */
	var fullResponse strings.Builder
	sw := &streamWriter{
		builder:  &fullResponse,
		callback: callback,
	}

	err = r.llm.GenerateStream(ctx, agent.ModelName, prompt, agent.Config, sw)
	if err != nil {
		promptTokens := EstimateTokens(prompt)
		return nil, fmt.Errorf("agent execution failed at step 4 (LLM streaming): session_id='%s', agent_id='%s', agent_name='%s', model_name='%s', prompt_length=%d, prompt_tokens=%d, user_message_length=%d, error=%w",
			sessionID.String(), agent.ID.String(), agent.Name, agent.ModelName, len(prompt), promptTokens, len(userMessage), err)
	}

	responseContent := fullResponse.String()
	state.FinalAnswer = responseContent

	/* Parse tool calls from response */
	toolCalls, err := ParseToolCalls(responseContent)
	if err == nil && len(toolCalls) > 0 {
		state.ToolCalls = toolCalls
		/* Notify about tool calls */
		if callback != nil {
			toolCallsJSON, err := json.Marshal(toolCalls)
			if err == nil {
				if callbackErr := callback(string(toolCallsJSON), "tool_calls"); callbackErr != nil {
					metrics.WarnWithContext(ctx, "Callback error when notifying about tool calls", map[string]interface{}{
						"session_id": sessionID.String(),
						"agent_id":   agent.ID.String(),
						"error":      callbackErr.Error(),
					})
				}
			} else {
				metrics.WarnWithContext(ctx, "Failed to marshal tool calls for callback", map[string]interface{}{
					"session_id": sessionID.String(),
					"agent_id":   agent.ID.String(),
					"error":      err.Error(),
				})
			}
		}

		/* Execute tools */
		toolCtx := WithSessionID(WithAgentID(ctx, agent.ID), sessionID)
		toolResults, err := r.executeTools(toolCtx, agent, toolCalls, sessionID)
		if err != nil {
			return nil, fmt.Errorf("tool execution failed: %w", err)
		}
		state.ToolResults = toolResults

		/* Notify about tool results */
		if callback != nil {
			toolResultsJSON, err := json.Marshal(toolResults)
			if err == nil {
				if callbackErr := callback(string(toolResultsJSON), "tool_results"); callbackErr != nil {
					metrics.WarnWithContext(ctx, "Callback error when notifying about tool results", map[string]interface{}{
						"session_id": sessionID.String(),
						"agent_id":   agent.ID.String(),
						"error":      callbackErr.Error(),
					})
				}
			} else {
				metrics.WarnWithContext(ctx, "Failed to marshal tool results for callback", map[string]interface{}{
					"session_id": sessionID.String(),
					"agent_id":   agent.ID.String(),
					"error":      err.Error(),
				})
			}
		}

		/* Build final prompt with tool results */
		finalPrompt, err := r.prompt.BuildWithToolResults(agent, agentContext, userMessage, &LLMResponse{Content: responseContent}, toolResults)
		if err != nil {
			return nil, fmt.Errorf("failed to build final prompt: %w", err)
		}

		/* Stream final response */
		var finalResponseBuilder strings.Builder
		finalSW := &streamWriter{
			builder:  &finalResponseBuilder,
			callback: callback,
		}

		err = r.llm.GenerateStream(ctx, agent.ModelName, finalPrompt, agent.Config, finalSW)
		if err != nil {
			return nil, fmt.Errorf("final LLM streaming failed: %w", err)
		}

		state.FinalAnswer = finalResponseBuilder.String()
	}

	/* Estimate token usage */
	state.TokensUsed = EstimateTokens(prompt) + EstimateTokens(state.FinalAnswer)

	/* Store in short-term memory */
	if r.hierMemory != nil && state.FinalAnswer != "" {
		importance := 0.5
		if len(state.FinalAnswer) > 200 {
			importance = 0.7
		}
		if _, err := r.hierMemory.StoreSTM(ctx, agent.ID, sessionID, state.FinalAnswer, importance); err != nil {
			/* Log error but continue - STM storage is non-critical */
			metrics.WarnWithContext(ctx, "Failed to store in short-term memory (streaming)", map[string]interface{}{
				"session_id": sessionID.String(),
				"agent_id":   agent.ID.String(),
				"error":      err.Error(),
			})
		}
	}

	/* Queue output for verification */
	if r.verifier != nil && state.FinalAnswer != "" {
		if _, err := r.verifier.QueueVerification(ctx, sessionID, nil, state.FinalAnswer, "medium"); err != nil {
			/* Log error but continue - verification queuing is non-critical */
			metrics.WarnWithContext(ctx, "Failed to queue output for verification (streaming)", map[string]interface{}{
				"session_id": sessionID.String(),
				"agent_id":   agent.ID.String(),
				"error":      err.Error(),
			})
		}
	}

	/* Store messages */
	if err := r.storeMessages(ctx, sessionID, userMessage, state.FinalAnswer, state.ToolCalls, state.ToolResults, state.TokensUsed); err != nil {
		/* Log error but continue - message storage failure shouldn't break streaming */
		metrics.WarnWithContext(ctx, "Failed to store messages in streaming execution", map[string]interface{}{
			"session_id": sessionID.String(),
			"agent_id":   agent.ID.String(),
			"error":      err.Error(),
		})
	}

	/* Store memory chunks (async) */
	go func() {
		bgCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		/* Recover from any panics in background goroutine */
		defer func() {
			if r := recover(); r != nil {
				metrics.ErrorWithContext(bgCtx, "Panic in background memory storage goroutine (streaming)", fmt.Errorf("panic: %v", r), map[string]interface{}{
					"session_id": sessionID.String(),
					"agent_id":   agent.ID.String(),
				})
			}
		}()

		/* Store chunks - errors are handled internally by StoreChunks */
		r.memory.StoreChunks(bgCtx, agent.ID, sessionID, state.FinalAnswer, state.ToolResults)

		/* Check if context was cancelled or timed out */
		if bgCtx.Err() != nil {
			metrics.WarnWithContext(bgCtx, "Memory storage context cancelled or timed out (streaming)", map[string]interface{}{
				"session_id": sessionID.String(),
				"agent_id":   agent.ID.String(),
				"error":      bgCtx.Err().Error(),
			})
		}
	}()

	/* Notify completion */
	if callback != nil {
		if callbackErr := callback("", "done"); callbackErr != nil {
			metrics.WarnWithContext(ctx, "Callback error when notifying completion", map[string]interface{}{
				"session_id": sessionID.String(),
				"agent_id":   agent.ID.String(),
				"error":      callbackErr.Error(),
			})
		}
	}

	return state, nil
}
