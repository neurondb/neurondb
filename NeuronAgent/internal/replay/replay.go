/*-------------------------------------------------------------------------
 *
 * replay.go
 *    Replay system for NeuronAgent
 *
 * Provides functionality to store and replay agent executions from stored snapshots.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/replay/replay.go
 *
 *-------------------------------------------------------------------------
 */

package replay

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/agent"
	"github.com/neurondb/NeuronAgent/internal/db"
)

type ReplayManager struct {
	queries *db.Queries
	runtime *agent.Runtime
}

func NewReplayManager(queries *db.Queries, runtime *agent.Runtime) *ReplayManager {
	return &ReplayManager{
		queries: queries,
		runtime: runtime,
	}
}

/* StoreExecutionSnapshot stores a complete execution snapshot for replay */
func (r *ReplayManager) StoreExecutionSnapshot(ctx context.Context, state *agent.ExecutionState, deterministicMode bool) error {
	executionState := map[string]interface{}{
		"user_message": state.UserMessage,
		"final_answer": state.FinalAnswer,
		"tokens_used":  state.TokensUsed,
		"tool_calls":   state.ToolCalls,
		"tool_results": r.serializeToolResults(state.ToolResults),
		"llm_response": r.serializeLLMResponse(state.LLMResponse),
	}

	snapshot := &db.ExecutionSnapshot{
		SessionID:         state.SessionID,
		AgentID:           state.AgentID,
		UserMessage:       state.UserMessage,
		ExecutionState:    executionState,
		DeterministicMode: deterministicMode,
	}

	return r.queries.CreateExecutionSnapshot(ctx, snapshot)
}

/* ReplayExecution replays an execution from a stored snapshot */
func (r *ReplayManager) ReplayExecution(ctx context.Context, snapshotID uuid.UUID) (*agent.ExecutionState, error) {
	snapshot, err := r.queries.GetExecutionSnapshotByID(ctx, snapshotID)
	if err != nil {
		return nil, fmt.Errorf("failed to get execution snapshot: %w", err)
	}

	/* In deterministic mode, we could use the stored results directly */
	/* For now, we'll re-execute with the stored user message */
	if snapshot.DeterministicMode {
		/* In deterministic mode, return stored state */
		return r.deserializeExecutionState(snapshot), nil
	}

	/* Otherwise, re-execute the agent */
	state, err := r.runtime.Execute(ctx, snapshot.SessionID, snapshot.UserMessage)
	if err != nil {
		return nil, fmt.Errorf("failed to replay execution: %w", err)
	}

	return state, nil
}

/* serializeToolResults serializes tool results for storage */
func (r *ReplayManager) serializeToolResults(results []agent.ToolResult) []map[string]interface{} {
	serialized := make([]map[string]interface{}, len(results))
	for i, result := range results {
		serialized[i] = map[string]interface{}{
			"tool_call_id": result.ToolCallID,
			"content":      result.Content,
			"error":        r.serializeError(result.Error),
		}
	}
	return serialized
}

/* serializeLLMResponse serializes LLM response for storage */
func (r *ReplayManager) serializeLLMResponse(resp *agent.LLMResponse) map[string]interface{} {
	if resp == nil {
		return nil
	}
	return map[string]interface{}{
		"content":    resp.Content,
		"tool_calls": resp.ToolCalls,
		"usage": map[string]interface{}{
			"prompt_tokens":     resp.Usage.PromptTokens,
			"completion_tokens": resp.Usage.CompletionTokens,
			"total_tokens":      resp.Usage.TotalTokens,
		},
	}
}

/* serializeError serializes an error */
func (r *ReplayManager) serializeError(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

/* deserializeExecutionState deserializes execution state from snapshot */
func (r *ReplayManager) deserializeExecutionState(snapshot *db.ExecutionSnapshot) *agent.ExecutionState {
	state := &agent.ExecutionState{
		SessionID:   snapshot.SessionID,
		AgentID:     snapshot.AgentID,
		UserMessage: snapshot.UserMessage,
	}

	if finalAnswer, ok := snapshot.ExecutionState["final_answer"].(string); ok {
		state.FinalAnswer = finalAnswer
	}
	if tokensUsed, ok := snapshot.ExecutionState["tokens_used"].(float64); ok {
		state.TokensUsed = int(tokensUsed)
	}

	/* Deserialize tool calls */
	if toolCallsData, ok := snapshot.ExecutionState["tool_calls"].([]interface{}); ok {
		toolCalls := make([]agent.ToolCall, 0, len(toolCallsData))
		for _, tc := range toolCallsData {
			if tcMap, ok := tc.(map[string]interface{}); ok {
				toolCall := agent.ToolCall{
					ID:        getString(tcMap, "id"),
					Name:      getString(tcMap, "name"),
					Arguments: getMap(tcMap, "arguments"),
				}
				toolCalls = append(toolCalls, toolCall)
			}
		}
		state.ToolCalls = toolCalls
	}

	/* Deserialize tool results */
	if toolResultsData, ok := snapshot.ExecutionState["tool_results"].([]interface{}); ok {
		toolResults := make([]agent.ToolResult, 0, len(toolResultsData))
		for _, tr := range toolResultsData {
			if trMap, ok := tr.(map[string]interface{}); ok {
				toolResult := agent.ToolResult{
					ToolCallID: getString(trMap, "tool_call_id"),
					Content:    getString(trMap, "content"),
				}
				if errStr := getString(trMap, "error"); errStr != "" {
					toolResult.Error = fmt.Errorf("%s", errStr)
				}
				toolResults = append(toolResults, toolResult)
			}
		}
		state.ToolResults = toolResults
	}

	return state
}

/* Helper functions for deserialization */
func getString(m map[string]interface{}, key string) string {
	if val, ok := m[key].(string); ok {
		return val
	}
	return ""
}

func getMap(m map[string]interface{}, key string) map[string]interface{} {
	if val, ok := m[key].(map[string]interface{}); ok {
		return val
	}
	/* Try JSON string */
	if valStr, ok := m[key].(string); ok {
		var result map[string]interface{}
		if err := json.Unmarshal([]byte(valStr), &result); err == nil {
			return result
		}
	}
	return make(map[string]interface{})
}
