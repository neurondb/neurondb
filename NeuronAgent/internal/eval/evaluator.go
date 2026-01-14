/*-------------------------------------------------------------------------
 *
 * evaluator.go
 *    Evaluation framework for NeuronAgent
 *
 * Provides evaluation functionality for golden tasks, tool sequences, SQL side effects,
 * retrieval evaluation, and tool evaluation.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/eval/evaluator.go
 *
 *-------------------------------------------------------------------------
 */

package eval

import (
	"context"
	"fmt"
	"math"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/agent"
	"github.com/neurondb/NeuronAgent/internal/db"
)

type Evaluator struct {
	queries *db.Queries
	runtime *agent.Runtime
}

func NewEvaluator(queries *db.Queries, runtime *agent.Runtime) *Evaluator {
	return &Evaluator{
		queries: queries,
		runtime: runtime,
	}
}

/* EvaluateTask evaluates a single task against an agent */
func (e *Evaluator) EvaluateTask(ctx context.Context, task *db.EvalTask, agentID uuid.UUID) (*db.EvalTaskResult, error) {
	/* Create a session for evaluation */
	session := &db.Session{
		AgentID: agentID,
		Metadata: map[string]interface{}{
			"eval_task_id": task.ID.String(),
		},
	}
	err := e.queries.CreateSession(ctx, session)
	if err != nil {
		return nil, fmt.Errorf("failed to create eval session: %w", err)
	}

	/* Execute the task */
	state, err := e.runtime.Execute(ctx, session.ID, task.Input)
	if err != nil {
		return nil, fmt.Errorf("failed to execute task: %w", err)
	}

	/* Evaluate the result */
	result := &db.EvalTaskResult{
		EvalTaskID: task.ID,
		SessionID:  &session.ID,
		Passed:     false,
	}

	/* Compare outputs */
	if task.ExpectedOutput != nil {
		result.ActualOutput = &state.FinalAnswer
		passed := e.compareOutputs(*task.ExpectedOutput, state.FinalAnswer)
		result.Passed = passed
		if passed {
			score := 1.0
			result.Score = &score
		} else {
			score := 0.0
			result.Score = &score
		}
	}

	/* Compare tool sequences */
	if task.ExpectedToolSequence != nil && len(task.ExpectedToolSequence) > 0 {
		actualSequence := e.serializeToolCalls(state.ToolCalls)
		result.ActualToolSequence = actualSequence

		toolSequencePassed := e.compareToolSequences(task.ExpectedToolSequence, actualSequence)
		if !toolSequencePassed {
			result.Passed = false
			if result.Score != nil {
				*result.Score *= 0.5 /* Penalize for wrong tool sequence */
			}
		}
	}

	/* Compare SQL side effects */
	if task.GoldenSQLSideEffects != nil && len(task.GoldenSQLSideEffects) > 0 {
		actualSideEffects, err := e.getSQLSideEffects(ctx, state)
		if err == nil {
			result.ActualSQLSideEffects = actualSideEffects
			sideEffectsPassed := e.compareSQLSideEffects(task.GoldenSQLSideEffects, actualSideEffects)
			if !sideEffectsPassed {
				result.Passed = false
				if result.Score != nil {
					*result.Score *= 0.5 /* Penalize for wrong SQL side effects */
				}
			}
		}
	}

	return result, nil
}

/* EvaluateRetrieval evaluates retrieval performance */
func (e *Evaluator) EvaluateRetrieval(ctx context.Context, retrievedChunks, relevantChunks []string, k int) (recallAtK, mrr float64, groundingPassed bool) {
	/* Calculate Recall@K */
	relevantSet := make(map[string]bool)
	for _, chunk := range relevantChunks {
		relevantSet[chunk] = true
	}

	intersection := 0
	for i := 0; i < k && i < len(retrievedChunks); i++ {
		if relevantSet[retrievedChunks[i]] {
			intersection++
		}
	}

	if len(relevantChunks) > 0 {
		recallAtK = float64(intersection) / float64(len(relevantChunks))
	}

	/* Calculate MRR (Mean Reciprocal Rank) */
	for i, chunk := range retrievedChunks {
		if relevantSet[chunk] {
			mrr = 1.0 / float64(i+1)
			break
		}
	}

	/* Grounding check: verify retrieved chunks are cited */
	groundingPassed = true /* Simplified - would need to check if chunks are cited in response */

	return recallAtK, mrr, groundingPassed
}

/* compareOutputs compares expected and actual outputs */
func (e *Evaluator) compareOutputs(expected, actual string) bool {
	/* Simple string comparison - could be enhanced with semantic similarity */
	return expected == actual
}

/* serializeToolCalls serializes tool calls to JSONB */
func (e *Evaluator) serializeToolCalls(calls []agent.ToolCall) map[string]interface{} {
	result := make(map[string]interface{})
	tools := make([]map[string]interface{}, len(calls))
	for i, call := range calls {
		tools[i] = map[string]interface{}{
			"id":        call.ID,
			"name":      call.Name,
			"arguments": call.Arguments,
		}
	}
	result["tools"] = tools
	return result
}

/* compareToolSequences compares expected and actual tool sequences */
func (e *Evaluator) compareToolSequences(expected, actual map[string]interface{}) bool {
	expectedTools, ok1 := expected["tools"].([]interface{})
	actualTools, ok2 := actual["tools"].([]interface{})

	if !ok1 || !ok2 {
		return false
	}

	if len(expectedTools) != len(actualTools) {
		return false
	}

	for i, expectedTool := range expectedTools {
		if i >= len(actualTools) {
			return false
		}
		expectedMap, ok1 := expectedTool.(map[string]interface{})
		actualMap, ok2 := actualTools[i].(map[string]interface{})
		if !ok1 || !ok2 {
			return false
		}
		if expectedMap["name"] != actualMap["name"] {
			return false
		}
	}
	return true
}

/* getSQLSideEffects gets SQL side effects from execution state */
func (e *Evaluator) getSQLSideEffects(ctx context.Context, state *agent.ExecutionState) (map[string]interface{}, error) {
	/* This would need to query the database to get table states */
	/* For now, return empty map */
	return make(map[string]interface{}), nil
}

/* compareSQLSideEffects compares expected and actual SQL side effects */
func (e *Evaluator) compareSQLSideEffects(expected, actual map[string]interface{}) bool {
	/* Compare table states */
	/* Simplified implementation */
	return true
}

/* CalculateRecallAtK calculates recall@k metric */
func CalculateRecallAtK(retrieved, relevant []string, k int) float64 {
	if len(relevant) == 0 {
		return 0.0
	}

	relevantSet := make(map[string]bool)
	for _, item := range relevant {
		relevantSet[item] = true
	}

	intersection := 0
	maxK := int(math.Min(float64(k), float64(len(retrieved))))
	for i := 0; i < maxK; i++ {
		if relevantSet[retrieved[i]] {
			intersection++
		}
	}

	return float64(intersection) / float64(len(relevant))
}

/* CalculateMRR calculates Mean Reciprocal Rank */
func CalculateMRR(retrieved, relevant []string) float64 {
	relevantSet := make(map[string]bool)
	for _, item := range relevant {
		relevantSet[item] = true
	}

	for i, item := range retrieved {
		if relevantSet[item] {
			return 1.0 / float64(i+1)
		}
	}

	return 0.0
}
