/*-------------------------------------------------------------------------
 *
 * composer.go
 *    Tool composition and chaining
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/tools/composer.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"fmt"
	"sync"
)

/* ToolCall represents a tool call (duplicated from agent to avoid import cycle) */
type ToolCall struct {
	ID        string
	Name      string
	Arguments map[string]interface{}
}

/* ToolResult represents the result of a tool call (duplicated from agent to avoid import cycle) */
type ToolResult struct {
	ToolCallID string
	Content    string
	Error      error
}

type ToolComposer struct {
	registry *Registry
}

/* NewToolComposer creates a new tool composer */
func NewToolComposer(registry *Registry) *ToolComposer {
	return &ToolComposer{registry: registry}
}

/* Compose executes multiple tools in sequence */
func (c *ToolComposer) Compose(ctx context.Context, tools []ToolCall) ([]ToolResult, error) {
	results := make([]ToolResult, len(tools))

		for i, call := range tools {
		tool, err := c.registry.Get(ctx, call.Name)
		if err != nil {
			results[i] = ToolResult{
				ToolCallID: call.ID,
				Error:      err,
			}
			continue
		}

		result, err := c.registry.ExecuteTool(ctx, tool, call.Arguments)
		if err != nil {
			results[i] = ToolResult{
				ToolCallID: call.ID,
				Content:    result,
				Error:      err,
			}
			continue
		}

		results[i] = ToolResult{
			ToolCallID: call.ID,
			Content:    result,
			Error:      nil,
		}

		/* Use previous result as input for next tool if needed */
		if i < len(tools)-1 && tools[i+1].Arguments == nil {
			tools[i+1].Arguments = map[string]interface{}{
				"previous_result": result,
			}
		}
	}

	return results, nil
}

/* ComposeParallel executes multiple tools in parallel */
func (c *ToolComposer) ComposeParallel(ctx context.Context, tools []ToolCall) ([]ToolResult, error) {
	results := make([]ToolResult, len(tools))
	var wg sync.WaitGroup
	var mu sync.Mutex

	for i, call := range tools {
		wg.Add(1)
		go func(idx int, toolCall ToolCall) {
			defer wg.Done()

			tool, err := c.registry.Get(ctx, toolCall.Name)
			if err != nil {
				mu.Lock()
				results[idx] = ToolResult{
					ToolCallID: toolCall.ID,
					Error:      err,
				}
				mu.Unlock()
				return
			}

			result, err := c.registry.ExecuteTool(ctx, tool, toolCall.Arguments)
			mu.Lock()
			results[idx] = ToolResult{
				ToolCallID: toolCall.ID,
				Content:    result,
				Error:      err,
			}
			mu.Unlock()
		}(i, call)
	}

	wg.Wait()
	return results, nil
}

/* ComposeConditional executes tools conditionally based on previous results */
func (c *ToolComposer) ComposeConditional(ctx context.Context, tools []ConditionalToolCall) ([]ToolResult, error) {
	var results []ToolResult

	for _, call := range tools {
		/* Check condition */
		if call.Condition != nil {
			shouldExecute, err := c.evaluateCondition(call.Condition, results)
			if err != nil {
				return results, fmt.Errorf("condition evaluation failed: error=%w", err)
			}
			if !shouldExecute {
				continue
			}
		}

		tool, err := c.registry.Get(ctx, call.ToolCall.Name)
		if err != nil {
			results = append(results, ToolResult{
				ToolCallID: call.ToolCall.ID,
				Error:      err,
			})
			continue
		}

		result, err := c.registry.ExecuteTool(ctx, tool, call.ToolCall.Arguments)
		results = append(results, ToolResult{
			ToolCallID: call.ToolCall.ID,
			Content:    result,
			Error:      err,
		})
	}

	return results, nil
}

/* evaluateCondition evaluates a condition based on previous results */
func (c *ToolComposer) evaluateCondition(condition map[string]interface{}, previousResults []ToolResult) (bool, error) {
	/* Simple condition evaluation */
	/* In production, this would support more complex logic */
	if op, ok := condition["operator"].(string); ok {
		switch op {
		case "equals":
			value := condition["value"]
			/* Check last result */
			if len(previousResults) > 0 {
				lastResult := previousResults[len(previousResults)-1]
				/* Simple string comparison */
				return lastResult.Content == fmt.Sprintf("%v", value), nil
			}
		case "contains":
			value := condition["value"].(string)
			if len(previousResults) > 0 {
				lastResult := previousResults[len(previousResults)-1]
				return containsString(lastResult.Content, value), nil
			}
		}
	}

	return true, nil
}

func containsString(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(substr) == 0 ||
		(len(s) > len(substr) && (s[:len(substr)] == substr || s[len(s)-len(substr):] == substr ||
			containsString(s[1:], substr))))
}

type ConditionalToolCall struct {
	ToolCall  ToolCall
	Condition map[string]interface{}
}
