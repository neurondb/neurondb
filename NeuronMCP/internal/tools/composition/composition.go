/*-------------------------------------------------------------------------
 *
 * composition.go
 *    Tool composition utilities for NeuronMCP
 *
 * Provides tools for chaining, parallel execution, conditional execution,
 * and retry logic for tool composition.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/composition/composition.go
 *
 *-------------------------------------------------------------------------
 */

package composition

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/pkg/mcp"
)

/* ToolChainTool chains multiple tools sequentially */
type ToolChainTool struct {
	baseTool     *BaseToolWrapper
	toolRegistry ToolRegistryInterface
	logger       *logging.Logger
}

/* NewToolChainTool creates a new tool chain tool */
func NewToolChainTool(toolRegistry ToolRegistryInterface, logger *logging.Logger) *ToolChainTool {
	return &ToolChainTool{
		baseTool: &BaseToolWrapper{
			name:        "tool_chain",
			description: "Chain multiple tools sequentially, passing output from one to the next",
			inputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"tools": map[string]interface{}{
						"type":        "array",
						"description": "Array of tool calls to execute in sequence",
						"items": map[string]interface{}{
							"type": "object",
							"properties": map[string]interface{}{
								"tool": map[string]interface{}{
									"type":        "string",
									"description": "Tool name",
								},
								"arguments": map[string]interface{}{
									"type":        "object",
									"description": "Tool arguments (can reference previous results with {{previous.result}})",
								},
							},
							"required": []interface{}{"tool"},
						},
					},
					"stop_on_error": map[string]interface{}{
						"type":        "boolean",
						"description": "Stop chain execution on first error",
						"default":     true,
					},
				},
				"required": []interface{}{"tools"},
			},
			version: "2.0.0",
		},
		toolRegistry: toolRegistry,
		logger:       logger,
	}
}

/* Name returns the tool name */
func (t *ToolChainTool) Name() string { return t.baseTool.Name() }

/* Description returns the tool description */
func (t *ToolChainTool) Description() string { return t.baseTool.Description() }

/* InputSchema returns the input schema */
func (t *ToolChainTool) InputSchema() map[string]interface{} { return t.baseTool.InputSchema() }

/* Version returns the tool version */
func (t *ToolChainTool) Version() string { return t.baseTool.Version() }

/* OutputSchema returns the output schema */
func (t *ToolChainTool) OutputSchema() map[string]interface{} { return t.baseTool.OutputSchema() }

/* Deprecated returns whether the tool is deprecated */
func (t *ToolChainTool) Deprecated() bool { return t.baseTool.Deprecated() }

/* Deprecation returns deprecation information */
func (t *ToolChainTool) Deprecation() *mcp.DeprecationInfo { 
	if dep := t.baseTool.Deprecation(); dep != nil {
		if d, ok := dep.(*mcp.DeprecationInfo); ok {
			return d
		}
	}
	return nil
}

/* BaseToolWrapper wraps base tool functionality */
type BaseToolWrapper struct {
	name         string
	description  string
	inputSchema  map[string]interface{}
	outputSchema map[string]interface{}
	version      string
}

/* Name returns the tool name */
func (b *BaseToolWrapper) Name() string { return b.name }

/* Description returns the tool description */
func (b *BaseToolWrapper) Description() string { return b.description }

/* InputSchema returns the input schema */
func (b *BaseToolWrapper) InputSchema() map[string]interface{} { return b.inputSchema }

/* OutputSchema returns the output schema */
func (b *BaseToolWrapper) OutputSchema() map[string]interface{} { return b.outputSchema }

/* Version returns the tool version */
func (b *BaseToolWrapper) Version() string { return b.version }

/* Deprecated returns whether the tool is deprecated */
func (b *BaseToolWrapper) Deprecated() bool { return false }

/* Deprecation returns deprecation information */
func (b *BaseToolWrapper) Deprecation() interface{} { return nil }

/* Execute chains tools sequentially */
func (t *ToolChainTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	toolsList, _ := params["tools"].([]interface{})
	stopOnError, _ := params["stop_on_error"].(bool)
	if !stopOnError {
		stopOnError = true /* Default to true */
	}

	if len(toolsList) == 0 {
		return errorResult("at least one tool is required", "VALIDATION_ERROR", nil), nil
	}

	results := []map[string]interface{}{}
	previousResult := map[string]interface{}{}

	for i, toolCall := range toolsList {
		toolMap, ok := toolCall.(map[string]interface{})
		if !ok {
			return errorResult(fmt.Sprintf("invalid tool definition at index %d", i), "VALIDATION_ERROR", nil), nil
		}

		toolName, _ := toolMap["tool"].(string)
		if toolName == "" {
			return errorResult(fmt.Sprintf("tool name is required at index %d", i), "VALIDATION_ERROR", nil), nil
		}

		/* Get tool from registry */
		tool := t.toolRegistry.GetTool(toolName)
		if tool == nil {
			if stopOnError {
				return errorResult(fmt.Sprintf("tool not found: %s (at index %d)", toolName, i), "TOOL_NOT_FOUND", nil), nil
			}
			results = append(results, map[string]interface{}{
				"index":     i,
				"tool":      toolName,
				"success":   false,
				"error":     fmt.Sprintf("tool not found: %s", toolName),
			})
			continue
		}

		/* Prepare arguments with variable substitution */
		arguments, _ := toolMap["arguments"].(map[string]interface{})
		arguments = t.substituteVariables(arguments, previousResult)

		/* Execute tool */
		result, err := tool.Execute(ctx, arguments)
		if err != nil {
			if stopOnError {
				return errorResult(fmt.Sprintf("tool execution failed at index %d: %v", i, err), "EXECUTION_ERROR", map[string]interface{}{
					"tool":    toolName,
					"index":   i,
					"results": results,
				}), nil
			}
			results = append(results, map[string]interface{}{
				"index":   i,
				"tool":    toolName,
				"success": false,
				"error":   err.Error(),
			})
			continue
		}

		/* Store result */
		resultData := map[string]interface{}{
			"index":   i,
			"tool":    toolName,
			"success": result != nil && result.Success,
		}

		if result != nil {
			resultData["data"] = result.Data
			resultData["metadata"] = result.Metadata
			if result.Error != nil {
				resultData["error"] = map[string]interface{}{
					"message": result.Error.Message,
					"code":    result.Error.Code,
				}
			}
			/* Use result data as previous result for next tool */
			if result.Success && result.Data != nil {
				if dataMap, ok := result.Data.(map[string]interface{}); ok {
					previousResult = dataMap
				} else {
					previousResult = map[string]interface{}{
						"result": result.Data,
					}
				}
			}
		}

		results = append(results, resultData)
	}

	return successResult(map[string]interface{}{
		"results":      results,
		"total_tools":  len(toolsList),
		"successful":   len(results),
		"final_result": previousResult,
	}), nil
}

/* substituteVariables substitutes variables in arguments */
func (t *ToolChainTool) substituteVariables(args map[string]interface{}, previousResult map[string]interface{}) map[string]interface{} {
	if args == nil {
		return nil
	}

	result := make(map[string]interface{})
	for k, v := range args {
		result[k] = t.substituteValue(v, previousResult)
	}

	return result
}

/* substituteValue substitutes variables in a value */
func (t *ToolChainTool) substituteValue(value interface{}, previousResult map[string]interface{}) interface{} {
	switch v := value.(type) {
	case string:
		/* Check if it's a variable reference {{previous.field}} */
		if len(v) > 4 && v[:2] == "{{" && v[len(v)-2:] == "}}" {
			varPath := v[2 : len(v)-2]
			/* Support dot notation: previous.result.field */
			if varPath[:9] == "previous." {
				fieldPath := varPath[9:]
				if val := getNestedValue(previousResult, fieldPath); val != nil {
					return val
				}
			}
		}
		return v
	case map[string]interface{}:
		result := make(map[string]interface{})
		for k, val := range v {
			result[k] = t.substituteValue(val, previousResult)
		}
		return result
	case []interface{}:
		result := make([]interface{}, len(v))
		for i, val := range v {
			result[i] = t.substituteValue(val, previousResult)
		}
		return result
	default:
		return v
	}
}

/* getNestedValue gets a nested value from a map using dot notation */
func getNestedValue(m map[string]interface{}, path string) interface{} {
	parts := strings.Split(path, ".")
	current := interface{}(m)

	for _, part := range parts {
		if m, ok := current.(map[string]interface{}); ok {
			current = m[part]
		} else {
			return nil
		}
	}

	return current
}

/* ToolParallelTool executes tools in parallel */
type ToolParallelTool struct {
	baseTool     *BaseToolWrapper
	toolRegistry ToolRegistryInterface
	logger       *logging.Logger
}

/* NewToolParallelTool creates a new parallel tool executor */
func NewToolParallelTool(toolRegistry ToolRegistryInterface, logger *logging.Logger) *ToolParallelTool {
	return &ToolParallelTool{
		baseTool: &BaseToolWrapper{
			name:        "tool_parallel",
			description: "Execute multiple tools in parallel",
			inputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"tools": map[string]interface{}{
						"type":        "array",
						"description": "Array of tool calls to execute in parallel",
						"items": map[string]interface{}{
							"type": "object",
							"properties": map[string]interface{}{
								"id": map[string]interface{}{
									"type":        "string",
									"description": "Unique ID for this tool call",
								},
								"tool": map[string]interface{}{
									"type":        "string",
									"description": "Tool name",
								},
								"arguments": map[string]interface{}{
									"type":        "object",
									"description": "Tool arguments",
								},
							},
							"required": []interface{}{"id", "tool"},
						},
					},
					"timeout_seconds": map[string]interface{}{
						"type":        "number",
						"description": "Timeout for all parallel executions",
						"default":     60,
					},
				},
				"required": []interface{}{"tools"},
			},
			version: "2.0.0",
		},
		toolRegistry: toolRegistry,
		logger:       logger,
	}
}

/* Name returns the tool name */
func (t *ToolParallelTool) Name() string { return t.baseTool.Name() }

/* Description returns the tool description */
func (t *ToolParallelTool) Description() string { return t.baseTool.Description() }

/* InputSchema returns the input schema */
func (t *ToolParallelTool) InputSchema() map[string]interface{} { return t.baseTool.InputSchema() }

/* Version returns the tool version */
func (t *ToolParallelTool) Version() string { return t.baseTool.Version() }

/* OutputSchema returns the output schema */
func (t *ToolParallelTool) OutputSchema() map[string]interface{} { return t.baseTool.OutputSchema() }

/* Deprecated returns whether the tool is deprecated */
func (t *ToolParallelTool) Deprecated() bool { return t.baseTool.Deprecated() }

/* Deprecation returns deprecation information */
func (t *ToolParallelTool) Deprecation() *mcp.DeprecationInfo { 
	if dep := t.baseTool.Deprecation(); dep != nil {
		if d, ok := dep.(*mcp.DeprecationInfo); ok {
			return d
		}
	}
	return nil
}

/* Execute executes tools in parallel */
func (t *ToolParallelTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	toolsList, _ := params["tools"].([]interface{})
	timeoutSeconds, _ := params["timeout_seconds"].(float64)
	if timeoutSeconds == 0 {
		timeoutSeconds = 60
	}

	if len(toolsList) == 0 {
		return errorResult("at least one tool is required", "VALIDATION_ERROR", nil), nil
	}

	/* Create context with timeout */
	timeoutCtx, cancel := context.WithTimeout(ctx, time.Duration(timeoutSeconds)*time.Second)
	defer cancel()

	/* Execute tools in parallel */
	var wg sync.WaitGroup
	results := make(map[string]map[string]interface{})
	var mu sync.Mutex

	for _, toolCall := range toolsList {
		toolMap, ok := toolCall.(map[string]interface{})
		if !ok {
			continue
		}

		toolID, _ := toolMap["id"].(string)
		toolName, _ := toolMap["tool"].(string)
		arguments, _ := toolMap["arguments"].(map[string]interface{})

		if toolID == "" || toolName == "" {
			continue
		}

		wg.Add(1)
		go func(id, name string, args map[string]interface{}) {
			defer wg.Done()

			/* Get tool */
			tool := t.toolRegistry.GetTool(name)
			if tool == nil {
				mu.Lock()
				results[id] = map[string]interface{}{
					"id":      id,
					"tool":    name,
					"success": false,
					"error":   fmt.Sprintf("tool not found: %s", name),
				}
				mu.Unlock()
				return
			}

			/* Execute tool */
			startTime := time.Now()
			result, err := tool.Execute(timeoutCtx, args)
			duration := time.Since(startTime)

			mu.Lock()
			resultData := map[string]interface{}{
				"id":              id,
				"tool":            name,
				"execution_time_ms": duration.Milliseconds(),
			}

			if err != nil {
				resultData["success"] = false
				resultData["error"] = err.Error()
			} else if result != nil {
				resultData["success"] = result.Success
				resultData["data"] = result.Data
				if result.Error != nil {
					resultData["error"] = map[string]interface{}{
						"message": result.Error.Message,
						"code":    result.Error.Code,
					}
				}
			} else {
				resultData["success"] = false
				resultData["error"] = "tool returned nil result"
			}

			results[id] = resultData
			mu.Unlock()
		}(toolID, toolName, arguments)
	}

	/* Wait for all goroutines to complete or timeout */
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		/* All tools completed */
	case <-timeoutCtx.Done():
		/* Timeout occurred */
		return errorResult("parallel execution timeout", "TIMEOUT", map[string]interface{}{
			"timeout_seconds": timeoutSeconds,
			"results":         results,
		}), nil
	}

	/* Collect results in order */
	orderedResults := []map[string]interface{}{}
	for _, toolCall := range toolsList {
		if toolMap, ok := toolCall.(map[string]interface{}); ok {
			if toolID, ok := toolMap["id"].(string); ok {
				if result, exists := results[toolID]; exists {
					orderedResults = append(orderedResults, result)
				}
			}
		}
	}

	return successResult(map[string]interface{}{
		"results":     orderedResults,
		"total_tools": len(toolsList),
		"completed":   len(orderedResults),
	}), nil
}

/* ToolConditionalTool executes tools conditionally */
type ToolConditionalTool struct {
	baseTool     *BaseToolWrapper
	toolRegistry ToolRegistryInterface
	logger       *logging.Logger
}

/* NewToolConditionalTool creates a new conditional tool executor */
func NewToolConditionalTool(toolRegistry ToolRegistryInterface, logger *logging.Logger) *ToolConditionalTool {
	return &ToolConditionalTool{
		baseTool: &BaseToolWrapper{
			name:        "tool_conditional",
			description: "Execute a tool conditionally based on a condition",
			inputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"condition": map[string]interface{}{
						"type":        "object",
						"description": "Condition to evaluate",
						"properties": map[string]interface{}{
							"type": map[string]interface{}{
								"type":        "string",
								"description": "Condition type: equals, not_equals, greater_than, less_than, contains, exists",
								"enum":        []string{"equals", "not_equals", "greater_than", "less_than", "contains", "exists"},
							},
							"field": map[string]interface{}{
								"type":        "string",
								"description": "Field path to check (dot notation)",
							},
							"value": map[string]interface{}{
								"description": "Value to compare against",
							},
						},
						"required": []interface{}{"type", "field"},
					},
					"if_true": map[string]interface{}{
						"type":        "object",
						"description": "Tool to execute if condition is true",
						"properties": map[string]interface{}{
							"tool": map[string]interface{}{
								"type": "string",
							},
							"arguments": map[string]interface{}{
								"type": "object",
							},
						},
						"required": []interface{}{"tool"},
					},
					"if_false": map[string]interface{}{
						"type":        "object",
						"description": "Tool to execute if condition is false (optional)",
						"properties": map[string]interface{}{
							"tool": map[string]interface{}{
								"type": "string",
							},
							"arguments": map[string]interface{}{
								"type": "object",
							},
						},
					},
					"context": map[string]interface{}{
						"type":        "object",
						"description": "Context data for condition evaluation",
					},
				},
				"required": []interface{}{"condition", "if_true", "context"},
			},
			version: "2.0.0",
		},
		toolRegistry: toolRegistry,
		logger:       logger,
	}
}

/* Name returns the tool name */
func (t *ToolConditionalTool) Name() string { return t.baseTool.Name() }

/* Description returns the tool description */
func (t *ToolConditionalTool) Description() string { return t.baseTool.Description() }

/* InputSchema returns the input schema */
func (t *ToolConditionalTool) InputSchema() map[string]interface{} { return t.baseTool.InputSchema() }

/* Version returns the tool version */
func (t *ToolConditionalTool) Version() string { return t.baseTool.Version() }

/* OutputSchema returns the output schema */
func (t *ToolConditionalTool) OutputSchema() map[string]interface{} { return t.baseTool.OutputSchema() }

/* Deprecated returns whether the tool is deprecated */
func (t *ToolConditionalTool) Deprecated() bool { return t.baseTool.Deprecated() }

/* Deprecation returns deprecation information */
func (t *ToolConditionalTool) Deprecation() *mcp.DeprecationInfo { 
	if dep := t.baseTool.Deprecation(); dep != nil {
		if d, ok := dep.(*mcp.DeprecationInfo); ok {
			return d
		}
	}
	return nil
}

/* Execute executes tool conditionally */
func (t *ToolConditionalTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	condition, _ := params["condition"].(map[string]interface{})
	ifTrue, _ := params["if_true"].(map[string]interface{})
	ifFalse, _ := params["if_false"].(map[string]interface{})
	contextData, _ := params["context"].(map[string]interface{})

	if condition == nil || ifTrue == nil || contextData == nil {
		return errorResult("condition, if_true, and context are required", "VALIDATION_ERROR", nil), nil
	}

	/* Evaluate condition */
	conditionMet, err := t.evaluateCondition(condition, contextData)
	if err != nil {
		return errorResult(fmt.Sprintf("failed to evaluate condition: %v", err), "CONDITION_ERROR", nil), nil
	}

	/* Execute appropriate tool */
	var toolToExecute map[string]interface{}
	if conditionMet {
		toolToExecute = ifTrue
	} else {
		if ifFalse != nil {
			toolToExecute = ifFalse
		} else {
			return successResult(map[string]interface{}{
				"condition_met": false,
				"executed":      false,
				"message":       "condition was false and no if_false tool provided",
			}), nil
		}
	}

	toolName, _ := toolToExecute["tool"].(string)
	arguments, _ := toolToExecute["arguments"].(map[string]interface{})

	if toolName == "" {
		return errorResult("tool name is required", "VALIDATION_ERROR", nil), nil
	}

	tool := t.toolRegistry.GetTool(toolName)
	if tool == nil {
		return errorResult(fmt.Sprintf("tool not found: %s", toolName), "TOOL_NOT_FOUND", nil), nil
	}

	result, err := tool.Execute(ctx, arguments)
	if err != nil {
		return errorResult(fmt.Sprintf("tool execution failed: %v", err), "EXECUTION_ERROR", nil), nil
	}

	return successResult(map[string]interface{}{
		"condition_met": conditionMet,
		"executed_tool": toolName,
		"result":        result,
	}), nil
}

/* evaluateCondition evaluates a condition against context data */
func (t *ToolConditionalTool) evaluateCondition(condition map[string]interface{}, context map[string]interface{}) (bool, error) {
	conditionType, _ := condition["type"].(string)
	field, _ := condition["field"].(string)
	value := condition["value"]

	if conditionType == "" || field == "" {
		return false, fmt.Errorf("condition type and field are required")
	}

	/* Get field value from context */
	fieldValue := getNestedValue(context, field)

	switch conditionType {
	case "equals":
		return fieldValue == value, nil
	case "not_equals":
		return fieldValue != value, nil
	case "greater_than":
		return compareNumbers(fieldValue, value) > 0, nil
	case "less_than":
		return compareNumbers(fieldValue, value) < 0, nil
	case "contains":
		if str, ok := fieldValue.(string); ok {
			if valStr, ok := value.(string); ok {
				return strings.Contains(str, valStr), nil
			}
		}
		return false, nil
	case "exists":
		return fieldValue != nil, nil
	default:
		return false, fmt.Errorf("unknown condition type: %s", conditionType)
	}
}

/* compareNumbers compares two numbers */
func compareNumbers(a, b interface{}) int {
	aFloat := toFloat(a)
	bFloat := toFloat(b)
	if aFloat > bFloat {
		return 1
	} else if aFloat < bFloat {
		return -1
	}
	return 0
}

/* toFloat converts a value to float64 */
func toFloat(v interface{}) float64 {
	switch val := v.(type) {
	case float64:
		return val
	case float32:
		return float64(val)
	case int:
		return float64(val)
	case int64:
		return float64(val)
	default:
		return 0
	}
}

/* ToolRetryTool retries a tool with backoff */
type ToolRetryTool struct {
	baseTool     *BaseToolWrapper
	toolRegistry ToolRegistryInterface
	logger       *logging.Logger
}

/* NewToolRetryTool creates a new retry tool */
func NewToolRetryTool(toolRegistry ToolRegistryInterface, logger *logging.Logger) *ToolRetryTool {
	return &ToolRetryTool{
		baseTool: &BaseToolWrapper{
			name:        "tool_retry",
			description: "Retry a tool execution with exponential backoff",
			inputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"tool": map[string]interface{}{
						"type":        "string",
						"description": "Tool name to retry",
					},
					"arguments": map[string]interface{}{
						"type":        "object",
						"description": "Tool arguments",
					},
					"max_retries": map[string]interface{}{
						"type":        "number",
						"description": "Maximum number of retries",
						"default":     3,
					},
					"initial_delay_ms": map[string]interface{}{
						"type":        "number",
						"description": "Initial delay in milliseconds",
						"default":     1000,
					},
					"max_delay_ms": map[string]interface{}{
						"type":        "number",
						"description": "Maximum delay in milliseconds",
						"default":     10000,
					},
					"backoff_multiplier": map[string]interface{}{
						"type":        "number",
						"description": "Backoff multiplier",
						"default":     2.0,
					},
				},
				"required": []interface{}{"tool", "arguments"},
			},
			version: "2.0.0",
		},
		toolRegistry: toolRegistry,
		logger:       logger,
	}
}

/* Name returns the tool name */
func (t *ToolRetryTool) Name() string { return t.baseTool.Name() }

/* Description returns the tool description */
func (t *ToolRetryTool) Description() string { return t.baseTool.Description() }

/* InputSchema returns the input schema */
func (t *ToolRetryTool) InputSchema() map[string]interface{} { return t.baseTool.InputSchema() }

/* Version returns the tool version */
func (t *ToolRetryTool) Version() string { return t.baseTool.Version() }

/* OutputSchema returns the output schema */
func (t *ToolRetryTool) OutputSchema() map[string]interface{} { return t.baseTool.OutputSchema() }

/* Deprecated returns whether the tool is deprecated */
func (t *ToolRetryTool) Deprecated() bool { return t.baseTool.Deprecated() }

/* Deprecation returns deprecation information */
func (t *ToolRetryTool) Deprecation() *mcp.DeprecationInfo { 
	if dep := t.baseTool.Deprecation(); dep != nil {
		if d, ok := dep.(*mcp.DeprecationInfo); ok {
			return d
		}
	}
	return nil
}

/* Execute retries tool execution */
func (t *ToolRetryTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	toolName, _ := params["tool"].(string)
	arguments, _ := params["arguments"].(map[string]interface{})
	maxRetries, _ := params["max_retries"].(float64)
	if maxRetries == 0 {
		maxRetries = 3
	}
	initialDelay, _ := params["initial_delay_ms"].(float64)
	if initialDelay == 0 {
		initialDelay = 1000
	}
	maxDelay, _ := params["max_delay_ms"].(float64)
	if maxDelay == 0 {
		maxDelay = 10000
	}
	backoffMultiplier, _ := params["backoff_multiplier"].(float64)
	if backoffMultiplier == 0 {
		backoffMultiplier = 2.0
	}

	if toolName == "" {
		return errorResult("tool name is required", "VALIDATION_ERROR", nil), nil
	}

	tool := t.toolRegistry.GetTool(toolName)
	if tool == nil {
		return errorResult(fmt.Sprintf("tool not found: %s", toolName), "TOOL_NOT_FOUND", nil), nil
	}

	delay := time.Duration(initialDelay) * time.Millisecond
	var lastErr error
	var lastResult *ToolResult

	for attempt := 0; attempt <= int(maxRetries); attempt++ {
		if attempt > 0 {
			/* Wait before retry */
			select {
			case <-ctx.Done():
				return errorResult("context cancelled during retry", "CANCELLED", nil), nil
			case <-time.After(delay):
			}

			/* Increase delay for next retry */
			delay = time.Duration(float64(delay) * backoffMultiplier)
			if delay > time.Duration(maxDelay)*time.Millisecond {
				delay = time.Duration(maxDelay) * time.Millisecond
			}
		}

		result, err := tool.Execute(ctx, arguments)
		if err == nil && result != nil && result.Success {
			return successResult(map[string]interface{}{
				"result":   result,
				"attempts": attempt + 1,
				"success":  true,
			}), nil
		}

		lastErr = err
		lastResult = result

		if t.logger != nil {
			t.logger.Warn("Tool retry attempt failed", map[string]interface{}{
				"tool":       toolName,
				"attempt":    attempt + 1,
				"max_retries": maxRetries,
				"error":      err,
			})
		}
	}

	/* All retries exhausted */
	return errorResult(fmt.Sprintf("tool failed after %d attempts: %v", int(maxRetries)+1, lastErr), "RETRY_EXHAUSTED", map[string]interface{}{
		"attempts":    int(maxRetries) + 1,
		"last_result": lastResult,
		"last_error":  lastErr,
	}), nil
}
