/*-------------------------------------------------------------------------
 *
 * tool_parser.go
 *    Database operations
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/tool_parser.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

/* ParseToolCalls extracts tool calls from LLM response */
/* Supports OpenAI format and custom formats */
func ParseToolCalls(response string) ([]ToolCall, error) {
	/* Try OpenAI JSON format first */
	if strings.Contains(response, "tool_calls") || strings.Contains(response, "function") {
		return parseOpenAIFormat(response)
	}

	/* Try custom format: <tool:name:args> */
	return parseCustomFormat(response)
}

func parseOpenAIFormat(response string) ([]ToolCall, error) {
	/* Look for JSON structure with tool_calls */
	var toolCalls []ToolCall

	/* Try to find JSON object with tool_calls */
	jsonRegex := regexp.MustCompile(`\{[^{}]*"tool_calls"[^{}]*\}`)
	matches := jsonRegex.FindAllString(response, -1)

	for _, match := range matches {
		var data map[string]interface{}
		if err := json.Unmarshal([]byte(match), &data); err != nil {
			continue
		}

		if tc, ok := data["tool_calls"].([]interface{}); ok {
			for _, t := range tc {
				if tcMap, ok := t.(map[string]interface{}); ok {
					call := ToolCall{
						ID:        getString(tcMap, "id"),
						Name:      getString(tcMap, "function", "name"),
						Arguments: getMap(tcMap, "function", "arguments"),
					}
					if call.Name != "" {
						toolCalls = append(toolCalls, call)
					}
				}
			}
		}
	}

	if len(toolCalls) > 0 {
		return toolCalls, nil
	}

	return nil, fmt.Errorf("no tool calls found in OpenAI format")
}

func parseCustomFormat(response string) ([]ToolCall, error) {
	/* Custom format: <tool:name:{"arg":"value"}> */
	pattern := regexp.MustCompile(`<tool:([^:]+):([^>]+)>`)
	matches := pattern.FindAllStringSubmatch(response, -1)

	var toolCalls []ToolCall
	for i, match := range matches {
		if len(match) < 3 {
			continue
		}

		name := match[1]
		argsStr := match[2]

		var args map[string]interface{}
		if err := json.Unmarshal([]byte(argsStr), &args); err != nil {
			/* If not JSON, create a simple map */
			args = map[string]interface{}{
				"input": argsStr,
			}
		}

		toolCalls = append(toolCalls, ToolCall{
			ID:        fmt.Sprintf("call_%d", i),
			Name:      name,
			Arguments: args,
		})
	}

	return toolCalls, nil
}

func getString(m map[string]interface{}, keys ...string) string {
	current := m
	for i, key := range keys {
		if i == len(keys)-1 {
			if val, ok := current[key].(string); ok {
				return val
			}
			return ""
		}
		if next, ok := current[key].(map[string]interface{}); ok {
			current = next
		} else {
			return ""
		}
	}
	return ""
}

func getMap(m map[string]interface{}, keys ...string) map[string]interface{} {
	current := m
	for i, key := range keys {
		if i == len(keys)-1 {
			if val, ok := current[key].(map[string]interface{}); ok {
				return val
			}
			if valStr, ok := current[key].(string); ok {
				var result map[string]interface{}
				if err := json.Unmarshal([]byte(valStr), &result); err == nil {
					return result
				}
			}
			return make(map[string]interface{})
		}
		if next, ok := current[key].(map[string]interface{}); ok {
			current = next
		} else {
			return make(map[string]interface{})
		}
	}
	return make(map[string]interface{})
}
