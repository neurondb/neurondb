/*-------------------------------------------------------------------------
 *
 * adapter.go
 *    Tool executor adapter for workflow engine
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/workflow/adapter.go
 *
 *-------------------------------------------------------------------------
 */

package workflow

import (
	"context"
	"fmt"
)

/* ToolExecutorAdapter adapts a tool registry interface to ToolExecutor */
/* This avoids import cycles by using an interface */
type ToolRegistryInterface interface {
	GetTool(name string) ToolInterface
}

/* ToolInterface represents a tool that can be executed */
type ToolInterface interface {
	Execute(ctx context.Context, arguments map[string]interface{}) (*ToolResult, error)
}

/* ToolResult represents a tool execution result */
type ToolResult struct {
	Success bool
	Data     interface{}
	Error    *ToolError
}

/* ToolError represents a tool error */
type ToolError struct {
	Message string
	Code    string
}

/* ToolExecutorAdapter adapts ToolRegistry to ToolExecutor interface */
type ToolExecutorAdapter struct {
	Registry ToolRegistryInterface
}

/* ExecuteTool executes a tool from the registry */
func (a *ToolExecutorAdapter) ExecuteTool(ctx context.Context, toolName string, arguments map[string]interface{}) (interface{}, error) {
	if a.Registry == nil {
		return nil, fmt.Errorf("tool registry is nil")
	}

	tool := a.Registry.GetTool(toolName)
	if tool == nil {
		return nil, fmt.Errorf("tool not found: %s", toolName)
	}

	result, err := tool.Execute(ctx, arguments)
	if err != nil {
		return nil, err
	}

	if result == nil {
		return nil, fmt.Errorf("tool returned nil result")
	}

	if !result.Success {
		if result.Error != nil {
			return nil, fmt.Errorf("tool execution failed: %s (code: %s)", result.Error.Message, result.Error.Code)
		}
		return nil, fmt.Errorf("tool execution failed")
	}

	return result.Data, nil
}
