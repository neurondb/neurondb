/*-------------------------------------------------------------------------
 *
 * interfaces.go
 *    Interfaces for tool composition to avoid import cycles
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/composition/interfaces.go
 *
 *-------------------------------------------------------------------------
 */

package composition

import (
	"context"
)

/* ToolRegistryInterface provides access to tools without importing tools package */
type ToolRegistryInterface interface {
	GetTool(name string) ToolInterface
}

/* ToolInterface represents a tool that can be executed */
type ToolInterface interface {
	Execute(ctx context.Context, arguments map[string]interface{}) (*ToolResult, error)
}

/* ToolResult represents a tool execution result */
type ToolResult struct {
	Success  bool
	Data     interface{}
	Error    *ToolError
	Metadata map[string]interface{}
}

/* ToolError represents a tool error */
type ToolError struct {
	Message string
	Code     string
	Details  interface{}
}

/* BaseToolInterface provides base tool functionality */
type BaseToolInterface interface {
	Name() string
	Description() string
	InputSchema() map[string]interface{}
	OutputSchema() map[string]interface{}
	Version() string
	Deprecated() bool
	Deprecation() interface{}
}
