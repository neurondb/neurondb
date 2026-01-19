/*-------------------------------------------------------------------------
 *
 * types.go
 *    Types for workflow tools to avoid import cycles
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/workflow/types.go
 *
 *-------------------------------------------------------------------------
 */

package workflow

import "github.com/neurondb/NeuronMCP/pkg/mcp"

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
func (b *BaseToolWrapper) Deprecation() *mcp.DeprecationInfo { return nil }

/* successResult creates a successful tool result */
func successResult(data interface{}) *ToolResult {
	return &ToolResult{
		Success: true,
		Data:    data,
	}
}

/* errorResult creates an error tool result */
func errorResult(message, code string, details interface{}) *ToolResult {
	return &ToolResult{
		Success: false,
		Error: &ToolError{
			Message: message,
			Code:    code,
		},
	}
}
