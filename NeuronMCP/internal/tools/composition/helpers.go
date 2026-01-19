/*-------------------------------------------------------------------------
 *
 * helpers.go
 *    Helper functions for tool composition
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/composition/helpers.go
 *
 *-------------------------------------------------------------------------
 */

package composition

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
			Details: details,
		},
	}
}
