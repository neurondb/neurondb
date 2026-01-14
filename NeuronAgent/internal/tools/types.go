/*-------------------------------------------------------------------------
 *
 * types.go
 *    Tool implementation for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/tools/types.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"

	"github.com/neurondb/NeuronAgent/internal/db"
)

/* ToolHandler is the interface that all tool handlers must implement */
type ToolHandler interface {
	Execute(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error)
	Validate(args map[string]interface{}, schema map[string]interface{}) error
}

/* ExecutionResult represents the result of tool execution */
type ExecutionResult struct {
	Output string
	Error  error
}
