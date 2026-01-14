/*-------------------------------------------------------------------------
 *
 * interfaces.go
 *    Tool interfaces for NeuronMCP
 *
 * Defines the core interfaces that all tools must implement.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/interfaces.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"

	"github.com/neurondb/NeuronMCP/pkg/mcp"
)

/* Tool is the interface that all tools must implement */
type Tool interface {
	Name() string
	Description() string
	InputSchema() map[string]interface{}
	OutputSchema() map[string]interface{}
	Version() string
	Deprecated() bool
	Deprecation() *mcp.DeprecationInfo
	Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error)
}

/* ToolExecutor provides database query execution for tools */
type ToolExecutor interface {
	ExecuteQuery(ctx context.Context, query string, params []interface{}) ([]map[string]interface{}, error)
	ExecuteQueryOne(ctx context.Context, query string, params []interface{}) (map[string]interface{}, error)
	ExecuteVectorSearch(ctx context.Context, table, vectorColumn string, queryVector []interface{}, distanceMetric string, limit int, additionalColumns []interface{}) ([]map[string]interface{}, error)
}

