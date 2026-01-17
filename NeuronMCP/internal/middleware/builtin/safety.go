/*-------------------------------------------------------------------------
 *
 * safety.go
 *    Safety middleware for NeuronMCP
 *
 * Provides safety checks for tool calls, enforcing read-only mode by default
 * and validating statements against allowlists.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/middleware/builtin/safety.go
 *
 *-------------------------------------------------------------------------
 */

package builtin

import (
	"context"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/middleware"
	"github.com/neurondb/NeuronMCP/internal/safety"
)

/* SafetyMiddleware enforces safety modes */
type SafetyMiddleware struct {
	safetyManager *safety.SafetyManager
	logger        *logging.Logger
}

/* NewSafetyMiddleware creates a new safety middleware */
func NewSafetyMiddleware(safetyManager *safety.SafetyManager, logger *logging.Logger) *SafetyMiddleware {
	return &SafetyMiddleware{
		safetyManager: safetyManager,
		logger:        logger,
	}
}

/* Name returns the middleware name */
func (m *SafetyMiddleware) Name() string {
	return "safety"
}

/* Order returns the execution order - runs early to block unsafe operations */
func (m *SafetyMiddleware) Order() int {
	return 5
}

/* Enabled returns whether the middleware is enabled */
func (m *SafetyMiddleware) Enabled() bool {
	return m.safetyManager != nil
}

/* Execute executes the safety middleware */
func (m *SafetyMiddleware) Execute(ctx context.Context, req *middleware.MCPRequest, next middleware.Handler) (*middleware.MCPResponse, error) {
	/* Only check tools/call requests */
	if req.Method != "tools/call" {
		return next(ctx, req)
	}

	/* Extract tool name and arguments */
	toolName, _ := req.Params["name"].(string)
	arguments, _ := req.Params["arguments"].(map[string]interface{})

	if arguments == nil {
		arguments = make(map[string]interface{})
	}

	/* Check if explicit write access is requested */
	allowWrite := false
	if val, ok := arguments["allow_write"].(bool); ok {
		allowWrite = val
	}

	/* For execute_query tool, validate the SQL statement */
	if toolName == "postgresql_execute_query" {
		query, ok := arguments["query"].(string)
		if ok && query != "" {
			if err := m.safetyManager.ValidateStatement(query, allowWrite); err != nil {
				m.logger.Warn("Safety violation blocked", map[string]interface{}{
					"tool_name":    toolName,
					"error":        err.Error(),
					"query_preview": getQueryPreview(query),
				})

				return &middleware.MCPResponse{
					Content: []middleware.ContentBlock{
						{
							Type: "text",
							Text: fmt.Sprintf("Safety violation: %s", err.Error()),
						},
					},
					IsError: true,
					Metadata: map[string]interface{}{
						"error_code": "SAFETY_VIOLATION",
						"tool":       toolName,
						"error":      err.Error(),
					},
				}, nil
			}
		}
	}

	/* For other tools that perform writes, check if write access is allowed */
	/* This is a simplified check - in production, you'd want to mark tools as read/write */
	if !allowWrite && m.safetyManager.IsReadOnly() {
		/* Check if this is a known write tool */
		if isWriteTool(toolName) {
			m.logger.Warn("Write tool blocked in read-only mode", map[string]interface{}{
				"tool_name": toolName,
			})

			return &middleware.MCPResponse{
				Content: []middleware.ContentBlock{
					{
						Type: "text",
						Text: fmt.Sprintf("Read-only mode: tool '%s' requires write access. Set allow_write=true to override", toolName),
					},
				},
				IsError: true,
				Metadata: map[string]interface{}{
					"error_code": "READ_ONLY_VIOLATION",
					"tool":       toolName,
				},
			}, nil
		}
	}

	return next(ctx, req)
}

/* isWriteTool checks if a tool performs write operations */
func isWriteTool(toolName string) bool {
	writeTools := map[string]bool{
		"postgresql_insert":           true,
		"postgresql_update":           true,
		"postgresql_delete":           true,
		"postgresql_create_table":     true,
		"postgresql_alter_table":     true,
		"postgresql_drop_table":      true,
		"postgresql_create_index":     true,
		"postgresql_drop_index":      true,
		"postgresql_create_database":  true,
		"postgresql_drop_database":   true,
		"postgresql_create_schema":    true,
		"postgresql_drop_schema":     true,
		"train_model":                true,
		"delete_model":                true,
		"process_document":            true,
		"create_hnsw_index":          true,
		"create_ivf_index":            true,
		"drop_index":                  true,
	}
	return writeTools[toolName]
}

/* getQueryPreview returns a preview of the query */
func getQueryPreview(query string) string {
	previewLen := 100
	if len(query) < previewLen {
		previewLen = len(query)
	}
	if previewLen == 0 {
		return ""
	}
	return query[:previewLen]
}



