/*-------------------------------------------------------------------------
 *
 * gpu.go
 *    Tool implementation for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/gpu.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

/* GPUMonitoringTool monitors GPU information */
type GPUMonitoringTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewGPUMonitoringTool creates a new GPU monitoring tool */
func NewGPUMonitoringTool(db *database.Database, logger *logging.Logger) *GPUMonitoringTool {
	return &GPUMonitoringTool{
		BaseTool: NewBaseTool(
			"postgresql_gpu_info",
			"Get GPU information and monitoring data",
			map[string]interface{}{
				"type":       "object",
				"properties": map[string]interface{}{},
				"required":   []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes GPU info query */
func (t *GPUMonitoringTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	/* Check if function exists first */
	checkQuery := `
		SELECT EXISTS (
			SELECT FROM pg_proc p
			JOIN pg_namespace n ON p.pronamespace = n.oid
			WHERE n.nspname = 'neurondb' 
			AND p.proname = 'gpu_info'
		) AS function_exists
	`
	exists, err := t.executor.ExecuteQueryOne(ctx, checkQuery, nil)
	if err != nil {
		t.logger.Warn("Failed to check if gpu_info function exists", map[string]interface{}{
			"error": err.Error(),
		})
		return Success(map[string]interface{}{
			"gpu_available": false,
			"message":       "GPU info function not available",
		}, nil), nil
	}

	if functionExists, ok := exists["function_exists"].(bool); !ok || !functionExists {
		return Success(map[string]interface{}{
			"gpu_available": false,
			"message":       "GPU info function not available in this installation",
		}, nil), nil
	}

	query := "SELECT * FROM neurondb.gpu_info()"
	result, err := t.executor.ExecuteQueryOne(ctx, query, nil)
	if err != nil {
		t.logger.Error("GPU info query failed", err, nil)
		return Error(fmt.Sprintf("GPU info query failed: error=%v", err), "EXECUTION_ERROR", map[string]interface{}{
			"error": err.Error(),
		}), nil
	}

	return Success(result, nil), nil
}





