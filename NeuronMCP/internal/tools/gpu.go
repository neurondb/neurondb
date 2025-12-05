package tools

import (
	"context"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

// GPUMonitoringTool monitors GPU information
type GPUMonitoringTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewGPUMonitoringTool creates a new GPU monitoring tool
func NewGPUMonitoringTool(db *database.Database, logger *logging.Logger) *GPUMonitoringTool {
	return &GPUMonitoringTool{
		BaseTool: NewBaseTool(
			"gpu_info",
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

// Execute executes GPU info query
func (t *GPUMonitoringTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
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

