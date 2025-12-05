package tools

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

// WorkerManagementTool manages background workers
type WorkerManagementTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewWorkerManagementTool creates a new worker management tool
func NewWorkerManagementTool(db *database.Database, logger *logging.Logger) *WorkerManagementTool {
	return &WorkerManagementTool{
		BaseTool: NewBaseTool(
			"worker_management",
			"Manage background workers: status, jobs, queue",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"operation": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"status", "list_jobs", "queue_job", "cancel_job"},
						"description": "Worker operation",
					},
					"job_id": map[string]interface{}{
						"type":        "number",
						"description": "Job ID (for cancel_job)",
					},
					"job_type": map[string]interface{}{
						"type":        "string",
						"description": "Job type (for queue_job)",
					},
					"job_params": map[string]interface{}{
						"type":        "object",
						"description": "Job parameters (for queue_job)",
					},
				},
				"required": []interface{}{"operation"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes worker management operation
func (t *WorkerManagementTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for worker_management tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	operation, _ := params["operation"].(string)

	var query string
	var queryParams []interface{}

	switch operation {
	case "status":
		query = "SELECT * FROM neurondb.worker_status()"
		queryParams = []interface{}{}
	case "list_jobs":
		query = "SELECT * FROM neurondb.list_worker_jobs()"
		queryParams = []interface{}{}
	case "queue_job":
		jobType, _ := params["job_type"].(string)
		jobParams, _ := params["job_params"].(map[string]interface{})
		if jobType == "" {
			return Error("job_type is required for queue_job", "VALIDATION_ERROR", nil), nil
		}
		// Format job params as JSON
		paramsJSON := "{}"
		if len(jobParams) > 0 {
			paramsBytes, err := json.Marshal(jobParams)
			if err == nil {
				paramsJSON = string(paramsBytes)
			}
		}
		query = "SELECT neurondb.queue_worker_job($1::text, $2::jsonb) AS job_id"
		queryParams = []interface{}{jobType, paramsJSON}
	case "cancel_job":
		jobID, ok := params["job_id"].(float64)
		if !ok {
			return Error("job_id is required for cancel_job", "VALIDATION_ERROR", nil), nil
		}
		query = "SELECT neurondb.cancel_worker_job($1::int) AS success"
		queryParams = []interface{}{int(jobID)}
	default:
		return Error(fmt.Sprintf("Unknown operation: %s", operation), "VALIDATION_ERROR", nil), nil
	}

	result, err := t.executor.ExecuteQueryOne(ctx, query, queryParams)
	if err != nil {
		t.logger.Error("Worker management operation failed", err, params)
		return Error(fmt.Sprintf("Worker management operation failed: operation='%s', error=%v", operation, err), "EXECUTION_ERROR", map[string]interface{}{
			"operation": operation,
			"error":    err.Error(),
		}), nil
	}

	return Success(result, map[string]interface{}{
		"operation": operation,
	}), nil
}

