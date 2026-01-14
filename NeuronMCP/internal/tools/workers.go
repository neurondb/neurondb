/*-------------------------------------------------------------------------
 *
 * workers.go
 *    Tool implementation for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/workers.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

/* WorkerManagementTool manages background workers */
type WorkerManagementTool struct {
	*BaseTool
	executor     *QueryExecutor
	logger       *logging.Logger
	configHelper *database.ConfigHelper
}

/* NewWorkerManagementTool creates a new worker management tool */
func NewWorkerManagementTool(db *database.Database, logger *logging.Logger) *WorkerManagementTool {
	return &WorkerManagementTool{
		BaseTool: NewBaseTool(
			"postgresql_worker_management",
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
		executor:     NewQueryExecutor(db),
		logger:       logger,
		configHelper: database.NewConfigHelper(db),
	}
}

/* Execute executes worker management operation */
func (t *WorkerManagementTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for neurondb_worker_management tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	operation, _ := params["operation"].(string)

	var query string
	var queryParams []interface{}

	switch operation {
	case "status":
		/* Try to get worker configs from database */
		workerConfigs := make(map[string]interface{})
		workers := []string{"neuranq", "neuranmon", "neurandefrag"}
		for _, workerName := range workers {
			if config, err := t.configHelper.GetWorkerConfig(ctx, workerName); err == nil {
				workerConfigs[workerName] = map[string]interface{}{
					"enabled": config.Enabled,
					"naptime_ms": config.NaptimeMS,
				}
			}
		}
		
		/* Try neurondb.worker_status() first, fallback to checking if function exists */
		query = "SELECT * FROM neurondb.worker_status()"
		queryParams = []interface{}{}
		result, err := t.executor.ExecuteQueryOne(ctx, query, queryParams)
		if err != nil {
			/* Function might not exist, return configs from database instead */
			t.logger.Warn("worker_status() function not found, returning configs from database", map[string]interface{}{
				"error": err.Error(),
			})
			return Success(map[string]interface{}{
				"workers": workerConfigs,
				"note":   "Worker status function not available, showing configuration from database",
			}, map[string]interface{}{
				"operation": operation,
			}), nil
		}
		
		/* Merge database configs with status result */
		result["configs"] = workerConfigs
		
		return Success(result, map[string]interface{}{
			"operation": operation,
		}), nil
	case "list_jobs":
		query = "SELECT * FROM neurondb.list_worker_jobs()"
		queryParams = []interface{}{}
		result, err := t.executor.ExecuteQueryOne(ctx, query, queryParams)
		if err != nil {
			t.logger.Warn("list_worker_jobs() function not found", map[string]interface{}{
				"error": err.Error(),
			})
			return Success(map[string]interface{}{
				"jobs": []interface{}{},
				"note": "list_worker_jobs() function not available in this NeuronDB installation",
			}, map[string]interface{}{
				"operation": operation,
			}), nil
		}
		return Success(result, map[string]interface{}{
			"operation": operation,
		}), nil
	case "queue_job":
		jobType, _ := params["job_type"].(string)
		jobParams, _ := params["job_params"].(map[string]interface{})
		if jobType == "" {
			return Error("job_type is required for queue_job", "VALIDATION_ERROR", nil), nil
		}
   /* Format job params as JSON */
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

