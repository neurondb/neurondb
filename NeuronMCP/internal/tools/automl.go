/*-------------------------------------------------------------------------
 *
 * automl.go
 *    Tool implementation for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/automl.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/validation"
)

/* AutoMLTool performs automated machine learning */
type AutoMLTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewAutoMLTool creates a new AutoML tool */
func NewAutoMLTool(db *database.Database, logger *logging.Logger) *AutoMLTool {
	return &AutoMLTool{
		BaseTool: NewBaseTool(
			"postgresql_automl",
			"Automated machine learning: model selection, hyperparameter tuning",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"operation": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"select_model", "tune_hyperparameters", "auto_train"},
						"description": "AutoML operation",
					},
					"table": map[string]interface{}{
						"type":        "string",
						"description": "Training data table name",
					},
					"feature_col": map[string]interface{}{
						"type":        "string",
						"description": "Feature column name",
					},
					"label_col": map[string]interface{}{
						"type":        "string",
						"description": "Label column name",
					},
					"task_type": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"classification", "regression"},
						"description": "Task type",
					},
					"algorithms": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "string"},
						"description": "Algorithms to consider (optional)",
					},
				},
				"required": []interface{}{"operation", "table", "feature_col", "label_col", "task_type"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes AutoML operation */
func (t *AutoMLTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for neurondb_automl tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	operation, _ := params["operation"].(string)
	table, _ := params["table"].(string)
	featureCol, _ := params["feature_col"].(string)
	labelCol, _ := params["label_col"].(string)
	taskType, _ := params["task_type"].(string)

	/* Validate table name (SQL identifier) */
	if err := validation.ValidateSQLIdentifierRequired(table, "table"); err != nil {
		return Error(fmt.Sprintf("Invalid table parameter: %v", err), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "table",
			"error":     err.Error(),
			"params":    params,
		}), nil
	}

	/* Validate feature_col (SQL identifier) */
	if err := validation.ValidateSQLIdentifierRequired(featureCol, "feature_col"); err != nil {
		return Error(fmt.Sprintf("Invalid feature_col parameter: %v", err), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "feature_col",
			"table":     table,
			"error":     err.Error(),
			"params":    params,
		}), nil
	}

	/* Validate label_col (SQL identifier) */
	if err := validation.ValidateSQLIdentifierRequired(labelCol, "label_col"); err != nil {
		return Error(fmt.Sprintf("Invalid label_col parameter: %v", err), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "label_col",
			"table":     table,
			"error":     err.Error(),
			"params":    params,
		}), nil
	}

	/* Validate task_type */
	if taskType == "" {
		return Error("task_type is required and cannot be empty", "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "task_type",
			"table":     table,
			"params":    params,
		}), nil
	}

	var query string
	var queryParams []interface{}

	switch operation {
	case "select_model":
		query = "SELECT * FROM automl_select_model($1::text, $2::text, $3::text, $4::text)"
		queryParams = []interface{}{table, featureCol, labelCol, taskType}
	case "tune_hyperparameters":
		algorithm, _ := params["algorithm"].(string)
		if algorithm == "" {
			return Error("algorithm is required for tune_hyperparameters", "VALIDATION_ERROR", nil), nil
		}
		query = "SELECT * FROM automl_tune_hyperparameters($1::text, $2::text, $3::text, $4::text, $5::text)"
		queryParams = []interface{}{table, featureCol, labelCol, taskType, algorithm}
	case "auto_train":
		query = "SELECT * FROM automl_auto_train($1::text, $2::text, $3::text, $4::text)"
		queryParams = []interface{}{table, featureCol, labelCol, taskType}
	default:
		return Error(fmt.Sprintf("Unknown operation: %s", operation), "VALIDATION_ERROR", nil), nil
	}

	results, err := t.executor.ExecuteQuery(ctx, query, queryParams)
	if err != nil {
		t.logger.Error("AutoML operation failed", err, params)
		return Error(fmt.Sprintf("AutoML operation failed: operation='%s', error=%v", operation, err), "EXECUTION_ERROR", map[string]interface{}{
			"operation": operation,
			"error":    err.Error(),
		}), nil
	}

	return Success(map[string]interface{}{
		"results": results,
		"count":   len(results),
	}, map[string]interface{}{
		"operation": operation,
		"task_type": taskType,
		"count":     len(results),
	}), nil
}









