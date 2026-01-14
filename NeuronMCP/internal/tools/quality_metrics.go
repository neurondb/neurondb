/*-------------------------------------------------------------------------
 *
 * quality_metrics.go
 *    Tool implementation for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/quality_metrics.go
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

/* QualityMetricsTool computes quality metrics for search results */
type QualityMetricsTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewQualityMetricsTool creates a new quality metrics tool */
func NewQualityMetricsTool(db *database.Database, logger *logging.Logger) *QualityMetricsTool {
	return &QualityMetricsTool{
		BaseTool: NewBaseTool(
			"postgresql_quality_metrics",
			"Compute quality metrics: Recall@K, Precision@K, F1@K, MRR, Davies-Bouldin Index",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"metric": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"recall_at_k", "precision_at_k", "f1_at_k", "mrr", "davies_bouldin"},
						"description": "Quality metric to compute",
					},
					"table": map[string]interface{}{
						"type":        "string",
						"description": "Table name with results",
					},
					"k": map[string]interface{}{
						"type":        "number",
						"description": "K value for @K metrics",
					},
					"ground_truth_col": map[string]interface{}{
						"type":        "string",
						"description": "Ground truth column name",
					},
					"predicted_col": map[string]interface{}{
						"type":        "string",
						"description": "Predicted results column name",
					},
				},
				"required": []interface{}{"metric", "table"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes quality metrics computation */
func (t *QualityMetricsTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for neurondb_quality_metrics tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	metric, _ := params["metric"].(string)
	table, _ := params["table"].(string)

	/* Validate table name (SQL identifier) */
	if err := validation.ValidateSQLIdentifierRequired(table, "table"); err != nil {
		return Error(fmt.Sprintf("Invalid table parameter: %v", err), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "table",
			"error":     err.Error(),
			"params":    params,
		}), nil
	}

  /* Build query based on metric type */
	var query string
	var queryParams []interface{}

	switch metric {
	case "recall_at_k", "precision_at_k", "f1_at_k":
		k, ok := params["k"].(float64)
		if !ok {
			return Error("k is required for @K metrics", "VALIDATION_ERROR", nil), nil
		}
		groundTruthCol, _ := params["ground_truth_col"].(string)
		predictedCol, _ := params["predicted_col"].(string)
		if err := validation.ValidateSQLIdentifierRequired(groundTruthCol, "ground_truth_col"); err != nil {
			return Error(fmt.Sprintf("Invalid ground_truth_col parameter: %v", err), "VALIDATION_ERROR", map[string]interface{}{
				"parameter": "ground_truth_col",
				"table":     table,
				"error":     err.Error(),
				"params":    params,
			}), nil
		}
		if err := validation.ValidateSQLIdentifierRequired(predictedCol, "predicted_col"); err != nil {
			return Error(fmt.Sprintf("Invalid predicted_col parameter: %v", err), "VALIDATION_ERROR", map[string]interface{}{
				"parameter": "predicted_col",
				"table":     table,
				"error":     err.Error(),
				"params":    params,
			}), nil
		}
   /* Use appropriate NeuronDB function */
		funcName := "recall_at_k"
		if metric == "precision_at_k" {
			funcName = "precision_at_k"
		} else if metric == "f1_at_k" {
			funcName = "f1_at_k"
		}
		query = fmt.Sprintf("SELECT %s($1::text, $2::text, $3::text, $4::int) AS metric_value", funcName)
		queryParams = []interface{}{table, groundTruthCol, predictedCol, int(k)}
	case "mrr":
		groundTruthCol, _ := params["ground_truth_col"].(string)
		predictedCol, _ := params["predicted_col"].(string)
		if err := validation.ValidateSQLIdentifierRequired(groundTruthCol, "ground_truth_col"); err != nil {
			return Error(fmt.Sprintf("Invalid ground_truth_col parameter: %v", err), "VALIDATION_ERROR", map[string]interface{}{
				"parameter": "ground_truth_col",
				"table":     table,
				"error":     err.Error(),
				"params":    params,
			}), nil
		}
		if err := validation.ValidateSQLIdentifierRequired(predictedCol, "predicted_col"); err != nil {
			return Error(fmt.Sprintf("Invalid predicted_col parameter: %v", err), "VALIDATION_ERROR", map[string]interface{}{
				"parameter": "predicted_col",
				"table":     table,
				"error":     err.Error(),
				"params":    params,
			}), nil
		}
		query = "SELECT mrr($1::text, $2::text, $3::text) AS metric_value"
		queryParams = []interface{}{table, groundTruthCol, predictedCol}
	case "davies_bouldin":
		vectorCol, _ := params["vector_column"].(string)
		clusterCol, _ := params["cluster_column"].(string)
		if err := validation.ValidateSQLIdentifierRequired(vectorCol, "vector_column"); err != nil {
			return Error(fmt.Sprintf("Invalid vector_column parameter: %v", err), "VALIDATION_ERROR", map[string]interface{}{
				"parameter": "vector_column",
				"table":     table,
				"error":     err.Error(),
				"params":    params,
			}), nil
		}
		if err := validation.ValidateSQLIdentifierRequired(clusterCol, "cluster_column"); err != nil {
			return Error(fmt.Sprintf("Invalid cluster_column parameter: %v", err), "VALIDATION_ERROR", map[string]interface{}{
				"parameter": "cluster_column",
				"table":     table,
				"error":     err.Error(),
				"params":    params,
			}), nil
		}
		query = "SELECT davies_bouldin_index($1::text, $2::text, $3::text) AS metric_value"
		queryParams = []interface{}{table, vectorCol, clusterCol}
	default:
		return Error(fmt.Sprintf("Unknown metric: %s", metric), "VALIDATION_ERROR", nil), nil
	}

	result, err := t.executor.ExecuteQueryOne(ctx, query, queryParams)
	if err != nil {
		t.logger.Error("Quality metrics computation failed", err, params)
		return Error(fmt.Sprintf("Quality metrics computation failed: metric='%s', error=%v", metric, err), "EXECUTION_ERROR", map[string]interface{}{
			"metric": metric,
			"error":  err.Error(),
		}), nil
	}

	return Success(result, map[string]interface{}{
		"metric": metric,
	}), nil
}









