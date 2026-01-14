/*-------------------------------------------------------------------------
 *
 * drift_detection.go
 *    Tool implementation for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/drift_detection.go
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

/* DriftDetectionTool detects data drift */
type DriftDetectionTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewDriftDetectionTool creates a new drift detection tool */
func NewDriftDetectionTool(db *database.Database, logger *logging.Logger) *DriftDetectionTool {
	return &DriftDetectionTool{
		BaseTool: NewBaseTool(
			"postgresql_detect_drift",
			"Detect data drift: centroid drift, distribution divergence, temporal monitoring",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"method": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"centroid", "distribution", "temporal"},
						"description": "Drift detection method",
					},
					"table": map[string]interface{}{
						"type":        "string",
						"description": "Table name",
					},
					"vector_column": map[string]interface{}{
						"type":        "string",
						"description": "Vector column name",
					},
					"reference_table": map[string]interface{}{
						"type":        "string",
						"description": "Reference table for comparison",
					},
					"threshold": map[string]interface{}{
						"type":        "number",
						"description": "Drift threshold",
					},
				},
				"required": []interface{}{"method", "table", "vector_column"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes drift detection */
func (t *DriftDetectionTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for neurondb_detect_drift tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	method, _ := params["method"].(string)
	table, _ := params["table"].(string)
	vectorColumn, _ := params["vector_column"].(string)

	/* Validate table name (SQL identifier) */
	if err := validation.ValidateSQLIdentifierRequired(table, "table"); err != nil {
		return Error(fmt.Sprintf("Invalid table parameter: %v", err), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "table",
			"error":     err.Error(),
			"params":    params,
		}), nil
	}

	/* Validate vector_column (SQL identifier) */
	if err := validation.ValidateSQLIdentifierRequired(vectorColumn, "vector_column"); err != nil {
		return Error(fmt.Sprintf("Invalid vector_column parameter: %v", err), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "vector_column",
			"table":     table,
			"error":     err.Error(),
			"params":    params,
		}), nil
	}

  /* Build query based on method */
	var query string
	var queryParams []interface{}

	switch method {
	case "centroid":
		referenceTable, _ := params["reference_table"].(string)
		if err := validation.ValidateSQLIdentifierRequired(referenceTable, "reference_table"); err != nil {
			return Error(fmt.Sprintf("Invalid reference_table parameter: %v", err), "VALIDATION_ERROR", map[string]interface{}{
				"parameter": "reference_table",
				"table":     table,
				"error":     err.Error(),
				"params":    params,
			}), nil
		}
		query = "SELECT detect_centroid_drift($1::text, $2::text, $3::text, $4::text) AS drift_score"
		queryParams = []interface{}{table, vectorColumn, referenceTable, vectorColumn}
	case "distribution":
		referenceTable, _ := params["reference_table"].(string)
		if err := validation.ValidateSQLIdentifierRequired(referenceTable, "reference_table"); err != nil {
			return Error(fmt.Sprintf("Invalid reference_table parameter: %v", err), "VALIDATION_ERROR", map[string]interface{}{
				"parameter": "reference_table",
				"table":     table,
				"error":     err.Error(),
				"params":    params,
			}), nil
		}
		query = "SELECT detect_distribution_drift($1::text, $2::text, $3::text, $4::text) AS drift_score"
		queryParams = []interface{}{table, vectorColumn, referenceTable, vectorColumn}
	case "temporal":
		timestampCol, _ := params["timestamp_column"].(string)
		if err := validation.ValidateSQLIdentifierRequired(timestampCol, "timestamp_column"); err != nil {
			return Error(fmt.Sprintf("Invalid timestamp_column parameter: %v", err), "VALIDATION_ERROR", map[string]interface{}{
				"parameter": "timestamp_column",
				"table":     table,
				"error":     err.Error(),
				"params":    params,
			}), nil
		}
		query = "SELECT detect_temporal_drift($1::text, $2::text, $3::text) AS drift_score"
		queryParams = []interface{}{table, vectorColumn, timestampCol}
	default:
		return Error(fmt.Sprintf("Unknown method: %s", method), "VALIDATION_ERROR", nil), nil
	}

	result, err := t.executor.ExecuteQueryOne(ctx, query, queryParams)
	if err != nil {
		t.logger.Error("Drift detection failed", err, params)
		return Error(fmt.Sprintf("Drift detection failed: method='%s', error=%v", method, err), "EXECUTION_ERROR", map[string]interface{}{
			"method": method,
			"error":  err.Error(),
		}), nil
	}

	return Success(result, map[string]interface{}{
		"method": method,
	}), nil
}









