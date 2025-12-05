package tools

import (
	"context"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

// DriftDetectionTool detects data drift
type DriftDetectionTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewDriftDetectionTool creates a new drift detection tool
func NewDriftDetectionTool(db *database.Database, logger *logging.Logger) *DriftDetectionTool {
	return &DriftDetectionTool{
		BaseTool: NewBaseTool(
			"detect_drift",
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

// Execute executes drift detection
func (t *DriftDetectionTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for detect_drift tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	method, _ := params["method"].(string)
	table, _ := params["table"].(string)
	vectorColumn, _ := params["vector_column"].(string)

	if table == "" || vectorColumn == "" {
		return Error("table and vector_column are required", "VALIDATION_ERROR", nil), nil
	}

	// Build query based on method
	var query string
	var queryParams []interface{}

	switch method {
	case "centroid":
		referenceTable, _ := params["reference_table"].(string)
		if referenceTable == "" {
			return Error("reference_table is required for centroid drift", "VALIDATION_ERROR", nil), nil
		}
		query = "SELECT detect_centroid_drift($1::text, $2::text, $3::text, $4::text) AS drift_score"
		queryParams = []interface{}{table, vectorColumn, referenceTable, vectorColumn}
	case "distribution":
		referenceTable, _ := params["reference_table"].(string)
		if referenceTable == "" {
			return Error("reference_table is required for distribution drift", "VALIDATION_ERROR", nil), nil
		}
		query = "SELECT detect_distribution_drift($1::text, $2::text, $3::text, $4::text) AS drift_score"
		queryParams = []interface{}{table, vectorColumn, referenceTable, vectorColumn}
	case "temporal":
		timestampCol, _ := params["timestamp_column"].(string)
		if timestampCol == "" {
			return Error("timestamp_column is required for temporal drift", "VALIDATION_ERROR", nil), nil
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

