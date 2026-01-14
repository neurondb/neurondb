/*-------------------------------------------------------------------------
 *
 * ml_advanced.go
 *    Advanced ML features tools for NeuronMCP
 *
 * Implements advanced ML operations as specified in Phase 1.2
 * of the roadmap: versioning, A/B testing, explainability, monitoring.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/ml_advanced.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"fmt"
	"strings"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

/* MLModelVersioningTool manages model versions */
type MLModelVersioningTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewMLModelVersioningTool creates a new ML model versioning tool */
func NewMLModelVersioningTool(db *database.Database, logger *logging.Logger) *MLModelVersioningTool {
	return &MLModelVersioningTool{
		BaseTool: NewBaseTool(
			"ml_model_versioning",
			"Manage ML model versions: list, create, compare versions",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"operation": map[string]interface{}{
						"type":        "string",
						"enum":        []string{"list", "create", "compare", "get_latest"},
						"default":     "list",
						"description": "Operation to perform",
					},
					"model_id": map[string]interface{}{
						"type":        "number",
						"description": "Model ID (required for create, compare, get_latest)",
					},
					"version_tag": map[string]interface{}{
						"type":        "string",
						"description": "Version tag (e.g., 'v1.0.0', 'production')",
					},
					"version_id1": map[string]interface{}{
						"type":        "number",
						"description": "First version ID for comparison",
					},
					"version_id2": map[string]interface{}{
						"type":        "number",
						"description": "Second version ID for comparison",
					},
				},
				"required": []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute manages model versions */
func (t *MLModelVersioningTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	operation := "list"
	if val, ok := params["operation"].(string); ok {
		operation = val
	}

	switch operation {
	case "list":
		modelID, _ := params["model_id"].(float64)
		var query string
		var queryParams []interface{}

		if modelID > 0 {
			query = `
				SELECT 
					v.version_id,
					v.model_id,
					v.version_tag,
					v.created_at,
					v.created_by,
					m.algorithm,
					m.metrics
				FROM neurondb.model_versions v
				JOIN neurondb.ml_models m ON v.model_id = m.model_id
				WHERE v.model_id = $1
				ORDER BY v.created_at DESC
			`
			queryParams = []interface{}{int(modelID)}
		} else {
			query = `
				SELECT 
					v.version_id,
					v.model_id,
					v.version_tag,
					v.created_at,
					v.created_by,
					m.algorithm,
					m.metrics
				FROM neurondb.model_versions v
				JOIN neurondb.ml_models m ON v.model_id = m.model_id
				ORDER BY v.model_id, v.created_at DESC
			`
			queryParams = []interface{}{}
		}

		results, err := t.executor.ExecuteQuery(ctx, query, queryParams)
		if err != nil {
			return Error(
				fmt.Sprintf("Failed to list model versions: %v", err),
				"QUERY_ERROR",
				map[string]interface{}{"error": err.Error()},
			), nil
		}

		return Success(map[string]interface{}{
			"versions": results,
			"count":    len(results),
		}, map[string]interface{}{
			"tool": "ml_model_versioning",
		}), nil

	case "get_latest":
		modelID, ok := params["model_id"].(float64)
		if !ok || modelID <= 0 {
			return Error("model_id is required for get_latest operation", "INVALID_PARAMETER", nil), nil
		}

		query := `
			SELECT 
				v.version_id,
				v.model_id,
				v.version_tag,
				v.created_at,
				v.created_by,
				m.algorithm,
				m.metrics
			FROM neurondb.model_versions v
			JOIN neurondb.ml_models m ON v.model_id = m.model_id
			WHERE v.model_id = $1
			ORDER BY v.created_at DESC
			LIMIT 1
		`
		result, err := t.executor.ExecuteQueryOne(ctx, query, []interface{}{int(modelID)})
		if err != nil {
			return Error(
				fmt.Sprintf("Failed to get latest version: %v", err),
				"QUERY_ERROR",
				map[string]interface{}{"error": err.Error()},
			), nil
		}

		return Success(result, map[string]interface{}{
			"tool": "ml_model_versioning",
		}), nil

	case "compare":
		versionID1, ok1 := params["version_id1"].(float64)
		versionID2, ok2 := params["version_id2"].(float64)
		if !ok1 || !ok2 || versionID1 <= 0 || versionID2 <= 0 {
			return Error("version_id1 and version_id2 are required for compare operation", "INVALID_PARAMETER", nil), nil
		}

		query := `
			SELECT 
				v1.version_id AS version1_id,
				v1.version_tag AS version1_tag,
				v1.created_at AS version1_created,
				v1.metrics AS version1_metrics,
				v2.version_id AS version2_id,
				v2.version_tag AS version2_tag,
				v2.created_at AS version2_created,
				v2.metrics AS version2_metrics
			FROM neurondb.model_versions v1
			CROSS JOIN neurondb.model_versions v2
			WHERE v1.version_id = $1 AND v2.version_id = $2
		`
		result, err := t.executor.ExecuteQueryOne(ctx, query, []interface{}{int(versionID1), int(versionID2)})
		if err != nil {
			return Error(
				fmt.Sprintf("Failed to compare versions: %v", err),
				"QUERY_ERROR",
				map[string]interface{}{"error": err.Error()},
			), nil
		}

		return Success(result, map[string]interface{}{
			"tool": "ml_model_versioning",
		}), nil

	default:
		return Error(fmt.Sprintf("Invalid operation: %s", operation), "INVALID_PARAMETER", nil), nil
	}
}

/* MLModelABTestingTool manages A/B testing */
type MLModelABTestingTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewMLModelABTestingTool creates a new ML A/B testing tool */
func NewMLModelABTestingTool(db *database.Database, logger *logging.Logger) *MLModelABTestingTool {
	return &MLModelABTestingTool{
		BaseTool: NewBaseTool(
			"ml_model_ab_testing",
			"Manage A/B testing for ML models: create, monitor, analyze results",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"operation": map[string]interface{}{
						"type":        "string",
						"enum":        []string{"create", "list", "get_results", "stop"},
						"default":     "list",
						"description": "Operation to perform",
					},
					"test_name": map[string]interface{}{
						"type":        "string",
						"description": "A/B test name (required for create)",
					},
					"model_ids": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "number"},
						"description": "Array of model IDs to test (required for create)",
					},
					"traffic_split": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "number"},
						"description": "Traffic split percentages (must sum to 100)",
					},
					"test_id": map[string]interface{}{
						"type":        "number",
						"description": "Test ID (required for get_results, stop)",
					},
				},
				"required": []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute manages A/B tests */
func (t *MLModelABTestingTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	operation := "list"
	if val, ok := params["operation"].(string); ok {
		operation = val
	}

	switch operation {
	case "list":
		query := `
			SELECT 
				test_id,
				test_name,
				model_ids,
				traffic_split,
				start_time,
				end_time,
				status,
				results
			FROM neurondb.ab_tests
			ORDER BY start_time DESC
		`
		results, err := t.executor.ExecuteQuery(ctx, query, nil)
		if err != nil {
			return Error(
				fmt.Sprintf("Failed to list A/B tests: %v", err),
				"QUERY_ERROR",
				map[string]interface{}{"error": err.Error()},
			), nil
		}

		return Success(map[string]interface{}{
			"ab_tests": results,
			"count":    len(results),
		}, map[string]interface{}{
			"tool": "ml_model_ab_testing",
		}), nil

	case "create":
		testName, ok := params["test_name"].(string)
		if !ok || testName == "" {
			return Error("test_name is required for create operation", "INVALID_PARAMETER", nil), nil
		}

		modelIDs, ok := params["model_ids"].([]interface{})
		if !ok || len(modelIDs) == 0 {
			return Error("model_ids is required and must be a non-empty array", "INVALID_PARAMETER", nil), nil
		}

		trafficSplit, _ := params["traffic_split"].([]interface{})

		/* Convert model IDs to PostgreSQL array */
		modelIDsStr := "{"
		for i, id := range modelIDs {
			if i > 0 {
				modelIDsStr += ","
			}
			if idFloat, ok := id.(float64); ok {
				modelIDsStr += fmt.Sprintf("%d", int(idFloat))
			}
		}
		modelIDsStr += "}"

		var trafficSplitStr string
		if len(trafficSplit) > 0 {
			trafficSplitStr = "{"
			for i, split := range trafficSplit {
				if i > 0 {
					trafficSplitStr += ","
				}
				if splitFloat, ok := split.(float64); ok {
					trafficSplitStr += fmt.Sprintf("%.2f", splitFloat)
				}
			}
			trafficSplitStr += "}"
		} else {
			/* Default: equal split */
			trafficSplitStr = "{"
			for i := 0; i < len(modelIDs); i++ {
				if i > 0 {
					trafficSplitStr += ","
				}
				trafficSplitStr += fmt.Sprintf("%.2f", 100.0/float64(len(modelIDs)))
			}
			trafficSplitStr += "}"
		}

		query := fmt.Sprintf(`
			INSERT INTO neurondb.ab_tests (test_name, model_ids, traffic_split, status)
			VALUES ($1, $2::integer[], $3::float[], 'running')
			RETURNING test_id, test_name, model_ids, traffic_split, start_time, status
		`)

		result, err := t.executor.ExecuteQueryOne(ctx, query, []interface{}{testName, modelIDsStr, trafficSplitStr})
		if err != nil {
			return Error(
				fmt.Sprintf("Failed to create A/B test: %v", err),
				"QUERY_ERROR",
				map[string]interface{}{"error": err.Error()},
			), nil
		}

		return Success(result, map[string]interface{}{
			"tool": "ml_model_ab_testing",
		}), nil

	case "get_results":
		testID, ok := params["test_id"].(float64)
		if !ok || testID <= 0 {
			return Error("test_id is required for get_results operation", "INVALID_PARAMETER", nil), nil
		}

		query := `
			SELECT 
				test_id,
				test_name,
				model_ids,
				traffic_split,
				start_time,
				end_time,
				status,
				results
			FROM neurondb.ab_tests
			WHERE test_id = $1
		`
		result, err := t.executor.ExecuteQueryOne(ctx, query, []interface{}{int(testID)})
		if err != nil {
			return Error(
				fmt.Sprintf("Failed to get A/B test results: %v", err),
				"QUERY_ERROR",
				map[string]interface{}{"error": err.Error()},
			), nil
		}

		return Success(result, map[string]interface{}{
			"tool": "ml_model_ab_testing",
		}), nil

	case "stop":
		testID, ok := params["test_id"].(float64)
		if !ok || testID <= 0 {
			return Error("test_id is required for stop operation", "INVALID_PARAMETER", nil), nil
		}

		query := `
			UPDATE neurondb.ab_tests
			SET status = 'completed', end_time = NOW()
			WHERE test_id = $1
			RETURNING test_id, test_name, status, end_time
		`
		result, err := t.executor.ExecuteQueryOne(ctx, query, []interface{}{int(testID)})
		if err != nil {
			return Error(
				fmt.Sprintf("Failed to stop A/B test: %v", err),
				"QUERY_ERROR",
				map[string]interface{}{"error": err.Error()},
			), nil
		}

		return Success(result, map[string]interface{}{
			"tool": "ml_model_ab_testing",
		}), nil

	default:
		return Error(fmt.Sprintf("Invalid operation: %s", operation), "INVALID_PARAMETER", nil), nil
	}
}

/* MLModelExplainabilityTool provides model explainability */
type MLModelExplainabilityTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewMLModelExplainabilityTool creates a new ML explainability tool */
func NewMLModelExplainabilityTool(db *database.Database, logger *logging.Logger) *MLModelExplainabilityTool {
	return &MLModelExplainabilityTool{
		BaseTool: NewBaseTool(
			"ml_model_explainability",
			"Provide model explainability using SHAP, LIME, or feature importance",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"model_id": map[string]interface{}{
						"type":        "number",
						"description": "Model ID to explain",
					},
					"method": map[string]interface{}{
						"type":        "string",
						"enum":        []string{"shap", "lime", "feature_importance", "partial_dependence"},
						"default":     "feature_importance",
						"description": "Explainability method",
					},
					"input_data": map[string]interface{}{
						"type":        "array",
						"description": "Input feature vector for local explanation (optional)",
					},
				},
				"required": []interface{}{"model_id"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute provides model explainability */
func (t *MLModelExplainabilityTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	modelID, ok := params["model_id"].(float64)
	if !ok || modelID <= 0 {
		return Error("model_id parameter is required", "INVALID_PARAMETER", nil), nil
	}

	method := "feature_importance"
	if val, ok := params["method"].(string); ok {
		method = val
	}

	/* Get model information */
	modelQuery := `
		SELECT 
			model_id,
			algorithm,
			hyperparameters,
			metrics,
			feature_columns,
			target_column
		FROM neurondb.ml_models
		WHERE model_id = $1
	`
	modelInfo, err := t.executor.ExecuteQueryOne(ctx, modelQuery, []interface{}{int(modelID)})
	if err != nil {
		return Error(
			fmt.Sprintf("Failed to get model information: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	/* Generate explainability based on method */
	explanation := map[string]interface{}{
		"model_id":    int(modelID),
		"method":      method,
		"model_info":  modelInfo,
		"note":        "Full explainability requires external libraries (SHAP, LIME). This provides model metadata and feature information.",
	}

	switch method {
	case "feature_importance":
		/* For tree-based models, we can extract feature importance from metrics */
		if metrics, ok := modelInfo["metrics"].(map[string]interface{}); ok {
			if importance, ok := metrics["feature_importance"].(map[string]interface{}); ok {
				explanation["feature_importance"] = importance
			}
		}
		explanation["recommendation"] = "For detailed feature importance, check model metrics or use SHAP/LIME methods"

	case "shap", "lime":
		explanation["note"] = fmt.Sprintf("%s explainability requires external Python libraries. Use Python SDK or external tools for full %s analysis.", strings.ToUpper(method), method)
		explanation["recommendation"] = fmt.Sprintf("Install %s library and use Python client for detailed %s values", method, method)

	case "partial_dependence":
		explanation["note"] = "Partial dependence plots require model-specific analysis. Use external tools for visualization."
	}

	return Success(explanation, map[string]interface{}{
		"tool": "ml_model_explainability",
	}), nil
}

/* MLModelMonitoringTool monitors model performance */
type MLModelMonitoringTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewMLModelMonitoringTool creates a new ML model monitoring tool */
func NewMLModelMonitoringTool(db *database.Database, logger *logging.Logger) *MLModelMonitoringTool {
	return &MLModelMonitoringTool{
		BaseTool: NewBaseTool(
			"ml_model_monitoring",
			"Monitor ML model performance: latency, accuracy, error rates",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"model_id": map[string]interface{}{
						"type":        "number",
						"description": "Model ID to monitor (optional, monitors all if not provided)",
					},
					"time_range_hours": map[string]interface{}{
						"type":        "number",
						"default":     24,
						"minimum":     1,
						"maximum":     720,
						"description": "Time range in hours to analyze",
					},
					"metrics": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "string", "enum": []string{"latency", "accuracy", "error_rate", "throughput", "all"}},
						"description": "Metrics to compute (default: all)",
					},
				},
				"required": []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute monitors model performance */
func (t *MLModelMonitoringTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	modelID, _ := params["model_id"].(float64)
	timeRangeHours := 24.0
	if val, ok := params["time_range_hours"].(float64); ok {
		timeRangeHours = val
		if timeRangeHours < 1 {
			timeRangeHours = 1
		}
		if timeRangeHours > 720 {
			timeRangeHours = 720
		}
	}

	var query string
	var queryParams []interface{}

	if modelID > 0 {
		query = `
			SELECT 
				model_id,
				COUNT(*) AS prediction_count,
				AVG(latency_ms) AS avg_latency_ms,
				PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY latency_ms) AS p50_latency_ms,
				PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_latency_ms,
				PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms) AS p99_latency_ms,
				SUM(CASE WHEN success = false THEN 1 ELSE 0 END)::float / COUNT(*) * 100 AS error_rate_percent,
				AVG(confidence) AS avg_confidence
			FROM neurondb.model_monitoring
			WHERE model_id = $1
			AND prediction_time >= NOW() - INTERVAL '%d hours'
			GROUP BY model_id
		`
		queryParams = []interface{}{int(modelID)}
		query = fmt.Sprintf(query, int(timeRangeHours))
	} else {
		query = `
			SELECT 
				model_id,
				COUNT(*) AS prediction_count,
				AVG(latency_ms) AS avg_latency_ms,
				PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY latency_ms) AS p50_latency_ms,
				PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_latency_ms,
				PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms) AS p99_latency_ms,
				SUM(CASE WHEN success = false THEN 1 ELSE 0 END)::float / COUNT(*) * 100 AS error_rate_percent,
				AVG(confidence) AS avg_confidence
			FROM neurondb.model_monitoring
			WHERE prediction_time >= NOW() - INTERVAL '%d hours'
			GROUP BY model_id
			ORDER BY prediction_count DESC
		`
		queryParams = []interface{}{}
		query = fmt.Sprintf(query, int(timeRangeHours))
	}

	results, err := t.executor.ExecuteQuery(ctx, query, queryParams)
	if err != nil {
		return Error(
			fmt.Sprintf("Model monitoring query failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	return Success(map[string]interface{}{
		"monitoring_data": results,
		"count":           len(results),
		"time_range_hours": timeRangeHours,
	}, map[string]interface{}{
		"tool": "ml_model_monitoring",
	}), nil
}

/* MLModelRollbackTool rolls back to previous model version */
type MLModelRollbackTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewMLModelRollbackTool creates a new ML model rollback tool */
func NewMLModelRollbackTool(db *database.Database, logger *logging.Logger) *MLModelRollbackTool {
	return &MLModelRollbackTool{
		BaseTool: NewBaseTool(
			"ml_model_rollback",
			"Rollback to a previous model version",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"model_id": map[string]interface{}{
						"type":        "number",
						"description": "Model ID to rollback",
					},
					"version_id": map[string]interface{}{
						"type":        "number",
						"description": "Version ID to rollback to (optional, uses previous version if not specified)",
					},
				},
				"required": []interface{}{"model_id"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute rolls back the model */
func (t *MLModelRollbackTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	modelID, ok := params["model_id"].(float64)
	if !ok || modelID <= 0 {
		return Error("model_id parameter is required", "INVALID_PARAMETER", nil), nil
	}

	versionID, _ := params["version_id"].(float64)

	/* Get current model version */
	currentQuery := `
		SELECT version, status, is_deployed
		FROM neurondb.ml_models
		WHERE model_id = $1
	`
	currentModel, err := t.executor.ExecuteQueryOne(ctx, currentQuery, []interface{}{int(modelID)})
	if err != nil {
		return Error(
			fmt.Sprintf("Failed to get current model: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	/* Get version to rollback to */
	var targetVersion map[string]interface{}
	if versionID > 0 {
		versionQuery := `
			SELECT version_id, version_tag, created_at, metrics
			FROM neurondb.model_versions
			WHERE version_id = $1 AND model_id = $2
		`
		targetVersion, err = t.executor.ExecuteQueryOne(ctx, versionQuery, []interface{}{int(versionID), int(modelID)})
		if err != nil {
			return Error(
				fmt.Sprintf("Failed to get target version: %v", err),
				"QUERY_ERROR",
				map[string]interface{}{"error": err.Error()},
			), nil
		}
	} else {
		/* Get previous version */
		prevQuery := `
			SELECT version_id, version_tag, created_at, metrics
			FROM neurondb.model_versions
			WHERE model_id = $1
			ORDER BY created_at DESC
			OFFSET 1
			LIMIT 1
		`
		targetVersion, err = t.executor.ExecuteQueryOne(ctx, prevQuery, []interface{}{int(modelID)})
		if err != nil {
			return Error(
				fmt.Sprintf("Failed to get previous version: %v", err),
				"QUERY_ERROR",
				map[string]interface{}{"error": err.Error()},
			), nil
		}
	}

	/* Update model to use previous version (mark as rollback) */
	rollbackQuery := `
		UPDATE neurondb.ml_models
		SET status = 'deployed',
			updated_at = NOW(),
			notes = COALESCE(notes, '') || ' Rolled back to version ' || $2 || ' at ' || NOW()::text
		WHERE model_id = $1
		RETURNING model_id, version, status, updated_at
	`
	versionTag := "unknown"
	if tag, ok := targetVersion["version_tag"].(string); ok {
		versionTag = tag
	}

	result, err := t.executor.ExecuteQueryOne(ctx, rollbackQuery, []interface{}{int(modelID), versionTag})
	if err != nil {
		return Error(
			fmt.Sprintf("Rollback failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Model rolled back", map[string]interface{}{
		"model_id":   int(modelID),
		"version_id": targetVersion["version_id"],
	})

	return Success(map[string]interface{}{
		"model_id":      int(modelID),
		"current_model": currentModel,
		"target_version": targetVersion,
		"rollback_result": result,
	}, map[string]interface{}{
		"tool": "ml_model_rollback",
	}), nil
}

