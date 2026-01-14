/*-------------------------------------------------------------------------
 *
 * ml.go
 *    Machine learning tools for NeuronMCP
 *
 * Provides tools for training, predicting, evaluating, and managing ML models
 * using the NeuronDB unified train function.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/ml.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/validation"
)

/* TrainModelTool trains an ML model using the unified neurondb.train function */
type TrainModelTool struct {
	*BaseTool
	executor     *QueryExecutor
	logger       *logging.Logger
	configHelper *database.ConfigHelper
}

/* NewTrainModelTool creates a new train model tool */
func NewTrainModelTool(db *database.Database, logger *logging.Logger) *TrainModelTool {
	return &TrainModelTool{
		BaseTool: NewBaseTool(
			"postgresql_train_model",
			"Train an ML model using specified algorithm",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"algorithm": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"linear_regression", "ridge", "lasso", "logistic", "random_forest", "svm", "knn", "decision_tree", "naive_bayes"},
						"description": "ML algorithm to use",
					},
					"table": map[string]interface{}{
						"type":        "string",
						"description": "Training data table name",
					},
					"feature_col": map[string]interface{}{
						"type":        "string",
						"description": "Feature column name (vector type)",
					},
					"label_col": map[string]interface{}{
						"type":        "string",
						"description": "Label column name",
					},
					"params": map[string]interface{}{
						"type":        "object",
						"description": "Algorithm-specific parameters (optional)",
					},
					"project": map[string]interface{}{
						"type":        "string",
						"description": "ML project name (optional)",
					},
				},
				"required": []interface{}{"algorithm", "table", "feature_col", "label_col"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the model training */
func (t *TrainModelTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for neurondb_train_model tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	algorithm, _ := params["algorithm"].(string)
	table, _ := params["table"].(string)
	featureCol, _ := params["feature_col"].(string)
	labelCol, _ := params["label_col"].(string)

	/* Validate algorithm */
	if err := validation.ValidateRequired(algorithm, "algorithm"); err != nil {
		return Error(fmt.Sprintf("Invalid algorithm parameter: %v", err), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "algorithm",
			"error":     err.Error(),
			"params":    params,
		}), nil
	}
	if err := validation.ValidateIn(algorithm, "algorithm", "linear_regression", "ridge", "lasso", "logistic", "random_forest", "svm", "knn", "decision_tree", "naive_bayes"); err != nil {
		return Error(fmt.Sprintf("Invalid algorithm parameter: %v", err), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "algorithm",
			"error":     err.Error(),
			"params":    params,
		}), nil
	}

	/* Validate table name (SQL identifier) */
	if err := validation.ValidateSQLIdentifierRequired(table, "table"); err != nil {
		return Error(fmt.Sprintf("Invalid table parameter: %v", err), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "table",
			"algorithm": algorithm,
			"error":     err.Error(),
			"params":    params,
		}), nil
	}

	/* Validate feature column (SQL identifier) */
	if err := validation.ValidateSQLIdentifierRequired(featureCol, "feature_col"); err != nil {
		return Error(fmt.Sprintf("Invalid feature_col parameter: %v", err), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "feature_col",
			"algorithm": algorithm,
			"table":     table,
			"error":     err.Error(),
			"params":    params,
		}), nil
	}

	/* Validate label column (SQL identifier) */
	if err := validation.ValidateSQLIdentifierRequired(labelCol, "label_col"); err != nil {
		return Error(fmt.Sprintf("Invalid label_col parameter: %v", err), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":  "label_col",
			"algorithm": algorithm,
			"table":      table,
			"feature_col": featureCol,
			"error":      err.Error(),
			"params":     params,
		}), nil
	}

	/* Get ML defaults from database */
	defaultParams := make(map[string]interface{})
	if mlDefaults, err := t.configHelper.GetMLDefaults(ctx, algorithm); err == nil {
		/* Merge default hyperparameters */
		for k, v := range mlDefaults.Hyperparameters {
			defaultParams[k] = v
		}
		t.logger.Info("Using ML defaults from database", map[string]interface{}{
			"algorithm": algorithm,
			"defaults": defaultParams,
		})
	}
	
	/* Override with provided parameters */
	if p, ok := params["params"].(map[string]interface{}); ok && len(p) > 0 {
		for k, v := range p {
			defaultParams[k] = v
		}
	}
	
	paramsJSON := "{}"
	if len(defaultParams) > 0 {
		paramsBytes, err := json.Marshal(defaultParams)
		if err == nil {
			paramsJSON = string(paramsBytes)
		}
	}

	project := "default"
	if p, ok := params["project"].(string); ok && p != "" {
		project = p
	}

	/* NeuronDB train function signature: neurondb.train(project_name, algorithm, table_name, label_col, feature_columns[], params) */
	query := `SELECT neurondb.train($1, $2, $3, $4, $5::text[], $6::jsonb) AS model_id`
	result, err := t.executor.ExecuteQueryOne(ctx, query, []interface{}{
		project, algorithm, table, labelCol, []string{featureCol}, paramsJSON,
	})
	if err != nil {
		t.logger.Error("Model training failed", err, params)
		return Error(fmt.Sprintf("Model training execution failed: algorithm='%s', project='%s', table='%s', feature_col='%s', label_col='%s', params=%s, error=%v", algorithm, project, table, featureCol, labelCol, paramsJSON, err), "TRAINING_ERROR", map[string]interface{}{
			"algorithm":  algorithm,
			"project":    project,
			"table":      table,
			"feature_col": featureCol,
			"label_col":  labelCol,
			"params":     paramsJSON,
			"error":      err.Error(),
		}), nil
	}

	return Success(result, map[string]interface{}{
		"algorithm": algorithm,
		"project":   project,
	}), nil
}

/* PredictTool predicts using a trained ML model */
type PredictTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPredictTool creates a new predict tool */
func NewPredictTool(db *database.Database, logger *logging.Logger) *PredictTool {
	return &PredictTool{
		BaseTool: NewBaseTool(
			"postgresql_predict",
			"Predict using a trained ML model",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"model_id": map[string]interface{}{
						"type":        "number",
						"description": "Model ID from training",
					},
					"features": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "number"},
						"description": "Feature vector for prediction",
					},
				},
				"required": []interface{}{"model_id", "features"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the prediction */
func (t *PredictTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for neurondb_predict tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	modelID, ok := params["model_id"].(float64)
	if !ok {
		return Error(fmt.Sprintf("model_id parameter must be a number for neurondb_predict tool: received type %T, value=%v", params["model_id"], params["model_id"]), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "model_id",
			"received_type": fmt.Sprintf("%T", params["model_id"]),
			"received_value": params["model_id"],
			"params": params,
		}), nil
	}

	modelIDInt := int(modelID)
	if modelIDInt <= 0 {
		return Error(fmt.Sprintf("model_id must be a positive integer for neurondb_predict tool: received %d", modelIDInt), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "model_id",
			"value":     modelIDInt,
			"params":    params,
		}), nil
	}

	features, ok := params["features"].([]interface{})
	if !ok {
		return Error(fmt.Sprintf("features parameter must be an array for neurondb_predict tool: model_id=%d, received type %T, value=%v", modelIDInt, params["features"], params["features"]), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":     "features",
			"model_id":      modelIDInt,
			"received_type": fmt.Sprintf("%T", params["features"]),
			"params":        params,
		}), nil
	}

	if len(features) == 0 {
		return Error(fmt.Sprintf("features array cannot be empty for neurondb_predict tool: model_id=%d, features_count=0", modelIDInt), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":    "features",
			"model_id":     modelIDInt,
			"features_count": 0,
			"params":       params,
		}), nil
	}

	vectorStr := formatVectorFromInterface(features)

	query := `SELECT neurondb.predict($1::integer, $2::vector) AS prediction`
	result, err := t.executor.ExecuteQueryOne(ctx, query, []interface{}{modelIDInt, vectorStr})
	if err != nil {
		t.logger.Error("Prediction failed", err, params)
		return Error(fmt.Sprintf("Prediction execution failed: model_id=%d, features_dimension=%d, error=%v", modelIDInt, len(features), err), "PREDICTION_ERROR", map[string]interface{}{
			"model_id":          modelIDInt,
			"features_dimension": len(features),
			"error":             err.Error(),
		}), nil
	}

	return Success(result, map[string]interface{}{
		"model_id": int(modelID),
	}), nil
}

/* EvaluateModelTool evaluates a trained ML model */
type EvaluateModelTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewEvaluateModelTool creates a new evaluate model tool */
func NewEvaluateModelTool(db *database.Database, logger *logging.Logger) *EvaluateModelTool {
	return &EvaluateModelTool{
		BaseTool: NewBaseTool(
			"postgresql_evaluate_model",
			"Evaluate a trained ML model",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"model_id": map[string]interface{}{
						"type":        "number",
						"description": "Model ID to evaluate",
					},
					"test_table": map[string]interface{}{
						"type":        "string",
						"description": "Test data table name",
					},
					"feature_col": map[string]interface{}{
						"type":        "string",
						"description": "Feature column name",
					},
					"label_col": map[string]interface{}{
						"type":        "string",
						"description": "Label column name",
					},
				},
				"required": []interface{}{"model_id", "test_table", "feature_col", "label_col"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the model evaluation */
func (t *EvaluateModelTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for neurondb_evaluate_model tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	modelID, _ := params["model_id"].(float64)
	modelIDInt := int(modelID)
	testTable, _ := params["test_table"].(string)
	featureCol, _ := params["feature_col"].(string)
	labelCol, _ := params["label_col"].(string)

	if modelIDInt <= 0 {
		return Error(fmt.Sprintf("model_id must be a positive integer for neurondb_evaluate_model tool: received %d", modelIDInt), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "model_id",
			"value":     modelIDInt,
			"params":    params,
		}), nil
	}

	if testTable == "" {
		return Error(fmt.Sprintf("test_table parameter is required and cannot be empty for neurondb_evaluate_model tool: model_id=%d", modelIDInt), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "test_table",
			"model_id":  modelIDInt,
			"params":    params,
		}), nil
	}

	if featureCol == "" {
		return Error(fmt.Sprintf("feature_col parameter is required and cannot be empty for neurondb_evaluate_model tool: model_id=%d, test_table='%s'", modelIDInt, testTable), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":  "feature_col",
			"model_id":   modelIDInt,
			"test_table": testTable,
			"params":     params,
		}), nil
	}

	if labelCol == "" {
		return Error(fmt.Sprintf("label_col parameter is required and cannot be empty for neurondb_evaluate_model tool: model_id=%d, test_table='%s', feature_col='%s'", modelIDInt, testTable, featureCol), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":  "label_col",
			"model_id":   modelIDInt,
			"test_table": testTable,
			"feature_col": featureCol,
			"params":     params,
		}), nil
	}

	query := `SELECT neurondb.evaluate($1::integer, $2, $3, $4) AS metrics`
	result, err := t.executor.ExecuteQueryOne(ctx, query, []interface{}{
		modelIDInt, testTable, featureCol, labelCol,
	})
	if err != nil {
		t.logger.Error("Model evaluation failed", err, params)
		return Error(fmt.Sprintf("Model evaluation execution failed: model_id=%d, test_table='%s', feature_col='%s', label_col='%s', error=%v", modelIDInt, testTable, featureCol, labelCol, err), "EVALUATION_ERROR", map[string]interface{}{
			"model_id":   modelIDInt,
			"test_table": testTable,
			"feature_col": featureCol,
			"label_col":  labelCol,
			"error":      err.Error(),
		}), nil
	}

	return Success(result, map[string]interface{}{
		"model_id":   modelIDInt,
		"test_table": testTable,
		"feature_col": featureCol,
		"label_col":  labelCol,
	}), nil
}

/* ListModelsTool lists all trained models */
type ListModelsTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewListModelsTool creates a new list models tool */
func NewListModelsTool(db *database.Database, logger *logging.Logger) *ListModelsTool {
	return &ListModelsTool{
		BaseTool: NewBaseTool(
			"postgresql_list_models",
			"List all trained ML models",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"project": map[string]interface{}{
						"type":        "string",
						"description": "Filter by project name (optional)",
					},
					"algorithm": map[string]interface{}{
						"type":        "string",
						"description": "Filter by algorithm (optional)",
					},
				},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the list models query */
func (t *ListModelsTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	query := `SELECT model_id, algorithm::text AS algorithm, training_table, created_at FROM neurondb.ml_models WHERE 1=1`
	queryParams := []interface{}{}
	paramIndex := 1

	if project, ok := params["project"].(string); ok && project != "" {
		query += fmt.Sprintf(" AND project_name = $%d", paramIndex)
		queryParams = append(queryParams, project)
		paramIndex++
	}

	if algorithm, ok := params["algorithm"].(string); ok && algorithm != "" {
		query += fmt.Sprintf(" AND algorithm = $%d", paramIndex)
		queryParams = append(queryParams, algorithm)
		paramIndex++
	}

	query += " ORDER BY created_at DESC"

	results, err := t.executor.ExecuteQuery(ctx, query, queryParams)
	if err != nil {
		t.logger.Error("List models failed", err, params)
		filterInfo := ""
		if project, ok := params["project"].(string); ok && project != "" {
			filterInfo += fmt.Sprintf("project='%s'", project)
		}
		if algorithm, ok := params["algorithm"].(string); ok && algorithm != "" {
			if filterInfo != "" {
				filterInfo += ", "
			}
			filterInfo += fmt.Sprintf("algorithm='%s'", algorithm)
		}
		if filterInfo == "" {
			filterInfo = "no filters"
		}
		return Error(fmt.Sprintf("List models query execution failed: filters=[%s], error=%v", filterInfo, err), "QUERY_ERROR", map[string]interface{}{
			"filters": filterInfo,
			"error":   err.Error(),
		}), nil
	}

	return Success(results, map[string]interface{}{
		"count": len(results),
	}), nil
}

/* GetModelInfoTool gets detailed information about a model */
type GetModelInfoTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewGetModelInfoTool creates a new get model info tool */
func NewGetModelInfoTool(db *database.Database, logger *logging.Logger) *GetModelInfoTool {
	return &GetModelInfoTool{
		BaseTool: NewBaseTool(
			"postgresql_get_model_info",
			"Get detailed information about a trained model",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"model_id": map[string]interface{}{
						"type":        "number",
						"description": "Model ID",
					},
				},
				"required": []interface{}{"model_id"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the get model info query */
func (t *GetModelInfoTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for neurondb_get_model_info tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	modelID, _ := params["model_id"].(float64)
	modelIDInt := int(modelID)

	if modelIDInt <= 0 {
		return Error(fmt.Sprintf("model_id must be a positive integer for neurondb_get_model_info tool: received %d", modelIDInt), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "model_id",
			"value":     modelIDInt,
			"params":    params,
		}), nil
	}

	query := `SELECT * FROM neurondb.ml_models WHERE model_id = $1`
	result, err := t.executor.ExecuteQueryOne(ctx, query, []interface{}{modelIDInt})
	if err != nil {
		t.logger.Error("Get model info failed", err, params)
		return Error(fmt.Sprintf("Get model info query execution failed: model_id=%d, error=%v", modelIDInt, err), "QUERY_ERROR", map[string]interface{}{
			"model_id": modelIDInt,
			"error":    err.Error(),
		}), nil
	}

	if result == nil || len(result) == 0 {
		return Error(fmt.Sprintf("Model not found: model_id=%d does not exist in neurondb.ml_models table", modelIDInt), "NOT_FOUND", map[string]interface{}{
			"model_id": modelIDInt,
			"table":    "neurondb.ml_models",
		}), nil
	}

	return Success(result, nil), nil
}

/* DeleteModelTool deletes a trained model */
type DeleteModelTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewDeleteModelTool creates a new delete model tool */
func NewDeleteModelTool(db *database.Database, logger *logging.Logger) *DeleteModelTool {
	return &DeleteModelTool{
		BaseTool: NewBaseTool(
			"postgresql_delete_model",
			"Delete a trained ML model",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"model_id": map[string]interface{}{
						"type":        "number",
						"description": "Model ID to delete",
					},
				},
				"required": []interface{}{"model_id"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the model deletion */
func (t *DeleteModelTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for neurondb_delete_model tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	modelID, _ := params["model_id"].(float64)
	modelIDInt := int(modelID)

	if modelIDInt <= 0 {
		return Error(fmt.Sprintf("model_id must be a positive integer for delete_model tool: received %d", modelIDInt), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "model_id",
			"value":     modelIDInt,
			"params":    params,
		}), nil
	}

	query := `DELETE FROM neurondb.ml_models WHERE model_id = $1 RETURNING model_id`
	result, err := t.executor.ExecuteQueryOne(ctx, query, []interface{}{modelIDInt})
	if err != nil {
		t.logger.Error("Delete model failed", err, params)
		return Error(fmt.Sprintf("Delete model execution failed: model_id=%d, table='neurondb.ml_models', error=%v", modelIDInt, err), "DELETE_ERROR", map[string]interface{}{
			"model_id": modelIDInt,
			"table":    "neurondb.ml_models",
			"error":    err.Error(),
		}), nil
	}

	if result == nil || len(result) == 0 {
		return Error(fmt.Sprintf("Model not found for deletion: model_id=%d does not exist in neurondb.ml_models table", modelIDInt), "NOT_FOUND", map[string]interface{}{
			"model_id": modelIDInt,
			"table":    "neurondb.ml_models",
		}), nil
	}

	return Success(result, map[string]interface{}{
		"deleted":  true,
		"model_id": modelIDInt,
	}), nil
}

/* formatVectorFromInterface formats vector from interface slice */
func formatVectorFromInterface(vec []interface{}) string {
	var parts []string
	for _, v := range vec {
		var val float64
		switch v := v.(type) {
		case float64:
			val = v
		case float32:
			val = float64(v)
		case int:
			val = float64(v)
		case int64:
			val = float64(v)
		default:
			val = 0
		}
		parts = append(parts, fmt.Sprintf("%g", val))
	}
	return "[" + strings.Join(parts, ",") + "]"
}

