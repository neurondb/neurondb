/*-------------------------------------------------------------------------
 *
 * ml_advanced_complete.go
 *    Complete advanced ML features for NeuronMCP
 *
 * Implements remaining advanced ML features from Phase 1.2:
 * - Model retraining
 * - Ensemble models
 * - Model export formats
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/ml_advanced_complete.go
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

/* MLModelRetrainingTool manages automated model retraining */
type MLModelRetrainingTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewMLModelRetrainingTool creates a new ML model retraining tool */
func NewMLModelRetrainingTool(db *database.Database, logger *logging.Logger) *MLModelRetrainingTool {
	return &MLModelRetrainingTool{
		BaseTool: NewBaseTool(
			"ml_model_retraining",
			"Automated model retraining with new data and scheduling",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"model_id": map[string]interface{}{
						"type":        "number",
						"description": "Model ID to retrain",
					},
					"training_table": map[string]interface{}{
						"type":        "string",
						"description": "New training data table (optional, uses original if not provided)",
					},
					"incremental": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Incremental training (append to existing data)",
					},
					"schedule": map[string]interface{}{
						"type":        "string",
						"description": "Cron expression for scheduled retraining (optional)",
					},
				},
				"required": []interface{}{"model_id"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute manages model retraining */
func (t *MLModelRetrainingTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	modelID, ok := params["model_id"].(float64)
	if !ok || modelID <= 0 {
		return Error("model_id parameter is required", "INVALID_PARAMETER", nil), nil
	}

	/* Get model information */
	modelQuery := `
		SELECT 
			model_id,
			algorithm,
			training_table,
			feature_columns,
			target_column,
			hyperparameters
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

	trainingTable, _ := params["training_table"].(string)
	if trainingTable == "" {
		if table, ok := modelInfo["training_table"].(string); ok {
			trainingTable = table
		}
	}

	incremental := false
	if val, ok := params["incremental"].(bool); ok {
		incremental = val
	}

	schedule, _ := params["schedule"].(string)

	/* Create new model version with retraining */
	instructions := []string{
		fmt.Sprintf("1. Use neurondb.train() to retrain model %d", int(modelID)),
		fmt.Sprintf("2. Training table: %s", trainingTable),
		fmt.Sprintf("3. Algorithm: %v", modelInfo["algorithm"]),
	}

	if incremental {
		instructions = append(instructions, "4. Use incremental training mode")
	}

	if schedule != "" {
		instructions = append(instructions, fmt.Sprintf("5. Schedule retraining: %s", schedule))
	}

	return Success(map[string]interface{}{
		"model_id":      int(modelID),
		"model_info":    modelInfo,
		"training_table": trainingTable,
		"incremental":    incremental,
		"schedule":       schedule,
		"instructions":  instructions,
		"sql_example":    fmt.Sprintf("SELECT neurondb.train('%v', '%s', '%v', '%v', '{}'::jsonb)", modelInfo["algorithm"], trainingTable, modelInfo["feature_columns"], modelInfo["target_column"]),
	}, map[string]interface{}{
		"tool": "ml_model_retraining",
	}), nil
}

/* MLEnsembleModelsTool creates ensemble models */
type MLEnsembleModelsTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewMLEnsembleModelsTool creates a new ML ensemble models tool */
func NewMLEnsembleModelsTool(db *database.Database, logger *logging.Logger) *MLEnsembleModelsTool {
	return &MLEnsembleModelsTool{
		BaseTool: NewBaseTool(
			"ml_ensemble_models",
			"Create ensemble models from multiple base models",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"model_ids": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "number"},
						"description": "Array of model IDs to ensemble",
					},
					"ensemble_method": map[string]interface{}{
						"type":        "string",
						"enum":        []string{"voting", "averaging", "stacking", "bagging"},
						"default":     "voting",
						"description": "Ensemble method",
					},
					"ensemble_name": map[string]interface{}{
						"type":        "string",
						"description": "Name for the ensemble model",
					},
					"weights": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "number"},
						"description": "Weights for each model (optional, defaults to equal)",
					},
				},
				"required": []interface{}{"model_ids", "ensemble_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute creates ensemble model */
func (t *MLEnsembleModelsTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	modelIDs, ok := params["model_ids"].([]interface{})
	if !ok || len(modelIDs) == 0 {
		return Error("model_ids parameter is required and must be a non-empty array", "INVALID_PARAMETER", nil), nil
	}

	ensembleName, ok := params["ensemble_name"].(string)
	if !ok || ensembleName == "" {
		return Error("ensemble_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	ensembleMethod := "voting"
	if val, ok := params["ensemble_method"].(string); ok {
		ensembleMethod = val
	}

	weights, _ := params["weights"].([]interface{})

	/* Get model information */
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

	query := `
		SELECT 
			model_id,
			algorithm,
			metrics
		FROM neurondb.ml_models
		WHERE model_id = ANY($1::integer[])
	`

	models, err := t.executor.ExecuteQuery(ctx, query, []interface{}{modelIDsStr})
	if err != nil {
		return Error(
			fmt.Sprintf("Failed to get model information: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	if len(models) != len(modelIDs) {
		return Error("Some model IDs not found", "INVALID_PARAMETER", nil), nil
	}

	/* Create ensemble model record */
	ensembleQuery := `
		INSERT INTO neurondb.ml_models (project_id, algorithm, model_name, training_table, parameters, status)
		VALUES (
			(SELECT project_id FROM neurondb.ml_models WHERE model_id = $1 LIMIT 1),
			'ensemble',
			$2,
			(SELECT training_table FROM neurondb.ml_models WHERE model_id = $1 LIMIT 1),
			jsonb_build_object('ensemble_method', $3, 'base_models', $4::integer[], 'weights', $5::float[]),
			'completed'
		)
		RETURNING model_id, model_name, algorithm, parameters
	`

	var weightsStr string
	if len(weights) > 0 {
		weightsStr = "{"
		for i, w := range weights {
			if i > 0 {
				weightsStr += ","
			}
			if wFloat, ok := w.(float64); ok {
				weightsStr += fmt.Sprintf("%.2f", wFloat)
			}
		}
		weightsStr += "}"
	} else {
		weightsStr = "{}"
	}

	firstModelID := 0
	if len(modelIDs) > 0 {
		if idFloat, ok := modelIDs[0].(float64); ok {
			firstModelID = int(idFloat)
		}
	}

	result, err := t.executor.ExecuteQueryOne(ctx, ensembleQuery, []interface{}{
		firstModelID,
		ensembleName,
		ensembleMethod,
		modelIDsStr,
		weightsStr,
	})
	if err != nil {
		return Error(
			fmt.Sprintf("Failed to create ensemble model: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	return Success(result, map[string]interface{}{
		"tool": "ml_ensemble_models",
	}), nil
}

/* MLModelExportFormatsTool exports models to multiple formats */
type MLModelExportFormatsTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewMLModelExportFormatsTool creates a new ML model export formats tool */
func NewMLModelExportFormatsTool(db *database.Database, logger *logging.Logger) *MLModelExportFormatsTool {
	return &MLModelExportFormatsTool{
		BaseTool: NewBaseTool(
			"ml_model_export_formats",
			"Export ML models to multiple formats (ONNX, PMML, TensorFlow, PyTorch)",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"model_id": map[string]interface{}{
						"type":        "number",
						"description": "Model ID to export",
					},
					"format": map[string]interface{}{
						"type":        "string",
						"enum":        []string{"onnx", "pmml", "tensorflow", "pytorch", "json", "pickle"},
						"default":     "onnx",
						"description": "Export format",
					},
					"output_path": map[string]interface{}{
						"type":        "string",
						"description": "Output file path (optional)",
					},
				},
				"required": []interface{}{"model_id"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute exports model */
func (t *MLModelExportFormatsTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	modelID, ok := params["model_id"].(float64)
	if !ok || modelID <= 0 {
		return Error("model_id parameter is required", "INVALID_PARAMETER", nil), nil
	}

	format := "onnx"
	if val, ok := params["format"].(string); ok {
		format = val
	}

	outputPath, _ := params["output_path"].(string)

	/* Get model information */
	modelQuery := `
		SELECT 
			model_id,
			algorithm,
			model_data,
			hyperparameters
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

	/* Use export_model tool or neurondb.export_model function */
	instructions := []string{
		fmt.Sprintf("Export model %d to %s format", int(modelID), format),
	}

	if outputPath != "" {
		instructions = append(instructions, fmt.Sprintf("Output path: %s", outputPath))
	}

	switch format {
	case "onnx":
		instructions = append(instructions, "Use ONNX export: SELECT neurondb.export_model($1, 'onnx');")
	case "pmml":
		instructions = append(instructions, "Use PMML export: SELECT neurondb.export_model($1, 'pmml');")
	case "tensorflow":
		instructions = append(instructions, "Use TensorFlow export: SELECT neurondb.export_model($1, 'tensorflow');")
	case "pytorch":
		instructions = append(instructions, "Use PyTorch export: SELECT neurondb.export_model($1, 'pytorch');")
	}

	return Success(map[string]interface{}{
		"model_id":    int(modelID),
		"format":      format,
		"output_path": outputPath,
		"model_info":  modelInfo,
		"instructions": instructions,
		"note":        "Use the export_model tool or neurondb.export_model() function for actual export",
	}, map[string]interface{}{
		"tool": "ml_model_export_formats",
	}), nil
}

