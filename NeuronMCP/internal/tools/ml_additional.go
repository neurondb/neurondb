/*-------------------------------------------------------------------------
 *
 * ml_additional.go
 *    Tool implementation for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/ml_additional.go
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

/* PredictBatchTool performs batch prediction using a trained ML model */
type PredictBatchTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

/* NewPredictBatchTool creates a new PredictBatchTool */
func NewPredictBatchTool(db *database.Database, logger *logging.Logger) *PredictBatchTool {
	return &PredictBatchTool{
		BaseTool: NewBaseTool(
			"postgresql_predict_batch",
			"Perform batch prediction using a trained machine learning model for multiple feature vectors",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"model_id": map[string]interface{}{
						"type":        "number",
						"description": "The ID of the trained model",
					},
					"features_array": map[string]interface{}{
						"type":        "array",
						"items": map[string]interface{}{
							"type": "array",
							"items": map[string]interface{}{"type": "number"},
						},
						"description": "Array of feature vectors for batch prediction",
						"minItems":    1,
						"maxItems":    1000,
					},
				},
				"required": []interface{}{"model_id", "features_array"},
			},
		),
		db:     db,
		logger: logger,
	}
}

/* Execute performs batch prediction */
func (t *PredictBatchTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for neurondb_neurondb_predict_batch tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	modelID, ok := params["model_id"].(float64)
	if !ok {
		return Error(fmt.Sprintf("model_id parameter must be a number for neurondb_neurondb_predict_batch tool: received type %T, value=%v", params["model_id"], params["model_id"]), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":     "model_id",
			"received_type": fmt.Sprintf("%T", params["model_id"]),
			"received_value": params["model_id"],
			"params":        params,
		}), nil
	}

	modelIDInt := int(modelID)
	if modelIDInt <= 0 {
		return Error(fmt.Sprintf("model_id must be a positive integer for neurondb_predict_batch tool: received %d", modelIDInt), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "model_id",
			"value":     modelIDInt,
			"params":    params,
		}), nil
	}

	featuresArray, ok := params["features_array"].([]interface{})
	if !ok {
		return Error(fmt.Sprintf("features_array parameter must be an array for neurondb_predict_batch tool: model_id=%d, received type %T, value=%v", modelIDInt, params["features_array"], params["features_array"]), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":     "features_array",
			"model_id":      modelIDInt,
			"received_type": fmt.Sprintf("%T", params["features_array"]),
			"params":        params,
		}), nil
	}

	if len(featuresArray) == 0 {
		return Error(fmt.Sprintf("features_array cannot be empty for neurondb_predict_batch tool: model_id=%d, features_count=0", modelIDInt), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":     "features_array",
			"model_id":      modelIDInt,
			"features_count": 0,
			"params":        params,
		}), nil
	}

	if len(featuresArray) > 1000 {
		return Error(fmt.Sprintf("features_array exceeds maximum size of 1000 for neurondb_predict_batch tool: model_id=%d, features_count=%d", modelIDInt, len(featuresArray)), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":     "features_array",
			"model_id":      modelIDInt,
			"features_count": len(featuresArray),
			"max_count":     1000,
			"params":        params,
		}), nil
	}

  /* Convert features_array to vector array format for PostgreSQL */
  /* Format: ARRAY['[1,2,3]'::vector, '[4,5,6]'::vector] */
	var vectorStrings []string
	for i, features := range featuresArray {
		featureVec, ok := features.([]interface{})
		if !ok {
			return Error(fmt.Sprintf("features_array element at index %d must be an array for neurondb_predict_batch tool: model_id=%d, features_count=%d, element_type=%T", i, modelIDInt, len(featuresArray), features), "VALIDATION_ERROR", map[string]interface{}{
				"parameter":     "features_array",
				"model_id":      modelIDInt,
				"features_count": len(featuresArray),
				"invalid_index": i,
				"received_type": fmt.Sprintf("%T", features),
				"params":        params,
			}), nil
		}

		if len(featureVec) == 0 {
			return Error(fmt.Sprintf("features_array element at index %d cannot be empty for neurondb_predict_batch tool: model_id=%d, features_count=%d", i, modelIDInt, len(featuresArray)), "VALIDATION_ERROR", map[string]interface{}{
				"parameter":     "features_array",
				"model_id":      modelIDInt,
				"features_count": len(featuresArray),
				"empty_index":   i,
				"params":        params,
			}), nil
		}

		vectorStr := formatVectorFromInterface(featureVec)
		vectorStrings = append(vectorStrings, vectorStr)
	}

  /* Build query: SELECT neurondb.predict_batch(model_id, ARRAY[vector1, vector2, ...]::vector[]) */
  /* Need to build array of vectors properly */
	if len(vectorStrings) == 0 {
		return Error(fmt.Sprintf("No valid vectors in features_array for neurondb_predict_batch tool: model_id=%d", modelIDInt), "VALIDATION_ERROR", map[string]interface{}{
			"model_id": modelIDInt,
			"params":   params,
		}), nil
	}

  /* Build array literal: ARRAY['[1,2,3]'::vector, '[4,5,6]'::vector]::vector[] */
	var arrayParts []string
	for _, vecStr := range vectorStrings {
		arrayParts = append(arrayParts, fmt.Sprintf("'%s'::vector", vecStr))
	}
	
	arrayLiteral := "ARRAY[" + arrayParts[0]
	for i := 1; i < len(arrayParts); i++ {
		arrayLiteral += ", " + arrayParts[i]
	}
	arrayLiteral += "]::vector[]"

	query := fmt.Sprintf("SELECT neurondb.predict_batch($1, %s) AS predictions", arrayLiteral)

	executor := NewQueryExecutor(t.db)
	result, err := executor.ExecuteQueryOne(ctx, query, []interface{}{modelIDInt})
	if err != nil {
		t.logger.Error("Batch prediction failed", err, params)
		return Error(fmt.Sprintf("Batch prediction execution failed: model_id=%d, features_count=%d, error=%v", modelIDInt, len(featuresArray), err), "PREDICTION_ERROR", map[string]interface{}{
			"model_id":       modelIDInt,
			"features_count": len(featuresArray),
			"error":          err.Error(),
		}), nil
	}

	return Success(result, map[string]interface{}{
		"model_id":       modelIDInt,
		"features_count": len(featuresArray),
		"predictions_count": len(featuresArray),
	}), nil
}

/* ExportModelTool exports a trained ML model */
type ExportModelTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

/* NewExportModelTool creates a new ExportModelTool */
func NewExportModelTool(db *database.Database, logger *logging.Logger) *ExportModelTool {
	return &ExportModelTool{
		BaseTool: NewBaseTool(
			"postgresql_export_model",
			"Export a trained machine learning model to various formats (ONNX, PMML, JSON)",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"model_id": map[string]interface{}{
						"type":        "number",
						"description": "The ID of the model to export",
					},
					"format": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"onnx", "pmml", "json"},
						"default":     "json",
						"description": "Export format (onnx, pmml, json)",
					},
					"path": map[string]interface{}{
						"type":        "string",
						"description": "Optional file path for export (if not provided, returns model data)",
					},
				},
				"required": []interface{}{"model_id"},
			},
		),
		db:     db,
		logger: logger,
	}
}

/* Execute exports the model */
func (t *ExportModelTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for neurondb_export_model tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	modelID, ok := params["model_id"].(float64)
	if !ok {
		return Error(fmt.Sprintf("model_id parameter must be a number for neurondb_export_model tool: received type %T, value=%v", params["model_id"], params["model_id"]), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":     "model_id",
			"received_type": fmt.Sprintf("%T", params["model_id"]),
			"received_value": params["model_id"],
			"params":        params,
		}), nil
	}

	modelIDInt := int(modelID)
	if modelIDInt <= 0 {
		return Error(fmt.Sprintf("model_id must be a positive integer for neurondb_export_model tool: received %d", modelIDInt), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "model_id",
			"value":     modelIDInt,
			"params":    params,
		}), nil
	}

	format, _ := params["format"].(string)
	if format == "" {
		format = "json"
	}

	validFormats := map[string]bool{"onnx": true, "pmml": true, "json": true}
	if !validFormats[format] {
		return Error(fmt.Sprintf("invalid format '%s' for neurondb_export_model tool: model_id=%d, valid formats are: onnx, pmml, json", format, modelIDInt), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":    "format",
			"model_id":     modelIDInt,
			"format":       format,
			"valid_formats": []string{"onnx", "pmml", "json"},
			"params":       params,
		}), nil
	}

	path, _ := params["path"].(string)

	var query string
	var queryParams []interface{}

	if format == "onnx" {
		if path != "" {
			query = `SELECT neurondb.export_model_to_onnx($1, $2) AS export_result`
			queryParams = []interface{}{modelIDInt, path}
		} else {
			query = `SELECT neurondb.export_model($1, 'onnx', NULL) AS export_result`
			queryParams = []interface{}{modelIDInt}
		}
	} else {
		if path != "" {
			query = `SELECT neurondb.export_model($1, $2, $3) AS export_result`
			queryParams = []interface{}{modelIDInt, format, path}
		} else {
			query = `SELECT neurondb.export_model($1, $2, NULL) AS export_result`
			queryParams = []interface{}{modelIDInt, format}
		}
	}

	executor := NewQueryExecutor(t.db)
	result, err := executor.ExecuteQueryOne(ctx, query, queryParams)
	if err != nil {
		t.logger.Error("Model export failed", err, params)
		return Error(fmt.Sprintf("Model export execution failed: model_id=%d, format='%s', path='%s', error=%v", modelIDInt, format, path, err), "EXPORT_ERROR", map[string]interface{}{
			"model_id": modelIDInt,
			"format":   format,
			"path":     path,
			"error":    err.Error(),
		}), nil
	}

	return Success(result, map[string]interface{}{
		"model_id": modelIDInt,
		"format":   format,
		"path":     path,
	}), nil
}

/* Helper function to format vector array for PostgreSQL */
func formatVectorArray(vectorStrings []string) string {
	if len(vectorStrings) == 0 {
		return ""
	}
	var parts []string
	for _, vecStr := range vectorStrings {
		parts = append(parts, fmt.Sprintf("'%s'::vector", vecStr))
	}
	return fmt.Sprintf("%s", parts[0])
}


