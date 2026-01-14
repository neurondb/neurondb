/*-------------------------------------------------------------------------
 *
 * ml_tool.go
 *    ML tool handler for NeuronAgent
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/tools/ml_tool.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"

	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/pkg/neurondb"
)

/* MLTool handles ML operations */
type MLTool struct {
	client *neurondb.MLClient
}

/* NewMLTool creates a new ML tool */
func NewMLTool(client *neurondb.MLClient) *MLTool {
	return &MLTool{client: client}
}

/* Execute executes an ML operation */
func (t *MLTool) Execute(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	operation, ok := args["operation"].(string)
	if !ok {
		return "", fmt.Errorf("ML tool execution failed: tool_name='%s', missing_operation=true", tool.Name)
	}

	switch operation {
	case "train":
		return t.train(ctx, args)
	case "predict":
		return t.predict(ctx, args)
	case "predict_batch":
		return t.predictBatch(ctx, args)
	case "evaluate":
		return t.evaluate(ctx, args)
	case "list_models":
		return t.listModels(ctx, args)
	case "get_model_info":
		return t.getModelInfo(ctx, args)
	case "delete_model":
		return t.deleteModel(ctx, args)
	default:
		return "", fmt.Errorf("ML tool execution failed: tool_name='%s', operation='%s', unsupported_operation=true", tool.Name, operation)
	}
}

/* Validate validates ML tool arguments */
func (t *MLTool) Validate(args map[string]interface{}, schema map[string]interface{}) error {
	/* Basic validation - full validation handled by validator */
	return nil
}

func (t *MLTool) train(ctx context.Context, args map[string]interface{}) (string, error) {
	project := getString(args, "project", "default")
	algorithm := getString(args, "algorithm", "")
	tableName := getString(args, "table_name", "")
	labelCol := getString(args, "label_col", "")

	/* Validate required fields */
	if algorithm == "" {
		return "", fmt.Errorf("ML training failed: algorithm_required=true")
	}
	if tableName == "" {
		return "", fmt.Errorf("ML training failed: table_name_required=true")
	}
	if labelCol == "" {
		return "", fmt.Errorf("ML training failed: label_col_required=true")
	}

	featureColsInterface, ok := args["feature_cols"].([]interface{})
	if !ok {
		return "", fmt.Errorf("ML training failed: feature_cols_missing_or_invalid=true")
	}
	if len(featureColsInterface) == 0 {
		return "", fmt.Errorf("ML training failed: feature_cols_empty=true")
	}

	featureCols := make([]string, len(featureColsInterface))
	for i, col := range featureColsInterface {
		featureCols[i] = fmt.Sprintf("%v", col)
	}

	params := make(map[string]interface{})
	if paramsInterface, ok := args["params"].(map[string]interface{}); ok {
		params = paramsInterface
	}

	modelID, err := t.client.TrainModel(ctx, project, algorithm, tableName, labelCol, featureCols, params)
	if err != nil {
		return "", err
	}

	result := map[string]interface{}{
		"model_id": modelID,
		"status":   "trained",
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON), nil
}

func (t *MLTool) predict(ctx context.Context, args map[string]interface{}) (string, error) {
	modelID, err := getInt(args, "model_id")
	if err != nil {
		return "", fmt.Errorf("ML prediction failed: model_id_missing_or_invalid=true, error=%w", err)
	}

	featuresInterface, ok := args["features"].([]interface{})
	if !ok {
		return "", fmt.Errorf("ML prediction failed: features_missing_or_invalid=true")
	}

	features := make([]float32, len(featuresInterface))
	for i, f := range featuresInterface {
		if val, ok := f.(float64); ok {
			features[i] = float32(val)
		} else {
			return "", fmt.Errorf("ML prediction failed: feature_index=%d, invalid_type=true", i)
		}
	}

	prediction, err := t.client.Predict(ctx, modelID, features)
	if err != nil {
		return "", err
	}

	result := map[string]interface{}{
		"prediction": prediction,
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON), nil
}

func (t *MLTool) predictBatch(ctx context.Context, args map[string]interface{}) (string, error) {
	modelID, err := getInt(args, "model_id")
	if err != nil {
		return "", fmt.Errorf("ML batch prediction failed: model_id_missing_or_invalid=true, error=%w", err)
	}

	featuresInterface, ok := args["features"].([]interface{})
	if !ok {
		return "", fmt.Errorf("ML batch prediction failed: features_missing_or_invalid=true")
	}

	features := make([][]float32, len(featuresInterface))
	for i, featInterface := range featuresInterface {
		featArray, ok := featInterface.([]interface{})
		if !ok {
			return "", fmt.Errorf("ML batch prediction failed: feature_row_index=%d, invalid_type=true", i)
		}
		features[i] = make([]float32, len(featArray))
		for j, f := range featArray {
			if val, ok := f.(float64); ok {
				features[i][j] = float32(val)
			} else {
				return "", fmt.Errorf("ML batch prediction failed: feature_row_index=%d, feature_col_index=%d, invalid_type=true", i, j)
			}
		}
	}

	predictions, err := t.client.PredictBatch(ctx, modelID, features)
	if err != nil {
		return "", err
	}

	result := map[string]interface{}{
		"predictions": predictions,
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON), nil
}

func (t *MLTool) evaluate(ctx context.Context, args map[string]interface{}) (string, error) {
	modelID, err := getInt(args, "model_id")
	if err != nil {
		return "", fmt.Errorf("ML evaluation failed: model_id_missing_or_invalid=true, error=%w", err)
	}

	testTable := getString(args, "test_table", "")
	labelCol := getString(args, "label_col", "")

	featureColsInterface, ok := args["feature_cols"].([]interface{})
	if !ok {
		return "", fmt.Errorf("ML evaluation failed: feature_cols_missing_or_invalid=true")
	}

	featureCols := make([]string, len(featureColsInterface))
	for i, col := range featureColsInterface {
		featureCols[i] = fmt.Sprintf("%v", col)
	}

	metrics, err := t.client.EvaluateModel(ctx, modelID, testTable, labelCol, featureCols)
	if err != nil {
		return "", err
	}

	resultJSON, _ := json.Marshal(metrics)
	return string(resultJSON), nil
}

func (t *MLTool) listModels(ctx context.Context, args map[string]interface{}) (string, error) {
	project := getString(args, "project", "default")

	models, err := t.client.ListModels(ctx, project)
	if err != nil {
		return "", err
	}

	resultJSON, _ := json.Marshal(models)
	return string(resultJSON), nil
}

func (t *MLTool) getModelInfo(ctx context.Context, args map[string]interface{}) (string, error) {
	modelID, err := getInt(args, "model_id")
	if err != nil {
		return "", fmt.Errorf("ML model info retrieval failed: model_id_missing_or_invalid=true, error=%w", err)
	}

	model, err := t.client.GetModelInfo(ctx, modelID)
	if err != nil {
		return "", err
	}

	resultJSON, _ := json.Marshal(model)
	return string(resultJSON), nil
}

func (t *MLTool) deleteModel(ctx context.Context, args map[string]interface{}) (string, error) {
	modelID, err := getInt(args, "model_id")
	if err != nil {
		return "", fmt.Errorf("ML model deletion failed: model_id_missing_or_invalid=true, error=%w", err)
	}

	err = t.client.DeleteModel(ctx, modelID)
	if err != nil {
		return "", err
	}

	result := map[string]interface{}{
		"status":   "deleted",
		"model_id": modelID,
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON), nil
}

/* Helper functions */
func getString(args map[string]interface{}, key, defaultValue string) string {
	if val, ok := args[key].(string); ok {
		return val
	}
	return defaultValue
}

func getInt(args map[string]interface{}, key string) (int, error) {
	if val, ok := args[key].(int); ok {
		return val, nil
	}
	if val, ok := args[key].(float64); ok {
		return int(val), nil
	}
	if val, ok := args[key].(string); ok {
		return strconv.Atoi(val)
	}
	return 0, fmt.Errorf("invalid int value for key: %s", key)
}
