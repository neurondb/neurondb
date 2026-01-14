/*-------------------------------------------------------------------------
 *
 * analytics_tool.go
 *    Analytics tool handler for NeuronAgent
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/tools/analytics_tool.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/pkg/neurondb"
)

/* AnalyticsTool handles analytics operations */
type AnalyticsTool struct {
	client *neurondb.AnalyticsClient
}

/* NewAnalyticsTool creates a new analytics tool */
func NewAnalyticsTool(client *neurondb.AnalyticsClient) *AnalyticsTool {
	return &AnalyticsTool{client: client}
}

/* Execute executes an analytics operation */
func (t *AnalyticsTool) Execute(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	operation, ok := args["operation"].(string)
	if !ok {
		return "", fmt.Errorf("analytics tool execution failed: tool_name='%s', missing_operation=true", tool.Name)
	}

	switch operation {
	case "cluster":
		return t.cluster(ctx, args)
	case "detect_outliers":
		return t.detectOutliers(ctx, args)
	case "reduce_dimensionality":
		return t.reduceDimensionality(ctx, args)
	case "analyze_data":
		return t.analyzeData(ctx, args)
	default:
		return "", fmt.Errorf("analytics tool execution failed: tool_name='%s', operation='%s', unsupported_operation=true", tool.Name, operation)
	}
}

/* Validate validates analytics tool arguments */
func (t *AnalyticsTool) Validate(args map[string]interface{}, schema map[string]interface{}) error {
	return nil
}

func (t *AnalyticsTool) cluster(ctx context.Context, args map[string]interface{}) (string, error) {
	tableName := getString(args, "table_name", "")
	featureCol := getString(args, "feature_col", "")
	algorithm := getString(args, "algorithm", "kmeans")

	params := make(map[string]interface{})
	if paramsInterface, ok := args["params"].(map[string]interface{}); ok {
		params = paramsInterface
	}

	results, err := t.client.ClusterData(ctx, tableName, featureCol, algorithm, params)
	if err != nil {
		return "", err
	}

	resultJSON, _ := json.Marshal(results)
	return string(resultJSON), nil
}

func (t *AnalyticsTool) detectOutliers(ctx context.Context, args map[string]interface{}) (string, error) {
	tableName := getString(args, "table_name", "")
	featureCol := getString(args, "feature_col", "")
	method := getString(args, "method", "z_score")

	params := make(map[string]interface{})
	if paramsInterface, ok := args["params"].(map[string]interface{}); ok {
		params = paramsInterface
	}

	results, err := t.client.DetectOutliers(ctx, tableName, featureCol, method, params)
	if err != nil {
		return "", err
	}

	resultJSON, _ := json.Marshal(results)
	return string(resultJSON), nil
}

func (t *AnalyticsTool) reduceDimensionality(ctx context.Context, args map[string]interface{}) (string, error) {
	tableName := getString(args, "table_name", "")
	featureCol := getString(args, "feature_col", "")
	method := getString(args, "method", "pca")
	targetDim := getIntDefault(args, "target_dim", 2)

	params := make(map[string]interface{})
	if paramsInterface, ok := args["params"].(map[string]interface{}); ok {
		params = paramsInterface
	}

	results, err := t.client.ReduceDimensionality(ctx, tableName, featureCol, method, targetDim, params)
	if err != nil {
		return "", err
	}

	resultJSON, _ := json.Marshal(results)
	return string(resultJSON), nil
}

func (t *AnalyticsTool) analyzeData(ctx context.Context, args map[string]interface{}) (string, error) {
	tableName := getString(args, "table_name", "")

	columnsInterface, ok := args["columns"].([]interface{})
	if !ok {
		return "", fmt.Errorf("data analysis failed: columns_missing_or_invalid=true")
	}

	columns := make([]string, len(columnsInterface))
	for i, col := range columnsInterface {
		columns[i] = fmt.Sprintf("%v", col)
	}

	analysis, err := t.client.AnalyzeData(ctx, tableName, columns)
	if err != nil {
		return "", err
	}

	resultJSON, _ := json.Marshal(analysis)
	return string(resultJSON), nil
}
