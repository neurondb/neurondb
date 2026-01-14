/*-------------------------------------------------------------------------
 *
 * vector_tool.go
 *    Vector operations tool handler for NeuronAgent
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/tools/vector_tool.go
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

/* VectorTool handles vector operations */
type VectorTool struct {
	client *neurondb.VectorClient
}

/* NewVectorTool creates a new vector tool */
func NewVectorTool(client *neurondb.VectorClient) *VectorTool {
	return &VectorTool{client: client}
}

/* Execute executes a vector operation */
func (t *VectorTool) Execute(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	operation, ok := args["operation"].(string)
	if !ok {
		return "", fmt.Errorf("vector tool execution failed: tool_name='%s', missing_operation=true", tool.Name)
	}

	switch operation {
	case "search":
		return t.search(ctx, args)
	case "create_hnsw_index":
		return t.createHNSWIndex(ctx, args)
	case "create_ivf_index":
		return t.createIVFIndex(ctx, args)
	case "drop_index":
		return t.dropIndex(ctx, args)
	case "quantize":
		return t.quantize(ctx, args)
	default:
		return "", fmt.Errorf("vector tool execution failed: tool_name='%s', operation='%s', unsupported_operation=true", tool.Name, operation)
	}
}

/* Validate validates vector tool arguments */
func (t *VectorTool) Validate(args map[string]interface{}, schema map[string]interface{}) error {
	return nil
}

func (t *VectorTool) search(ctx context.Context, args map[string]interface{}) (string, error) {
	tableName := getString(args, "table_name", "")
	vectorCol := getString(args, "vector_col", "embedding")
	limit := getIntDefault(args, "limit", 10)
	metric := getString(args, "metric", "l2")

	queryVectorInterface, ok := args["query_vector"].([]interface{})
	if !ok {
		return "", fmt.Errorf("vector search failed: query_vector_missing_or_invalid=true")
	}

	queryVector := make(neurondb.Vector, len(queryVectorInterface))
	for i, v := range queryVectorInterface {
		if val, ok := v.(float64); ok {
			queryVector[i] = float32(val)
		} else {
			return "", fmt.Errorf("vector search failed: query_vector_index=%d, invalid_type=true", i)
		}
	}

	results, err := t.client.VectorSearch(ctx, tableName, vectorCol, queryVector, limit, metric)
	if err != nil {
		return "", err
	}

	resultJSON, _ := json.Marshal(results)
	return string(resultJSON), nil
}

func (t *VectorTool) createHNSWIndex(ctx context.Context, args map[string]interface{}) (string, error) {
	indexName := getString(args, "index_name", "")
	tableName := getString(args, "table_name", "")
	vectorCol := getString(args, "vector_col", "embedding")

	params := make(map[string]interface{})
	if paramsInterface, ok := args["params"].(map[string]interface{}); ok {
		params = paramsInterface
	}

	err := t.client.CreateHNSWIndex(ctx, indexName, tableName, vectorCol, params)
	if err != nil {
		return "", err
	}

	result := map[string]interface{}{
		"status":     "created",
		"index_name": indexName,
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON), nil
}

func (t *VectorTool) createIVFIndex(ctx context.Context, args map[string]interface{}) (string, error) {
	indexName := getString(args, "index_name", "")
	tableName := getString(args, "table_name", "")
	vectorCol := getString(args, "vector_col", "embedding")

	params := make(map[string]interface{})
	if paramsInterface, ok := args["params"].(map[string]interface{}); ok {
		params = paramsInterface
	}

	err := t.client.CreateIVFIndex(ctx, indexName, tableName, vectorCol, params)
	if err != nil {
		return "", err
	}

	result := map[string]interface{}{
		"status":     "created",
		"index_name": indexName,
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON), nil
}

func (t *VectorTool) dropIndex(ctx context.Context, args map[string]interface{}) (string, error) {
	indexName := getString(args, "index_name", "")

	err := t.client.DropIndex(ctx, indexName)
	if err != nil {
		return "", err
	}

	result := map[string]interface{}{
		"status":     "deleted",
		"index_name": indexName,
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON), nil
}

func (t *VectorTool) quantize(ctx context.Context, args map[string]interface{}) (string, error) {
	method := getString(args, "method", "int8")

	vectorInterface, ok := args["vector"].([]interface{})
	if !ok {
		return "", fmt.Errorf("vector quantization failed: vector_missing_or_invalid=true")
	}

	vector := make(neurondb.Vector, len(vectorInterface))
	for i, v := range vectorInterface {
		if val, ok := v.(float64); ok {
			vector[i] = float32(val)
		} else {
			return "", fmt.Errorf("vector quantization failed: vector_index=%d, invalid_type=true", i)
		}
	}

	quantized, err := t.client.QuantizeVector(ctx, vector, method)
	if err != nil {
		return "", err
	}

	result := map[string]interface{}{
		"quantized": quantized,
		"method":    method,
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON), nil
}

func getIntDefault(args map[string]interface{}, key string, defaultValue int) int {
	if val, ok := args[key].(int); ok {
		return val
	}
	if val, ok := args[key].(float64); ok {
		return int(val)
	}
	return defaultValue
}
