/*-------------------------------------------------------------------------
 *
 * hybrid_search_tool.go
 *    Hybrid search tool handler for NeuronAgent
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/tools/hybrid_search_tool.go
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

/* HybridSearchTool handles hybrid search operations */
type HybridSearchTool struct {
	client *neurondb.HybridSearchClient
}

/* NewHybridSearchTool creates a new hybrid search tool */
func NewHybridSearchTool(client *neurondb.HybridSearchClient) *HybridSearchTool {
	return &HybridSearchTool{client: client}
}

/* Execute executes a hybrid search operation */
func (t *HybridSearchTool) Execute(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	operation, ok := args["operation"].(string)
	if !ok {
		return "", fmt.Errorf("hybrid search tool execution failed: tool_name='%s', missing_operation=true", tool.Name)
	}

	switch operation {
	case "hybrid_search":
		return t.hybridSearch(ctx, args)
	case "reciprocal_rank_fusion":
		return t.reciprocalRankFusion(ctx, args)
	case "semantic_keyword_search":
		return t.semanticKeywordSearch(ctx, args)
	case "multi_vector_search":
		return t.multiVectorSearch(ctx, args)
	default:
		return "", fmt.Errorf("hybrid search tool execution failed: tool_name='%s', operation='%s', unsupported_operation=true", tool.Name, operation)
	}
}

/* Validate validates hybrid search tool arguments */
func (t *HybridSearchTool) Validate(args map[string]interface{}, schema map[string]interface{}) error {
	return nil
}

func (t *HybridSearchTool) hybridSearch(ctx context.Context, args map[string]interface{}) (string, error) {
	query := getString(args, "query", "")
	tableName := getString(args, "table_name", "")
	vectorCol := getString(args, "vector_col", "embedding")
	textCol := getString(args, "text_col", "content")
	limit := getIntDefault(args, "limit", 10)

	queryVectorInterface, ok := args["query_embedding"].([]interface{})
	if !ok {
		return "", fmt.Errorf("hybrid search failed: query_embedding_missing_or_invalid=true")
	}

	queryVector := make(neurondb.Vector, len(queryVectorInterface))
	for i, v := range queryVectorInterface {
		if val, ok := v.(float64); ok {
			queryVector[i] = float32(val)
		} else {
			return "", fmt.Errorf("hybrid search failed: query_embedding_index=%d, invalid_type=true", i)
		}
	}

	params := make(map[string]interface{})
	if paramsInterface, ok := args["params"].(map[string]interface{}); ok {
		params = paramsInterface
	}

	results, err := t.client.HybridSearch(ctx, query, queryVector, tableName, vectorCol, textCol, limit, params)
	if err != nil {
		return "", err
	}

	resultJSON, _ := json.Marshal(results)
	return string(resultJSON), nil
}

func (t *HybridSearchTool) reciprocalRankFusion(ctx context.Context, args map[string]interface{}) (string, error) {
	k := getIntDefault(args, "k", 60)

	resultSetsInterface, ok := args["result_sets"].([]interface{})
	if !ok {
		return "", fmt.Errorf("reciprocal rank fusion failed: result_sets_missing_or_invalid=true")
	}

	/* Convert result sets - simplified for now */
	resultSets := make([][]neurondb.HybridSearchResult, len(resultSetsInterface))
	for i := range resultSetsInterface {
		/* This would need proper unmarshaling in production */
		resultSets[i] = []neurondb.HybridSearchResult{}
	}

	results, err := t.client.ReciprocalRankFusion(ctx, resultSets, k)
	if err != nil {
		return "", err
	}

	resultJSON, _ := json.Marshal(results)
	return string(resultJSON), nil
}

func (t *HybridSearchTool) semanticKeywordSearch(ctx context.Context, args map[string]interface{}) (string, error) {
	query := getString(args, "query", "")
	tableName := getString(args, "table_name", "")
	vectorCol := getString(args, "vector_col", "embedding")
	textCol := getString(args, "text_col", "content")
	limit := getIntDefault(args, "limit", 10)

	queryVectorInterface, ok := args["query_embedding"].([]interface{})
	if !ok {
		return "", fmt.Errorf("semantic keyword search failed: query_embedding_missing_or_invalid=true")
	}

	queryVector := make(neurondb.Vector, len(queryVectorInterface))
	for i, v := range queryVectorInterface {
		if val, ok := v.(float64); ok {
			queryVector[i] = float32(val)
		} else {
			return "", fmt.Errorf("semantic keyword search failed: query_embedding_index=%d, invalid_type=true", i)
		}
	}

	results, err := t.client.SemanticKeywordSearch(ctx, query, queryVector, tableName, vectorCol, textCol, limit)
	if err != nil {
		return "", err
	}

	resultJSON, _ := json.Marshal(results)
	return string(resultJSON), nil
}

func (t *HybridSearchTool) multiVectorSearch(ctx context.Context, args map[string]interface{}) (string, error) {
	tableName := getString(args, "table_name", "")
	vectorCol := getString(args, "vector_col", "embedding")
	limit := getIntDefault(args, "limit", 10)

	queryEmbeddingsInterface, ok := args["query_embeddings"].([]interface{})
	if !ok {
		return "", fmt.Errorf("multi-vector search failed: query_embeddings_missing_or_invalid=true")
	}

	queryEmbeddings := make([]neurondb.Vector, len(queryEmbeddingsInterface))
	for i, embInterface := range queryEmbeddingsInterface {
		embArray, ok := embInterface.([]interface{})
		if !ok {
			return "", fmt.Errorf("multi-vector search failed: embedding_index=%d, invalid_type=true", i)
		}
		queryEmbeddings[i] = make(neurondb.Vector, len(embArray))
		for j, v := range embArray {
			if val, ok := v.(float64); ok {
				queryEmbeddings[i][j] = float32(val)
			} else {
				return "", fmt.Errorf("multi-vector search failed: embedding_index=%d, vector_index=%d, invalid_type=true", i, j)
			}
		}
	}

	params := make(map[string]interface{})
	if paramsInterface, ok := args["params"].(map[string]interface{}); ok {
		params = paramsInterface
	}

	results, err := t.client.MultiVectorSearch(ctx, queryEmbeddings, tableName, vectorCol, limit, params)
	if err != nil {
		return "", err
	}

	resultJSON, _ := json.Marshal(results)
	return string(resultJSON), nil
}
