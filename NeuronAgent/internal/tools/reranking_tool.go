/*-------------------------------------------------------------------------
 *
 * reranking_tool.go
 *    Reranking tool handler for NeuronAgent
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/tools/reranking_tool.go
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

/* RerankingTool handles reranking operations */
type RerankingTool struct {
	client *neurondb.RerankingClient
}

/* NewRerankingTool creates a new reranking tool */
func NewRerankingTool(client *neurondb.RerankingClient) *RerankingTool {
	return &RerankingTool{client: client}
}

/* Execute executes a reranking operation */
func (t *RerankingTool) Execute(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	operation, ok := args["operation"].(string)
	if !ok {
		return "", fmt.Errorf("reranking tool execution failed: tool_name='%s', missing_operation=true", tool.Name)
	}

	switch operation {
	case "rerank_cross_encoder":
		return t.rerankCrossEncoder(ctx, args)
	case "rerank_llm":
		return t.rerankLLM(ctx, args)
	case "rerank_colbert":
		return t.rerankColBERT(ctx, args)
	case "rerank_ensemble":
		return t.rerankEnsemble(ctx, args)
	default:
		return "", fmt.Errorf("reranking tool execution failed: tool_name='%s', operation='%s', unsupported_operation=true", tool.Name, operation)
	}
}

/* Validate validates reranking tool arguments */
func (t *RerankingTool) Validate(args map[string]interface{}, schema map[string]interface{}) error {
	return nil
}

func (t *RerankingTool) rerankCrossEncoder(ctx context.Context, args map[string]interface{}) (string, error) {
	query := getString(args, "query", "")
	model := getString(args, "model", "cross-encoder")
	topK := getIntDefault(args, "top_k", 5)

	documentsInterface, ok := args["documents"].([]interface{})
	if !ok {
		return "", fmt.Errorf("cross-encoder reranking failed: documents_missing_or_invalid=true")
	}

	documents := make([]string, len(documentsInterface))
	for i, doc := range documentsInterface {
		documents[i] = fmt.Sprintf("%v", doc)
	}

	results, err := t.client.RerankCrossEncoder(ctx, query, documents, model, topK)
	if err != nil {
		return "", err
	}

	resultJSON, _ := json.Marshal(results)
	return string(resultJSON), nil
}

func (t *RerankingTool) rerankLLM(ctx context.Context, args map[string]interface{}) (string, error) {
	query := getString(args, "query", "")
	model := getString(args, "model", "gpt-4")
	topK := getIntDefault(args, "top_k", 5)

	documentsInterface, ok := args["documents"].([]interface{})
	if !ok {
		return "", fmt.Errorf("LLM reranking failed: documents_missing_or_invalid=true")
	}

	documents := make([]string, len(documentsInterface))
	for i, doc := range documentsInterface {
		documents[i] = fmt.Sprintf("%v", doc)
	}

	params := make(map[string]interface{})
	if paramsInterface, ok := args["params"].(map[string]interface{}); ok {
		params = paramsInterface
	}

	results, err := t.client.RerankLLM(ctx, query, documents, model, topK, params)
	if err != nil {
		return "", err
	}

	resultJSON, _ := json.Marshal(results)
	return string(resultJSON), nil
}

func (t *RerankingTool) rerankColBERT(ctx context.Context, args map[string]interface{}) (string, error) {
	query := getString(args, "query", "")
	model := getString(args, "model", "colbert")
	topK := getIntDefault(args, "top_k", 5)

	documentsInterface, ok := args["documents"].([]interface{})
	if !ok {
		return "", fmt.Errorf("ColBERT reranking failed: documents_missing_or_invalid=true")
	}

	documents := make([]string, len(documentsInterface))
	for i, doc := range documentsInterface {
		documents[i] = fmt.Sprintf("%v", doc)
	}

	results, err := t.client.RerankColBERT(ctx, query, documents, model, topK)
	if err != nil {
		return "", err
	}

	resultJSON, _ := json.Marshal(results)
	return string(resultJSON), nil
}

func (t *RerankingTool) rerankEnsemble(ctx context.Context, args map[string]interface{}) (string, error) {
	query := getString(args, "query", "")
	topK := getIntDefault(args, "top_k", 5)

	documentsInterface, ok := args["documents"].([]interface{})
	if !ok {
		return "", fmt.Errorf("ensemble reranking failed: documents_missing_or_invalid=true")
	}

	documents := make([]string, len(documentsInterface))
	for i, doc := range documentsInterface {
		documents[i] = fmt.Sprintf("%v", doc)
	}

	methodsInterface, ok := args["methods"].([]interface{})
	if !ok {
		return "", fmt.Errorf("ensemble reranking failed: methods_missing_or_invalid=true")
	}

	methods := make([]string, len(methodsInterface))
	for i, m := range methodsInterface {
		methods[i] = fmt.Sprintf("%v", m)
	}

	weightsInterface, ok := args["weights"].([]interface{})
	if !ok {
		/* Default equal weights */
		weights := make([]float64, len(methods))
		for i := range weights {
			weights[i] = 1.0 / float64(len(methods))
		}
		return t.rerankEnsembleWithWeights(ctx, query, documents, methods, weights, topK)
	}

	weights := make([]float64, len(weightsInterface))
	for i, w := range weightsInterface {
		if val, ok := w.(float64); ok {
			weights[i] = val
		} else {
			return "", fmt.Errorf("ensemble reranking failed: weight_index=%d, invalid_type=true", i)
		}
	}

	return t.rerankEnsembleWithWeights(ctx, query, documents, methods, weights, topK)
}

func (t *RerankingTool) rerankEnsembleWithWeights(ctx context.Context, query string, documents []string, methods []string, weights []float64, topK int) (string, error) {
	results, err := t.client.RerankEnsemble(ctx, query, documents, methods, weights, topK)
	if err != nil {
		return "", err
	}

	resultJSON, _ := json.Marshal(results)
	return string(resultJSON), nil
}
