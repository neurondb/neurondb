/*-------------------------------------------------------------------------
 *
 * rag_tool.go
 *    RAG pipeline tool handler for NeuronAgent
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/tools/rag_tool.go
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

/* RAGTool handles RAG pipeline operations */
type RAGTool struct {
	client *neurondb.RAGClient
}

/* NewRAGTool creates a new RAG tool */
func NewRAGTool(client *neurondb.RAGClient) *RAGTool {
	return &RAGTool{client: client}
}

/* Execute executes a RAG operation */
func (t *RAGTool) Execute(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	operation, ok := args["operation"].(string)
	if !ok {
		return "", fmt.Errorf("RAG tool execution failed: tool_name='%s', missing_operation=true", tool.Name)
	}

	switch operation {
	case "chunk_document":
		return t.chunkDocument(ctx, args)
	case "retrieve_context":
		return t.retrieveContext(ctx, args)
	case "rerank_results":
		return t.rerankResults(ctx, args)
	case "generate_answer":
		return t.generateAnswer(ctx, args)
	default:
		return "", fmt.Errorf("RAG tool execution failed: tool_name='%s', operation='%s', unsupported_operation=true", tool.Name, operation)
	}
}

/* Validate validates RAG tool arguments */
func (t *RAGTool) Validate(args map[string]interface{}, schema map[string]interface{}) error {
	return nil
}

func (t *RAGTool) chunkDocument(ctx context.Context, args map[string]interface{}) (string, error) {
	text := getString(args, "text", "")
	chunkSize := getIntDefault(args, "chunk_size", 500)
	overlap := getIntDefault(args, "overlap", 50)

	chunks, err := t.client.ChunkDocument(ctx, text, chunkSize, overlap)
	if err != nil {
		return "", err
	}

	result := map[string]interface{}{
		"chunks": chunks,
		"count":  len(chunks),
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON), nil
}

func (t *RAGTool) retrieveContext(ctx context.Context, args map[string]interface{}) (string, error) {
	tableName := getString(args, "table_name", "")
	vectorCol := getString(args, "vector_col", "embedding")
	limit := getIntDefault(args, "limit", 5)

	queryVectorInterface, ok := args["query_embedding"].([]interface{})
	if !ok {
		return "", fmt.Errorf("RAG context retrieval failed: query_embedding_missing_or_invalid=true")
	}

	queryVector := make(neurondb.Vector, len(queryVectorInterface))
	for i, v := range queryVectorInterface {
		if val, ok := v.(float64); ok {
			queryVector[i] = float32(val)
		} else {
			return "", fmt.Errorf("RAG context retrieval failed: query_embedding_index=%d, invalid_type=true", i)
		}
	}

	contexts, err := t.client.RetrieveContext(ctx, queryVector, tableName, vectorCol, limit)
	if err != nil {
		return "", err
	}

	resultJSON, _ := json.Marshal(contexts)
	return string(resultJSON), nil
}

func (t *RAGTool) rerankResults(ctx context.Context, args map[string]interface{}) (string, error) {
	query := getString(args, "query", "")
	model := getString(args, "model", "cross-encoder")
	topK := getIntDefault(args, "top_k", 5)

	documentsInterface, ok := args["documents"].([]interface{})
	if !ok {
		return "", fmt.Errorf("RAG reranking failed: documents_missing_or_invalid=true")
	}

	documents := make([]string, len(documentsInterface))
	for i, doc := range documentsInterface {
		documents[i] = fmt.Sprintf("%v", doc)
	}

	results, err := t.client.RerankResults(ctx, query, documents, model, topK)
	if err != nil {
		return "", err
	}

	resultJSON, _ := json.Marshal(results)
	return string(resultJSON), nil
}

func (t *RAGTool) generateAnswer(ctx context.Context, args map[string]interface{}) (string, error) {
	query := getString(args, "query", "")
	model := getString(args, "model", "gpt-4")

	contextInterface, ok := args["context"].([]interface{})
	if !ok {
		return "", fmt.Errorf("RAG answer generation failed: context_missing_or_invalid=true")
	}

	context := make([]string, len(contextInterface))
	for i, ctx := range contextInterface {
		context[i] = fmt.Sprintf("%v", ctx)
	}

	params := make(map[string]interface{})
	if paramsInterface, ok := args["params"].(map[string]interface{}); ok {
		params = paramsInterface
	}

	answer, err := t.client.GenerateAnswer(ctx, query, context, model, params)
	if err != nil {
		return "", err
	}

	result := map[string]interface{}{
		"answer": answer,
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON), nil
}
