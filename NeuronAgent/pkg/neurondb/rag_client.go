/*-------------------------------------------------------------------------
 *
 * rag_client.go
 *    RAG pipeline operations via NeuronDB
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/pkg/neurondb/rag_client.go
 *
 *-------------------------------------------------------------------------
 */

package neurondb

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/jmoiron/sqlx"
)

/* RAGClient handles RAG pipeline operations via NeuronDB */
type RAGClient struct {
	db *sqlx.DB
}

/* NewRAGClient creates a new RAG client */
func NewRAGClient(db *sqlx.DB) *RAGClient {
	return &RAGClient{db: db}
}

/* ChunkDocument chunks a document into smaller pieces */
func (c *RAGClient) ChunkDocument(ctx context.Context, text string, chunkSize, overlap int) ([]string, error) {
	query := `SELECT neurondb_chunk_text($1, $2, $3) AS chunks`

	var chunksJSON string
	err := c.db.GetContext(ctx, &chunksJSON, query, text, chunkSize, overlap)
	if err != nil {
		return nil, fmt.Errorf("document chunking failed via NeuronDB: text_length=%d, chunk_size=%d, overlap=%d, function='neurondb_chunk_text', error=%w",
			len(text), chunkSize, overlap, err)
	}

	var chunks []string
	if err := json.Unmarshal([]byte(chunksJSON), &chunks); err != nil {
		return nil, fmt.Errorf("document chunking result parsing failed: text_length=%d, chunks_json_length=%d, error=%w",
			len(text), len(chunksJSON), err)
	}

	return chunks, nil
}

/* RetrieveContext retrieves relevant context for a query */
func (c *RAGClient) RetrieveContext(ctx context.Context, queryEmbedding Vector, tableName, vectorCol string, limit int) ([]RAGContext, error) {
	query := fmt.Sprintf(`
		SELECT id, content, metadata, 1 - (embedding <=> $1::vector) AS similarity
		FROM %s
		ORDER BY embedding <=> $1::vector
		LIMIT $2`,
		tableName)

	var contexts []RAGContext
	err := c.db.SelectContext(ctx, &contexts, query, queryEmbedding, limit)
	if err != nil {
		return nil, fmt.Errorf("context retrieval failed via NeuronDB: table_name='%s', vector_col='%s', query_embedding_dimension=%d, limit=%d, error=%w",
			tableName, vectorCol, len(queryEmbedding), limit, err)
	}

	return contexts, nil
}

/* RerankResults reranks search results using a reranking model */
func (c *RAGClient) RerankResults(ctx context.Context, query string, documents []string, model string, topK int) ([]RerankResult, error) {
	querySQL := `SELECT neurondb_rerank_results($1, $2::text[], $3, $4) AS reranked`

	var resultsJSON string
	err := c.db.GetContext(ctx, &resultsJSON, querySQL, query, documents, model, topK)
	if err != nil {
		return nil, fmt.Errorf("reranking failed via NeuronDB: query_length=%d, documents_count=%d, model='%s', top_k=%d, function='neurondb_rerank_results', error=%w",
			len(query), len(documents), model, topK, err)
	}

	var results []RerankResult
	if err := json.Unmarshal([]byte(resultsJSON), &results); err != nil {
		return nil, fmt.Errorf("reranking result parsing failed: query_length=%d, documents_count=%d, results_json_length=%d, error=%w",
			len(query), len(documents), len(resultsJSON), err)
	}

	return results, nil
}

/* GenerateAnswer generates an answer using RAG pipeline */
func (c *RAGClient) GenerateAnswer(ctx context.Context, query string, context []string, model string, params map[string]interface{}) (string, error) {
	paramsJSON, err := json.Marshal(params)
	if err != nil {
		return "", fmt.Errorf("RAG answer generation failed: query_length=%d, context_count=%d, model='%s', parameter_marshaling_error=true, error=%w",
			len(query), len(context), model, err)
	}

	var answer string
	querySQL := `SELECT neurondb_generate_answer($1, $2::text[], $3, $4::jsonb) AS answer`

	err = c.db.GetContext(ctx, &answer, querySQL, query, context, model, paramsJSON)
	if err != nil {
		return "", fmt.Errorf("RAG answer generation failed via NeuronDB: query_length=%d, context_count=%d, model='%s', function='neurondb_generate_answer', error=%w",
			len(query), len(context), model, err)
	}

	return answer, nil
}

/* RAGContext represents retrieved context for RAG */
type RAGContext struct {
	ID         interface{}            `db:"id"`
	Content    string                 `db:"content"`
	Metadata   map[string]interface{} `db:"metadata"`
	Similarity float64                `db:"similarity"`
}

/* RerankResult represents a reranked result */
type RerankResult struct {
	Document string  `json:"document"`
	Score    float64 `json:"score"`
	Rank     int     `json:"rank"`
}
