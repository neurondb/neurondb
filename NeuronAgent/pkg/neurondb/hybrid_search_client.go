/*-------------------------------------------------------------------------
 *
 * hybrid_search_client.go
 *    Hybrid search operations via NeuronDB
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/pkg/neurondb/hybrid_search_client.go
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

/* HybridSearchClient handles hybrid search operations via NeuronDB */
type HybridSearchClient struct {
	db *sqlx.DB
}

/* NewHybridSearchClient creates a new hybrid search client */
func NewHybridSearchClient(db *sqlx.DB) *HybridSearchClient {
	return &HybridSearchClient{db: db}
}

/* HybridSearch performs hybrid search combining vector and full-text search */
func (c *HybridSearchClient) HybridSearch(ctx context.Context, query string, queryEmbedding Vector, tableName, vectorCol, textCol string, limit int, params map[string]interface{}) ([]HybridSearchResult, error) {
	paramsJSON, err := json.Marshal(params)
	if err != nil {
		return nil, fmt.Errorf("hybrid search failed: query_length=%d, query_embedding_dimension=%d, table_name='%s', parameter_marshaling_error=true, error=%w",
			len(query), len(queryEmbedding), tableName, err)
	}

	var resultsJSON string
	querySQL := `SELECT neurondb_hybrid_search($1, $2::vector, $3, $4, $5, $6, $7::jsonb) AS results`

	err = c.db.GetContext(ctx, &resultsJSON, querySQL, query, queryEmbedding, tableName, vectorCol, textCol, limit, paramsJSON)
	if err != nil {
		return nil, fmt.Errorf("hybrid search failed via NeuronDB: query_length=%d, query_embedding_dimension=%d, table_name='%s', vector_col='%s', text_col='%s', limit=%d, function='neurondb_hybrid_search', error=%w",
			len(query), len(queryEmbedding), tableName, vectorCol, textCol, limit, err)
	}

	var results []HybridSearchResult
	if err := json.Unmarshal([]byte(resultsJSON), &results); err != nil {
		return nil, fmt.Errorf("hybrid search result parsing failed: query_length=%d, results_json_length=%d, error=%w",
			len(query), len(resultsJSON), err)
	}

	return results, nil
}

/* ReciprocalRankFusion performs reciprocal rank fusion on multiple result sets */
func (c *HybridSearchClient) ReciprocalRankFusion(ctx context.Context, resultSets [][]HybridSearchResult, k int) ([]HybridSearchResult, error) {
	resultSetsJSON, err := json.Marshal(resultSets)
	if err != nil {
		return nil, fmt.Errorf("reciprocal rank fusion failed: result_sets_count=%d, k=%d, parameter_marshaling_error=true, error=%w",
			len(resultSets), k, err)
	}

	var resultsJSON string
	query := `SELECT neurondb_reciprocal_rank_fusion($1::jsonb, $2) AS fused_results`

	err = c.db.GetContext(ctx, &resultsJSON, query, resultSetsJSON, k)
	if err != nil {
		return nil, fmt.Errorf("reciprocal rank fusion failed via NeuronDB: result_sets_count=%d, k=%d, function='neurondb_reciprocal_rank_fusion', error=%w",
			len(resultSets), k, err)
	}

	var results []HybridSearchResult
	if err := json.Unmarshal([]byte(resultsJSON), &results); err != nil {
		return nil, fmt.Errorf("reciprocal rank fusion result parsing failed: result_sets_count=%d, results_json_length=%d, error=%w",
			len(resultSets), len(resultsJSON), err)
	}

	return results, nil
}

/* SemanticKeywordSearch performs semantic + keyword search */
func (c *HybridSearchClient) SemanticKeywordSearch(ctx context.Context, query string, queryEmbedding Vector, tableName, vectorCol, textCol string, limit int) ([]HybridSearchResult, error) {
	var resultsJSON string
	querySQL := `SELECT neurondb_semantic_keyword_search($1, $2::vector, $3, $4, $5, $6) AS results`

	err := c.db.GetContext(ctx, &resultsJSON, querySQL, query, queryEmbedding, tableName, vectorCol, textCol, limit)
	if err != nil {
		return nil, fmt.Errorf("semantic keyword search failed via NeuronDB: query_length=%d, query_embedding_dimension=%d, table_name='%s', vector_col='%s', text_col='%s', limit=%d, function='neurondb_semantic_keyword_search', error=%w",
			len(query), len(queryEmbedding), tableName, vectorCol, textCol, limit, err)
	}

	var results []HybridSearchResult
	if err := json.Unmarshal([]byte(resultsJSON), &results); err != nil {
		return nil, fmt.Errorf("semantic keyword search result parsing failed: query_length=%d, results_json_length=%d, error=%w",
			len(query), len(resultsJSON), err)
	}

	return results, nil
}

/* MultiVectorSearch performs search using multiple vectors */
func (c *HybridSearchClient) MultiVectorSearch(ctx context.Context, queryEmbeddings []Vector, tableName, vectorCol string, limit int, params map[string]interface{}) ([]HybridSearchResult, error) {
	paramsJSON, err := json.Marshal(params)
	if err != nil {
		return nil, fmt.Errorf("multi-vector search failed: query_embeddings_count=%d, table_name='%s', parameter_marshaling_error=true, error=%w",
			len(queryEmbeddings), tableName, err)
	}

	var resultsJSON string
	querySQL := `SELECT neurondb_multi_vector_search($1::vector[], $2, $3, $4, $5::jsonb) AS results`

	err = c.db.GetContext(ctx, &resultsJSON, querySQL, queryEmbeddings, tableName, vectorCol, limit, paramsJSON)
	if err != nil {
		return nil, fmt.Errorf("multi-vector search failed via NeuronDB: query_embeddings_count=%d, table_name='%s', vector_col='%s', limit=%d, function='neurondb_multi_vector_search', error=%w",
			len(queryEmbeddings), tableName, vectorCol, limit, err)
	}

	var results []HybridSearchResult
	if err := json.Unmarshal([]byte(resultsJSON), &results); err != nil {
		return nil, fmt.Errorf("multi-vector search result parsing failed: query_embeddings_count=%d, results_json_length=%d, error=%w",
			len(queryEmbeddings), len(resultsJSON), err)
	}

	return results, nil
}

/* HybridSearchResult represents a hybrid search result */
type HybridSearchResult struct {
	ID            interface{}            `json:"id"`
	Content       string                 `json:"content"`
	VectorScore   float64                `json:"vector_score"`
	TextScore     float64                `json:"text_score"`
	CombinedScore float64                `json:"combined_score"`
	Metadata      map[string]interface{} `json:"metadata"`
}
