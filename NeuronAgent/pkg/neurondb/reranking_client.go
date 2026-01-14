/*-------------------------------------------------------------------------
 *
 * reranking_client.go
 *    Reranking operations via NeuronDB
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/pkg/neurondb/reranking_client.go
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

/* RerankingClient handles reranking operations via NeuronDB */
type RerankingClient struct {
	db *sqlx.DB
}

/* NewRerankingClient creates a new reranking client */
func NewRerankingClient(db *sqlx.DB) *RerankingClient {
	return &RerankingClient{db: db}
}

/* RerankCrossEncoder reranks using a cross-encoder model */
func (c *RerankingClient) RerankCrossEncoder(ctx context.Context, query string, documents []string, model string, topK int) ([]RerankResult, error) {
	var resultsJSON string
	querySQL := `SELECT neurondb_rerank_cross_encoder($1, $2::text[], $3, $4) AS reranked`

	err := c.db.GetContext(ctx, &resultsJSON, querySQL, query, documents, model, topK)
	if err != nil {
		return nil, fmt.Errorf("cross-encoder reranking failed via NeuronDB: query_length=%d, documents_count=%d, model='%s', top_k=%d, function='neurondb_rerank_cross_encoder', error=%w",
			len(query), len(documents), model, topK, err)
	}

	var results []RerankResult
	if err := json.Unmarshal([]byte(resultsJSON), &results); err != nil {
		return nil, fmt.Errorf("cross-encoder reranking result parsing failed: query_length=%d, documents_count=%d, results_json_length=%d, error=%w",
			len(query), len(documents), len(resultsJSON), err)
	}

	return results, nil
}

/* RerankLLM reranks using an LLM */
func (c *RerankingClient) RerankLLM(ctx context.Context, query string, documents []string, model string, topK int, params map[string]interface{}) ([]RerankResult, error) {
	paramsJSON, err := json.Marshal(params)
	if err != nil {
		return nil, fmt.Errorf("LLM reranking failed: query_length=%d, documents_count=%d, model='%s', parameter_marshaling_error=true, error=%w",
			len(query), len(documents), model, err)
	}

	var resultsJSON string
	querySQL := `SELECT neurondb_rerank_llm($1, $2::text[], $3, $4, $5::jsonb) AS reranked`

	err = c.db.GetContext(ctx, &resultsJSON, querySQL, query, documents, model, topK, paramsJSON)
	if err != nil {
		return nil, fmt.Errorf("LLM reranking failed via NeuronDB: query_length=%d, documents_count=%d, model='%s', top_k=%d, function='neurondb_rerank_llm', error=%w",
			len(query), len(documents), model, topK, err)
	}

	var results []RerankResult
	if err := json.Unmarshal([]byte(resultsJSON), &results); err != nil {
		return nil, fmt.Errorf("LLM reranking result parsing failed: query_length=%d, documents_count=%d, results_json_length=%d, error=%w",
			len(query), len(documents), len(resultsJSON), err)
	}

	return results, nil
}

/* RerankColBERT reranks using ColBERT model */
func (c *RerankingClient) RerankColBERT(ctx context.Context, query string, documents []string, model string, topK int) ([]RerankResult, error) {
	var resultsJSON string
	querySQL := `SELECT neurondb_rerank_colbert($1, $2::text[], $3, $4) AS reranked`

	err := c.db.GetContext(ctx, &resultsJSON, querySQL, query, documents, model, topK)
	if err != nil {
		return nil, fmt.Errorf("ColBERT reranking failed via NeuronDB: query_length=%d, documents_count=%d, model='%s', top_k=%d, function='neurondb_rerank_colbert', error=%w",
			len(query), len(documents), model, topK, err)
	}

	var results []RerankResult
	if err := json.Unmarshal([]byte(resultsJSON), &results); err != nil {
		return nil, fmt.Errorf("ColBERT reranking result parsing failed: query_length=%d, documents_count=%d, results_json_length=%d, error=%w",
			len(query), len(documents), len(resultsJSON), err)
	}

	return results, nil
}

/* RerankEnsemble reranks using ensemble of multiple methods */
func (c *RerankingClient) RerankEnsemble(ctx context.Context, query string, documents []string, methods []string, weights []float64, topK int) ([]RerankResult, error) {
	methodsJSON, err := json.Marshal(methods)
	if err != nil {
		return nil, fmt.Errorf("ensemble reranking failed: query_length=%d, documents_count=%d, methods_count=%d, parameter_marshaling_error=true, error=%w",
			len(query), len(documents), len(methods), err)
	}

	weightsJSON, err := json.Marshal(weights)
	if err != nil {
		return nil, fmt.Errorf("ensemble reranking failed: query_length=%d, documents_count=%d, weights_count=%d, parameter_marshaling_error=true, error=%w",
			len(query), len(documents), len(weights), err)
	}

	var resultsJSON string
	querySQL := `SELECT neurondb_rerank_ensemble($1, $2::text[], $3::jsonb, $4::jsonb, $5) AS reranked`

	err = c.db.GetContext(ctx, &resultsJSON, querySQL, query, documents, methodsJSON, weightsJSON, topK)
	if err != nil {
		return nil, fmt.Errorf("ensemble reranking failed via NeuronDB: query_length=%d, documents_count=%d, methods_count=%d, top_k=%d, function='neurondb_rerank_ensemble', error=%w",
			len(query), len(documents), len(methods), topK, err)
	}

	var results []RerankResult
	if err := json.Unmarshal([]byte(resultsJSON), &results); err != nil {
		return nil, fmt.Errorf("ensemble reranking result parsing failed: query_length=%d, documents_count=%d, results_json_length=%d, error=%w",
			len(query), len(documents), len(resultsJSON), err)
	}

	return results, nil
}
