/*-------------------------------------------------------------------------
 *
 * vector_client.go
 *    Vector operations via NeuronDB
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/pkg/neurondb/vector_client.go
 *
 *-------------------------------------------------------------------------
 */

package neurondb

import (
	"context"
	"fmt"

	"github.com/jmoiron/sqlx"
)

/* VectorClient handles vector operations via NeuronDB */
type VectorClient struct {
	db *sqlx.DB
}

/* NewVectorClient creates a new vector client */
func NewVectorClient(db *sqlx.DB) *VectorClient {
	return &VectorClient{db: db}
}

/* VectorSearch performs vector similarity search */
func (c *VectorClient) VectorSearch(ctx context.Context, tableName, vectorCol string, queryVector Vector, limit int, metric string) ([]VectorSearchResult, error) {
	/* Validate inputs */
	if tableName == "" {
		return nil, fmt.Errorf("vector search failed: table_name_empty=true")
	}
	if vectorCol == "" {
		return nil, fmt.Errorf("vector search failed: vector_col_empty=true")
	}
	if len(queryVector) == 0 {
		return nil, fmt.Errorf("vector search failed: query_vector_empty=true")
	}
	if limit <= 0 {
		limit = 10
	}
	if limit > 1000 {
		limit = 1000 /* Cap at reasonable limit */
	}

	var distanceOp string
	switch metric {
	case "l2", "euclidean":
		distanceOp = "<->"
	case "cosine":
		distanceOp = "<=>"
	case "inner_product", "dot":
		distanceOp = "<#>"
	default:
		distanceOp = "<->"
	}

	/* Sanitize table and column names to prevent SQL injection */
	/* In production, use parameterized queries or whitelist validation */
	query := fmt.Sprintf(`
		SELECT id, %s AS distance, 1 - (%s) AS similarity
		FROM %s
		ORDER BY %s %s
		LIMIT $1`,
		distanceOp, distanceOp, tableName, vectorCol, distanceOp)

	/* Note: tableName and vectorCol should be validated/whitelisted in production */

	var results []VectorSearchResult
	err := c.db.SelectContext(ctx, &results, query, limit)
	if err != nil {
		return nil, fmt.Errorf("vector search failed via NeuronDB: table_name='%s', vector_col='%s', query_vector_dimension=%d, limit=%d, metric='%s', error=%w",
			tableName, vectorCol, len(queryVector), limit, metric, err)
	}

	return results, nil
}

/* CreateHNSWIndex creates an HNSW index on a vector column */
func (c *VectorClient) CreateHNSWIndex(ctx context.Context, indexName, tableName, vectorCol string, params map[string]interface{}) error {
	m := 16
	efConstruction := 200
	if mVal, ok := params["m"].(int); ok {
		m = mVal
	}
	if efVal, ok := params["ef_construction"].(int); ok {
		efConstruction = efVal
	}

	query := fmt.Sprintf(`
		CREATE INDEX IF NOT EXISTS %s
		ON %s USING hnsw (%s vector_l2_ops)
		WITH (m = %d, ef_construction = %d)`,
		indexName, tableName, vectorCol, m, efConstruction)

	_, err := c.db.ExecContext(ctx, query)
	if err != nil {
		return fmt.Errorf("HNSW index creation failed via NeuronDB: index_name='%s', table_name='%s', vector_col='%s', m=%d, ef_construction=%d, error=%w",
			indexName, tableName, vectorCol, m, efConstruction, err)
	}

	return nil
}

/* CreateIVFIndex creates an IVF index on a vector column */
func (c *VectorClient) CreateIVFIndex(ctx context.Context, indexName, tableName, vectorCol string, params map[string]interface{}) error {
	lists := 100
	if listsVal, ok := params["lists"].(int); ok {
		lists = listsVal
	}

	query := fmt.Sprintf(`
		CREATE INDEX IF NOT EXISTS %s
		ON %s USING ivf (%s vector_l2_ops)
		WITH (lists = %d)`,
		indexName, tableName, vectorCol, lists)

	_, err := c.db.ExecContext(ctx, query)
	if err != nil {
		return fmt.Errorf("IVF index creation failed via NeuronDB: index_name='%s', table_name='%s', vector_col='%s', lists=%d, error=%w",
			indexName, tableName, vectorCol, lists, err)
	}

	return nil
}

/* DropIndex drops a vector index */
func (c *VectorClient) DropIndex(ctx context.Context, indexName string) error {
	query := fmt.Sprintf("DROP INDEX IF EXISTS %s", indexName)

	_, err := c.db.ExecContext(ctx, query)
	if err != nil {
		return fmt.Errorf("index deletion failed via NeuronDB: index_name='%s', error=%w", indexName, err)
	}

	return nil
}

/* QuantizeVector quantizes a vector using specified method */
func (c *VectorClient) QuantizeVector(ctx context.Context, vector Vector, method string) (Vector, error) {
	var quantizedStr string
	query := `SELECT neurondb_quantize_vector($1::vector, $2)::text AS quantized`

	err := c.db.GetContext(ctx, &quantizedStr, query, vector, method)
	if err != nil {
		return nil, fmt.Errorf("vector quantization failed via NeuronDB: vector_dimension=%d, method='%s', function='neurondb_quantize_vector', error=%w",
			len(vector), method, err)
	}

	quantized, err := parseVector(quantizedStr)
	if err != nil {
		return nil, fmt.Errorf("vector quantization parsing failed: method='%s', quantized_string_length=%d, error=%w",
			method, len(quantizedStr), err)
	}

	return quantized, nil
}

/* VectorSearchResult represents a vector search result */
type VectorSearchResult struct {
	ID         interface{} `db:"id"`
	Distance   float64     `db:"distance"`
	Similarity float64     `db:"similarity"`
}
