package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/neurondb/NeuronMCP/internal/database"
)

const (
	// Default query timeout for all database operations
	DefaultQueryTimeout = 60 * time.Second
	// Embedding query timeout (embeddings can take longer)
	EmbeddingQueryTimeout = 120 * time.Second
	// Vector search timeout
	VectorSearchTimeout = 30 * time.Second
)

// QueryExecutor executes database queries for tools
type QueryExecutor struct {
	db *database.Database
}

// NewQueryExecutor creates a new query executor
func NewQueryExecutor(db *database.Database) *QueryExecutor {
	return &QueryExecutor{db: db}
}

// ExecuteVectorSearch executes a vector search query
func (e *QueryExecutor) ExecuteVectorSearch(ctx context.Context, table, vectorColumn string, queryVector []interface{}, distanceMetric string, limit int, additionalColumns []interface{}) ([]map[string]interface{}, error) {
	if e.db == nil {
		return nil, fmt.Errorf("query executor database instance is nil: cannot execute vector search on table '%s', column '%s'", table, vectorColumn)
	}
	
	if !e.db.IsConnected() {
		return nil, fmt.Errorf("database connection not available: cannot execute vector search on table '%s', column '%s' (database connection pool is not initialized)", table, vectorColumn)
	}
	
	if table == "" {
		return nil, fmt.Errorf("table name is required for vector search: table parameter is empty")
	}
	
	if vectorColumn == "" {
		return nil, fmt.Errorf("vector column name is required for vector search: vector_column parameter is empty for table '%s'", table)
	}
	
	if len(queryVector) == 0 {
		return nil, fmt.Errorf("query vector cannot be empty: vector search on table '%s', column '%s' requires a non-empty query vector", table, vectorColumn)
	}
	
	// Convert queryVector to []float32
	vec := make([]float32, 0, len(queryVector))
	for i, v := range queryVector {
		if f, ok := v.(float64); ok {
			vec = append(vec, float32(f))
		} else if f, ok := v.(float32); ok {
			vec = append(vec, f)
		} else {
			return nil, fmt.Errorf("invalid vector element type at index %d: expected float64 or float32, got %T (value: %v) for vector search on table '%s', column '%s'", i, v, v, table, vectorColumn)
		}
	}

	// Convert additional columns to []string
	cols := make([]string, 0, len(additionalColumns))
	for i, col := range additionalColumns {
		if str, ok := col.(string); ok {
			if str == "" {
				return nil, fmt.Errorf("additional column at index %d is empty string for vector search on table '%s', column '%s'", i, table, vectorColumn)
			}
			cols = append(cols, str)
		} else {
			return nil, fmt.Errorf("additional column at index %d has invalid type: expected string, got %T (value: %v) for vector search on table '%s', column '%s'", i, col, col, table, vectorColumn)
		}
	}

	// Validate distance metric
	validMetrics := map[string]bool{"l2": true, "cosine": true, "inner_product": true, "l1": true, "hamming": true, "chebyshev": true, "minkowski": true}
	if !validMetrics[distanceMetric] {
		return nil, fmt.Errorf("invalid distance metric '%s' for vector search on table '%s', column '%s': valid metrics are l2, cosine, inner_product, l1, hamming, chebyshev, minkowski", distanceMetric, table, vectorColumn)
	}

	// Validate limit
	if limit <= 0 {
		return nil, fmt.Errorf("invalid limit %d for vector search on table '%s', column '%s': limit must be greater than 0", limit, table, vectorColumn)
	}
	if limit > 10000 {
		return nil, fmt.Errorf("limit %d exceeds maximum allowed value of 10000 for vector search on table '%s', column '%s'", limit, table, vectorColumn)
	}

	qb := &database.QueryBuilder{}
	query, params := qb.VectorSearch(table, vectorColumn, vec, distanceMetric, limit, cols, nil)

	// Create timeout context for vector search
	queryCtx, cancel := context.WithTimeout(ctx, VectorSearchTimeout)
	defer cancel()

	rows, err := e.db.Query(queryCtx, query, params...)
	if err != nil {
		if queryCtx.Err() != nil {
			return nil, fmt.Errorf("vector search timeout after %v: table='%s', vector_column='%s', distance_metric='%s', limit=%d, error=%w", VectorSearchTimeout, table, vectorColumn, distanceMetric, limit, queryCtx.Err())
		}
		return nil, fmt.Errorf("vector search execution failed: table='%s', vector_column='%s', distance_metric='%s', limit=%d, vector_dimension=%d, additional_columns=%v, error=%w", table, vectorColumn, distanceMetric, limit, len(vec), cols, err)
	}
	defer rows.Close()

	results, err := scanRowsToMaps(rows)
	if err != nil {
		return nil, fmt.Errorf("failed to scan vector search results: table='%s', vector_column='%s', distance_metric='%s', limit=%d, error=%w", table, vectorColumn, distanceMetric, limit, err)
	}

	return results, nil
}

// ExecuteQuery executes a query and returns all rows
func (e *QueryExecutor) ExecuteQuery(ctx context.Context, query string, params []interface{}) ([]map[string]interface{}, error) {
	if e.db == nil {
		return nil, fmt.Errorf("query executor database instance is nil: cannot execute query '%s' with %d parameters", query, len(params))
	}
	
	if !e.db.IsConnected() {
		return nil, fmt.Errorf("database connection not available: cannot execute query '%s' with %d parameters (database connection pool is not initialized)", query, len(params))
	}
	
	if query == "" {
		return nil, fmt.Errorf("query string is empty: cannot execute empty query")
	}
	
	// Create timeout context
	queryCtx, cancel := context.WithTimeout(ctx, DefaultQueryTimeout)
	defer cancel()
	
	rows, err := e.db.Query(queryCtx, query, params...)
	if err != nil {
		if queryCtx.Err() != nil {
			return nil, fmt.Errorf("query timeout after %v: query='%s', parameter_count=%d, error=%w", DefaultQueryTimeout, query, len(params), queryCtx.Err())
		}
		return nil, fmt.Errorf("query execution failed: query='%s', parameter_count=%d, parameters=%v, error=%w", query, len(params), params, err)
	}
	defer rows.Close()

	results, err := scanRowsToMaps(rows)
	if err != nil {
		return nil, fmt.Errorf("failed to scan query results: query='%s', parameter_count=%d, error=%w", query, len(params), err)
	}

	return results, nil
}

// ExecuteQueryOne executes a query and returns a single row
func (e *QueryExecutor) ExecuteQueryOne(ctx context.Context, query string, params []interface{}) (map[string]interface{}, error) {
	return e.ExecuteQueryOneWithTimeout(ctx, query, params, DefaultQueryTimeout)
}

// ExecuteQueryOneWithTimeout executes a query with a specific timeout
func (e *QueryExecutor) ExecuteQueryOneWithTimeout(ctx context.Context, query string, params []interface{}, timeout time.Duration) (map[string]interface{}, error) {
	if e.db == nil {
		return nil, fmt.Errorf("query executor database instance is nil: cannot execute single-row query '%s' with %d parameters", query, len(params))
	}
	
	if !e.db.IsConnected() {
		return nil, fmt.Errorf("database connection not available: cannot execute single-row query '%s' with %d parameters (database connection pool is not initialized)", query, len(params))
	}
	
	if query == "" {
		return nil, fmt.Errorf("query string is empty: cannot execute empty query for single row result")
	}
	
	// Create timeout context
	queryCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	
	rows, err := e.db.Query(queryCtx, query, params...)
	if err != nil {
		return nil, fmt.Errorf("single-row query execution failed: query='%s', parameter_count=%d, parameters=%v, error=%w", query, len(params), params, err)
	}
	defer rows.Close()

	if !rows.Next() {
		return nil, fmt.Errorf("no rows returned from single-row query: query='%s', parameter_count=%d, parameters=%v (expected exactly one row)", query, len(params), params)
	}

	result, err := scanRowToMap(rows)
	if err != nil {
		return nil, fmt.Errorf("failed to scan single row result: query='%s', parameter_count=%d, error=%w", query, len(params), err)
	}

	if rows.Next() {
		return nil, fmt.Errorf("multiple rows returned from single-row query: query='%s', parameter_count=%d, parameters=%v (expected exactly one row, got at least two)", query, len(params), params)
	}

	// Check if context was cancelled (timeout)
	if queryCtx.Err() != nil {
		return nil, fmt.Errorf("query timeout after %v: query='%s', parameter_count=%d, error=%w", timeout, query, len(params), queryCtx.Err())
	}

	return result, nil
}

// Exec executes a query without returning rows (for DDL statements)
func (e *QueryExecutor) Exec(ctx context.Context, query string, params []interface{}) error {
	if e.db == nil {
		return fmt.Errorf("query executor database instance is nil: cannot execute DDL query '%s' with %d parameters", query, len(params))
	}
	
	if !e.db.IsConnected() {
		return fmt.Errorf("database connection not available: cannot execute DDL query '%s' with %d parameters (database connection pool is not initialized)", query, len(params))
	}
	
	if query == "" {
		return fmt.Errorf("query string is empty: cannot execute empty DDL query")
	}
	
	_, err := e.db.Exec(ctx, query, params...)
	if err != nil {
		return fmt.Errorf("DDL query execution failed: query='%s', parameter_count=%d, parameters=%v, error=%w", query, len(params), params, err)
	}
	return nil
}

// scanRowsToMaps scans all rows into maps
func scanRowsToMaps(rows pgx.Rows) ([]map[string]interface{}, error) {
	var results []map[string]interface{}
	rowNum := 0

	for rows.Next() {
		rowNum++
		row, err := scanRowToMap(rows)
		if err != nil {
			fieldDescs := rows.FieldDescriptions()
			fieldNames := make([]string, len(fieldDescs))
			for i, desc := range fieldDescs {
				fieldNames[i] = string(desc.Name)
			}
			return nil, fmt.Errorf("failed to scan row %d: expected columns=%v, error=%w", rowNum, fieldNames, err)
		}
		results = append(results, row)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error while iterating rows: scanned %d rows successfully before error, error=%w", len(results), err)
	}

	return results, nil
}

// scanRowToMap scans a single row into a map
func scanRowToMap(rows pgx.Rows) (map[string]interface{}, error) {
	fieldDescriptions := rows.FieldDescriptions()
	if len(fieldDescriptions) == 0 {
		return nil, fmt.Errorf("row has no columns: cannot scan empty result set")
	}

	values := make([]interface{}, len(fieldDescriptions))
	valuePointers := make([]interface{}, len(fieldDescriptions))
	fieldNames := make([]string, len(fieldDescriptions))

	for i, desc := range fieldDescriptions {
		fieldNames[i] = string(desc.Name)
		valuePointers[i] = &values[i]
	}

	if err := rows.Scan(valuePointers...); err != nil {
		return nil, fmt.Errorf("failed to scan row values: columns=%v, error=%w", fieldNames, err)
	}

	result := make(map[string]interface{})
	for i, desc := range fieldDescriptions {
		val := values[i]
		// Handle byte arrays (JSON, text, etc.)
		if bytes, ok := val.([]byte); ok {
			// Try to parse as JSON
			var jsonVal interface{}
			if err := json.Unmarshal(bytes, &jsonVal); err == nil {
				val = jsonVal
			} else {
				val = string(bytes)
			}
		}
		result[string(desc.Name)] = val
	}

	return result, nil
}

