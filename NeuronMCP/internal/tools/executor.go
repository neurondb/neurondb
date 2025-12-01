package tools

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/jackc/pgx/v5"
	"github.com/neurondb/NeuronMCP/internal/database"
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
	// Convert queryVector to []float32
	vec := make([]float32, 0, len(queryVector))
	for _, v := range queryVector {
		if f, ok := v.(float64); ok {
			vec = append(vec, float32(f))
		} else if f, ok := v.(float32); ok {
			vec = append(vec, f)
		} else {
			return nil, fmt.Errorf("invalid vector element type")
		}
	}

	// Convert additional columns to []string
	cols := make([]string, 0, len(additionalColumns))
	for _, col := range additionalColumns {
		if str, ok := col.(string); ok {
			cols = append(cols, str)
		}
	}

	qb := &database.QueryBuilder{}
	query, params := qb.VectorSearch(table, vectorColumn, vec, distanceMetric, limit, cols, nil)

	rows, err := e.db.Query(ctx, query, params...)
	if err != nil {
		return nil, fmt.Errorf("failed to execute vector search: %w", err)
	}
	defer rows.Close()

	return scanRowsToMaps(rows)
}

// ExecuteQuery executes a query and returns all rows
func (e *QueryExecutor) ExecuteQuery(ctx context.Context, query string, params []interface{}) ([]map[string]interface{}, error) {
	rows, err := e.db.Query(ctx, query, params...)
	if err != nil {
		return nil, fmt.Errorf("failed to execute query: %w", err)
	}
	defer rows.Close()

	return scanRowsToMaps(rows)
}

// ExecuteQueryOne executes a query and returns a single row
func (e *QueryExecutor) ExecuteQueryOne(ctx context.Context, query string, params []interface{}) (map[string]interface{}, error) {
	rows, err := e.db.Query(ctx, query, params...)
	if err != nil {
		return nil, fmt.Errorf("failed to execute query: %w", err)
	}
	defer rows.Close()

	if !rows.Next() {
		return nil, fmt.Errorf("no rows returned")
	}

	result, err := scanRowToMap(rows)
	if err != nil {
		return nil, err
	}

	if rows.Next() {
		return nil, fmt.Errorf("multiple rows returned")
	}

	return result, nil
}

// scanRowsToMaps scans all rows into maps
func scanRowsToMaps(rows pgx.Rows) ([]map[string]interface{}, error) {
	var results []map[string]interface{}

	for rows.Next() {
		row, err := scanRowToMap(rows)
		if err != nil {
			return nil, err
		}
		results = append(results, row)
	}

	return results, rows.Err()
}

// scanRowToMap scans a single row into a map
func scanRowToMap(rows pgx.Rows) (map[string]interface{}, error) {
	fieldDescriptions := rows.FieldDescriptions()
	values := make([]interface{}, len(fieldDescriptions))
	valuePointers := make([]interface{}, len(fieldDescriptions))

	for i := range values {
		valuePointers[i] = &values[i]
	}

	if err := rows.Scan(valuePointers...); err != nil {
		return nil, err
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

