/*-------------------------------------------------------------------------
 *
 * vector_advanced_operations.go
 *    Advanced vector operations tools for NeuronMCP
 *
 * Implements advanced vector operations as specified in Phase 1.2
 * of the roadmap: aggregation, normalization, similarity matrices, etc.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/vector_advanced_operations.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

/* VectorAggregateTool performs vector aggregation operations */
type VectorAggregateTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewVectorAggregateTool creates a new vector aggregate tool */
func NewVectorAggregateTool(db *database.Database, logger *logging.Logger) *VectorAggregateTool {
	return &VectorAggregateTool{
		BaseTool: NewBaseTool(
			"vector_aggregate",
			"Perform vector aggregation operations (mean, max, min, sum) on vector columns",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table": map[string]interface{}{
						"type":        "string",
						"description": "Table name containing vectors",
					},
					"vector_column": map[string]interface{}{
						"type":        "string",
						"description": "Name of the vector column",
					},
					"operation": map[string]interface{}{
						"type":        "string",
						"enum":        []string{"mean", "max", "min", "sum"},
						"description": "Aggregation operation to perform",
					},
					"filter": map[string]interface{}{
						"type":        "string",
						"description": "Optional WHERE clause filter (SQL condition)",
					},
				},
				"required": []interface{}{"table", "vector_column", "operation"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute performs vector aggregation */
func (t *VectorAggregateTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	table, ok := params["table"].(string)
	if !ok || table == "" {
		return Error("table parameter is required", "INVALID_PARAMETER", nil), nil
	}

	vectorColumn, ok := params["vector_column"].(string)
	if !ok || vectorColumn == "" {
		return Error("vector_column parameter is required", "INVALID_PARAMETER", nil), nil
	}

	operation, ok := params["operation"].(string)
	if !ok || operation == "" {
		return Error("operation parameter is required", "INVALID_PARAMETER", nil), nil
	}

	filter, _ := params["filter"].(string)

	/* Build aggregation query */
	var aggFunc string
	switch operation {
	case "mean":
		aggFunc = "vector_mean"
	case "max":
		aggFunc = "vector_max"
	case "min":
		aggFunc = "vector_min"
	case "sum":
		aggFunc = "vector_sum"
	default:
		return Error(fmt.Sprintf("Invalid operation: %s", operation), "INVALID_PARAMETER", nil), nil
	}

	whereClause := ""
	if filter != "" {
		whereClause = "WHERE " + filter
	}

	query := fmt.Sprintf(`
		SELECT %s(%s) AS aggregated_vector
		FROM %s
		%s
	`, aggFunc, vectorColumn, table, whereClause)

	result, err := t.executor.ExecuteQueryOne(ctx, query, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("Vector aggregation failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	return Success(result, map[string]interface{}{
		"tool": "vector_aggregate",
	}), nil
}

/* VectorNormalizeBatchTool normalizes vectors in batch */
type VectorNormalizeBatchTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewVectorNormalizeBatchTool creates a new vector normalize batch tool */
func NewVectorNormalizeBatchTool(db *database.Database, logger *logging.Logger) *VectorNormalizeBatchTool {
	return &VectorNormalizeBatchTool{
		BaseTool: NewBaseTool(
			"vector_normalize_batch",
			"Normalize vectors in a table column in batch",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table": map[string]interface{}{
						"type":        "string",
						"description": "Table name containing vectors",
					},
					"vector_column": map[string]interface{}{
						"type":        "string",
						"description": "Name of the vector column to normalize",
					},
					"target_column": map[string]interface{}{
						"type":        "string",
						"description": "Target column name for normalized vectors (optional, updates source if not provided)",
					},
					"in_place": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Update vectors in place (requires target_column to be same as vector_column)",
					},
				},
				"required": []interface{}{"table", "vector_column"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute normalizes vectors in batch */
func (t *VectorNormalizeBatchTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	table, ok := params["table"].(string)
	if !ok || table == "" {
		return Error("table parameter is required", "INVALID_PARAMETER", nil), nil
	}

	vectorColumn, ok := params["vector_column"].(string)
	if !ok || vectorColumn == "" {
		return Error("vector_column parameter is required", "INVALID_PARAMETER", nil), nil
	}

	targetColumn, _ := params["target_column"].(string)
	inPlace := false
	if val, ok := params["in_place"].(bool); ok {
		inPlace = val
	}

	if inPlace {
		if targetColumn == "" {
			targetColumn = vectorColumn
		}
		if targetColumn != vectorColumn {
			return Error("in_place requires target_column to be same as vector_column", "INVALID_PARAMETER", nil), nil
		}

		/* Update in place */
		query := fmt.Sprintf(`
			UPDATE %s
			SET %s = vector_normalize(%s)
			WHERE %s IS NOT NULL
		`, table, vectorColumn, vectorColumn, vectorColumn)

		_, err := t.executor.ExecuteQuery(ctx, query, nil)
		if err != nil {
			return Error(
				fmt.Sprintf("Batch normalization failed: %v", err),
				"QUERY_ERROR",
				map[string]interface{}{"error": err.Error()},
			), nil
		}

		/* Get count of updated rows */
		countQuery := fmt.Sprintf("SELECT COUNT(*) AS count FROM %s WHERE %s IS NOT NULL", table, vectorColumn)
		countResult, _ := t.executor.ExecuteQueryOne(ctx, countQuery, nil)

		return Success(map[string]interface{}{
			"operation":    "normalize_batch",
			"table":        table,
			"vector_column": vectorColumn,
			"in_place":     true,
			"rows_updated": countResult,
		}, map[string]interface{}{
			"tool": "vector_normalize_batch",
		}), nil
	}

	/* Create new column with normalized vectors */
	if targetColumn == "" {
		targetColumn = vectorColumn + "_normalized"
	}

	/* Check if target column exists, if not create it */
	checkQuery := fmt.Sprintf(`
		SELECT column_name 
		FROM information_schema.columns 
		WHERE table_name = '%s' AND column_name = '%s'
	`, table, targetColumn)
	checkResult, _ := t.executor.ExecuteQueryOne(ctx, checkQuery, nil)

	if checkResult == nil || len(checkResult) == 0 {
		/* Get vector dimension from first row */
		dimQuery := fmt.Sprintf(`
			SELECT vector_dims(%s) AS dims 
			FROM %s 
			WHERE %s IS NOT NULL 
			LIMIT 1
		`, vectorColumn, table, vectorColumn)
		dimResult, err := t.executor.ExecuteQueryOne(ctx, dimQuery, nil)
		if err != nil {
			return Error("Failed to determine vector dimensions", "QUERY_ERROR", map[string]interface{}{"error": err.Error()}), nil
		}

		dims := 0
		if dimVal, ok := dimResult["dims"].(int64); ok {
			dims = int(dimVal)
		} else if dimVal, ok := dimResult["dims"].(float64); ok {
			dims = int(dimVal)
		}

		if dims == 0 {
			return Error("Could not determine vector dimensions", "QUERY_ERROR", nil), nil
		}

		/* Add column */
		addColQuery := fmt.Sprintf("ALTER TABLE %s ADD COLUMN %s vector(%d)", table, targetColumn, dims)
		_, err = t.executor.ExecuteQuery(ctx, addColQuery, nil)
		if err != nil {
			return Error(
				fmt.Sprintf("Failed to add target column: %v", err),
				"QUERY_ERROR",
				map[string]interface{}{"error": err.Error()},
			), nil
		}
	}

	/* Update with normalized vectors */
	updateQuery := fmt.Sprintf(`
		UPDATE %s
		SET %s = vector_normalize(%s)
		WHERE %s IS NOT NULL
	`, table, targetColumn, vectorColumn, vectorColumn)

	_, err := t.executor.ExecuteQuery(ctx, updateQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("Batch normalization failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	countQuery := fmt.Sprintf("SELECT COUNT(*) AS count FROM %s WHERE %s IS NOT NULL", table, targetColumn)
	countResult, _ := t.executor.ExecuteQueryOne(ctx, countQuery, nil)

	return Success(map[string]interface{}{
		"operation":     "normalize_batch",
		"table":         table,
		"vector_column": vectorColumn,
		"target_column": targetColumn,
		"in_place":      false,
		"rows_updated":  countResult,
	}, map[string]interface{}{
		"tool": "vector_normalize_batch",
	}), nil
}

/* VectorSimilarityMatrixTool computes similarity matrices */
type VectorSimilarityMatrixTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewVectorSimilarityMatrixTool creates a new vector similarity matrix tool */
func NewVectorSimilarityMatrixTool(db *database.Database, logger *logging.Logger) *VectorSimilarityMatrixTool {
	return &VectorSimilarityMatrixTool{
		BaseTool: NewBaseTool(
			"vector_similarity_matrix",
			"Compute similarity matrix between vectors in a table",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table": map[string]interface{}{
						"type":        "string",
						"description": "Table name containing vectors",
					},
					"vector_column": map[string]interface{}{
						"type":        "string",
						"description": "Name of the vector column",
					},
					"id_column": map[string]interface{}{
						"type":        "string",
						"description": "ID column for identifying vectors",
					},
					"distance_metric": map[string]interface{}{
						"type":        "string",
						"enum":        []string{"l2", "cosine", "inner_product"},
						"default":     "cosine",
						"description": "Distance metric to use",
					},
					"limit": map[string]interface{}{
						"type":        "number",
						"default":     100,
						"minimum":     1,
						"maximum":     1000,
						"description": "Maximum number of vectors to compare",
					},
				},
				"required": []interface{}{"table", "vector_column", "id_column"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute computes similarity matrix */
func (t *VectorSimilarityMatrixTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	table, ok := params["table"].(string)
	if !ok || table == "" {
		return Error("table parameter is required", "INVALID_PARAMETER", nil), nil
	}

	vectorColumn, ok := params["vector_column"].(string)
	if !ok || vectorColumn == "" {
		return Error("vector_column parameter is required", "INVALID_PARAMETER", nil), nil
	}

	idColumn, ok := params["id_column"].(string)
	if !ok || idColumn == "" {
		return Error("id_column parameter is required", "INVALID_PARAMETER", nil), nil
	}

	distanceMetric := "cosine"
	if val, ok := params["distance_metric"].(string); ok {
		distanceMetric = val
	}

	limit := 100
	if val, ok := params["limit"].(float64); ok {
		limit = int(val)
		if limit < 1 {
			limit = 1
		}
		if limit > 1000 {
			limit = 1000
		}
	}

	/* Build similarity matrix query */
	var distanceOp string
	switch distanceMetric {
	case "l2":
		distanceOp = "<->"
	case "cosine":
		distanceOp = "<=>"
	case "inner_product":
		distanceOp = "<#>"
	default:
		return Error(fmt.Sprintf("Invalid distance metric: %s", distanceMetric), "INVALID_PARAMETER", nil), nil
	}

	query := fmt.Sprintf(`
		SELECT 
			a.%s AS id1,
			b.%s AS id2,
			(a.%s %s b.%s) AS distance
		FROM %s a
		CROSS JOIN %s b
		WHERE a.%s < b.%s
		ORDER BY distance
		LIMIT %d
	`, idColumn, idColumn, vectorColumn, distanceOp, vectorColumn, table, table, idColumn, idColumn, limit*limit)

	results, err := t.executor.ExecuteQuery(ctx, query, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("Similarity matrix computation failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	return Success(map[string]interface{}{
		"similarity_matrix": results,
		"count":            len(results),
		"distance_metric":  distanceMetric,
	}, map[string]interface{}{
		"tool": "vector_similarity_matrix",
	}), nil
}

/* VectorBatchDistanceTool computes batch distances */
type VectorBatchDistanceTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewVectorBatchDistanceTool creates a new vector batch distance tool */
func NewVectorBatchDistanceTool(db *database.Database, logger *logging.Logger) *VectorBatchDistanceTool {
	return &VectorBatchDistanceTool{
		BaseTool: NewBaseTool(
			"vector_batch_distance",
			"Compute distances between a query vector and multiple vectors in batch",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table": map[string]interface{}{
						"type":        "string",
						"description": "Table name containing vectors",
					},
					"vector_column": map[string]interface{}{
						"type":        "string",
						"description": "Name of the vector column",
					},
					"query_vector": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "number"},
						"description": "Query vector",
					},
					"distance_metric": map[string]interface{}{
						"type":        "string",
						"enum":        []string{"l2", "cosine", "inner_product", "l1", "hamming"},
						"default":     "l2",
						"description": "Distance metric to use",
					},
					"limit": map[string]interface{}{
						"type":        "number",
						"default":     100,
						"minimum":     1,
						"maximum":     10000,
						"description": "Maximum number of results",
					},
				},
				"required": []interface{}{"table", "vector_column", "query_vector"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute computes batch distances */
func (t *VectorBatchDistanceTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	table, ok := params["table"].(string)
	if !ok || table == "" {
		return Error("table parameter is required", "INVALID_PARAMETER", nil), nil
	}

	vectorColumn, ok := params["vector_column"].(string)
	if !ok || vectorColumn == "" {
		return Error("vector_column parameter is required", "INVALID_PARAMETER", nil), nil
	}

	queryVector, ok := params["query_vector"].([]interface{})
	if !ok || len(queryVector) == 0 {
		return Error("query_vector parameter is required and must be a non-empty array", "INVALID_PARAMETER", nil), nil
	}

	distanceMetric := "l2"
	if val, ok := params["distance_metric"].(string); ok {
		distanceMetric = val
	}

	limit := 100
	if val, ok := params["limit"].(float64); ok {
		limit = int(val)
		if limit < 1 {
			limit = 1
		}
		if limit > 10000 {
			limit = 10000
		}
	}

	/* Convert query vector to PostgreSQL array format */
	vecStr := "["
	for i, v := range queryVector {
		if i > 0 {
			vecStr += ","
		}
		if f, ok := v.(float64); ok {
			vecStr += fmt.Sprintf("%.6f", f)
		} else {
			vecStr += fmt.Sprintf("%v", v)
		}
	}
	vecStr += "]"

	/* Build distance query */
	var distanceOp string
	switch distanceMetric {
	case "l2":
		distanceOp = "<->"
	case "cosine":
		distanceOp = "<=>"
	case "inner_product":
		distanceOp = "<#>"
	case "l1":
		distanceOp = "<+>"
	case "hamming":
		distanceOp = "<%>"
	default:
		return Error(fmt.Sprintf("Invalid distance metric: %s", distanceMetric), "INVALID_PARAMETER", nil), nil
	}

	query := fmt.Sprintf(`
		SELECT 
			*,
			(%s %s '%s'::vector) AS distance
		FROM %s
		ORDER BY distance
		LIMIT %d
	`, vectorColumn, distanceOp, vecStr, table, limit)

	results, err := t.executor.ExecuteQuery(ctx, query, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("Batch distance computation failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	return Success(map[string]interface{}{
		"results":        results,
		"count":          len(results),
		"distance_metric": distanceMetric,
	}, map[string]interface{}{
		"tool": "vector_batch_distance",
	}), nil
}

/* VectorIndexStatisticsTool provides detailed index statistics */
type VectorIndexStatisticsTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewVectorIndexStatisticsTool creates a new vector index statistics tool */
func NewVectorIndexStatisticsTool(db *database.Database, logger *logging.Logger) *VectorIndexStatisticsTool {
	return &VectorIndexStatisticsTool{
		BaseTool: NewBaseTool(
			"vector_index_statistics",
			"Get detailed statistics for vector indexes (HNSW, IVF)",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table": map[string]interface{}{
						"type":        "string",
						"description": "Table name (optional)",
					},
					"index_name": map[string]interface{}{
						"type":        "string",
						"description": "Index name (optional)",
					},
				},
				"required": []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute retrieves index statistics */
func (t *VectorIndexStatisticsTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	table, _ := params["table"].(string)
	indexName, _ := params["index_name"].(string)

	var query string
	var queryParams []interface{}

	if indexName != "" {
		query = `
			SELECT 
				i.schemaname,
				i.tablename,
				i.indexname,
				i.indexdef,
				pg_relation_size(i.schemaname||'.'||i.indexname) AS index_size_bytes,
				pg_size_pretty(pg_relation_size(i.schemaname||'.'||i.indexname)) AS index_size_pretty,
				COALESCE(s.idx_scan, 0) AS index_scans,
				COALESCE(s.idx_tup_read, 0) AS tuples_read,
				COALESCE(s.idx_tup_fetch, 0) AS tuples_fetched
			FROM pg_indexes i
			LEFT JOIN pg_stat_user_indexes s ON s.schemaname = i.schemaname AND s.indexrelname = i.indexname
			WHERE i.indexname = $1
		`
		queryParams = []interface{}{indexName}
	} else if table != "" {
		query = `
			SELECT 
				i.schemaname,
				i.tablename,
				i.indexname,
				i.indexdef,
				pg_relation_size(i.schemaname||'.'||i.indexname) AS index_size_bytes,
				pg_size_pretty(pg_relation_size(i.schemaname||'.'||i.indexname)) AS index_size_pretty,
				COALESCE(s.idx_scan, 0) AS index_scans,
				COALESCE(s.idx_tup_read, 0) AS tuples_read,
				COALESCE(s.idx_tup_fetch, 0) AS tuples_fetched
			FROM pg_indexes i
			LEFT JOIN pg_stat_user_indexes s ON s.schemaname = i.schemaname AND s.indexrelname = i.indexname
			WHERE i.tablename = $1 AND (i.indexdef LIKE '%hnsw%' OR i.indexdef LIKE '%ivf%' OR i.indexdef LIKE '%vector%')
		`
		queryParams = []interface{}{table}
	} else {
		query = `
			SELECT 
				i.schemaname,
				i.tablename,
				i.indexname,
				i.indexdef,
				pg_relation_size(i.schemaname||'.'||i.indexname) AS index_size_bytes,
				pg_size_pretty(pg_relation_size(i.schemaname||'.'||i.indexname)) AS index_size_pretty,
				COALESCE(s.idx_scan, 0) AS index_scans,
				COALESCE(s.idx_tup_read, 0) AS tuples_read,
				COALESCE(s.idx_tup_fetch, 0) AS tuples_fetched
			FROM pg_indexes i
			LEFT JOIN pg_stat_user_indexes s ON s.schemaname = i.schemaname AND s.indexrelname = i.indexname
			WHERE i.indexdef LIKE '%hnsw%' OR i.indexdef LIKE '%ivf%' OR i.indexdef LIKE '%vector%'
		`
		queryParams = []interface{}{}
	}

	results, err := t.executor.ExecuteQuery(ctx, query, queryParams)
	if err != nil {
		return Error(
			fmt.Sprintf("Index statistics query failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	return Success(map[string]interface{}{
		"index_statistics": results,
		"count":            len(results),
	}, map[string]interface{}{
		"tool": "vector_index_statistics",
	}), nil
}



