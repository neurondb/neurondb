/*-------------------------------------------------------------------------
 *
 * vector_additional.go
 *    Tool implementation for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/vector_additional.go
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

/* VectorSimilarityTool computes similarity between two vectors */
type VectorSimilarityTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewVectorSimilarityTool creates a new vector similarity tool */
func NewVectorSimilarityTool(db *database.Database, logger *logging.Logger) *VectorSimilarityTool {
	return &VectorSimilarityTool{
		BaseTool: NewBaseTool(
			"postgresql_vector_similarity",
			"Compute similarity between two vectors",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"vector1": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "number"},
						"description": "First vector",
					},
					"vector2": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "number"},
						"description": "Second vector",
					},
					"distance_metric": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"l2", "cosine", "inner_product", "l1"},
						"default":     "cosine",
						"description": "Distance metric to use",
					},
				},
				"required": []interface{}{"vector1", "vector2"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the vector similarity computation */
func (t *VectorSimilarityTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for neurondb_vector_similarity tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	vector1, _ := params["vector1"].([]interface{})
	vector2, _ := params["vector2"].([]interface{})
	distanceMetric := "cosine"
	if d, ok := params["distance_metric"].(string); ok {
		distanceMetric = d
	}

	if vector1 == nil || len(vector1) == 0 {
		return Error("vector1 parameter is required and cannot be empty array for neurondb_vector_similarity tool", "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "vector1",
			"vector1_dimension": 0,
			"params": params,
		}), nil
	}

	if vector2 == nil || len(vector2) == 0 {
		return Error(fmt.Sprintf("vector2 parameter is required and cannot be empty array for neurondb_vector_similarity tool: vector1_dimension=%d", len(vector1)), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "vector2",
			"vector1_dimension": len(vector1),
			"vector2_dimension": 0,
			"params": params,
		}), nil
	}

	if len(vector1) != len(vector2) {
		return Error(fmt.Sprintf("vector1 and vector2 must have the same dimension for neurondb_vector_similarity tool: vector1_dimension=%d, vector2_dimension=%d, distance_metric='%s'", len(vector1), len(vector2), distanceMetric), "VALIDATION_ERROR", map[string]interface{}{
			"vector1_dimension": len(vector1),
			"vector2_dimension": len(vector2),
			"distance_metric":   distanceMetric,
			"params":            params,
		}), nil
	}

	validMetrics := map[string]bool{"l2": true, "cosine": true, "inner_product": true, "l1": true}
	if !validMetrics[distanceMetric] {
		validList := []string{"l2", "cosine", "inner_product", "l1"}
		return Error(fmt.Sprintf("invalid distance_metric '%s' for neurondb_vector_similarity tool: vector1_dimension=%d, vector2_dimension=%d. Valid metrics are: %v", distanceMetric, len(vector1), len(vector2), validList), "VALIDATION_ERROR", map[string]interface{}{
			"distance_metric":   distanceMetric,
			"vector1_dimension": len(vector1),
			"vector2_dimension": len(vector2),
			"valid_metrics":     validList,
			"params":            params,
		}), nil
	}

	vec1Str := formatVectorFromInterface(vector1)
	vec2Str := formatVectorFromInterface(vector2)

	var query string
	switch distanceMetric {
	case "cosine":
		query = `SELECT $1::vector <=> $2::vector AS similarity`
	case "inner_product":
		query = `SELECT $1::vector <#> $2::vector AS similarity`
	case "l1":
		query = `SELECT vector_l1_distance($1::vector, $2::vector) AS similarity`
 	default: /* l2 */
		query = `SELECT $1::vector <-> $2::vector AS similarity`
	}

	result, err := t.executor.ExecuteQueryOne(ctx, query, []interface{}{vec1Str, vec2Str})
	if err != nil {
		t.logger.Error("Vector similarity computation failed", err, params)
		return Error(fmt.Sprintf("Vector similarity computation execution failed: vector1_dimension=%d, vector2_dimension=%d, distance_metric='%s', query='%s', error=%v", len(vector1), len(vector2), distanceMetric, query, err), "SIMILARITY_ERROR", map[string]interface{}{
			"vector1_dimension": len(vector1),
			"vector2_dimension": len(vector2),
			"distance_metric":   distanceMetric,
			"query":             query,
			"error":             err.Error(),
		}), nil
	}

	return Success(result, map[string]interface{}{
		"distance_metric": distanceMetric,
	}), nil
}

/* CreateVectorIndexTool creates a vector index (generic, uses HNSW by default) */
type CreateVectorIndexTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewCreateVectorIndexTool creates a new create vector index tool */
func NewCreateVectorIndexTool(db *database.Database, logger *logging.Logger) *CreateVectorIndexTool {
	return &CreateVectorIndexTool{
		BaseTool: NewBaseTool(
			"postgresql_create_vector_index",
			"Create a vector index (HNSW or IVF) for a vector column",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table": map[string]interface{}{
						"type":        "string",
						"description": "Table name",
					},
					"vector_column": map[string]interface{}{
						"type":        "string",
						"description": "Vector column name",
					},
					"index_name": map[string]interface{}{
						"type":        "string",
						"description": "Name for the index",
					},
					"index_type": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"hnsw", "ivf"},
						"default":     "hnsw",
						"description": "Index type",
					},
					"m": map[string]interface{}{
						"type":        "number",
						"default":     16,
						"description": "HNSW parameter M (for HNSW index)",
					},
					"ef_construction": map[string]interface{}{
						"type":        "number",
						"default":     200,
						"description": "HNSW parameter ef_construction (for HNSW index)",
					},
					"num_lists": map[string]interface{}{
						"type":        "number",
						"default":     100,
						"description": "Number of lists (for IVF index)",
					},
				},
				"required": []interface{}{"table", "vector_column", "index_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the vector index creation */
func (t *CreateVectorIndexTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for neurondb_create_vector_index tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	indexType := "hnsw"
	if it, ok := params["index_type"].(string); ok {
		indexType = it
	}

	table, _ := params["table"].(string)
	vectorColumn, _ := params["vector_column"].(string)
	indexName, _ := params["index_name"].(string)

	if table == "" {
		return Error(fmt.Sprintf("table parameter is required and cannot be empty for neurondb_create_vector_index tool: index_type='%s'", indexType), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":  "table",
			"index_type": indexType,
			"params":     params,
		}), nil
	}

	if vectorColumn == "" {
		return Error(fmt.Sprintf("vector_column parameter is required and cannot be empty for neurondb_create_vector_index tool: index_type='%s', table='%s'", indexType, table), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":  "vector_column",
			"index_type": indexType,
			"table":      table,
			"params":     params,
		}), nil
	}

	if indexName == "" {
		return Error(fmt.Sprintf("index_name parameter is required and cannot be empty for neurondb_create_vector_index tool: index_type='%s', table='%s', vector_column='%s'", indexType, table, vectorColumn), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":     "index_name",
			"index_type":    indexType,
			"table":         table,
			"vector_column": vectorColumn,
			"params":        params,
		}), nil
	}

	validTypes := map[string]bool{"hnsw": true, "ivf": true}
	if !validTypes[indexType] {
		return Error(fmt.Sprintf("invalid index_type '%s' for neurondb_create_vector_index tool: table='%s', vector_column='%s', index_name='%s'. Valid types are: hnsw, ivf", indexType, table, vectorColumn, indexName), "VALIDATION_ERROR", map[string]interface{}{
			"index_type":    indexType,
			"table":         table,
			"vector_column": vectorColumn,
			"index_name":    indexName,
			"valid_types":   []string{"hnsw", "ivf"},
			"params":        params,
		}), nil
	}

	var result map[string]interface{}
	var err error

	if indexType == "ivf" {
		numLists := 100
		if n, ok := params["num_lists"].(float64); ok {
			numLists = int(n)
		}
		query := `SELECT ivf_create_index($1, $2, $3, $4) AS result`
		result, err = t.executor.ExecuteQueryOne(ctx, query, []interface{}{
			table, vectorColumn, indexName, numLists,
		})
	} else {
   /* Default to HNSW */
		m := 16
		if mVal, ok := params["m"].(float64); ok {
			m = int(mVal)
		}
		efConstruction := 200
		if ef, ok := params["ef_construction"].(float64); ok {
			efConstruction = int(ef)
		}
		query := `SELECT hnsw_create_index($1, $2, $3, $4, $5) AS result`
		result, err = t.executor.ExecuteQueryOne(ctx, query, []interface{}{
			table, vectorColumn, indexName, m, efConstruction,
		})
	}

	if err != nil {
		t.logger.Error("Vector index creation failed", err, params)
		var indexParams string
		if indexType == "ivf" {
			numLists := 100
			if n, ok := params["num_lists"].(float64); ok {
				numLists = int(n)
			}
			indexParams = fmt.Sprintf("num_lists=%d", numLists)
		} else {
			m := 16
			efConstruction := 200
			if mVal, ok := params["m"].(float64); ok {
				m = int(mVal)
			}
			if ef, ok := params["ef_construction"].(float64); ok {
				efConstruction = int(ef)
			}
			indexParams = fmt.Sprintf("m=%d, ef_construction=%d", m, efConstruction)
		}
		return Error(fmt.Sprintf("Vector index creation execution failed: index_type='%s', table='%s', vector_column='%s', index_name='%s', %s, error=%v", indexType, table, vectorColumn, indexName, indexParams, err), "INDEX_ERROR", map[string]interface{}{
			"index_type":    indexType,
			"table":         table,
			"vector_column": vectorColumn,
			"index_name":    indexName,
			"error":         err.Error(),
		}), nil
	}

	return Success(result, map[string]interface{}{
		"index_name": indexName,
		"index_type": indexType,
	}), nil
}

