/*-------------------------------------------------------------------------
 *
 * vector.go
 *    Vector search and embedding tools for NeuronMCP
 *
 * Provides tools for vector similarity search with multiple distance metrics
 * and text embedding generation.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/vector.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/validation"
)

/* VectorSearchTool performs vector similarity search */
type VectorSearchTool struct {
	*BaseTool
	executor     *QueryExecutor
	logger       *logging.Logger
	configHelper *database.ConfigHelper
}

/* NewVectorSearchTool creates a new vector search tool */
func NewVectorSearchTool(db *database.Database, logger *logging.Logger) *VectorSearchTool {
	tool := NewBaseTool(
		"postgresql_vector_search",
		"Perform vector similarity search using L2, cosine, inner product, L1, Hamming, Chebyshev, or Minkowski distance",
		map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"table": map[string]interface{}{
					"type":        "string",
					"description": "Table name containing vectors",
					"minLength":   1,
				},
				"vector_column": map[string]interface{}{
					"type":        "string",
					"description": "Name of the vector column",
					"minLength":   1,
				},
				"query_vector": map[string]interface{}{
					"type":        "array",
					"items":       map[string]interface{}{"type": "number"},
					"description": "Query vector for similarity search",
					"minItems":    1,
				},
				"limit": map[string]interface{}{
					"type":        "integer",
					"default":     10,
					"minimum":     1,
					"maximum":     1000,
					"description": "Maximum number of results",
				},
				"distance_metric": map[string]interface{}{
					"type":        "string",
					"enum":        []interface{}{"l2", "cosine", "inner_product", "l1", "hamming", "chebyshev", "minkowski"},
					"default":     "l2",
					"description": "Distance metric to use",
				},
				"additional_columns": map[string]interface{}{
					"type":        "array",
					"items":       map[string]interface{}{"type": "string"},
					"description": "Additional columns to return in results",
				},
			},
			"required": []interface{}{"table", "vector_column", "query_vector"},
			"additionalProperties": false,
		},
	)
	return &VectorSearchTool{
		BaseTool:     tool,
		executor:     NewQueryExecutor(db),
		logger:       logger,
		configHelper: database.NewConfigHelper(db),
	}
}

/* Execute executes the vector search */
func (t *VectorSearchTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for neurondb_vector_search tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	table, _ := params["table"].(string)
	vectorColumn, _ := params["vector_column"].(string)
	queryVector, _ := params["query_vector"].([]interface{})
	limit := 10
	if l, ok := params["limit"].(float64); ok {
		limit = int(l)
	}
	distanceMetric := "l2"
	if dm, ok := params["distance_metric"].(string); ok {
		distanceMetric = dm
	}
	additionalColumns := []interface{}{}
	if ac, ok := params["additional_columns"].([]interface{}); ok {
		additionalColumns = ac
	}

	/* Validate table name (SQL identifier) */
	if err := validation.ValidateSQLIdentifierRequired(table, "table"); err != nil {
		return Error(fmt.Sprintf("Invalid table parameter: %v", err), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "table",
			"error":     err.Error(),
			"params":    params,
		}), nil
	}

	/* Validate vector column name (SQL identifier) */
	if err := validation.ValidateSQLIdentifierRequired(vectorColumn, "vector_column"); err != nil {
		return Error(fmt.Sprintf("Invalid vector_column parameter: %v", err), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "vector_column",
			"table":     table,
			"error":     err.Error(),
			"params":    params,
		}), nil
	}

	/* Validate query vector */
	if err := validation.ValidateVectorRequired(queryVector, "query_vector", 1, 10000); err != nil {
		return Error(fmt.Sprintf("Invalid query_vector parameter: %v", err), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":    "query_vector",
			"table":        table,
			"vector_column": vectorColumn,
			"error":        err.Error(),
			"params":       params,
		}), nil
	}

	/* Validate limit */
	if err := validation.ValidateIntRange(limit, 1, 1000, "limit"); err != nil {
		return Error(fmt.Sprintf("Invalid limit parameter: %v", err), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "limit",
			"error":     err.Error(),
			"params":    params,
		}), nil
	}

	/* Validate distance metric */
	if err := validation.ValidateIn(distanceMetric, "distance_metric", "l2", "cosine", "inner_product", "l1", "hamming", "chebyshev", "minkowski"); err != nil {
		return Error(fmt.Sprintf("Invalid distance_metric parameter: %v", err), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "distance_metric",
			"error":     err.Error(),
			"params":    params,
		}), nil
	}

	/* Validate additional columns (if provided) */
	for i, col := range additionalColumns {
		colStr, ok := col.(string)
		if !ok {
			return Error(fmt.Sprintf("additional_columns[%d] must be a string", i), "VALIDATION_ERROR", map[string]interface{}{
				"parameter": "additional_columns",
				"index":      i,
				"params":     params,
			}), nil
		}
		if err := validation.ValidateSQLIdentifier(colStr, fmt.Sprintf("additional_columns[%d]", i)); err != nil {
			return Error(fmt.Sprintf("Invalid additional_columns[%d]: %v", i, err), "VALIDATION_ERROR", map[string]interface{}{
				"parameter": "additional_columns",
				"index":      i,
				"error":      err.Error(),
				"params":     params,
			}), nil
		}
	}

	results, err := t.executor.ExecuteVectorSearch(ctx, table, vectorColumn, queryVector, distanceMetric, limit, additionalColumns)
	if err != nil {
		t.logger.Error("Vector search failed", err, params)
		return Error(fmt.Sprintf("Vector search execution failed: table='%s', vector_column='%s', distance_metric='%s', limit=%d, query_vector_dimension=%d, additional_columns_count=%d, error=%v", table, vectorColumn, distanceMetric, limit, len(queryVector), len(additionalColumns), err), "SEARCH_ERROR", map[string]interface{}{
			"table":             table,
			"vector_column":     vectorColumn,
			"distance_metric":   distanceMetric,
			"limit":            limit,
			"query_vector_size": len(queryVector),
			"error":            err.Error(),
		}), nil
	}

	return Success(results, map[string]interface{}{
		"count":          len(results),
		"distance_metric": distanceMetric,
		"table":          table,
		"vector_column":  vectorColumn,
		"limit":         limit,
	}), nil
}

/* VectorSearchL2Tool performs L2 distance vector search */
type VectorSearchL2Tool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewVectorSearchL2Tool creates a new L2 vector search tool */
func NewVectorSearchL2Tool(db *database.Database, logger *logging.Logger) *VectorSearchL2Tool {
	return &VectorSearchL2Tool{
		BaseTool: NewBaseTool(
			"postgresql_vector_search_l2",
			"Perform vector similarity search using L2 (Euclidean) distance",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table":         map[string]interface{}{"type": "string"},
					"vector_column": map[string]interface{}{"type": "string"},
					"query_vector":  map[string]interface{}{"type": "array", "items": map[string]interface{}{"type": "number"}},
					"limit":         map[string]interface{}{"type": "number", "default": 10, "minimum": 1, "maximum": 1000},
				},
				"required": []interface{}{"table", "vector_column", "query_vector"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the L2 vector search */
func (t *VectorSearchL2Tool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error("Invalid parameters", "VALIDATION_ERROR", map[string]interface{}{"errors": errors}), nil
	}

	table, _ := params["table"].(string)
	vectorColumn, _ := params["vector_column"].(string)
	queryVector, _ := params["query_vector"].([]interface{})
	limit := 10
	if l, ok := params["limit"].(float64); ok {
		limit = int(l)
	}

	results, err := t.executor.ExecuteVectorSearch(ctx, table, vectorColumn, queryVector, "l2", limit, nil)
	if err != nil {
		t.logger.Error("L2 vector search failed", err, params)
		return Error(fmt.Sprintf("L2 vector search execution failed: table='%s', vector_column='%s', limit=%d, query_vector_dimension=%d, error=%v", table, vectorColumn, limit, len(queryVector), err), "SEARCH_ERROR", map[string]interface{}{
			"table":             table,
			"vector_column":     vectorColumn,
			"distance_metric":   "l2",
			"limit":            limit,
			"query_vector_size": len(queryVector),
			"error":            err.Error(),
		}), nil
	}

	return Success(results, map[string]interface{}{
		"count":          len(results),
		"distance_metric": "l2",
	}), nil
}

/* VectorSearchCosineTool performs cosine distance vector search */
type VectorSearchCosineTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewVectorSearchCosineTool creates a new cosine vector search tool */
func NewVectorSearchCosineTool(db *database.Database, logger *logging.Logger) *VectorSearchCosineTool {
	return &VectorSearchCosineTool{
		BaseTool: NewBaseTool(
			"postgresql_vector_search_cosine",
			"Perform vector similarity search using cosine distance",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table":         map[string]interface{}{"type": "string"},
					"vector_column": map[string]interface{}{"type": "string"},
					"query_vector":  map[string]interface{}{"type": "array", "items": map[string]interface{}{"type": "number"}},
					"limit":         map[string]interface{}{"type": "number", "default": 10, "minimum": 1, "maximum": 1000},
				},
				"required": []interface{}{"table", "vector_column", "query_vector"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the cosine vector search */
func (t *VectorSearchCosineTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error("Invalid parameters", "VALIDATION_ERROR", map[string]interface{}{"errors": errors}), nil
	}

	table, _ := params["table"].(string)
	vectorColumn, _ := params["vector_column"].(string)
	queryVector, _ := params["query_vector"].([]interface{})
	limit := 10
	if l, ok := params["limit"].(float64); ok {
		limit = int(l)
	}

	results, err := t.executor.ExecuteVectorSearch(ctx, table, vectorColumn, queryVector, "cosine", limit, nil)
	if err != nil {
		t.logger.Error("Cosine vector search failed", err, params)
		return Error(fmt.Sprintf("Cosine vector search execution failed: table='%s', vector_column='%s', limit=%d, query_vector_dimension=%d, error=%v", table, vectorColumn, limit, len(queryVector), err), "SEARCH_ERROR", map[string]interface{}{
			"table":             table,
			"vector_column":     vectorColumn,
			"distance_metric":   "cosine",
			"limit":            limit,
			"query_vector_size": len(queryVector),
			"error":            err.Error(),
		}), nil
	}

	return Success(results, map[string]interface{}{
		"count":          len(results),
		"distance_metric": "cosine",
	}), nil
}

/* VectorSearchInnerProductTool performs inner product distance vector search */
type VectorSearchInnerProductTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewVectorSearchInnerProductTool creates a new inner product vector search tool */
func NewVectorSearchInnerProductTool(db *database.Database, logger *logging.Logger) *VectorSearchInnerProductTool {
	return &VectorSearchInnerProductTool{
		BaseTool: NewBaseTool(
			"postgresql_vector_search_inner_product",
			"Perform vector similarity search using inner product distance",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table":         map[string]interface{}{"type": "string"},
					"vector_column": map[string]interface{}{"type": "string"},
					"query_vector":  map[string]interface{}{"type": "array", "items": map[string]interface{}{"type": "number"}},
					"limit":         map[string]interface{}{"type": "number", "default": 10, "minimum": 1, "maximum": 1000},
				},
				"required": []interface{}{"table", "vector_column", "query_vector"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the inner product vector search */
func (t *VectorSearchInnerProductTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error("Invalid parameters", "VALIDATION_ERROR", map[string]interface{}{"errors": errors}), nil
	}

	table, _ := params["table"].(string)
	vectorColumn, _ := params["vector_column"].(string)
	queryVector, _ := params["query_vector"].([]interface{})
	limit := 10
	if l, ok := params["limit"].(float64); ok {
		limit = int(l)
	}

	results, err := t.executor.ExecuteVectorSearch(ctx, table, vectorColumn, queryVector, "inner_product", limit, nil)
	if err != nil {
		t.logger.Error("Inner product vector search failed", err, params)
		return Error(fmt.Sprintf("Inner product vector search execution failed: table='%s', vector_column='%s', limit=%d, query_vector_dimension=%d, error=%v", table, vectorColumn, limit, len(queryVector), err), "SEARCH_ERROR", map[string]interface{}{
			"table":             table,
			"vector_column":     vectorColumn,
			"distance_metric":   "inner_product",
			"limit":            limit,
			"query_vector_size": len(queryVector),
			"error":            err.Error(),
		}), nil
	}

	return Success(results, map[string]interface{}{
		"count":          len(results),
		"distance_metric": "inner_product",
	}), nil
}

/* VectorSearchL1Tool performs L1 (Manhattan) distance vector search */
type VectorSearchL1Tool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewVectorSearchL1Tool creates a new L1 vector search tool */
func NewVectorSearchL1Tool(db *database.Database, logger *logging.Logger) *VectorSearchL1Tool {
	return &VectorSearchL1Tool{
		BaseTool: NewBaseTool(
			"postgresql_vector_search_l1",
			"Perform vector similarity search using L1 (Manhattan) distance",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table":         map[string]interface{}{"type": "string"},
					"vector_column": map[string]interface{}{"type": "string"},
					"query_vector":  map[string]interface{}{"type": "array", "items": map[string]interface{}{"type": "number"}},
					"limit":         map[string]interface{}{"type": "number", "default": 10, "minimum": 1, "maximum": 1000},
				},
				"required": []interface{}{"table", "vector_column", "query_vector"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the L1 vector search */
func (t *VectorSearchL1Tool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error("Invalid parameters", "VALIDATION_ERROR", map[string]interface{}{"errors": errors}), nil
	}

	table, _ := params["table"].(string)
	vectorColumn, _ := params["vector_column"].(string)
	queryVector, _ := params["query_vector"].([]interface{})
	limit := 10
	if l, ok := params["limit"].(float64); ok {
		limit = int(l)
	}

	results, err := t.executor.ExecuteVectorSearch(ctx, table, vectorColumn, queryVector, "l1", limit, nil)
	if err != nil {
		t.logger.Error("L1 vector search failed", err, params)
		return Error(fmt.Sprintf("L1 vector search execution failed: table='%s', vector_column='%s', limit=%d, query_vector_dimension=%d, error=%v", table, vectorColumn, limit, len(queryVector), err), "SEARCH_ERROR", map[string]interface{}{
			"table":             table,
			"vector_column":     vectorColumn,
			"distance_metric":   "l1",
			"limit":            limit,
			"query_vector_size": len(queryVector),
			"error":            err.Error(),
		}), nil
	}

	return Success(results, map[string]interface{}{
		"count":          len(results),
		"distance_metric": "l1",
	}), nil
}

/* VectorSearchHammingTool performs Hamming distance vector search */
type VectorSearchHammingTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewVectorSearchHammingTool creates a new Hamming vector search tool */
func NewVectorSearchHammingTool(db *database.Database, logger *logging.Logger) *VectorSearchHammingTool {
	return &VectorSearchHammingTool{
		BaseTool: NewBaseTool(
			"postgresql_vector_search_hamming",
			"Perform vector similarity search using Hamming distance",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table":         map[string]interface{}{"type": "string"},
					"vector_column": map[string]interface{}{"type": "string"},
					"query_vector":  map[string]interface{}{"type": "array", "items": map[string]interface{}{"type": "number"}},
					"limit":         map[string]interface{}{"type": "number", "default": 10, "minimum": 1, "maximum": 1000},
				},
				"required": []interface{}{"table", "vector_column", "query_vector"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the Hamming vector search */
func (t *VectorSearchHammingTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error("Invalid parameters", "VALIDATION_ERROR", map[string]interface{}{"errors": errors}), nil
	}

	table, _ := params["table"].(string)
	vectorColumn, _ := params["vector_column"].(string)
	queryVector, _ := params["query_vector"].([]interface{})
	limit := 10
	if l, ok := params["limit"].(float64); ok {
		limit = int(l)
	}

	results, err := t.executor.ExecuteVectorSearch(ctx, table, vectorColumn, queryVector, "hamming", limit, nil)
	if err != nil {
		t.logger.Error("Hamming vector search failed", err, params)
		return Error(fmt.Sprintf("Hamming vector search execution failed: table='%s', vector_column='%s', limit=%d, query_vector_dimension=%d, error=%v", table, vectorColumn, limit, len(queryVector), err), "SEARCH_ERROR", map[string]interface{}{
			"table":             table,
			"vector_column":     vectorColumn,
			"distance_metric":   "hamming",
			"limit":            limit,
			"query_vector_size": len(queryVector),
			"error":            err.Error(),
		}), nil
	}

	return Success(results, map[string]interface{}{
		"count":          len(results),
		"distance_metric": "hamming",
	}), nil
}

/* VectorSearchChebyshevTool performs Chebyshev distance vector search */
type VectorSearchChebyshevTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewVectorSearchChebyshevTool creates a new Chebyshev vector search tool */
func NewVectorSearchChebyshevTool(db *database.Database, logger *logging.Logger) *VectorSearchChebyshevTool {
	return &VectorSearchChebyshevTool{
		BaseTool: NewBaseTool(
			"postgresql_vector_search_chebyshev",
			"Perform vector similarity search using Chebyshev distance",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table":         map[string]interface{}{"type": "string"},
					"vector_column": map[string]interface{}{"type": "string"},
					"query_vector":  map[string]interface{}{"type": "array", "items": map[string]interface{}{"type": "number"}},
					"limit":         map[string]interface{}{"type": "number", "default": 10, "minimum": 1, "maximum": 1000},
				},
				"required": []interface{}{"table", "vector_column", "query_vector"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the Chebyshev vector search */
func (t *VectorSearchChebyshevTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error("Invalid parameters", "VALIDATION_ERROR", map[string]interface{}{"errors": errors}), nil
	}

	table, _ := params["table"].(string)
	vectorColumn, _ := params["vector_column"].(string)
	queryVector, _ := params["query_vector"].([]interface{})
	limit := 10
	if l, ok := params["limit"].(float64); ok {
		limit = int(l)
	}

	results, err := t.executor.ExecuteVectorSearch(ctx, table, vectorColumn, queryVector, "chebyshev", limit, nil)
	if err != nil {
		t.logger.Error("Chebyshev vector search failed", err, params)
		return Error(fmt.Sprintf("Chebyshev vector search execution failed: table='%s', vector_column='%s', limit=%d, query_vector_dimension=%d, error=%v", table, vectorColumn, limit, len(queryVector), err), "SEARCH_ERROR", map[string]interface{}{
			"table":             table,
			"vector_column":     vectorColumn,
			"distance_metric":   "chebyshev",
			"limit":            limit,
			"query_vector_size": len(queryVector),
			"error":            err.Error(),
		}), nil
	}

	return Success(results, map[string]interface{}{
		"count":          len(results),
		"distance_metric": "chebyshev",
	}), nil
}

/* VectorSearchMinkowskiTool performs Minkowski distance vector search */
type VectorSearchMinkowskiTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewVectorSearchMinkowskiTool creates a new Minkowski vector search tool */
func NewVectorSearchMinkowskiTool(db *database.Database, logger *logging.Logger) *VectorSearchMinkowskiTool {
	return &VectorSearchMinkowskiTool{
		BaseTool: NewBaseTool(
			"postgresql_vector_search_minkowski",
			"Perform vector similarity search using Minkowski distance with configurable p parameter",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table":         map[string]interface{}{"type": "string"},
					"vector_column": map[string]interface{}{"type": "string"},
					"query_vector":  map[string]interface{}{"type": "array", "items": map[string]interface{}{"type": "number"}},
					"p_value": map[string]interface{}{
						"type":        "number",
						"default":     2.0,
						"minimum":     1.0,
						"description": "Minkowski distance parameter p (p=1 is Manhattan, p=2 is Euclidean, p=∞ is Chebyshev)",
					},
					"limit":         map[string]interface{}{"type": "number", "default": 10, "minimum": 1, "maximum": 1000},
				},
				"required": []interface{}{"table", "vector_column", "query_vector"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the Minkowski vector search */
func (t *VectorSearchMinkowskiTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error("Invalid parameters", "VALIDATION_ERROR", map[string]interface{}{"errors": errors}), nil
	}

	table, _ := params["table"].(string)
	vectorColumn, _ := params["vector_column"].(string)
	queryVector, _ := params["query_vector"].([]interface{})
	pValue := 2.0
	if p, ok := params["p_value"].(float64); ok {
		pValue = p
	}
	limit := 10
	if l, ok := params["limit"].(float64); ok {
		limit = int(l)
	}

	/* Pass p_value for Minkowski distance */
	results, err := t.executor.ExecuteVectorSearchWithMinkowski(ctx, table, vectorColumn, queryVector, "minkowski", limit, nil, &pValue)
	if err != nil {
		t.logger.Error("Minkowski vector search failed", err, params)
		return Error(fmt.Sprintf("Minkowski vector search execution failed: table='%s', vector_column='%s', limit=%d, p_value=%.2f, query_vector_dimension=%d, error=%v", table, vectorColumn, limit, pValue, len(queryVector), err), "SEARCH_ERROR", map[string]interface{}{
			"table":             table,
			"vector_column":     vectorColumn,
			"distance_metric":   "minkowski",
			"p_value":          pValue,
			"limit":            limit,
			"query_vector_size": len(queryVector),
			"error":            err.Error(),
		}), nil
	}

	return Success(results, map[string]interface{}{
		"count":          len(results),
		"distance_metric": "minkowski",
		"p_value":       pValue,
	}), nil
}

/* GenerateEmbeddingTool generates text embeddings */
type GenerateEmbeddingTool struct {
	*BaseTool
	executor     *QueryExecutor
	logger       *logging.Logger
	configHelper *database.ConfigHelper
}

/* NewGenerateEmbeddingTool creates a new embedding generation tool */
func NewGenerateEmbeddingTool(db *database.Database, logger *logging.Logger) *GenerateEmbeddingTool {
	return &GenerateEmbeddingTool{
		BaseTool: NewBaseTool(
			"postgresql_generate_embedding",
			"Generate text embedding using configured model",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"text": map[string]interface{}{
						"type":        "string",
						"description": "Text to embed",
					},
					"model": map[string]interface{}{
						"type":        "string",
						"description": "Model name (optional, uses default if not specified)",
					},
				},
				"required": []interface{}{"text"},
			},
		),
		executor:     NewQueryExecutor(db),
		logger:       logger,
		configHelper: database.NewConfigHelper(db),
	}
}

/* Execute executes the embedding generation */
func (t *GenerateEmbeddingTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error("Invalid parameters", "VALIDATION_ERROR", map[string]interface{}{"errors": errors}), nil
	}

	text, _ := params["text"].(string)
	model, _ := params["model"].(string)
	
	textLen := len(text)
	if textLen == 0 {
		return Error("text parameter is required and cannot be empty for generate_embedding tool", "VALIDATION_ERROR", map[string]interface{}{
			"parameter":   "text",
			"text_length": 0,
			"params":      params,
		}), nil
	}

	/* Resolve model name - try to get from database config, fallback to provided or default */
	modelName := model
	if modelName == "" {
		/* Try to get default model for embedding operation */
		if defaultModel, err := t.configHelper.GetDefaultModel(ctx, "embedding"); err == nil {
			modelName = defaultModel
		} else {
			modelName = "default"
		}
	}

	var result interface{}
	var err error
	startTime := time.Now()
	
	query := "SELECT embed_text($1, $2)::text AS embedding"
	queryParams := []interface{}{text, modelName}
	
	result, err = t.executor.ExecuteQueryOneWithTimeout(ctx, query, queryParams, EmbeddingQueryTimeout)
	if err != nil {
		/* Check if error is about configuration (API key, etc.) - check first method's error */
		errStr := err.Error()
		if strings.Contains(errStr, "llm_api_key") || strings.Contains(errStr, "llm_provider") || strings.Contains(errStr, "Configure neurondb") || strings.Contains(errStr, "embedding generation failed") {
			t.logger.Error("Embedding generation failed - configuration issue", err, params)
			return Error(fmt.Sprintf("Embedding generation failed: text_length=%d, model='%s'. The embedding function requires proper configuration. Error: %v. Note: Configuration is managed via PostgreSQL GUC settings (neurondb.llm_api_key and neurondb.llm_provider).", textLen, modelName, err), "CONFIGURATION_ERROR", map[string]interface{}{
				"text_length": textLen,
				"model":       modelName,
				"error":       err.Error(),
				"methods_tried": []string{"embed_text"},
				"hint":        "Check PostgreSQL GUC settings: SHOW neurondb.llm_api_key; SHOW neurondb.llm_provider;",
			}), nil
		}
		
		t.logger.Warn("embed_text failed, trying neurondb.embed fallback", map[string]interface{}{
			"error": err.Error(),
			"model": modelName,
		})
		
		/* neurondb.embed(model text, input_text text, task text) - function signature: model, input_text, task */
		query = "SELECT neurondb.embed($1, $2, $3)::text AS embedding"
		queryParams = []interface{}{modelName, text, "embedding"}
		
		result, err = t.executor.ExecuteQueryOneWithTimeout(ctx, query, queryParams, EmbeddingQueryTimeout)
		if err != nil {
			/* Check if error is "no rows returned" - function exists but requires configuration */
			errStr = err.Error()
			if strings.Contains(errStr, "no rows returned") {
				t.logger.Error("Embedding generation failed - function returned no rows (configuration required)", err, params)
				return Error(fmt.Sprintf("Embedding generation requires configuration: text_length=%d, model='%s'. The embedding function exists but returned no results, which typically means embedding model configuration is required. Configuration is managed via PostgreSQL GUC settings.", textLen, modelName), "CONFIGURATION_ERROR", map[string]interface{}{
					"text_length": textLen,
					"model":       modelName,
					"error":       err.Error(),
					"methods_tried": []string{"embed_text", "neurondb.embed"},
					"hint":        "Check PostgreSQL GUC settings: SHOW neurondb.llm_api_key; SHOW neurondb.llm_provider;",
				}), nil
			}
			/* Check if error is about configuration (API key, etc.) */
			if strings.Contains(errStr, "llm_api_key") || strings.Contains(errStr, "llm_provider") || strings.Contains(errStr, "Configure neurondb") || strings.Contains(errStr, "embedding generation failed") {
				t.logger.Error("Embedding generation failed - configuration required", err, params)
				return Error(fmt.Sprintf("Embedding generation requires configuration: text_length=%d, model='%s'. Error: %v. Configuration is managed via PostgreSQL GUC settings (neurondb.llm_api_key and neurondb.llm_provider).", textLen, modelName, err), "CONFIGURATION_ERROR", map[string]interface{}{
					"text_length": textLen,
					"model":       modelName,
					"error":       err.Error(),
					"methods_tried": []string{"embed_text", "neurondb.embed"},
					"hint":        "Check PostgreSQL GUC settings: SHOW neurondb.llm_api_key; SHOW neurondb.llm_provider;",
				}), nil
			}
			t.logger.Error("Embedding generation failed with both methods", err, params)
			return Error(fmt.Sprintf("Embedding generation failed: text_length=%d, model='%s', error=%v", textLen, modelName, err), "EMBEDDING_ERROR", map[string]interface{}{
				"text_length": textLen,
				"model":       modelName,
				"error":       err.Error(),
				"methods_tried": []string{"embed_text", "neurondb.embed"},
			}), nil
		}
	}

	/* Log usage metrics if successful */
	if result != nil {
		/* Estimate tokens (rough approximation: 1 token ≈ 4 characters) */
		tokensEstimate := textLen / 4
		tokensInput := &tokensEstimate
		latency := int(time.Since(startTime).Milliseconds())
		latencyMS := &latency
		
		/* Log asynchronously (don't fail if logging fails) */
		/* Use a timeout context to prevent goroutine leak if logging hangs */
		go func() {
			logCtx, logCancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer logCancel()
			
			if err := t.configHelper.LogModelUsage(logCtx, modelName, "embedding", tokensInput, nil, latencyMS, true, nil); err != nil {
				if logCtx.Err() != nil {
					/* Timeout occurred, don't log error */
					return
				}
				if t.logger != nil {
					t.logger.Warn("Failed to log model usage", map[string]interface{}{"error": err.Error()})
				}
			}
		}()
	}

	return Success(result, map[string]interface{}{"model": modelName}), nil
}

/* BatchEmbeddingTool generates embeddings for multiple texts */
type BatchEmbeddingTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewBatchEmbeddingTool creates a new batch embedding tool */
func NewBatchEmbeddingTool(db *database.Database, logger *logging.Logger) *BatchEmbeddingTool {
	return &BatchEmbeddingTool{
		BaseTool: NewBaseTool(
			"postgresql_batch_embedding",
			"Generate embeddings for multiple texts efficiently",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"texts": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "string"},
						"description": "Array of texts to embed",
						"minItems":    1,
						"maxItems":    1000,
					},
					"model": map[string]interface{}{
						"type":        "string",
						"description": "Model name (optional)",
					},
				},
				"required": []interface{}{"texts"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the batch embedding */
func (t *BatchEmbeddingTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error("Invalid parameters", "VALIDATION_ERROR", map[string]interface{}{"errors": errors}), nil
	}

	texts, _ := params["texts"].([]interface{})
	model, _ := params["model"].(string)
	
	textsCount := len(texts)
	if textsCount == 0 {
		return Error("texts parameter is required and cannot be empty array for neurondb_batch_embedding tool", "VALIDATION_ERROR", map[string]interface{}{
			"parameter":   "texts",
			"texts_count": 0,
			"params":      params,
		}), nil
	}

	if textsCount > 1000 {
		return Error(fmt.Sprintf("texts array exceeds maximum size of 1000: received %d texts for neurondb_batch_embedding tool", textsCount), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":   "texts",
			"texts_count": textsCount,
			"max_count":   1000,
			"params":      params,
		}), nil
	}

	modelName := model
	if modelName == "" {
		modelName = "default"
	}

	textStrings := make([]string, 0, textsCount)
	for i, text := range texts {
		if textStr, ok := text.(string); ok {
			if textStr == "" {
				return Error(fmt.Sprintf("texts array element at index %d is empty string for neurondb_batch_embedding tool", i), "VALIDATION_ERROR", map[string]interface{}{
					"parameter":   "texts",
					"texts_count": textsCount,
					"empty_index": i,
					"params":      params,
				}), nil
			}
			textStrings = append(textStrings, textStr)
		} else {
			return Error(fmt.Sprintf("texts array element at index %d must be a string for neurondb_batch_embedding tool: got %T", i, text), "VALIDATION_ERROR", map[string]interface{}{
				"parameter":     "texts",
				"texts_count":   textsCount,
				"invalid_index": i,
				"received_type": fmt.Sprintf("%T", text),
				"params":        params,
			}), nil
		}
	}

	// embed_text_batch takes (text[], text) - array of texts first, then optional model name
	var query string
	var queryParams []interface{}
	if modelName != "" && modelName != "default" {
		query = "SELECT json_agg(embedding::text) AS embeddings FROM unnest(embed_text_batch($1::text[], $2)) AS embedding"
		queryParams = []interface{}{textStrings, modelName}
	} else {
		query = "SELECT json_agg(embedding::text) AS embeddings FROM unnest(embed_text_batch($1::text[])) AS embedding"
		queryParams = []interface{}{textStrings}
	}

	result, err := t.executor.ExecuteQueryOneWithTimeout(ctx, query, queryParams, EmbeddingQueryTimeout)
	if err != nil {
		t.logger.Error("Batch embedding failed", err, params)
		return Error(fmt.Sprintf("Batch embedding generation failed: texts_count=%d, model='%s', error=%v", textsCount, modelName, err), "EMBEDDING_ERROR", map[string]interface{}{
			"texts_count": textsCount,
			"model":       modelName,
			"error":       err.Error(),
		}), nil
	}

	return Success(result, map[string]interface{}{
		"count": len(texts),
		"model": modelName,
	}), nil
}

