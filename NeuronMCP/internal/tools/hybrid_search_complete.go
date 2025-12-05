package tools

import (
	"context"
	"fmt"
	"strings"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

// ReciprocalRankFusionTool performs reciprocal rank fusion
type ReciprocalRankFusionTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewReciprocalRankFusionTool creates a new RRF tool
func NewReciprocalRankFusionTool(db *database.Database, logger *logging.Logger) *ReciprocalRankFusionTool {
	return &ReciprocalRankFusionTool{
		BaseTool: NewBaseTool(
			"reciprocal_rank_fusion",
			"Perform reciprocal rank fusion on multiple ranking arrays",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"rankings": map[string]interface{}{
						"type":        "array",
						"items": map[string]interface{}{
							"type": "array",
							"items": map[string]interface{}{"type": "number"},
						},
						"description": "Array of ranking arrays",
					},
					"k": map[string]interface{}{
						"type":        "number",
						"default":     60.0,
						"description": "RRF k parameter (default: 60.0)",
					},
				},
				"required": []interface{}{"rankings"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes RRF
func (t *ReciprocalRankFusionTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for reciprocal_rank_fusion tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	rankings, _ := params["rankings"].([]interface{})
	k := 60.0
	if kVal, ok := params["k"].(float64); ok {
		k = kVal
	}

	if len(rankings) == 0 {
		return Error("rankings cannot be empty", "VALIDATION_ERROR", nil), nil
	}

	// Format rankings array for PostgreSQL
	var rankingStrs []string
	for _, ranking := range rankings {
		if arr, ok := ranking.([]interface{}); ok {
			var parts []string
			for _, v := range arr {
				if num, ok := v.(float64); ok {
					parts = append(parts, fmt.Sprintf("%g", num))
				}
			}
			rankingStrs = append(rankingStrs, "{"+strings.Join(parts, ",")+"}")
		}
	}
	rankingsStr := "ARRAY[" + strings.Join(rankingStrs, ",") + "]"

	query := fmt.Sprintf("SELECT reciprocal_rank_fusion(%s, $1::float8) AS result", rankingsStr)
	queryParams := []interface{}{k}

	result, err := t.executor.ExecuteQueryOne(ctx, query, queryParams)
	if err != nil {
		t.logger.Error("Reciprocal rank fusion failed", err, params)
		return Error(fmt.Sprintf("Reciprocal rank fusion failed: error=%v", err), "EXECUTION_ERROR", map[string]interface{}{
			"error": err.Error(),
		}), nil
	}

	return Success(result, map[string]interface{}{
		"k": k,
	}), nil
}

// SemanticKeywordSearchTool performs semantic + keyword search
type SemanticKeywordSearchTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewSemanticKeywordSearchTool creates a new semantic-keyword search tool
func NewSemanticKeywordSearchTool(db *database.Database, logger *logging.Logger) *SemanticKeywordSearchTool {
	return &SemanticKeywordSearchTool{
		BaseTool: NewBaseTool(
			"semantic_keyword_search",
			"Perform semantic + keyword search",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table": map[string]interface{}{
						"type":        "string",
						"description": "Table name",
					},
					"semantic_query": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "number"},
						"description": "Semantic query vector",
					},
					"keyword_query": map[string]interface{}{
						"type":        "string",
						"description": "Keyword query text",
					},
					"top_k": map[string]interface{}{
						"type":        "number",
						"default":     10,
						"minimum":     1,
						"maximum":     1000,
						"description": "Number of results",
					},
				},
				"required": []interface{}{"table", "semantic_query", "keyword_query"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes semantic-keyword search
func (t *SemanticKeywordSearchTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for semantic_keyword_search tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	table, _ := params["table"].(string)
	semanticQuery, _ := params["semantic_query"].([]interface{})
	keywordQuery, _ := params["keyword_query"].(string)
	topK := 10
	if k, ok := params["top_k"].(float64); ok {
		topK = int(k)
	}

	if table == "" || len(semanticQuery) == 0 || keywordQuery == "" {
		return Error("table, semantic_query, and keyword_query are required", "VALIDATION_ERROR", nil), nil
	}

	vecStr := formatVectorFromInterface(semanticQuery)
	query := "SELECT * FROM semantic_keyword_search($1::text, $2::vector, $3::text, $4::int)"
	queryParams := []interface{}{table, vecStr, keywordQuery, topK}

	results, err := t.executor.ExecuteQuery(ctx, query, queryParams)
	if err != nil {
		t.logger.Error("Semantic-keyword search failed", err, params)
		return Error(fmt.Sprintf("Semantic-keyword search failed: error=%v", err), "EXECUTION_ERROR", map[string]interface{}{
			"error": err.Error(),
		}), nil
	}

	return Success(map[string]interface{}{
		"results": results,
		"count":   len(results),
	}, map[string]interface{}{
		"count": len(results),
	}), nil
}

// MultiVectorSearchTool performs multi-vector search
type MultiVectorSearchTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewMultiVectorSearchTool creates a new multi-vector search tool
func NewMultiVectorSearchTool(db *database.Database, logger *logging.Logger) *MultiVectorSearchTool {
	return &MultiVectorSearchTool{
		BaseTool: NewBaseTool(
			"multi_vector_search",
			"Perform search with multiple query vectors",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table": map[string]interface{}{
						"type":        "string",
						"description": "Table name",
					},
					"query_vectors": map[string]interface{}{
						"type":        "array",
						"items": map[string]interface{}{
							"type": "array",
							"items": map[string]interface{}{"type": "number"},
						},
						"description": "Array of query vectors",
					},
					"agg_method": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"max", "min", "avg", "sum"},
						"default":     "max",
						"description": "Aggregation method",
					},
					"top_k": map[string]interface{}{
						"type":        "number",
						"default":     10,
						"minimum":     1,
						"maximum":     1000,
						"description": "Number of results",
					},
				},
				"required": []interface{}{"table", "query_vectors"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes multi-vector search
func (t *MultiVectorSearchTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for multi_vector_search tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	table, _ := params["table"].(string)
	queryVectors, _ := params["query_vectors"].([]interface{})
	aggMethod := "max"
	if m, ok := params["agg_method"].(string); ok {
		aggMethod = m
	}
	topK := 10
	if k, ok := params["top_k"].(float64); ok {
		topK = int(k)
	}

	if table == "" || len(queryVectors) == 0 {
		return Error("table and query_vectors are required", "VALIDATION_ERROR", nil), nil
	}

	// Format vectors array
	var vecStrs []string
	for _, vec := range queryVectors {
		if arr, ok := vec.([]interface{}); ok {
			vecStr := formatVectorFromInterface(arr)
			vecStrs = append(vecStrs, vecStr+"::vector")
		}
	}
	vectorsStr := "ARRAY[" + strings.Join(vecStrs, ",") + "]"

	query := fmt.Sprintf("SELECT * FROM multi_vector_search($1::text, %s, $2::text, $3::int)", vectorsStr)
	queryParams := []interface{}{table, aggMethod, topK}

	results, err := t.executor.ExecuteQuery(ctx, query, queryParams)
	if err != nil {
		t.logger.Error("Multi-vector search failed", err, params)
		return Error(fmt.Sprintf("Multi-vector search failed: error=%v", err), "EXECUTION_ERROR", map[string]interface{}{
			"error": err.Error(),
		}), nil
	}

	return Success(map[string]interface{}{
		"results": results,
		"count":   len(results),
	}, map[string]interface{}{
		"count": len(results),
	}), nil
}

// FacetedVectorSearchTool performs faceted search
type FacetedVectorSearchTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewFacetedVectorSearchTool creates a new faceted search tool
func NewFacetedVectorSearchTool(db *database.Database, logger *logging.Logger) *FacetedVectorSearchTool {
	return &FacetedVectorSearchTool{
		BaseTool: NewBaseTool(
			"faceted_vector_search",
			"Perform faceted vector search",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table": map[string]interface{}{
						"type":        "string",
						"description": "Table name",
					},
					"query_vec": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "number"},
						"description": "Query vector",
					},
					"facet_column": map[string]interface{}{
						"type":        "string",
						"description": "Facet column name",
					},
					"per_facet_limit": map[string]interface{}{
						"type":        "number",
						"default":     3,
						"minimum":     1,
						"maximum":     100,
						"description": "Results per facet",
					},
				},
				"required": []interface{}{"table", "query_vec", "facet_column"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes faceted search
func (t *FacetedVectorSearchTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for faceted_vector_search tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	table, _ := params["table"].(string)
	queryVec, _ := params["query_vec"].([]interface{})
	facetColumn, _ := params["facet_column"].(string)
	perFacetLimit := 3
	if l, ok := params["per_facet_limit"].(float64); ok {
		perFacetLimit = int(l)
	}

	if table == "" || len(queryVec) == 0 || facetColumn == "" {
		return Error("table, query_vec, and facet_column are required", "VALIDATION_ERROR", nil), nil
	}

	vecStr := formatVectorFromInterface(queryVec)
	query := "SELECT * FROM faceted_vector_search($1::text, $2::vector, $3::text, $4::int)"
	queryParams := []interface{}{table, vecStr, facetColumn, perFacetLimit}

	results, err := t.executor.ExecuteQuery(ctx, query, queryParams)
	if err != nil {
		t.logger.Error("Faceted vector search failed", err, params)
		return Error(fmt.Sprintf("Faceted vector search failed: error=%v", err), "EXECUTION_ERROR", map[string]interface{}{
			"error": err.Error(),
		}), nil
	}

	return Success(map[string]interface{}{
		"results": results,
		"count":   len(results),
	}, map[string]interface{}{
		"count": len(results),
	}), nil
}

// TemporalVectorSearchTool performs temporal search
type TemporalVectorSearchTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewTemporalVectorSearchTool creates a new temporal search tool
func NewTemporalVectorSearchTool(db *database.Database, logger *logging.Logger) *TemporalVectorSearchTool {
	return &TemporalVectorSearchTool{
		BaseTool: NewBaseTool(
			"temporal_vector_search",
			"Perform temporal vector search with time decay",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table": map[string]interface{}{
						"type":        "string",
						"description": "Table name",
					},
					"query_vec": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "number"},
						"description": "Query vector",
					},
					"timestamp_col": map[string]interface{}{
						"type":        "string",
						"description": "Timestamp column name",
					},
					"decay_rate": map[string]interface{}{
						"type":        "number",
						"default":     0.01,
						"description": "Time decay rate",
					},
					"top_k": map[string]interface{}{
						"type":        "number",
						"default":     10,
						"minimum":     1,
						"maximum":     1000,
						"description": "Number of results",
					},
				},
				"required": []interface{}{"table", "query_vec", "timestamp_col"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes temporal search
func (t *TemporalVectorSearchTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for temporal_vector_search tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	table, _ := params["table"].(string)
	queryVec, _ := params["query_vec"].([]interface{})
	timestampCol, _ := params["timestamp_col"].(string)
	decayRate := 0.01
	if d, ok := params["decay_rate"].(float64); ok {
		decayRate = d
	}
	topK := 10
	if k, ok := params["top_k"].(float64); ok {
		topK = int(k)
	}

	if table == "" || len(queryVec) == 0 || timestampCol == "" {
		return Error("table, query_vec, and timestamp_col are required", "VALIDATION_ERROR", nil), nil
	}

	vecStr := formatVectorFromInterface(queryVec)
	query := "SELECT * FROM temporal_vector_search($1::text, $2::vector, $3::text, $4::float8, $5::int)"
	queryParams := []interface{}{table, vecStr, timestampCol, decayRate, topK}

	results, err := t.executor.ExecuteQuery(ctx, query, queryParams)
	if err != nil {
		t.logger.Error("Temporal vector search failed", err, params)
		return Error(fmt.Sprintf("Temporal vector search failed: error=%v", err), "EXECUTION_ERROR", map[string]interface{}{
			"error": err.Error(),
		}), nil
	}

	return Success(map[string]interface{}{
		"results": results,
		"count":   len(results),
	}, map[string]interface{}{
		"count": len(results),
	}), nil
}

// DiverseVectorSearchTool performs diverse search
type DiverseVectorSearchTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewDiverseVectorSearchTool creates a new diverse search tool
func NewDiverseVectorSearchTool(db *database.Database, logger *logging.Logger) *DiverseVectorSearchTool {
	return &DiverseVectorSearchTool{
		BaseTool: NewBaseTool(
			"diverse_vector_search",
			"Perform diverse vector search to maximize result diversity",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table": map[string]interface{}{
						"type":        "string",
						"description": "Table name",
					},
					"query_vec": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "number"},
						"description": "Query vector",
					},
					"diversity": map[string]interface{}{
						"type":        "number",
						"default":     0.5,
						"minimum":     0.0,
						"maximum":     1.0,
						"description": "Diversity parameter (0.0-1.0)",
					},
					"top_k": map[string]interface{}{
						"type":        "number",
						"default":     10,
						"minimum":     1,
						"maximum":     1000,
						"description": "Number of results",
					},
				},
				"required": []interface{}{"table", "query_vec"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes diverse search
func (t *DiverseVectorSearchTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for diverse_vector_search tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	table, _ := params["table"].(string)
	queryVec, _ := params["query_vec"].([]interface{})
	diversity := 0.5
	if d, ok := params["diversity"].(float64); ok {
		diversity = d
	}
	topK := 10
	if k, ok := params["top_k"].(float64); ok {
		topK = int(k)
	}

	if table == "" || len(queryVec) == 0 {
		return Error("table and query_vec are required", "VALIDATION_ERROR", nil), nil
	}

	vecStr := formatVectorFromInterface(queryVec)
	query := "SELECT * FROM diverse_vector_search($1::text, $2::vector, $3::float8, $4::int)"
	queryParams := []interface{}{table, vecStr, diversity, topK}

	results, err := t.executor.ExecuteQuery(ctx, query, queryParams)
	if err != nil {
		t.logger.Error("Diverse vector search failed", err, params)
		return Error(fmt.Sprintf("Diverse vector search failed: error=%v", err), "EXECUTION_ERROR", map[string]interface{}{
			"error": err.Error(),
		}), nil
	}

	return Success(map[string]interface{}{
		"results": results,
		"count":   len(results),
	}, map[string]interface{}{
		"count": len(results),
	}), nil
}

