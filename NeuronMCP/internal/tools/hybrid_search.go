/*-------------------------------------------------------------------------
 *
 * hybrid_search.go
 *    Tool implementation for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/hybrid_search.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/validation"
)

/* HybridSearchTool performs hybrid semantic + lexical search */
type HybridSearchTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

/* NewHybridSearchTool creates a new HybridSearchTool */
func NewHybridSearchTool(db *database.Database, logger *logging.Logger) *HybridSearchTool {
	return &HybridSearchTool{
		BaseTool: NewBaseTool(
			"postgresql_hybrid_search",
			"Perform hybrid search combining semantic (vector) and lexical (BM25/text) search",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table": map[string]interface{}{
						"type":        "string",
						"description": "The name of the table to search",
					},
					"query_vector": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "number"},
						"description": "The query vector for semantic search",
					},
					"query_text": map[string]interface{}{
						"type":        "string",
						"description": "The text query for lexical search",
					},
					"vector_column": map[string]interface{}{
						"type":        "string",
						"description": "The name of the vector column",
					},
					"text_column": map[string]interface{}{
						"type":        "string",
						"description": "The name of the text column for lexical search",
					},
					"vector_weight": map[string]interface{}{
						"type":        "number",
						"default":     0.7,
						"minimum":     0.0,
						"maximum":     1.0,
						"description": "Weight for vector search (0.0-1.0), lexical weight is 1.0 - vector_weight",
					},
					"limit": map[string]interface{}{
						"type":        "number",
						"default":     10,
						"minimum":     1,
						"maximum":     1000,
						"description": "Maximum number of results to return",
					},
					"filters": map[string]interface{}{
						"type":        "object",
						"description": "Optional filters as JSON object",
					},
					"query_type": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"plain", "to", "phrase"},
						"default":     "plain",
						"description": "FTS query type: 'plain' (plainto_tsquery), 'to' (to_tsquery), 'phrase' (phraseto_tsquery)",
					},
				},
				"required": []interface{}{"table", "query_vector", "query_text", "vector_column", "text_column"},
			},
		),
		db:     db,
		logger: logger,
	}
}

/* Execute performs hybrid search */
func (t *HybridSearchTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for neurondb_hybrid_search tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	table, _ := params["table"].(string)
	queryVector, _ := params["query_vector"].([]interface{})
	queryText, _ := params["query_text"].(string)
	vectorColumn, _ := params["vector_column"].(string)
	textColumn, _ := params["text_column"].(string)

	/* Validate table (SQL identifier) */
	if err := validation.ValidateSQLIdentifierRequired(table, "table"); err != nil {
		return Error(fmt.Sprintf("Invalid table parameter: %v", err), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "table",
			"error":     err.Error(),
			"params":    params,
		}), nil
	}

	/* Validate vector_column (SQL identifier) */
	if err := validation.ValidateSQLIdentifierRequired(vectorColumn, "vector_column"); err != nil {
		return Error(fmt.Sprintf("Invalid vector_column parameter: %v", err), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "vector_column",
			"table":     table,
			"error":     err.Error(),
			"params":    params,
		}), nil
	}

	/* Validate text_column (SQL identifier) */
	if err := validation.ValidateSQLIdentifierRequired(textColumn, "text_column"); err != nil {
		return Error(fmt.Sprintf("Invalid text_column parameter: %v", err), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "text_column",
			"table":     table,
			"error":     err.Error(),
			"params":    params,
		}), nil
	}

	/* Validate query_vector */
	if err := validation.ValidateVectorRequired(queryVector, "query_vector", 1, 10000); err != nil {
		return Error(fmt.Sprintf("Invalid query_vector parameter: %v", err), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "query_vector",
			"table":     table,
			"error":     err.Error(),
			"params":    params,
		}), nil
	}

	/* Validate query_text */
	if err := validation.ValidateRequired(queryText, "query_text"); err != nil {
		return Error(fmt.Sprintf("Invalid query_text parameter: %v", err), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "query_text",
			"table":     table,
			"error":     err.Error(),
			"params":    params,
		}), nil
	}
	if err := validation.ValidateMaxLength(queryText, "query_text", 10000); err != nil {
		return Error(fmt.Sprintf("Invalid query_text parameter: %v", err), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "query_text",
			"table":     table,
			"error":     err.Error(),
			"params":    params,
		}), nil
	}
	vectorWeight := 0.7
	if w, ok := params["vector_weight"].(float64); ok {
		vectorWeight = w
	}
	limit := 10
	if l, ok := params["limit"].(float64); ok {
		limit = int(l)
	}
	filters, _ := params["filters"].(map[string]interface{})
	queryType := "plain"
	if qt, ok := params["query_type"].(string); ok && qt != "" {
		queryType = qt
	}

	if table == "" {
		return Error("table parameter is required and cannot be empty for neurondb_hybrid_search tool", "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "table",
			"params":    params,
		}), nil
	}

	if queryText == "" {
		return Error(fmt.Sprintf("query_text parameter is required and cannot be empty for neurondb_hybrid_search tool: table='%s'", table), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "query_text",
			"table":     table,
			"params":    params,
		}), nil
	}

	if vectorColumn == "" {
		return Error(fmt.Sprintf("vector_column parameter is required and cannot be empty for neurondb_hybrid_search tool: table='%s', query_text_length=%d", table, len(queryText)), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":      "vector_column",
			"table":          table,
			"query_text_length": len(queryText),
			"params":         params,
		}), nil
	}

	if textColumn == "" {
		return Error(fmt.Sprintf("text_column parameter is required and cannot be empty for neurondb_hybrid_search tool: table='%s', query_text_length=%d, vector_column='%s'", table, len(queryText), vectorColumn), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":      "text_column",
			"table":          table,
			"query_text_length": len(queryText),
			"vector_column": vectorColumn,
			"params":         params,
		}), nil
	}

	if len(queryVector) == 0 {
		return Error(fmt.Sprintf("query_vector parameter is required and cannot be empty for neurondb_hybrid_search tool: table='%s', query_text_length=%d, vector_column='%s', text_column='%s'", table, len(queryText), vectorColumn, textColumn), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":      "query_vector",
			"table":          table,
			"query_text_length": len(queryText),
			"vector_column": vectorColumn,
			"text_column":   textColumn,
			"params":        params,
		}), nil
	}

	if vectorWeight < 0.0 || vectorWeight > 1.0 {
		return Error(fmt.Sprintf("vector_weight must be between 0.0 and 1.0 for neurondb_hybrid_search tool: table='%s', received vector_weight=%g", table, vectorWeight), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":    "vector_weight",
			"table":        table,
			"vector_weight": vectorWeight,
			"params":       params,
		}), nil
	}

  /* Format query vector */
	vectorStr := formatVectorFromInterface(queryVector)

  /* Format filters as JSON string */
	filtersJSON := "{}"
	if len(filters) > 0 {
		filtersBytes, err := json.Marshal(filters)
		if err == nil {
			filtersJSON = string(filtersBytes)
		}
	}

  /* Use NeuronDB's hybrid_search function: hybrid_search(table, query_vec, query_text, filters, vector_weight, limit, query_type) */
	query := `SELECT hybrid_search($1, $2::vector, $3, $4::text, $5, $6, $7) AS results`
	executor := NewQueryExecutor(t.db)
	result, err := executor.ExecuteQueryOne(ctx, query, []interface{}{
		table, vectorStr, queryText, filtersJSON, vectorWeight, limit, queryType,
	})
	if err != nil {
		t.logger.Error("Hybrid search failed", err, params)
		return Error(fmt.Sprintf("Hybrid search execution failed: table='%s', query_text_length=%d, vector_column='%s', text_column='%s', vector_weight=%g, limit=%d, error=%v", 
			table, len(queryText), vectorColumn, textColumn, vectorWeight, limit, err), "SEARCH_ERROR", map[string]interface{}{
			"table":          table,
			"query_text_length": len(queryText),
			"vector_column":  vectorColumn,
			"text_column":    textColumn,
			"vector_weight":  vectorWeight,
			"limit":          limit,
			"error":          err.Error(),
		}), nil
	}

	return Success(result, map[string]interface{}{
		"table":         table,
		"vector_weight": vectorWeight,
		"query_type":    queryType,
		"limit":         limit,
	}), nil
}

