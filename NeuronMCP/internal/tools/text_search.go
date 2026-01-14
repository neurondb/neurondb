/*-------------------------------------------------------------------------
 *
 * text_search.go
 *    Tool implementation for text-only search in NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/text_search.go
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

/* TextSearchTool performs text-only full-text search */
type TextSearchTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

/* NewTextSearchTool creates a new TextSearchTool */
func NewTextSearchTool(db *database.Database, logger *logging.Logger) *TextSearchTool {
	return &TextSearchTool{
		BaseTool: NewBaseTool(
			"postgresql_text_search",
			"Perform full-text search using PostgreSQL FTS (text-only, no vectors required)",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table": map[string]interface{}{
						"type":        "string",
						"description": "The name of the table to search",
					},
					"query_text": map[string]interface{}{
						"type":        "string",
						"description": "The text query for full-text search",
					},
					"text_column": map[string]interface{}{
						"type":        "string",
						"default":     "fts_vector",
						"description": "The name of the text column (tsvector type) for search. Default: 'fts_vector'",
					},
					"query_type": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"plain", "to", "phrase"},
						"default":     "plain",
						"description": "Query type: 'plain' (plainto_tsquery), 'to' (to_tsquery), 'phrase' (phraseto_tsquery)",
					},
					"limit": map[string]interface{}{
						"type":        "number",
						"default":     10,
						"minimum":     1,
						"maximum":     10000,
						"description": "Maximum number of results to return",
					},
					"filters": map[string]interface{}{
						"type":        "object",
						"description": "Optional metadata filters as JSON object",
					},
				},
				"required": []interface{}{"table", "query_text"},
			},
		),
		db:     db,
		logger: logger,
	}
}

/* Execute performs text-only search */
func (t *TextSearchTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for neurondb_text_search tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	table, _ := params["table"].(string)
	queryText, _ := params["query_text"].(string)
	textColumn := "fts_vector"
	if tc, ok := params["text_column"].(string); ok && tc != "" {
		textColumn = tc
	}
	queryType := "plain"
	if qt, ok := params["query_type"].(string); ok && qt != "" {
		queryType = qt
	}

	/* Validate table (SQL identifier) */
	if err := validation.ValidateSQLIdentifierRequired(table, "table"); err != nil {
		return Error(fmt.Sprintf("Invalid table parameter: %v", err), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "table",
			"error":     err.Error(),
			"params":    params,
		}), nil
	}

	/* Validate text_column (SQL identifier) */
	if err := validation.ValidateSQLIdentifier(textColumn, "text_column"); err != nil {
		return Error(fmt.Sprintf("Invalid text_column parameter: %v", err), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "text_column",
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

	/* Validate query_type */
	if err := validation.ValidateIn(queryType, "query_type", "plain", "to", "phrase"); err != nil {
		return Error(fmt.Sprintf("Invalid query_type parameter: %v", err), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "query_type",
			"table":     table,
			"error":     err.Error(),
			"params":    params,
		}), nil
	}
	limit := 10
	if l, ok := params["limit"].(float64); ok {
		limit = int(l)
	}
	filters, _ := params["filters"].(map[string]interface{})

	if table == "" {
		return Error("table parameter is required and cannot be empty for neurondb_text_search tool", "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "table",
			"params":    params,
		}), nil
	}

	if queryText == "" {
		return Error(fmt.Sprintf("query_text parameter is required and cannot be empty for neurondb_text_search tool: table='%s'", table), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "query_text",
			"table":     table,
			"params":    params,
		}), nil
	}

	if limit < 1 || limit > 10000 {
		return Error(fmt.Sprintf("limit must be between 1 and 10000 for neurondb_text_search tool: table='%s', received limit=%d", table, limit), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "limit",
			"table":      table,
			"limit":      limit,
			"params":     params,
		}), nil
	}

	/* Format filters as JSON string */
	filtersJSON := "{}"
	if len(filters) > 0 {
		filtersBytes, err := json.Marshal(filters)
		if err == nil {
			filtersJSON = string(filtersBytes)
		}
	}

	/* Use NeuronDB's full_text_search function: full_text_search(table, query_text, text_column, query_type, filters, limit) */
	query := `SELECT * FROM full_text_search($1, $2, $3, $4, $5, $6)`
	executor := NewQueryExecutor(t.db)
	result, err := executor.ExecuteQuery(ctx, query, []interface{}{
		table, queryText, textColumn, queryType, filtersJSON, limit,
	})
	if err != nil {
		t.logger.Error("Text search failed", err, params)
		return Error(fmt.Sprintf("Text search execution failed: table='%s', query_text_length=%d, text_column='%s', query_type='%s', limit=%d, error=%v",
			table, len(queryText), textColumn, queryType, limit, err), "SEARCH_ERROR", map[string]interface{}{
			"table":          table,
			"query_text_length": len(queryText),
			"text_column":    textColumn,
			"query_type":     queryType,
			"limit":          limit,
			"error":          err.Error(),
		}), nil
	}

	return Success(result, map[string]interface{}{
		"table":      table,
		"query_type": queryType,
		"limit":      limit,
		"count":      len(result),
	}), nil
}



