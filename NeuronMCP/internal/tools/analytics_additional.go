/*-------------------------------------------------------------------------
 *
 * analytics_additional.go
 *    Tool implementation for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/analytics_additional.go
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
)

/* AnalyzeDataTool performs comprehensive data analysis */
type AnalyzeDataTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

/* NewAnalyzeDataTool creates a new AnalyzeDataTool */
func NewAnalyzeDataTool(db *database.Database, logger *logging.Logger) *AnalyzeDataTool {
	return &AnalyzeDataTool{
		BaseTool: NewBaseTool(
			"postgresql_analyze_data",
			"Perform comprehensive data analysis including statistics, distributions, and data quality metrics",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table": map[string]interface{}{
						"type":        "string",
						"description": "The name of the table to analyze",
					},
					"columns": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "string"},
						"description": "Optional array of column names to analyze (if not provided, analyzes all columns)",
					},
					"include_stats": map[string]interface{}{
						"type":        "boolean",
						"default":     true,
						"description": "Include statistical measures (mean, median, stddev, etc.)",
					},
					"include_distribution": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Include distribution analysis",
					},
				},
				"required": []interface{}{"table"},
			},
		),
		db:     db,
		logger: logger,
	}
}

/* Execute performs data analysis */
func (t *AnalyzeDataTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for neurondb_analyze_data tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	table, _ := params["table"].(string)
	columns, _ := params["columns"].([]interface{})
	
	/* Handle string array format from command parser */
	if columns == nil {
		if colsStr, ok := params["columns"].(string); ok && colsStr != "" {
			/* Try to parse as JSON array */
			var colsArray []interface{}
			if err := json.Unmarshal([]byte(colsStr), &colsArray); err == nil {
				columns = colsArray
			}
		}
	}
	
	includeStats, _ := params["include_stats"].(bool)
	includeDistribution, _ := params["include_distribution"].(bool)

	if table == "" {
		return Error("table parameter is required and cannot be empty for neurondb_analyze_data tool", "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "table",
			"params":    params,
		}), nil
	}

  /* Build comprehensive analysis query */
	var query string
	var queryParams []interface{}

	if len(columns) > 0 {
   /* Analyze specific columns */
		colNames := make([]string, len(columns))
		for i, col := range columns {
			colNames[i] = col.(string)
		}

   /* Build statistics query for each column */
		statsQueries := []string{}
		for _, col := range colNames {
			statsQueries = append(statsQueries, fmt.Sprintf(`
				SELECT 
					'%s' AS column_name,
					COUNT(*) AS row_count,
					COUNT(%s) AS non_null_count,
					COUNT(*) - COUNT(%s) AS null_count,
					AVG(%s) AS mean_value,
					PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY %s) AS median_value,
					STDDEV(%s) AS stddev_value,
					MIN(%s) AS min_value,
					MAX(%s) AS max_value
				FROM %s
			`, col, database.EscapeIdentifier(col), database.EscapeIdentifier(col),
				database.EscapeIdentifier(col), database.EscapeIdentifier(col),
				database.EscapeIdentifier(col), database.EscapeIdentifier(col),
				database.EscapeIdentifier(col), database.EscapeIdentifier(table)))
		}

		if len(statsQueries) == 0 {
			return Error(fmt.Sprintf("No valid columns specified for neurondb_analyze_data tool: table='%s'", table), "VALIDATION_ERROR", map[string]interface{}{
				"table": table,
			}), nil
		}
		unionQuery := statsQueries[0]
		for i := 1; i < len(statsQueries); i++ {
			unionQuery += " UNION ALL " + statsQueries[i]
		}
		query = "SELECT json_agg(row_to_json(t)) AS analysis FROM (" + unionQuery + ") t"
		queryParams = []interface{}{}
	} else {
   /* Analyze all columns - get column list first */
		colQuery := `
			SELECT column_name, data_type 
			FROM information_schema.columns 
			WHERE table_name = $1 AND table_schema = 'public'
			ORDER BY ordinal_position
		`
		executor := NewQueryExecutor(t.db)
		colResults, err := executor.ExecuteQuery(ctx, colQuery, []interface{}{table})
		if err != nil {
			return Error(fmt.Sprintf("Failed to get column list for analyze_data: table='%s', error=%v", table, err), "QUERY_ERROR", map[string]interface{}{
				"table": table,
				"error": err.Error(),
			}), nil
		}

		if len(colResults) == 0 {
			return Error(fmt.Sprintf("Table '%s' not found or has no columns for neurondb_analyze_data tool", table), "NOT_FOUND", map[string]interface{}{
				"table": table,
			}), nil
		}

   /* Build analysis for all numeric columns */
		statsParts := []string{}
		for _, colRow := range colResults {
			colName, _ := colRow["column_name"].(string)
			dataType, _ := colRow["data_type"].(string)
			
    /* Only analyze numeric types */
			if dataType == "real" || dataType == "double precision" || dataType == "integer" || 
			   dataType == "bigint" || dataType == "numeric" || dataType == "smallint" {
				statsParts = append(statsParts, fmt.Sprintf(`
					SELECT 
						'%s' AS column_name,
						'%s' AS data_type,
						COUNT(*) AS row_count,
						COUNT(%s) AS non_null_count,
						COUNT(*) - COUNT(%s) AS null_count,
						AVG(%s) AS mean_value,
						PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY %s) AS median_value,
						STDDEV(%s) AS stddev_value,
						MIN(%s) AS min_value,
						MAX(%s) AS max_value
					FROM %s
				`, colName, dataType, database.EscapeIdentifier(colName), database.EscapeIdentifier(colName),
					database.EscapeIdentifier(colName), database.EscapeIdentifier(colName),
					database.EscapeIdentifier(colName), database.EscapeIdentifier(colName),
					database.EscapeIdentifier(colName), database.EscapeIdentifier(table)))
			}
		}

		if len(statsParts) == 0 {
			return Error(fmt.Sprintf("No numeric columns found in table '%s' for neurondb_analyze_data tool", table), "VALIDATION_ERROR", map[string]interface{}{
				"table": table,
			}), nil
		}
		unionQuery := statsParts[0]
		for i := 1; i < len(statsParts); i++ {
			unionQuery += " UNION ALL " + statsParts[i]
		}
		query = "SELECT json_agg(row_to_json(t)) AS analysis FROM (" + unionQuery + ") t"
		queryParams = []interface{}{}
	}

	executor := NewQueryExecutor(t.db)
	result, err := executor.ExecuteQueryOne(ctx, query, queryParams)
	if err != nil {
		t.logger.Error("Data analysis failed", err, params)
		return Error(fmt.Sprintf("Data analysis execution failed: table='%s', columns_count=%d, include_stats=%v, include_distribution=%v, error=%v", 
			table, len(columns), includeStats, includeDistribution, err), "ANALYSIS_ERROR", map[string]interface{}{
			"table":               table,
			"columns_count":       len(columns),
			"include_stats":       includeStats,
			"include_distribution": includeDistribution,
			"error":               err.Error(),
		}), nil
	}

	return Success(result, map[string]interface{}{
		"table":               table,
		"columns_count":       len(columns),
		"include_stats":       includeStats,
		"include_distribution": includeDistribution,
	}), nil
}

