/*-------------------------------------------------------------------------
 *
 * postgresql_query_execution.go
 *    Query execution and management tools for NeuronMCP
 *
 * Implements query execution, cancellation, history, and optimization tools
 * as specified in Phase 1.1 of the roadmap.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/postgresql_query_execution.go
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
)

/* PostgreSQLExecuteQueryTool executes arbitrary SQL with safety checks */
type PostgreSQLExecuteQueryTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLExecuteQueryTool creates a new PostgreSQL execute query tool */
func NewPostgreSQLExecuteQueryTool(db *database.Database, logger *logging.Logger) *PostgreSQLExecuteQueryTool {
	return &PostgreSQLExecuteQueryTool{
		BaseTool: NewBaseTool(
			"postgresql_execute_query",
			"Execute arbitrary SQL query with safety checks and result limits",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]interface{}{
						"type":        "string",
						"description": "SQL query to execute",
					},
					"max_rows": map[string]interface{}{
						"type":        "number",
						"default":     1000,
						"minimum":     1,
						"maximum":     10000,
						"description": "Maximum number of rows to return",
					},
					"timeout_seconds": map[string]interface{}{
						"type":        "number",
						"default":     60,
						"minimum":     1,
						"maximum":     3600,
						"description": "Query timeout in seconds",
					},
					"read_only": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Enforce read-only mode (only SELECT queries allowed)",
					},
				},
				"required": []interface{}{"query"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the SQL query with safety checks */
func (t *PostgreSQLExecuteQueryTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	query, ok := params["query"].(string)
	if !ok || query == "" {
		return Error("Query parameter is required", "INVALID_PARAMETER", nil), nil
	}

	/* Safety checks */
	queryUpper := strings.ToUpper(strings.TrimSpace(query))
	
	/* Check for dangerous operations */
	dangerousOps := []string{"DROP DATABASE", "DROP SCHEMA", "TRUNCATE", "DELETE FROM", "UPDATE", "INSERT INTO"}
	readOnly := false
	if val, ok := params["read_only"].(bool); ok {
		readOnly = val
	}

	if readOnly {
		if !strings.HasPrefix(queryUpper, "SELECT") {
			return Error("Read-only mode: only SELECT queries are allowed", "READ_ONLY_VIOLATION", nil), nil
		}
	} else {
		for _, op := range dangerousOps {
			if strings.Contains(queryUpper, op) {
				previewLen := 100
				if len(query) < previewLen {
					previewLen = len(query)
				}
				t.logger.Warn("Potentially dangerous query detected", map[string]interface{}{
					"operation": op,
					"query_preview": query[:previewLen],
				})
			}
		}
	}

	maxRows := 1000
	if val, ok := params["max_rows"].(float64); ok {
		maxRows = int(val)
		if maxRows < 1 {
			maxRows = 1
		}
		if maxRows > 10000 {
			maxRows = 10000
		}
	}

	timeoutSeconds := 60
	if val, ok := params["timeout_seconds"].(float64); ok {
		timeoutSeconds = int(val)
		if timeoutSeconds < 1 {
			timeoutSeconds = 1
		}
		if timeoutSeconds > 3600 {
			timeoutSeconds = 3600
		}
	}

	/* Add LIMIT if not present and it's a SELECT query */
	if strings.HasPrefix(queryUpper, "SELECT") && !strings.Contains(queryUpper, "LIMIT") {
		query = fmt.Sprintf("%s LIMIT %d", query, maxRows)
	}

	/* Create context with timeout */
	queryCtx, cancel := context.WithTimeout(ctx, time.Duration(timeoutSeconds)*time.Second)
	defer cancel()

	/* Execute query */
	results, err := t.executor.ExecuteQuery(queryCtx, query, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("Query execution failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	/* Limit results */
	if len(results) > maxRows {
		results = results[:maxRows]
	}

	t.logger.Info("Query executed", map[string]interface{}{
		"rows_returned": len(results),
		"max_rows":      maxRows,
		"timeout":       timeoutSeconds,
	})

	return Success(map[string]interface{}{
		"rows":    results,
		"count":   len(results),
		"limited": len(results) >= maxRows,
	}, map[string]interface{}{
		"tool": "postgresql_execute_query",
	}), nil
}

/* minInt returns the minimum of two integers */
func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

/* PostgreSQLQueryPlanTool generates visual query plan */
type PostgreSQLQueryPlanTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLQueryPlanTool creates a new PostgreSQL query plan tool */
func NewPostgreSQLQueryPlanTool(db *database.Database, logger *logging.Logger) *PostgreSQLQueryPlanTool {
	return &PostgreSQLQueryPlanTool{
		BaseTool: NewBaseTool(
			"postgresql_query_plan",
			"Generate visual query execution plan in JSON format",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]interface{}{
						"type":        "string",
						"description": "SQL query to analyze",
					},
					"analyze": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Execute query and include actual statistics",
					},
				},
				"required": []interface{}{"query"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute generates the query plan */
func (t *PostgreSQLQueryPlanTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	query, ok := params["query"].(string)
	if !ok || query == "" {
		return Error("Query parameter is required", "INVALID_PARAMETER", nil), nil
	}

	analyze := false
	if val, ok := params["analyze"].(bool); ok {
		analyze = val
	}

	var explainQuery string
	if analyze {
		explainQuery = fmt.Sprintf("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) %s", query)
	} else {
		explainQuery = fmt.Sprintf("EXPLAIN (FORMAT JSON) %s", query)
	}

	result, err := t.executor.ExecuteQueryOne(ctx, explainQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("Query plan generation failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	return Success(result, map[string]interface{}{
		"tool": "postgresql_query_plan",
	}), nil
}

/* PostgreSQLCancelQueryTool cancels a running query */
type PostgreSQLCancelQueryTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLCancelQueryTool creates a new PostgreSQL cancel query tool */
func NewPostgreSQLCancelQueryTool(db *database.Database, logger *logging.Logger) *PostgreSQLCancelQueryTool {
	return &PostgreSQLCancelQueryTool{
		BaseTool: NewBaseTool(
			"postgresql_cancel_query",
			"Cancel a running query using pg_cancel_backend",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"pid": map[string]interface{}{
						"type":        "integer",
						"description": "Process ID of the query to cancel",
					},
				},
				"required": []interface{}{"pid"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute cancels the query */
func (t *PostgreSQLCancelQueryTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	pid, ok := params["pid"].(float64)
	if !ok {
		return Error("pid parameter must be a number", "INVALID_PARAMETER", nil), nil
	}

	query := fmt.Sprintf("SELECT pg_cancel_backend(%d)", int(pid))
	result, err := t.executor.ExecuteQueryOne(ctx, query, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("Cancel query failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Query cancelled", map[string]interface{}{
		"pid": int(pid),
	})

	return Success(map[string]interface{}{
		"pid":      int(pid),
		"cancelled": result,
	}, map[string]interface{}{
		"tool": "postgresql_cancel_query",
	}), nil
}

/* PostgreSQLQueryHistoryTool retrieves query execution history */
type PostgreSQLQueryHistoryTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLQueryHistoryTool creates a new PostgreSQL query history tool */
func NewPostgreSQLQueryHistoryTool(db *database.Database, logger *logging.Logger) *PostgreSQLQueryHistoryTool {
	return &PostgreSQLQueryHistoryTool{
		BaseTool: NewBaseTool(
			"postgresql_query_history",
			"Retrieve query execution history from pg_stat_statements",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"limit": map[string]interface{}{
						"type":        "number",
						"default":     100,
						"minimum":     1,
						"maximum":     1000,
						"description": "Maximum number of queries to return",
					},
					"order_by": map[string]interface{}{
						"type":        "string",
						"enum":        []string{"calls", "total_time", "mean_time", "max_time"},
						"default":     "total_time",
						"description": "Order results by this metric",
					},
				},
				"required": []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute retrieves query history */
func (t *PostgreSQLQueryHistoryTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	/* Check if pg_stat_statements extension is available */
	checkQuery := `SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements') AS extension_exists`
	checkResult, err := t.executor.ExecuteQueryOne(ctx, checkQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("Failed to check pg_stat_statements: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	exists, ok := checkResult["extension_exists"].(bool)
	if !ok || !exists {
		return Error(
			"pg_stat_statements extension is not installed. Install it with: CREATE EXTENSION pg_stat_statements;",
			"EXTENSION_MISSING",
			nil,
		), nil
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

	orderBy := "total_time"
	if val, ok := params["order_by"].(string); ok {
		orderBy = val
	}

	var orderClause string
	switch orderBy {
	case "calls":
		orderClause = "ORDER BY calls DESC"
	case "mean_time":
		orderClause = "ORDER BY mean_exec_time DESC"
	case "max_time":
		orderClause = "ORDER BY max_exec_time DESC"
	default:
		orderClause = "ORDER BY total_exec_time DESC"
	}

	query := fmt.Sprintf(`
		SELECT 
			query,
			calls,
			total_exec_time,
			mean_exec_time,
			max_exec_time,
			min_exec_time,
			stddev_exec_time,
			rows
		FROM pg_stat_statements
		%s
		LIMIT %d
	`, orderClause, limit)

	results, err := t.executor.ExecuteQuery(ctx, query, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("Query history retrieval failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	return Success(map[string]interface{}{
		"queries": results,
		"count":   len(results),
	}, map[string]interface{}{
		"tool": "postgresql_query_history",
	}), nil
}

/* PostgreSQLQueryOptimizationTool provides query optimization suggestions */
type PostgreSQLQueryOptimizationTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLQueryOptimizationTool creates a new PostgreSQL query optimization tool */
func NewPostgreSQLQueryOptimizationTool(db *database.Database, logger *logging.Logger) *PostgreSQLQueryOptimizationTool {
	return &PostgreSQLQueryOptimizationTool{
		BaseTool: NewBaseTool(
			"postgresql_query_optimization",
			"Analyze query and provide optimization suggestions",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]interface{}{
						"type":        "string",
						"description": "SQL query to optimize",
					},
				},
				"required": []interface{}{"query"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute provides optimization suggestions */
func (t *PostgreSQLQueryOptimizationTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	query, ok := params["query"].(string)
	if !ok || query == "" {
		return Error("Query parameter is required", "INVALID_PARAMETER", nil), nil
	}

	/* Get EXPLAIN ANALYZE results */
	explainQuery := fmt.Sprintf("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) %s", query)
	planResult, err := t.executor.ExecuteQueryOne(ctx, explainQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("Query analysis failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	/* Generate optimization suggestions */
	suggestions := []string{}

	queryUpper := strings.ToUpper(query)
	
	/* Check for missing indexes on WHERE clauses */
	if strings.Contains(queryUpper, "WHERE") {
		suggestions = append(suggestions, "Consider adding indexes on columns used in WHERE clauses")
	}

	/* Check for sequential scans */
	if strings.Contains(queryUpper, "SELECT") {
		suggestions = append(suggestions, "Review query plan for sequential scans - consider adding indexes")
	}

	/* Check for missing JOIN conditions */
	if strings.Contains(queryUpper, "JOIN") && !strings.Contains(queryUpper, "ON") {
		suggestions = append(suggestions, "Ensure all JOINs have proper ON conditions")
	}

	/* Check for SELECT * */
	if strings.Contains(queryUpper, "SELECT *") {
		suggestions = append(suggestions, "Avoid SELECT * - specify only needed columns")
	}

	return Success(map[string]interface{}{
		"query_plan":   planResult,
		"suggestions":  suggestions,
		"suggestion_count": len(suggestions),
	}, map[string]interface{}{
		"tool": "postgresql_query_optimization",
	}), nil
}


