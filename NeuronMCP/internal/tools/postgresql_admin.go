/*-------------------------------------------------------------------------
 *
 * postgresql_admin.go
 *    PostgreSQL administration tools for NeuronMCP
 *
 * Implements critical PostgreSQL administration tools:
 * - Query Analysis (explain, explain_analyze)
 * - Maintenance Operations (vacuum, analyze, reindex)
 * - Transaction Management (transactions, terminate_query)
 * - Configuration Management (set_config, reload_config)
 * - Performance Monitoring (slow_queries, cache_hit_ratio, buffer_stats)
 * - Advanced Features (partitions, FDW, logical replication)
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/postgresql_admin.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"fmt"
	"strings"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

/* ============================================================================
 * Query Analysis Tools
 * ============================================================================ */

/* PostgreSQLExplainTool explains query execution plan */
type PostgreSQLExplainTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLExplainTool creates a new PostgreSQL explain tool */
func NewPostgreSQLExplainTool(db *database.Database, logger *logging.Logger) *PostgreSQLExplainTool {
	return &PostgreSQLExplainTool{
		BaseTool: NewBaseTool(
			"postgresql_explain",
			"Explain query execution plan without executing the query",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]interface{}{
						"type":        "string",
						"description": "SQL query to explain",
					},
					"format": map[string]interface{}{
						"type":        "string",
						"enum":        []string{"text", "json", "xml", "yaml"},
						"default":     "text",
						"description": "Output format for explain plan",
					},
					"verbose": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Include verbose output",
					},
					"costs": map[string]interface{}{
						"type":        "boolean",
						"default":     true,
						"description": "Include cost estimates",
					},
					"buffers": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Include buffer usage (requires ANALYZE)",
					},
					"timing": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Include timing (requires ANALYZE)",
					},
				},
				"required": []interface{}{"query"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the EXPLAIN query */
func (t *PostgreSQLExplainTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	query, ok := params["query"].(string)
	if !ok || query == "" {
		return Error("Query parameter is required", "INVALID_PARAMETER", nil), nil
	}

	format := "text"
	if val, ok := params["format"].(string); ok {
		format = val
	}

	verbose := false
	if val, ok := params["verbose"].(bool); ok {
		verbose = val
	}

	costs := true
	if val, ok := params["costs"].(bool); ok {
		costs = val
	}

	buffers := false
	if val, ok := params["buffers"].(bool); ok {
		buffers = val
	}

	timing := false
	if val, ok := params["timing"].(bool); ok {
		timing = val
	}

	/* Build EXPLAIN statement */
	explainParts := []string{"EXPLAIN"}
	if format != "text" {
		explainParts = append(explainParts, fmt.Sprintf("(FORMAT %s)", strings.ToUpper(format)))
	}
	if verbose {
		explainParts = append(explainParts, "VERBOSE")
	}
	if !costs {
		explainParts = append(explainParts, "COSTS false")
	}
	if buffers {
		explainParts = append(explainParts, "BUFFERS")
	}
	if timing {
		explainParts = append(explainParts, "TIMING")
	}

	explainQuery := strings.Join(explainParts, " ") + " " + query

	result, err := t.executor.ExecuteQueryOne(ctx, explainQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("EXPLAIN query failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Query explained", map[string]interface{}{
		"format": format,
		"verbose": verbose,
	})

	return Success(result, map[string]interface{}{
		"tool": "postgresql_explain",
	}), nil
}

/* PostgreSQLExplainAnalyzeTool explains and analyzes query execution */
type PostgreSQLExplainAnalyzeTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLExplainAnalyzeTool creates a new PostgreSQL explain analyze tool */
func NewPostgreSQLExplainAnalyzeTool(db *database.Database, logger *logging.Logger) *PostgreSQLExplainAnalyzeTool {
	return &PostgreSQLExplainAnalyzeTool{
		BaseTool: NewBaseTool(
			"postgresql_explain_analyze",
			"Explain query execution plan and execute the query to get actual execution statistics",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]interface{}{
						"type":        "string",
						"description": "SQL query to explain and analyze",
					},
					"format": map[string]interface{}{
						"type":        "string",
						"enum":        []string{"text", "json", "xml", "yaml"},
						"default":     "text",
						"description": "Output format for explain plan",
					},
					"verbose": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Include verbose output",
					},
					"costs": map[string]interface{}{
						"type":        "boolean",
						"default":     true,
						"description": "Include cost estimates",
					},
					"buffers": map[string]interface{}{
						"type":        "boolean",
						"default":     true,
						"description": "Include buffer usage statistics",
					},
					"timing": map[string]interface{}{
						"type":        "boolean",
						"default":     true,
						"description": "Include timing information",
					},
				},
				"required": []interface{}{"query"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the EXPLAIN ANALYZE query */
func (t *PostgreSQLExplainAnalyzeTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	query, ok := params["query"].(string)
	if !ok || query == "" {
		return Error("Query parameter is required", "INVALID_PARAMETER", nil), nil
	}

	format := "text"
	if val, ok := params["format"].(string); ok {
		format = val
	}

	verbose := false
	if val, ok := params["verbose"].(bool); ok {
		verbose = val
	}

	costs := true
	if val, ok := params["costs"].(bool); ok {
		costs = val
	}

	buffers := true
	if val, ok := params["buffers"].(bool); ok {
		buffers = val
	}

	timing := true
	if val, ok := params["timing"].(bool); ok {
		timing = val
	}

	/* Build EXPLAIN ANALYZE statement */
	explainParts := []string{"EXPLAIN ANALYZE"}
	if format != "text" {
		explainParts = append(explainParts, fmt.Sprintf("(FORMAT %s)", strings.ToUpper(format)))
	}
	if verbose {
		explainParts = append(explainParts, "VERBOSE")
	}
	if !costs {
		explainParts = append(explainParts, "COSTS false")
	}
	if buffers {
		explainParts = append(explainParts, "BUFFERS")
	}
	if !timing {
		explainParts = append(explainParts, "TIMING false")
	}

	explainQuery := strings.Join(explainParts, " ") + " " + query

	result, err := t.executor.ExecuteQueryOne(ctx, explainQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("EXPLAIN ANALYZE query failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Query explained and analyzed", map[string]interface{}{
		"format": format,
	})

	return Success(result, map[string]interface{}{
		"tool": "postgresql_explain_analyze",
	}), nil
}

/* ============================================================================
 * Maintenance Operations
 * ============================================================================ */

/* PostgreSQLVacuumTool runs VACUUM operation */
type PostgreSQLVacuumTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLVacuumTool creates a new PostgreSQL vacuum tool */
func NewPostgreSQLVacuumTool(db *database.Database, logger *logging.Logger) *PostgreSQLVacuumTool {
	return &PostgreSQLVacuumTool{
		BaseTool: NewBaseTool(
			"postgresql_vacuum",
			"Run VACUUM operation to reclaim storage and update statistics",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table": map[string]interface{}{
						"type":        "string",
						"description": "Table name to vacuum (optional, if not provided vacuums all tables)",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"description": "Schema name (optional, defaults to public)",
					},
					"full": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Run VACUUM FULL (requires exclusive lock)",
					},
					"analyze": map[string]interface{}{
						"type":        "boolean",
						"default":     true,
						"description": "Run ANALYZE after vacuum",
					},
					"verbose": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Include verbose output",
					},
				},
				"required": []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the VACUUM operation */
func (t *PostgreSQLVacuumTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	table, _ := params["table"].(string)
	schema, _ := params["schema"].(string)
	full := false
	if val, ok := params["full"].(bool); ok {
		full = val
	}
	analyze := true
	if val, ok := params["analyze"].(bool); ok {
		analyze = val
	}
	verbose := false
	if val, ok := params["verbose"].(bool); ok {
		verbose = val
	}

	/* Build VACUUM statement */
	vacuumParts := []string{"VACUUM"}
	if full {
		vacuumParts = append(vacuumParts, "FULL")
	}
	if analyze {
		vacuumParts = append(vacuumParts, "ANALYZE")
	}
	if verbose {
		vacuumParts = append(vacuumParts, "VERBOSE")
	}

	var target string
	if table != "" {
		if schema != "" {
			target = fmt.Sprintf("%s.%s", schema, table)
		} else {
			target = table
		}
		vacuumParts = append(vacuumParts, target)
	}

	vacuumQuery := strings.Join(vacuumParts, " ")

	_, err := t.executor.ExecuteQuery(ctx, vacuumQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("VACUUM operation failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("VACUUM completed", map[string]interface{}{
		"table":   table,
		"schema":  schema,
		"full":    full,
		"analyze": analyze,
	})

	return Success(map[string]interface{}{
		"operation": "vacuum",
		"table":     table,
		"schema":    schema,
		"full":      full,
		"analyze":   analyze,
	}, map[string]interface{}{
		"tool": "postgresql_vacuum",
	}), nil
}

/* PostgreSQLAnalyzeTool runs ANALYZE operation */
type PostgreSQLAnalyzeTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLAnalyzeTool creates a new PostgreSQL analyze tool */
func NewPostgreSQLAnalyzeTool(db *database.Database, logger *logging.Logger) *PostgreSQLAnalyzeTool {
	return &PostgreSQLAnalyzeTool{
		BaseTool: NewBaseTool(
			"postgresql_analyze",
			"Run ANALYZE to update table statistics for query planner",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table": map[string]interface{}{
						"type":        "string",
						"description": "Table name to analyze (optional, if not provided analyzes all tables)",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"description": "Schema name (optional, defaults to public)",
					},
					"verbose": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Include verbose output",
					},
				},
				"required": []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the ANALYZE operation */
func (t *PostgreSQLAnalyzeTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	table, _ := params["table"].(string)
	schema, _ := params["schema"].(string)
	verbose := false
	if val, ok := params["verbose"].(bool); ok {
		verbose = val
	}

	/* Build ANALYZE statement */
	analyzeParts := []string{"ANALYZE"}
	if verbose {
		analyzeParts = append(analyzeParts, "VERBOSE")
	}

	var target string
	if table != "" {
		if schema != "" {
			target = fmt.Sprintf("%s.%s", schema, table)
		} else {
			target = table
		}
		analyzeParts = append(analyzeParts, target)
	}

	analyzeQuery := strings.Join(analyzeParts, " ")

	_, err := t.executor.ExecuteQuery(ctx, analyzeQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("ANALYZE operation failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("ANALYZE completed", map[string]interface{}{
		"table":  table,
		"schema": schema,
	})

	return Success(map[string]interface{}{
		"operation": "analyze",
		"table":     table,
		"schema":    schema,
	}, map[string]interface{}{
		"tool": "postgresql_analyze",
	}), nil
}

/* PostgreSQLReindexTool runs REINDEX operation */
type PostgreSQLReindexTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLReindexTool creates a new PostgreSQL reindex tool */
func NewPostgreSQLReindexTool(db *database.Database, logger *logging.Logger) *PostgreSQLReindexTool {
	return &PostgreSQLReindexTool{
		BaseTool: NewBaseTool(
			"postgresql_reindex",
			"Rebuild indexes to reclaim space and improve performance",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table": map[string]interface{}{
						"type":        "string",
						"description": "Table name to reindex (optional)",
					},
					"index": map[string]interface{}{
						"type":        "string",
						"description": "Index name to reindex (optional)",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"description": "Schema name (optional, defaults to public)",
					},
					"concurrently": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Rebuild index concurrently (non-blocking)",
					},
				},
				"required": []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the REINDEX operation */
func (t *PostgreSQLReindexTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	table, _ := params["table"].(string)
	index, _ := params["index"].(string)
	schema, _ := params["schema"].(string)
	concurrently := false
	if val, ok := params["concurrently"].(bool); ok {
		concurrently = val
	}

	/* Build REINDEX statement */
	reindexParts := []string{"REINDEX"}
	if concurrently {
		reindexParts = append(reindexParts, "TABLE CONCURRENTLY")
	} else {
		if index != "" {
			reindexParts = append(reindexParts, "INDEX")
		} else if table != "" {
			reindexParts = append(reindexParts, "TABLE")
		} else {
			reindexParts = append(reindexParts, "DATABASE")
		}
	}

	var target string
	if index != "" {
		if schema != "" {
			target = fmt.Sprintf("%s.%s", schema, index)
		} else {
			target = index
		}
	} else if table != "" {
		if schema != "" {
			target = fmt.Sprintf("%s.%s", schema, table)
		} else {
			target = table
		}
	} else {
		target = "CURRENT_DATABASE()"
	}

	if target != "" {
		reindexParts = append(reindexParts, target)
	}

	reindexQuery := strings.Join(reindexParts, " ")

	_, err := t.executor.ExecuteQuery(ctx, reindexQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("REINDEX operation failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("REINDEX completed", map[string]interface{}{
		"table":        table,
		"index":        index,
		"schema":       schema,
		"concurrently": concurrently,
	})

	return Success(map[string]interface{}{
		"operation":   "reindex",
		"table":       table,
		"index":       index,
		"schema":      schema,
		"concurrently": concurrently,
	}, map[string]interface{}{
		"tool": "postgresql_reindex",
	}), nil
}

/* ============================================================================
 * Transaction Management
 * ============================================================================ */

/* PostgreSQLTransactionsTool lists active transactions */
type PostgreSQLTransactionsTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLTransactionsTool creates a new PostgreSQL transactions tool */
func NewPostgreSQLTransactionsTool(db *database.Database, logger *logging.Logger) *PostgreSQLTransactionsTool {
	return &PostgreSQLTransactionsTool{
		BaseTool: NewBaseTool(
			"postgresql_transactions",
			"List all active transactions with details",
			map[string]interface{}{
				"type":       "object",
				"properties": map[string]interface{}{},
				"required":   []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the transactions query */
func (t *PostgreSQLTransactionsTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	query := `
		SELECT 
			pid,
			usename,
			application_name,
			client_addr,
			state,
			query_start,
			xact_start,
			state_change,
			wait_event_type,
			wait_event,
			query,
			backend_type
		FROM pg_stat_activity
		WHERE datname = current_database()
		AND state != 'idle'
		ORDER BY query_start DESC
	`

	results, err := t.executor.ExecuteQuery(ctx, query, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("Transactions query failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Active transactions retrieved", map[string]interface{}{
		"count": len(results),
	})

	return Success(map[string]interface{}{
		"transactions": results,
		"count":       len(results),
	}, map[string]interface{}{
		"tool": "postgresql_transactions",
	}), nil
}

/* PostgreSQLTerminateQueryTool terminates a running query */
type PostgreSQLTerminateQueryTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLTerminateQueryTool creates a new PostgreSQL terminate query tool */
func NewPostgreSQLTerminateQueryTool(db *database.Database, logger *logging.Logger) *PostgreSQLTerminateQueryTool {
	return &PostgreSQLTerminateQueryTool{
		BaseTool: NewBaseTool(
			"postgresql_terminate_query",
			"Terminate a running query or backend process",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"pid": map[string]interface{}{
						"type":        "integer",
						"description": "Process ID of the query to terminate",
					},
					"terminate_backend": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Use pg_terminate_backend instead of pg_cancel_backend (more forceful)",
					},
				},
				"required": []interface{}{"pid"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the terminate query operation */
func (t *PostgreSQLTerminateQueryTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	pid, ok := params["pid"].(float64)
	if !ok {
		return Error("pid parameter must be a number", "INVALID_PARAMETER", nil), nil
	}

	terminateBackend := false
	if val, ok := params["terminate_backend"].(bool); ok {
		terminateBackend = val
	}

	var query string
	if terminateBackend {
		query = fmt.Sprintf("SELECT pg_terminate_backend(%d)", int(pid))
	} else {
		query = fmt.Sprintf("SELECT pg_cancel_backend(%d)", int(pid))
	}

	result, err := t.executor.ExecuteQueryOne(ctx, query, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("Terminate query failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Query terminated", map[string]interface{}{
		"pid":              int(pid),
		"terminate_backend": terminateBackend,
	})

	return Success(map[string]interface{}{
		"pid":              int(pid),
		"terminated":       result,
		"terminate_backend": terminateBackend,
	}, map[string]interface{}{
		"tool": "postgresql_terminate_query",
	}), nil
}

/* ============================================================================
 * Configuration Management
 * ============================================================================ */

/* PostgreSQLSetConfigTool sets a configuration parameter */
type PostgreSQLSetConfigTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLSetConfigTool creates a new PostgreSQL set config tool */
func NewPostgreSQLSetConfigTool(db *database.Database, logger *logging.Logger) *PostgreSQLSetConfigTool {
	return &PostgreSQLSetConfigTool{
		BaseTool: NewBaseTool(
			"postgresql_set_config",
			"Set a PostgreSQL configuration parameter",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"parameter": map[string]interface{}{
						"type":        "string",
						"description": "Configuration parameter name",
					},
					"value": map[string]interface{}{
						"type":        "string",
						"description": "Configuration parameter value",
					},
					"scope": map[string]interface{}{
						"type":        "string",
						"enum":        []string{"session", "local", "transaction"},
						"default":     "session",
						"description": "Scope of the configuration change",
					},
				},
				"required": []interface{}{"parameter", "value"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the SET configuration operation */
func (t *PostgreSQLSetConfigTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	parameter, ok := params["parameter"].(string)
	if !ok || parameter == "" {
		return Error("parameter is required", "INVALID_PARAMETER", nil), nil
	}

	value, ok := params["value"].(string)
	if !ok || value == "" {
		return Error("value is required", "INVALID_PARAMETER", nil), nil
	}

	scope := "session"
	if val, ok := params["scope"].(string); ok {
		scope = val
	}

	var query string
	switch scope {
	case "local":
		query = fmt.Sprintf("SET LOCAL %s = %s", parameter, value)
	case "transaction":
		query = fmt.Sprintf("SET LOCAL %s = %s", parameter, value)
	default:
		query = fmt.Sprintf("SET %s = %s", parameter, value)
	}

	_, err := t.executor.ExecuteQuery(ctx, query, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("SET configuration failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Configuration set", map[string]interface{}{
		"parameter": parameter,
		"scope":     scope,
	})

	return Success(map[string]interface{}{
		"parameter": parameter,
		"value":     value,
		"scope":     scope,
	}, map[string]interface{}{
		"tool": "postgresql_set_config",
	}), nil
}

/* PostgreSQLReloadConfigTool reloads PostgreSQL configuration */
type PostgreSQLReloadConfigTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLReloadConfigTool creates a new PostgreSQL reload config tool */
func NewPostgreSQLReloadConfigTool(db *database.Database, logger *logging.Logger) *PostgreSQLReloadConfigTool {
	return &PostgreSQLReloadConfigTool{
		BaseTool: NewBaseTool(
			"postgresql_reload_config",
			"Reload PostgreSQL configuration file (postgresql.conf) without restarting the server",
			map[string]interface{}{
				"type":       "object",
				"properties": map[string]interface{}{},
				"required":   []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the reload configuration operation */
func (t *PostgreSQLReloadConfigTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	query := "SELECT pg_reload_conf()"

	result, err := t.executor.ExecuteQueryOne(ctx, query, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("Reload configuration failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Configuration reloaded", map[string]interface{}{})

	return Success(result, map[string]interface{}{
		"tool": "postgresql_reload_config",
	}), nil
}

/* ============================================================================
 * Performance Monitoring
 * ============================================================================ */

/* PostgreSQLSlowQueriesTool identifies slow queries */
type PostgreSQLSlowQueriesTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLSlowQueriesTool creates a new PostgreSQL slow queries tool */
func NewPostgreSQLSlowQueriesTool(db *database.Database, logger *logging.Logger) *PostgreSQLSlowQueriesTool {
	return &PostgreSQLSlowQueriesTool{
		BaseTool: NewBaseTool(
			"postgresql_slow_queries",
			"Identify slow-running queries from pg_stat_statements",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"min_duration_ms": map[string]interface{}{
						"type":        "number",
						"default":     1000,
						"description": "Minimum query duration in milliseconds",
					},
					"limit": map[string]interface{}{
						"type":        "integer",
						"default":     20,
						"description": "Maximum number of queries to return",
					},
				},
				"required": []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the slow queries query */
func (t *PostgreSQLSlowQueriesTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	minDuration := 1000.0
	if val, ok := params["min_duration_ms"].(float64); ok {
		minDuration = val
	}

	limit := 20
	if val, ok := params["limit"].(float64); ok {
		limit = int(val)
	}

	/* Check if pg_stat_statements extension is available */
	checkQuery := `
		SELECT EXISTS(
			SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements'
		) AS extension_exists
	`
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

	query := fmt.Sprintf(`
		SELECT 
			query,
			calls,
			total_exec_time,
			mean_exec_time,
			max_exec_time,
			min_exec_time,
			stddev_exec_time,
			rows,
			shared_blks_hit,
			shared_blks_read,
			shared_blks_dirtied,
			shared_blks_written
		FROM pg_stat_statements
		WHERE mean_exec_time >= %f
		ORDER BY mean_exec_time DESC
		LIMIT %d
	`, minDuration, limit)

	results, err := t.executor.ExecuteQuery(ctx, query, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("Slow queries query failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Slow queries retrieved", map[string]interface{}{
		"count":        len(results),
		"min_duration": minDuration,
	})

	return Success(map[string]interface{}{
		"slow_queries": results,
		"count":       len(results),
	}, map[string]interface{}{
		"tool": "postgresql_slow_queries",
	}), nil
}

/* PostgreSQLCacheHitRatioTool gets cache hit ratio statistics */
type PostgreSQLCacheHitRatioTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLCacheHitRatioTool creates a new PostgreSQL cache hit ratio tool */
func NewPostgreSQLCacheHitRatioTool(db *database.Database, logger *logging.Logger) *PostgreSQLCacheHitRatioTool {
	return &PostgreSQLCacheHitRatioTool{
		BaseTool: NewBaseTool(
			"postgresql_cache_hit_ratio",
			"Get cache hit ratio statistics for tables and indexes",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"schema": map[string]interface{}{
						"type":        "string",
						"description": "Filter by schema name (optional)",
					},
					"table": map[string]interface{}{
						"type":        "string",
						"description": "Filter by table name (optional)",
					},
				},
				"required": []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the cache hit ratio query */
func (t *PostgreSQLCacheHitRatioTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	schema, _ := params["schema"].(string)
	table, _ := params["table"].(string)

	var whereClause string
	var queryParams []interface{}

	if schema != "" && table != "" {
		whereClause = "WHERE schemaname = $1 AND relname = $2"
		queryParams = []interface{}{schema, table}
	} else if schema != "" {
		whereClause = "WHERE schemaname = $1"
		queryParams = []interface{}{schema}
	} else if table != "" {
		whereClause = "WHERE relname = $1"
		queryParams = []interface{}{table}
	}

	query := fmt.Sprintf(`
		SELECT 
			schemaname,
			relname AS tablename,
			heap_blks_hit,
			heap_blks_read,
			CASE 
				WHEN (heap_blks_hit + heap_blks_read) > 0 
				THEN (heap_blks_hit::float / (heap_blks_hit + heap_blks_read) * 100)
				ELSE 0
			END AS heap_cache_hit_ratio,
			idx_blks_hit,
			idx_blks_read,
			CASE 
				WHEN (idx_blks_hit + idx_blks_read) > 0 
				THEN (idx_blks_hit::float / (idx_blks_hit + idx_blks_read) * 100)
				ELSE 0
			END AS index_cache_hit_ratio,
			toast_blks_hit,
			toast_blks_read,
			CASE 
				WHEN (toast_blks_hit + toast_blks_read) > 0 
				THEN (toast_blks_hit::float / (toast_blks_hit + toast_blks_read) * 100)
				ELSE 0
			END AS toast_cache_hit_ratio
		FROM pg_statio_user_tables
		%s
		ORDER BY heap_cache_hit_ratio ASC, index_cache_hit_ratio ASC
	`, whereClause)

	results, err := t.executor.ExecuteQuery(ctx, query, queryParams)
	if err != nil {
		return Error(
			fmt.Sprintf("Cache hit ratio query failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	/* Calculate overall statistics */
	overallQuery := `
		SELECT 
			SUM(heap_blks_hit) AS total_heap_hit,
			SUM(heap_blks_read) AS total_heap_read,
			SUM(idx_blks_hit) AS total_idx_hit,
			SUM(idx_blks_read) AS total_idx_read,
			CASE 
				WHEN SUM(heap_blks_hit + heap_blks_read) > 0 
				THEN (SUM(heap_blks_hit)::float / SUM(heap_blks_hit + heap_blks_read) * 100)
				ELSE 0
			END AS overall_heap_ratio,
			CASE 
				WHEN SUM(idx_blks_hit + idx_blks_read) > 0 
				THEN (SUM(idx_blks_hit)::float / SUM(idx_blks_hit + idx_blks_read) * 100)
				ELSE 0
			END AS overall_index_ratio
		FROM pg_statio_user_tables
	`
	if whereClause != "" {
		overallQuery += " " + whereClause
	}

	overall, err := t.executor.ExecuteQueryOne(ctx, overallQuery, queryParams)
	if err == nil {
		return Success(map[string]interface{}{
			"tables": results,
			"overall": overall,
			"count":   len(results),
		}, map[string]interface{}{
			"tool": "postgresql_cache_hit_ratio",
		}), nil
	}

	return Success(map[string]interface{}{
		"tables": results,
		"count":  len(results),
	}, map[string]interface{}{
		"tool": "postgresql_cache_hit_ratio",
	}), nil
}

/* PostgreSQLBufferStatsTool gets buffer pool statistics */
type PostgreSQLBufferStatsTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLBufferStatsTool creates a new PostgreSQL buffer stats tool */
func NewPostgreSQLBufferStatsTool(db *database.Database, logger *logging.Logger) *PostgreSQLBufferStatsTool {
	return &PostgreSQLBufferStatsTool{
		BaseTool: NewBaseTool(
			"postgresql_buffer_stats",
			"Get PostgreSQL buffer pool statistics and usage",
			map[string]interface{}{
				"type":       "object",
				"properties": map[string]interface{}{},
				"required":   []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the buffer stats query */
func (t *PostgreSQLBufferStatsTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	query := `
		SELECT 
			current_setting('shared_buffers') AS shared_buffers,
			(SELECT setting FROM pg_settings WHERE name = 'shared_buffers') AS shared_buffers_setting,
			(SELECT setting FROM pg_settings WHERE name = 'effective_cache_size') AS effective_cache_size,
			pg_stat_get_db_numbackends((SELECT oid FROM pg_database WHERE datname = current_database())) AS active_backends,
			(SELECT count(*) FROM pg_buffercache) AS buffers_in_cache,
			(SELECT count(*) FROM pg_buffercache WHERE isdirty) AS dirty_buffers,
			(SELECT count(*) FROM pg_buffercache WHERE usagecount > 0) AS used_buffers
	`

	result, err := t.executor.ExecuteQueryOne(ctx, query, nil)
	if err != nil {
		/* Fallback to simpler query if pg_buffercache extension is not available */
		simpleQuery := `
			SELECT 
				current_setting('shared_buffers') AS shared_buffers,
				(SELECT setting FROM pg_settings WHERE name = 'shared_buffers') AS shared_buffers_setting,
				(SELECT setting FROM pg_settings WHERE name = 'effective_cache_size') AS effective_cache_size,
				pg_stat_get_db_numbackends((SELECT oid FROM pg_database WHERE datname = current_database())) AS active_backends
		`
		result, err = t.executor.ExecuteQueryOne(ctx, simpleQuery, nil)
		if err != nil {
			return Error(
				fmt.Sprintf("Buffer stats query failed: %v", err),
				"QUERY_ERROR",
				map[string]interface{}{"error": err.Error()},
			), nil
		}
	}

	t.logger.Info("Buffer statistics retrieved", map[string]interface{}{})

	return Success(result, map[string]interface{}{
		"tool": "postgresql_buffer_stats",
	}), nil
}

/* ============================================================================
 * Advanced Features
 * ============================================================================ */

/* PostgreSQLPartitionsTool lists table partitions */
type PostgreSQLPartitionsTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLPartitionsTool creates a new PostgreSQL partitions tool */
func NewPostgreSQLPartitionsTool(db *database.Database, logger *logging.Logger) *PostgreSQLPartitionsTool {
	return &PostgreSQLPartitionsTool{
		BaseTool: NewBaseTool(
			"postgresql_partitions",
			"List all table partitions with metadata",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"parent_table": map[string]interface{}{
						"type":        "string",
						"description": "Filter by parent table name (optional)",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"description": "Filter by schema name (optional)",
					},
				},
				"required": []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the partitions query */
func (t *PostgreSQLPartitionsTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	parentTable, _ := params["parent_table"].(string)
	schema, _ := params["schema"].(string)

	var whereClause string
	var queryParams []interface{}

	conditions := []string{}
	if parentTable != "" {
		conditions = append(conditions, fmt.Sprintf("parent.relname = $%d", len(queryParams)+1))
		queryParams = append(queryParams, parentTable)
	}
	if schema != "" {
		conditions = append(conditions, fmt.Sprintf("parent.relnamespace::regnamespace::text = $%d", len(queryParams)+1))
		queryParams = append(queryParams, schema)
	}

	if len(conditions) > 0 {
		whereClause = "WHERE " + strings.Join(conditions, " AND ")
	}

	query := fmt.Sprintf(`
		SELECT 
			parent.relnamespace::regnamespace::text AS parent_schema,
			parent.relname AS parent_table,
			part.relnamespace::regnamespace::text AS partition_schema,
			part.relname AS partition_name,
			pg_get_expr(part.relpartbound, part.oid) AS partition_bound,
			pg_total_relation_size(part.oid) AS partition_size_bytes,
			pg_size_pretty(pg_total_relation_size(part.oid)) AS partition_size_pretty,
			(SELECT n_live_tup FROM pg_stat_user_tables WHERE relid = part.oid) AS row_count
		FROM pg_class part
		JOIN pg_inherits inh ON inh.inhrelid = part.oid
		JOIN pg_class parent ON inh.inhparent = parent.oid
		%s
		ORDER BY parent.relname, part.relname
	`, whereClause)

	results, err := t.executor.ExecuteQuery(ctx, query, queryParams)
	if err != nil {
		return Error(
			fmt.Sprintf("Partitions query failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Partitions retrieved", map[string]interface{}{
		"count": len(results),
	})

	return Success(map[string]interface{}{
		"partitions": results,
		"count":     len(results),
	}, map[string]interface{}{
		"tool": "postgresql_partitions",
	}), nil
}

/* PostgreSQLPartitionStatsTool gets partition statistics */
type PostgreSQLPartitionStatsTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLPartitionStatsTool creates a new PostgreSQL partition stats tool */
func NewPostgreSQLPartitionStatsTool(db *database.Database, logger *logging.Logger) *PostgreSQLPartitionStatsTool {
	return &PostgreSQLPartitionStatsTool{
		BaseTool: NewBaseTool(
			"postgresql_partition_stats",
			"Get detailed statistics for table partitions",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"parent_table": map[string]interface{}{
						"type":        "string",
						"description": "Filter by parent table name (optional)",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"description": "Filter by schema name (optional)",
					},
				},
				"required": []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the partition stats query */
func (t *PostgreSQLPartitionStatsTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	parentTable, _ := params["parent_table"].(string)
	schema, _ := params["schema"].(string)

	var whereClause string
	var queryParams []interface{}

	conditions := []string{}
	if parentTable != "" {
		conditions = append(conditions, fmt.Sprintf("parent.relname = $%d", len(queryParams)+1))
		queryParams = append(queryParams, parentTable)
	}
	if schema != "" {
		conditions = append(conditions, fmt.Sprintf("parent.relnamespace::regnamespace::text = $%d", len(queryParams)+1))
		queryParams = append(queryParams, schema)
	}

	if len(conditions) > 0 {
		whereClause = "WHERE " + strings.Join(conditions, " AND ")
	}

	query := fmt.Sprintf(`
		SELECT 
			parent.relnamespace::regnamespace::text AS parent_schema,
			parent.relname AS parent_table,
			part.relname AS partition_name,
			s.n_live_tup AS live_tuples,
			s.n_dead_tup AS dead_tuples,
			s.n_tup_ins AS inserts,
			s.n_tup_upd AS updates,
			s.n_tup_del AS deletes,
			s.seq_scan AS sequential_scans,
			s.idx_scan AS index_scans,
			s.last_vacuum,
			s.last_autovacuum,
			s.last_analyze,
			s.last_autoanalyze,
			pg_total_relation_size(part.oid) AS total_size_bytes,
			pg_size_pretty(pg_total_relation_size(part.oid)) AS total_size_pretty
		FROM pg_class part
		JOIN pg_inherits inh ON inh.inhrelid = part.oid
		JOIN pg_class parent ON inh.inhparent = parent.oid
		LEFT JOIN pg_stat_user_tables s ON s.relid = part.oid
		%s
		ORDER BY parent.relname, part.relname
	`, whereClause)

	results, err := t.executor.ExecuteQuery(ctx, query, queryParams)
	if err != nil {
		return Error(
			fmt.Sprintf("Partition stats query failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Partition statistics retrieved", map[string]interface{}{
		"count": len(results),
	})

	return Success(map[string]interface{}{
		"partition_stats": results,
		"count":           len(results),
	}, map[string]interface{}{
		"tool": "postgresql_partition_stats",
	}), nil
}

/* PostgreSQLFDWServersTool lists foreign data wrapper servers */
type PostgreSQLFDWServersTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLFDWServersTool creates a new PostgreSQL FDW servers tool */
func NewPostgreSQLFDWServersTool(db *database.Database, logger *logging.Logger) *PostgreSQLFDWServersTool {
	return &PostgreSQLFDWServersTool{
		BaseTool: NewBaseTool(
			"postgresql_fdw_servers",
			"List all foreign data wrapper (FDW) servers",
			map[string]interface{}{
				"type":       "object",
				"properties": map[string]interface{}{},
				"required":   []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the FDW servers query */
func (t *PostgreSQLFDWServersTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	query := `
		SELECT 
			srvname AS server_name,
			srvowner::regrole AS owner,
			srvfdw::regproc AS foreign_data_wrapper,
			srvtype AS server_type,
			srvversion AS server_version,
			srvacl AS access_privileges,
			srvoptions AS server_options
		FROM pg_foreign_server
		ORDER BY srvname
	`

	results, err := t.executor.ExecuteQuery(ctx, query, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("FDW servers query failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("FDW servers retrieved", map[string]interface{}{
		"count": len(results),
	})

	return Success(map[string]interface{}{
		"fdw_servers": results,
		"count":      len(results),
	}, map[string]interface{}{
		"tool": "postgresql_fdw_servers",
	}), nil
}

/* PostgreSQLFDWTablesTool lists foreign tables */
type PostgreSQLFDWTablesTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLFDWTablesTool creates a new PostgreSQL FDW tables tool */
func NewPostgreSQLFDWTablesTool(db *database.Database, logger *logging.Logger) *PostgreSQLFDWTablesTool {
	return &PostgreSQLFDWTablesTool{
		BaseTool: NewBaseTool(
			"postgresql_fdw_tables",
			"List all foreign tables (FDW tables)",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"server": map[string]interface{}{
						"type":        "string",
						"description": "Filter by FDW server name (optional)",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"description": "Filter by schema name (optional)",
					},
				},
				"required": []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the FDW tables query */
func (t *PostgreSQLFDWTablesTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	server, _ := params["server"].(string)
	schema, _ := params["schema"].(string)

	var whereClause string
	var queryParams []interface{}

	conditions := []string{}
	if server != "" {
		conditions = append(conditions, fmt.Sprintf("s.srvname = $%d", len(queryParams)+1))
		queryParams = append(queryParams, server)
	}
	if schema != "" {
		conditions = append(conditions, fmt.Sprintf("n.nspname = $%d", len(queryParams)+1))
		queryParams = append(queryParams, schema)
	}

	if len(conditions) > 0 {
		whereClause = "WHERE " + strings.Join(conditions, " AND ")
	}

	query := fmt.Sprintf(`
		SELECT 
			n.nspname AS schema_name,
			c.relname AS table_name,
			s.srvname AS server_name,
			ft.ftoptions AS foreign_table_options,
			c.relowner::regrole AS owner
		FROM pg_foreign_table ft
		JOIN pg_class c ON ft.ftrelid = c.oid
		JOIN pg_namespace n ON c.relnamespace = n.oid
		JOIN pg_foreign_server s ON ft.ftserver = s.oid
		%s
		ORDER BY n.nspname, c.relname
	`, whereClause)

	results, err := t.executor.ExecuteQuery(ctx, query, queryParams)
	if err != nil {
		return Error(
			fmt.Sprintf("FDW tables query failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("FDW tables retrieved", map[string]interface{}{
		"count": len(results),
	})

	return Success(map[string]interface{}{
		"fdw_tables": results,
		"count":     len(results),
	}, map[string]interface{}{
		"tool": "postgresql_fdw_tables",
	}), nil
}

/* PostgreSQLLogicalReplicationSlotsTool lists logical replication slots */
type PostgreSQLLogicalReplicationSlotsTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLLogicalReplicationSlotsTool creates a new PostgreSQL logical replication slots tool */
func NewPostgreSQLLogicalReplicationSlotsTool(db *database.Database, logger *logging.Logger) *PostgreSQLLogicalReplicationSlotsTool {
	return &PostgreSQLLogicalReplicationSlotsTool{
		BaseTool: NewBaseTool(
			"postgresql_logical_replication_slots",
			"List all logical replication slots with status information",
			map[string]interface{}{
				"type":       "object",
				"properties": map[string]interface{}{},
				"required":   []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the logical replication slots query */
func (t *PostgreSQLLogicalReplicationSlotsTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	query := `
		SELECT 
			slot_name,
			plugin,
			slot_type,
			datoid,
			database,
			active,
			active_pid,
			xmin,
			catalog_xmin,
			restart_lsn,
			confirmed_flush_lsn,
			pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), restart_lsn)) AS lag_size,
			pg_wal_lsn_diff(pg_current_wal_lsn(), restart_lsn) AS lag_bytes
		FROM pg_replication_slots
		WHERE slot_type = 'logical'
		ORDER BY slot_name
	`

	results, err := t.executor.ExecuteQuery(ctx, query, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("Logical replication slots query failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Logical replication slots retrieved", map[string]interface{}{
		"count": len(results),
	})

	return Success(map[string]interface{}{
		"replication_slots": results,
		"count":            len(results),
	}, map[string]interface{}{
		"tool": "postgresql_logical_replication_slots",
	}), nil
}

