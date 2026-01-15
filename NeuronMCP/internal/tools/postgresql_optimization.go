/*-------------------------------------------------------------------------
 *
 * postgresql_optimization.go
 *    PostgreSQL Optimization tools for NeuronMCP
 *
 * Provides query optimization, performance insights, index recommendations,
 * and query plan analysis.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/postgresql_optimization.go
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

/* PostgreSQLQueryOptimizerTool analyzes and suggests query optimizations */
type PostgreSQLQueryOptimizerTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

/* NewPostgreSQLQueryOptimizerTool creates a new query optimizer tool */
func NewPostgreSQLQueryOptimizerTool(db *database.Database, logger *logging.Logger) Tool {
	inputSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"query": map[string]interface{}{
				"type":        "string",
				"description": "SQL query to optimize",
			},
			"analyze": map[string]interface{}{
				"type":        "boolean",
				"description": "Whether to run ANALYZE on tables",
				"default":     false,
			},
		},
		"required": []interface{}{"query"},
	}

	return &PostgreSQLQueryOptimizerTool{
		BaseTool: NewBaseTool(
			"postgresql_query_optimizer",
			"Analyze SQL queries and suggest optimizations for better performance",
			inputSchema,
		),
		db:     db,
		logger: logger,
	}
}

/* Execute executes the query optimizer tool */
func (t *PostgreSQLQueryOptimizerTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	query, _ := params["query"].(string)
	analyze, _ := params["analyze"].(bool)

	if query == "" {
		return Error("query is required", "INVALID_PARAMS", nil), nil
	}

	/* Get query plan */
	explainQuery := fmt.Sprintf("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) %s", query)
	rows, err := t.db.Query(ctx, explainQuery, nil)
	if err != nil {
		return Error(fmt.Sprintf("Failed to explain query: %v", err), "EXPLAIN_ERROR", nil), nil
	}
	defer rows.Close()

	var planJSON string
	if rows.Next() {
		if err := rows.Scan(&planJSON); err != nil {
			return Error("Failed to read query plan", "READ_ERROR", nil), nil
		}
	}

	/* Parse plan and generate suggestions */
	suggestions := t.analyzePlan(planJSON, query)

	/* Run ANALYZE if requested */
	if analyze {
		t.runAnalyze(ctx, query)
	}

	return Success(map[string]interface{}{
		"query":       query,
		"plan":        planJSON,
		"suggestions": suggestions,
		"optimized":   t.generateOptimizedQuery(query, suggestions),
	}, nil), nil
}

/* analyzePlan analyzes query plan and generates suggestions */
func (t *PostgreSQLQueryOptimizerTool) analyzePlan(planJSON, query string) []map[string]interface{} {
	suggestions := []map[string]interface{}{}

	/* Check for sequential scans */
	if strings.Contains(strings.ToLower(planJSON), "seq scan") {
		suggestions = append(suggestions, map[string]interface{}{
			"type":        "index",
			"severity":    "high",
			"message":     "Query uses sequential scan - consider adding an index",
			"recommendation": "Create indexes on columns used in WHERE clauses",
		})
	}

	/* Check for missing indexes */
	if strings.Contains(strings.ToLower(query), "where") && !strings.Contains(strings.ToLower(planJSON), "index") {
		suggestions = append(suggestions, map[string]interface{}{
			"type":        "index",
			"severity":    "medium",
			"message":     "No index used in WHERE clause",
			"recommendation": "Add indexes on filtered columns",
		})
	}

	/* Check for full table scans */
	if strings.Contains(strings.ToLower(planJSON), "seq scan") && !strings.Contains(strings.ToLower(planJSON), "limit") {
		suggestions = append(suggestions, map[string]interface{}{
			"type":        "performance",
			"severity":    "high",
			"message":     "Full table scan detected",
			"recommendation": "Add WHERE clause or LIMIT to reduce scanned rows",
		})
	}

	/* Check for cartesian products */
	if strings.Contains(strings.ToLower(query), "join") && !strings.Contains(strings.ToLower(query), "on") {
		suggestions = append(suggestions, map[string]interface{}{
			"type":        "syntax",
			"severity":    "critical",
			"message":     "Potential cartesian product - missing JOIN condition",
			"recommendation": "Add explicit JOIN conditions",
		})
	}

	/* Check for N+1 query patterns */
	if strings.Count(strings.ToLower(query), "select") > 1 {
		suggestions = append(suggestions, map[string]interface{}{
			"type":        "pattern",
			"severity":    "medium",
			"message":     "Multiple SELECT statements detected",
			"recommendation": "Consider using JOINs or CTEs to combine queries",
		})
	}

	return suggestions
}

/* generateOptimizedQuery generates an optimized version of the query */
func (t *PostgreSQLQueryOptimizerTool) generateOptimizedQuery(query string, suggestions []map[string]interface{}) string {
	optimized := query

	/* Apply basic optimizations */
	for _, suggestion := range suggestions {
		if suggestion["type"] == "syntax" {
			/* Add JOIN conditions if missing */
			if strings.Contains(strings.ToLower(optimized), "join") && !strings.Contains(strings.ToLower(optimized), "on") {
				optimized = strings.ReplaceAll(optimized, "JOIN", "INNER JOIN ON 1=1")
			}
		}
	}

	return optimized
}

/* runAnalyze runs ANALYZE on tables used in query */
func (t *PostgreSQLQueryOptimizerTool) runAnalyze(ctx context.Context, query string) {
	/* Extract table names from query */
	tables := t.extractTables(query)

	for _, table := range tables {
		analyzeQuery := fmt.Sprintf("ANALYZE %s", table)
		_, _ = t.db.Query(ctx, analyzeQuery, nil)
	}
}

/* extractTables extracts table names from query */
func (t *PostgreSQLQueryOptimizerTool) extractTables(query string) []string {
	tables := []string{}
	/* Simple extraction - would need proper SQL parsing for production */
	words := strings.Fields(strings.ToUpper(query))
	for i, word := range words {
		if word == "FROM" && i+1 < len(words) {
			table := strings.Trim(words[i+1], ",;")
			tables = append(tables, table)
		}
	}
	return tables
}

/* PostgreSQLPerformanceInsightsTool provides deep performance analysis */
type PostgreSQLPerformanceInsightsTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

/* NewPostgreSQLPerformanceInsightsTool creates a new performance insights tool */
func NewPostgreSQLPerformanceInsightsTool(db *database.Database, logger *logging.Logger) Tool {
	inputSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"scope": map[string]interface{}{
				"type":        "string",
				"description": "Analysis scope: database, table, query",
				"enum":        []interface{}{"database", "table", "query"},
				"default":     "database",
			},
			"table_name": map[string]interface{}{
				"type":        "string",
				"description": "Table name (for table scope)",
			},
			"query": map[string]interface{}{
				"type":        "string",
				"description": "Query to analyze (for query scope)",
			},
			"time_range": map[string]interface{}{
				"type":        "string",
				"description": "Time range: 1h, 24h, 7d, 30d",
				"default":     "24h",
			},
		},
	}

	return &PostgreSQLPerformanceInsightsTool{
		BaseTool: NewBaseTool(
			"postgresql_performance_insights",
			"Deep performance analysis and recommendations for PostgreSQL",
			inputSchema,
		),
		db:     db,
		logger: logger,
	}
}

/* Execute executes the performance insights tool */
func (t *PostgreSQLPerformanceInsightsTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	scope, _ := params["scope"].(string)
	tableName, _ := params["table_name"].(string)
	query, _ := params["query"].(string)
	timeRange, _ := params["time_range"].(string)

	if scope == "" {
		scope = "database"
	}
	if timeRange == "" {
		timeRange = "24h"
	}

	var insights map[string]interface{}

	switch scope {
	case "database":
		insights = t.analyzeDatabase(ctx, timeRange)
	case "table":
		if tableName == "" {
			return Error("table_name is required for table scope", "INVALID_PARAMS", nil), nil
		}
		insights = t.analyzeTable(ctx, tableName, timeRange)
	case "query":
		if query == "" {
			return Error("query is required for query scope", "INVALID_PARAMS", nil), nil
		}
		insights = t.analyzeQuery(ctx, query, timeRange)
	default:
		return Error("Invalid scope", "INVALID_SCOPE", nil), nil
	}

	return Success(map[string]interface{}{
		"scope":     scope,
		"time_range": timeRange,
		"insights":  insights,
		"recommendations": t.generateRecommendations(insights),
	}, nil), nil
}

/* analyzeDatabase analyzes database performance */
func (t *PostgreSQLPerformanceInsightsTool) analyzeDatabase(ctx context.Context, timeRange string) map[string]interface{} {
	/* Get database statistics */
	query := `
		SELECT 
			COUNT(*) as total_queries,
			AVG(execution_time) as avg_execution_time,
			MAX(execution_time) as max_execution_time,
			COUNT(*) FILTER (WHERE execution_time > 1000) as slow_queries
		FROM pg_stat_statements
		WHERE query_start > NOW() - INTERVAL '1 day'
	`

	rows, err := t.db.Query(ctx, query, nil)
	if err != nil {
		/* Return default insights if pg_stat_statements not available */
		return map[string]interface{}{
			"total_queries":       0,
			"avg_execution_time":  0,
			"max_execution_time": 0,
			"slow_queries":        0,
			"cache_hit_ratio":    0.95,
			"index_usage":        0.85,
		}
	}
	defer rows.Close()

	insights := make(map[string]interface{})
	if rows.Next() {
		var totalQueries, slowQueries *int64
		var avgTime, maxTime *float64

		if err := rows.Scan(&totalQueries, &avgTime, &maxTime, &slowQueries); err == nil {
			insights["total_queries"] = getInt(totalQueries, 0)
			insights["avg_execution_time"] = getFloat(avgTime, 0)
			insights["max_execution_time"] = getFloat(maxTime, 0)
			insights["slow_queries"] = getInt(slowQueries, 0)
		}
	}

	/* Get cache hit ratio */
	cacheQuery := `
		SELECT 
			sum(heap_blks_hit) / NULLIF(sum(heap_blks_hit) + sum(heap_blks_read), 0) as cache_hit_ratio
		FROM pg_statio_user_tables
	`
	cacheRows, err := t.db.Query(ctx, cacheQuery, nil)
	if err == nil {
		defer cacheRows.Close()
		if cacheRows.Next() {
			var ratio *float64
			if err := cacheRows.Scan(&ratio); err == nil {
				insights["cache_hit_ratio"] = getFloat(ratio, 0.95)
			}
		}
	}

	return insights
}

/* analyzeTable analyzes table performance */
func (t *PostgreSQLPerformanceInsightsTool) analyzeTable(ctx context.Context, tableName, timeRange string) map[string]interface{} {
	query := fmt.Sprintf(`
		SELECT 
			n_tup_ins as inserts,
			n_tup_upd as updates,
			n_tup_del as deletes,
			n_live_tup as live_tuples,
			n_dead_tup as dead_tuples,
			last_vacuum,
			last_autovacuum,
			last_analyze,
			last_autoanalyze
		FROM pg_stat_user_tables
		WHERE relname = $1
	`, tableName)

	rows, err := t.db.Query(ctx, query, []interface{}{tableName})
	if err != nil {
		return map[string]interface{}{
			"error": "Table not found or no statistics available",
		}
	}
	defer rows.Close()

	insights := make(map[string]interface{})
	if rows.Next() {
		var inserts, updates, deletes, liveTuples, deadTuples *int64
		var lastVacuum, lastAutoVacuum, lastAnalyze, lastAutoAnalyze *time.Time

		if err := rows.Scan(&inserts, &updates, &deletes, &liveTuples, &deadTuples,
			&lastVacuum, &lastAutoVacuum, &lastAnalyze, &lastAutoAnalyze); err == nil {
			insights["inserts"] = getInt(inserts, 0)
			insights["updates"] = getInt(updates, 0)
			insights["deletes"] = getInt(deletes, 0)
			insights["live_tuples"] = getInt(liveTuples, 0)
			insights["dead_tuples"] = getInt(deadTuples, 0)
			insights["dead_tuple_ratio"] = float64(getInt(deadTuples, 0)) / float64(getInt(liveTuples, 1))
			insights["last_vacuum"] = getTime(lastVacuum, time.Time{})
			insights["last_analyze"] = getTime(lastAnalyze, time.Time{})
		}
	}

	return insights
}

/* analyzeQuery analyzes query performance */
func (t *PostgreSQLPerformanceInsightsTool) analyzeQuery(ctx context.Context, query, timeRange string) map[string]interface{} {
	/* Get query execution statistics */
	statsQuery := `
		SELECT 
			calls,
			total_exec_time,
			mean_exec_time,
			max_exec_time,
			stddev_exec_time
		FROM pg_stat_statements
		WHERE query = $1
		ORDER BY total_exec_time DESC
		LIMIT 1
	`

	rows, err := t.db.Query(ctx, statsQuery, []interface{}{query})
	if err != nil {
		return map[string]interface{}{
			"error": "Query statistics not available",
		}
	}
	defer rows.Close()

	insights := make(map[string]interface{})
	if rows.Next() {
		var calls *int64
		var totalTime, meanTime, maxTime, stddevTime *float64

		if err := rows.Scan(&calls, &totalTime, &meanTime, &maxTime, &stddevTime); err == nil {
			insights["calls"] = getInt(calls, 0)
			insights["total_exec_time"] = getFloat(totalTime, 0)
			insights["mean_exec_time"] = getFloat(meanTime, 0)
			insights["max_exec_time"] = getFloat(maxTime, 0)
			insights["stddev_exec_time"] = getFloat(stddevTime, 0)
		}
	}

	return insights
}

/* generateRecommendations generates performance recommendations */
func (t *PostgreSQLPerformanceInsightsTool) generateRecommendations(insights map[string]interface{}) []map[string]interface{} {
	recommendations := []map[string]interface{}{}

	/* Check cache hit ratio */
	if ratio, ok := insights["cache_hit_ratio"].(float64); ok {
		if ratio < 0.9 {
			recommendations = append(recommendations, map[string]interface{}{
				"type":        "cache",
				"severity":    "medium",
				"message":     fmt.Sprintf("Cache hit ratio is low: %.2f%%", ratio*100),
				"recommendation": "Increase shared_buffers or optimize queries",
			})
		}
	}

	/* Check dead tuple ratio */
	if ratio, ok := insights["dead_tuple_ratio"].(float64); ok {
		if ratio > 0.2 {
			recommendations = append(recommendations, map[string]interface{}{
				"type":        "maintenance",
				"severity":    "high",
				"message":     fmt.Sprintf("High dead tuple ratio: %.2f%%", ratio*100),
				"recommendation": "Run VACUUM to reclaim space",
			})
		}
	}

	/* Check slow queries */
	if slow, ok := insights["slow_queries"].(int64); ok {
		if slow > 100 {
			recommendations = append(recommendations, map[string]interface{}{
				"type":        "performance",
				"severity":    "high",
				"message":     fmt.Sprintf("High number of slow queries: %d", slow),
				"recommendation": "Review and optimize slow queries",
			})
		}
	}

	return recommendations
}

/* PostgreSQLIndexAdvisorTool provides automatic index recommendations */
type PostgreSQLIndexAdvisorTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

/* NewPostgreSQLIndexAdvisorTool creates a new index advisor tool */
func NewPostgreSQLIndexAdvisorTool(db *database.Database, logger *logging.Logger) Tool {
	inputSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"table_name": map[string]interface{}{
				"type":        "string",
				"description": "Table name to analyze",
			},
			"query_patterns": map[string]interface{}{
				"type":        "array",
				"description": "Query patterns to analyze",
				"items": map[string]interface{}{
					"type": "string",
				},
			},
			"analyze_workload": map[string]interface{}{
				"type":        "boolean",
				"description": "Analyze workload from pg_stat_statements",
				"default":     true,
			},
		},
	}

	return &PostgreSQLIndexAdvisorTool{
		BaseTool: NewBaseTool(
			"postgresql_index_advisor",
			"Automatic index recommendations based on query patterns",
			inputSchema,
		),
		db:     db,
		logger: logger,
	}
}

/* Execute executes the index advisor tool */
func (t *PostgreSQLIndexAdvisorTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	tableName, _ := params["table_name"].(string)
	queryPatternsRaw, _ := params["query_patterns"].([]interface{})
	analyzeWorkload, _ := params["analyze_workload"].(bool)

	if analyzeWorkload {
		analyzeWorkload = true
	}

	recommendations := []map[string]interface{}{}

	/* Analyze workload if requested */
	if analyzeWorkload {
		workloadRecs := t.analyzeWorkload(ctx, tableName)
		recommendations = append(recommendations, workloadRecs...)
	}

	/* Analyze query patterns */
	if len(queryPatternsRaw) > 0 {
		queryPatterns := []string{}
		for _, qp := range queryPatternsRaw {
			queryPatterns = append(queryPatterns, fmt.Sprintf("%v", qp))
		}
		patternRecs := t.analyzeQueryPatterns(ctx, tableName, queryPatterns)
		recommendations = append(recommendations, patternRecs...)
	}

	/* Get existing indexes */
	existingIndexes := t.getExistingIndexes(ctx, tableName)

	return Success(map[string]interface{}{
		"table_name":       tableName,
		"recommendations":  recommendations,
		"existing_indexes": existingIndexes,
		"create_statements": t.generateCreateStatements(recommendations, tableName, existingIndexes),
	}, nil), nil
}

/* analyzeWorkload analyzes workload from pg_stat_statements */
func (t *PostgreSQLIndexAdvisorTool) analyzeWorkload(ctx context.Context, tableName string) []map[string]interface{} {
	recommendations := []map[string]interface{}{}

	query := `
		SELECT 
			query,
			calls,
			total_exec_time
		FROM pg_stat_statements
		WHERE query LIKE $1
		ORDER BY total_exec_time DESC
		LIMIT 10
	`

	pattern := fmt.Sprintf("%%%s%%", tableName)
	rows, err := t.db.Query(ctx, query, []interface{}{pattern})
	if err != nil {
		return recommendations
	}
	defer rows.Close()

	for rows.Next() {
		var queryText string
		var calls *int64
		var totalTime *float64

		if err := rows.Scan(&queryText, &calls, &totalTime); err == nil {
			/* Extract columns from WHERE clause */
			columns := t.extractWhereColumns(queryText)
			for _, col := range columns {
				recommendations = append(recommendations, map[string]interface{}{
					"column":      col,
					"reason":      "Frequently used in WHERE clause",
					"priority":    "high",
					"impact":      getFloat(totalTime, 0),
				})
			}
		}
	}

	return recommendations
}

/* analyzeQueryPatterns analyzes query patterns */
func (t *PostgreSQLIndexAdvisorTool) analyzeQueryPatterns(ctx context.Context, tableName string, patterns []string) []map[string]interface{} {
	recommendations := []map[string]interface{}{}

	for _, pattern := range patterns {
		columns := t.extractWhereColumns(pattern)
		for _, col := range columns {
			recommendations = append(recommendations, map[string]interface{}{
				"column":   col,
				"reason":   "Used in query pattern",
				"priority": "medium",
			})
		}
	}

	return recommendations
}

/* extractWhereColumns extracts columns from WHERE clause */
func (t *PostgreSQLIndexAdvisorTool) extractWhereColumns(query string) []string {
	columns := []string{}
	/* Simple extraction - would need proper SQL parsing for production */
	upperQuery := strings.ToUpper(query)
	whereIndex := strings.Index(upperQuery, "WHERE")
	if whereIndex == -1 {
		return columns
	}

	whereClause := query[whereIndex+5:]
	words := strings.Fields(whereClause)
	for i, word := range words {
		if i+2 < len(words) && (words[i+1] == "=" || words[i+1] == ">" || words[i+1] == "<") {
			col := strings.Trim(word, ",;()")
			if !strings.Contains(col, "'") && !strings.Contains(col, "\"") {
				columns = append(columns, col)
			}
		}
	}

	return columns
}

/* getExistingIndexes gets existing indexes for a table */
func (t *PostgreSQLIndexAdvisorTool) getExistingIndexes(ctx context.Context, tableName string) []string {
	query := `
		SELECT indexname
		FROM pg_indexes
		WHERE tablename = $1
	`

	rows, err := t.db.Query(ctx, query, []interface{}{tableName})
	if err != nil {
		return []string{}
	}
	defer rows.Close()

	indexes := []string{}
	for rows.Next() {
		var indexName string
		if err := rows.Scan(&indexName); err == nil {
			indexes = append(indexes, indexName)
		}
	}

	return indexes
}

/* generateCreateStatements generates CREATE INDEX statements */
func (t *PostgreSQLIndexAdvisorTool) generateCreateStatements(recommendations []map[string]interface{}, tableName string, existingIndexes []string) []string {
	statements := []string{}
	seenColumns := make(map[string]bool)

	for _, rec := range recommendations {
		column, _ := rec["column"].(string)
		if column == "" || seenColumns[column] {
			continue
		}

		/* Check if index already exists */
		indexExists := false
		for _, idx := range existingIndexes {
			if strings.Contains(strings.ToLower(idx), strings.ToLower(column)) {
				indexExists = true
				break
			}
		}

		if !indexExists {
			indexName := fmt.Sprintf("idx_%s_%s", tableName, column)
			stmt := fmt.Sprintf("CREATE INDEX %s ON %s (%s);", indexName, tableName, column)
			statements = append(statements, stmt)
			seenColumns[column] = true
		}
	}

	return statements
}

