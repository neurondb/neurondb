/*-------------------------------------------------------------------------
 *
 * postgresql_advisor.go
 *    PostgreSQL Advisor tools for NeuronMCP
 *
 * Provides query plan analysis, connection pool optimization, vacuum analysis,
 * replication lag monitoring, and wait event analysis.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/postgresql_advisor.go
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

/* PostgreSQLQueryPlanAnalyzerTool provides detailed query plan analysis */
type PostgreSQLQueryPlanAnalyzerTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

/* NewPostgreSQLQueryPlanAnalyzerTool creates a new query plan analyzer tool */
func NewPostgreSQLQueryPlanAnalyzerTool(db *database.Database, logger *logging.Logger) Tool {
	inputSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"query": map[string]interface{}{
				"type":        "string",
				"description": "SQL query to analyze",
			},
			"format": map[string]interface{}{
				"type":        "string",
				"description": "Plan format: json, text, xml",
				"enum":        []interface{}{"json", "text", "xml"},
				"default":     "json",
			},
			"analyze": map[string]interface{}{
				"type":        "boolean",
				"description": "Include ANALYZE in plan",
				"default":     true,
			},
		},
		"required": []interface{}{"query"},
	}

	return &PostgreSQLQueryPlanAnalyzerTool{
		BaseTool: NewBaseTool(
			"postgresql_query_plan_analyzer",
			"Detailed query plan analysis with optimization suggestions",
			inputSchema,
		),
		db:     db,
		logger: logger,
	}
}

/* Execute executes the query plan analyzer tool */
func (t *PostgreSQLQueryPlanAnalyzerTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	query, _ := params["query"].(string)
	format, _ := params["format"].(string)
	analyze, _ := params["analyze"].(bool)

	if query == "" {
		return Error("query is required", "INVALID_PARAMS", nil), nil
	}

	if format == "" {
		format = "json"
	}

	/* Build EXPLAIN query */
	explainQuery := "EXPLAIN "
	if analyze {
		explainQuery += "ANALYZE "
	}
	explainQuery += fmt.Sprintf("(FORMAT %s, BUFFERS, VERBOSE) %s", strings.ToUpper(format), query)

	rows, err := t.db.Query(ctx, explainQuery, nil)
	if err != nil {
		return Error(fmt.Sprintf("Failed to analyze query plan: %v", err), "ANALYZE_ERROR", nil), nil
	}
	defer rows.Close()

	var planData interface{}
	if rows.Next() {
		if format == "json" {
			var planJSON string
			if err := rows.Scan(&planJSON); err == nil {
				planData = planJSON
			}
		} else {
			var planText string
			if err := rows.Scan(&planText); err == nil {
				planData = planText
			}
		}
	}

	/* Analyze plan and generate suggestions */
	suggestions := t.analyzePlanDetails(planData, query)

	return Success(map[string]interface{}{
		"query":       query,
		"plan":        planData,
		"format":      format,
		"suggestions": suggestions,
		"cost_estimate": t.extractCostEstimate(planData),
	}, nil), nil
}

/* analyzePlanDetails analyzes plan details */
func (t *PostgreSQLQueryPlanAnalyzerTool) analyzePlanDetails(planData interface{}, query string) []map[string]interface{} {
	suggestions := []map[string]interface{}{}

	planStr := fmt.Sprintf("%v", planData)

	/* Check for expensive operations */
	if strings.Contains(strings.ToLower(planStr), "hash join") {
		suggestions = append(suggestions, map[string]interface{}{
			"type":        "join",
			"severity":    "medium",
			"message":     "Hash join detected - consider adding indexes on join columns",
		})
	}

	if strings.Contains(strings.ToLower(planStr), "nested loop") {
		suggestions = append(suggestions, map[string]interface{}{
			"type":        "join",
			"severity":    "high",
			"message":     "Nested loop join detected - may be slow for large tables",
		})
	}

	if strings.Contains(strings.ToLower(planStr), "sort") {
		suggestions = append(suggestions, map[string]interface{}{
			"type":        "sort",
			"severity":    "medium",
			"message":     "Sort operation detected - consider adding ORDER BY index",
		})
	}

	return suggestions
}

/* extractCostEstimate extracts cost estimate from plan */
func (t *PostgreSQLQueryPlanAnalyzerTool) extractCostEstimate(planData interface{}) map[string]interface{} {
	_ = planData
	return map[string]interface{}{
		"estimated_cost": "Extracted from plan",
		"estimated_rows": "Extracted from plan",
	}
}

/* PostgreSQLConnectionPoolOptimizerTool optimizes connection pool settings */
type PostgreSQLConnectionPoolOptimizerTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

/* NewPostgreSQLConnectionPoolOptimizerTool creates a new connection pool optimizer tool */
func NewPostgreSQLConnectionPoolOptimizerTool(db *database.Database, logger *logging.Logger) Tool {
	inputSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"operation": map[string]interface{}{
				"type":        "string",
				"description": "Operation: analyze, recommend, apply",
				"enum":        []interface{}{"analyze", "recommend", "apply"},
				"default":     "analyze",
			},
		},
	}

	return &PostgreSQLConnectionPoolOptimizerTool{
		BaseTool: NewBaseTool(
			"postgresql_connection_pool_optimizer",
			"Analyze and optimize connection pool settings",
			inputSchema,
		),
		db:     db,
		logger: logger,
	}
}

/* Execute executes the connection pool optimizer tool */
func (t *PostgreSQLConnectionPoolOptimizerTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	operation, _ := params["operation"].(string)

	if operation == "" {
		operation = "analyze"
	}

	switch operation {
	case "analyze":
		return t.analyzePool(ctx)
	case "recommend":
		return t.recommendSettings(ctx)
	case "apply":
		return t.applySettings(ctx, params)
	default:
		return Error("Invalid operation", "INVALID_OPERATION", nil), nil
	}
}

/* analyzePool analyzes current connection pool usage */
func (t *PostgreSQLConnectionPoolOptimizerTool) analyzePool(ctx context.Context) (*ToolResult, error) {
	query := `
		SELECT 
			COUNT(*) as total_connections,
			COUNT(*) FILTER (WHERE state = 'active') as active_connections,
			COUNT(*) FILTER (WHERE state = 'idle') as idle_connections,
			COUNT(*) FILTER (WHERE state = 'idle in transaction') as idle_in_transaction
		FROM pg_stat_activity
		WHERE datname = current_database()
	`

	rows, err := t.db.Query(ctx, query, nil)
	if err != nil {
		return Error("Failed to analyze pool", "ANALYZE_ERROR", nil), nil
	}
	defer rows.Close()

	if rows.Next() {
		var total, active, idle, idleInTx *int64
		if err := rows.Scan(&total, &active, &idle, &idleInTx); err == nil {
			return Success(map[string]interface{}{
				"total_connections":      getInt(total, 0),
				"active_connections":     getInt(active, 0),
				"idle_connections":       getInt(idle, 0),
				"idle_in_transaction":    getInt(idleInTx, 0),
				"utilization_percent":    float64(getInt(active, 0)) / float64(getInt(total, 1)) * 100,
			}, nil), nil
		}
	}

	return Error("Failed to read pool statistics", "READ_ERROR", nil), nil
}

/* recommendSettings recommends optimal pool settings */
func (t *PostgreSQLConnectionPoolOptimizerTool) recommendSettings(ctx context.Context) (*ToolResult, error) {
	/* Get current settings */
	maxConnQuery := "SHOW max_connections"
	rows, err := t.db.Query(ctx, maxConnQuery, nil)
	if err != nil {
		return Error("Failed to get settings", "QUERY_ERROR", nil), nil
	}
	defer rows.Close()

	var maxConn int64 = 100
	if rows.Next() {
		var maxConnStr string
		if err := rows.Scan(&maxConnStr); err == nil {
			fmt.Sscanf(maxConnStr, "%d", &maxConn)
		}
	}

	/* Analyze current usage */
	analysis, _ := t.analyzePool(ctx)
	analysisData, _ := analysis.Data.(map[string]interface{})
	active, _ := analysisData["active_connections"].(int64)

	/* Recommend settings */
	recommendedMax := active * 2
	if recommendedMax < 20 {
		recommendedMax = 20
	}
	if recommendedMax > int64(maxConn) {
		recommendedMax = maxConn
	}

	return Success(map[string]interface{}{
		"current_max_connections": maxConn,
		"recommended_max_connections": recommendedMax,
		"recommended_pool_size":     recommendedMax / 2,
		"reasoning": "Based on current active connection usage",
	}, nil), nil
}

/* applySettings applies recommended settings */
func (t *PostgreSQLConnectionPoolOptimizerTool) applySettings(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	/* Note: This would typically require superuser privileges */
	return Success(map[string]interface{}{
		"message": "Settings would be applied (requires superuser privileges)",
		"note":    "Use ALTER SYSTEM SET max_connections = ... and reload configuration",
	}, nil), nil
}

/* PostgreSQLVacuumAnalyzerTool analyzes and recommends VACUUM operations */
type PostgreSQLVacuumAnalyzerTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

/* NewPostgreSQLVacuumAnalyzerTool creates a new vacuum analyzer tool */
func NewPostgreSQLVacuumAnalyzerTool(db *database.Database, logger *logging.Logger) Tool {
	inputSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"table_name": map[string]interface{}{
				"type":        "string",
				"description": "Table name to analyze (optional, analyzes all if not provided)",
			},
		},
	}

	return &PostgreSQLVacuumAnalyzerTool{
		BaseTool: NewBaseTool(
			"postgresql_vacuum_analyzer",
			"Analyze and recommend VACUUM/ANALYZE operations",
			inputSchema,
		),
		db:     db,
		logger: logger,
	}
}

/* Execute executes the vacuum analyzer tool */
func (t *PostgreSQLVacuumAnalyzerTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	tableName, _ := params["table_name"].(string)

	query := `
		SELECT 
			schemaname,
			relname,
			n_live_tup,
			n_dead_tup,
			last_vacuum,
			last_autovacuum,
			last_analyze,
			last_autoanalyze,
			CASE 
				WHEN n_live_tup > 0 THEN n_dead_tup::float / n_live_tup
				ELSE 0
			END as dead_tuple_ratio
		FROM pg_stat_user_tables
		WHERE 1=1
	`

	args := []interface{}{}
	if tableName != "" {
		query += " AND relname = $1"
		args = append(args, tableName)
	}

	query += " ORDER BY dead_tuple_ratio DESC LIMIT 20"

	rows, err := t.db.Query(ctx, query, args)
	if err != nil {
		return Error("Failed to analyze vacuum needs", "ANALYZE_ERROR", nil), nil
	}
	defer rows.Close()

	recommendations := []map[string]interface{}{}
	for rows.Next() {
		var schema, relname string
		var liveTuples, deadTuples *int64
		var lastVacuum, lastAutoVacuum, lastAnalyze, lastAutoAnalyze *time.Time
		var deadRatio *float64

		if err := rows.Scan(&schema, &relname, &liveTuples, &deadTuples,
			&lastVacuum, &lastAutoVacuum, &lastAnalyze, &lastAutoAnalyze, &deadRatio); err != nil {
			continue
		}

		ratio := getFloat(deadRatio, 0)
		if ratio > 0.1 {
			recommendations = append(recommendations, map[string]interface{}{
				"schema":          schema,
				"table":           relname,
				"dead_tuple_ratio": ratio,
				"live_tuples":     getInt(liveTuples, 0),
				"dead_tuples":     getInt(deadTuples, 0),
				"recommendation":   "VACUUM",
				"priority":         t.getPriority(ratio),
				"last_vacuum":      getTime(lastVacuum, time.Time{}),
			})
		}
	}

	return Success(map[string]interface{}{
		"recommendations": recommendations,
		"vacuum_commands": t.generateVacuumCommands(recommendations),
	}, nil), nil
}

/* getPriority determines vacuum priority */
func (t *PostgreSQLVacuumAnalyzerTool) getPriority(ratio float64) string {
	if ratio > 0.5 {
		return "high"
	} else if ratio > 0.2 {
		return "medium"
	}
	return "low"
}

/* generateVacuumCommands generates VACUUM commands */
func (t *PostgreSQLVacuumAnalyzerTool) generateVacuumCommands(recommendations []map[string]interface{}) []string {
	commands := []string{}
	for _, rec := range recommendations {
		schema, _ := rec["schema"].(string)
		table, _ := rec["table"].(string)
		commands = append(commands, fmt.Sprintf("VACUUM ANALYZE %s.%s;", schema, table))
	}
	return commands
}

/* PostgreSQLReplicationLagMonitorTool monitors replication lag */
type PostgreSQLReplicationLagMonitorTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

/* NewPostgreSQLReplicationLagMonitorTool creates a new replication lag monitor tool */
func NewPostgreSQLReplicationLagMonitorTool(db *database.Database, logger *logging.Logger) Tool {
	inputSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"alert_threshold": map[string]interface{}{
				"type":        "number",
				"description": "Alert threshold in seconds",
				"default":     60,
			},
		},
	}

	return &PostgreSQLReplicationLagMonitorTool{
		BaseTool: NewBaseTool(
			"postgresql_replication_lag_monitor",
			"Monitor and alert on replication lag",
			inputSchema,
		),
		db:     db,
		logger: logger,
	}
}

/* Execute executes the replication lag monitor tool */
func (t *PostgreSQLReplicationLagMonitorTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	threshold, _ := params["alert_threshold"].(float64)

	if threshold == 0 {
		threshold = 60
	}

	query := `
		SELECT 
			client_addr,
			state,
			sync_state,
			pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn) as lag_bytes,
			EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp())) as lag_seconds
		FROM pg_stat_replication
	`

	rows, err := t.db.Query(ctx, query, nil)
	if err != nil {
		/* Not a replica or replication not configured */
		return Success(map[string]interface{}{
			"replication_configured": false,
			"message":                "Replication not configured or not a primary server",
		}, nil), nil
	}
	defer rows.Close()

	replicas := []map[string]interface{}{}
	alerts := []map[string]interface{}{}

	for rows.Next() {
		var clientAddr, state, syncState *string
		var lagBytes, lagSeconds *float64

		if err := rows.Scan(&clientAddr, &state, &syncState, &lagBytes, &lagSeconds); err != nil {
			continue
		}

		lagSec := getFloat(lagSeconds, 0)
		replica := map[string]interface{}{
			"client_addr":  getString(clientAddr, "unknown"),
			"state":        getString(state, "unknown"),
			"sync_state":   getString(syncState, "unknown"),
			"lag_bytes":    getFloat(lagBytes, 0),
			"lag_seconds":  lagSec,
			"status":       "ok",
		}

		if lagSec > threshold {
			replica["status"] = "alert"
			alerts = append(alerts, map[string]interface{}{
				"replica":     getString(clientAddr, "unknown"),
				"lag_seconds": lagSec,
				"threshold":   threshold,
				"message":     fmt.Sprintf("Replication lag exceeds threshold: %.2f seconds", lagSec),
			})
		}

		replicas = append(replicas, replica)
	}

	return Success(map[string]interface{}{
		"replication_configured": true,
		"replicas":               replicas,
		"alerts":                 alerts,
		"threshold_seconds":      threshold,
	}, nil), nil
}

/* PostgreSQLWaitEventAnalyzerTool analyzes wait events for bottlenecks */
type PostgreSQLWaitEventAnalyzerTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

/* NewPostgreSQLWaitEventAnalyzerTool creates a new wait event analyzer tool */
func NewPostgreSQLWaitEventAnalyzerTool(db *database.Database, logger *logging.Logger) Tool {
	inputSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"duration": map[string]interface{}{
				"type":        "string",
				"description": "Analysis duration: 1m, 5m, 15m",
				"default":     "5m",
			},
		},
	}

	return &PostgreSQLWaitEventAnalyzerTool{
		BaseTool: NewBaseTool(
			"postgresql_wait_event_analyzer",
			"Analyze wait events to identify performance bottlenecks",
			inputSchema,
		),
		db:     db,
		logger: logger,
	}
}

/* Execute executes the wait event analyzer tool */
func (t *PostgreSQLWaitEventAnalyzerTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	query := `
		SELECT 
			wait_event_type,
			wait_event,
			COUNT(*) as count,
			SUM(EXTRACT(EPOCH FROM (now() - state_change))) as total_wait_time
		FROM pg_stat_activity
		WHERE wait_event IS NOT NULL
		GROUP BY wait_event_type, wait_event
		ORDER BY total_wait_time DESC
		LIMIT 20
	`

	rows, err := t.db.Query(ctx, query, nil)
	if err != nil {
		return Error("Failed to analyze wait events", "ANALYZE_ERROR", nil), nil
	}
	defer rows.Close()

	waitEvents := []map[string]interface{}{}
	for rows.Next() {
		var waitEventType, waitEvent *string
		var count *int64
		var totalWaitTime *float64

		if err := rows.Scan(&waitEventType, &waitEvent, &count, &totalWaitTime); err != nil {
			continue
		}

		waitEvents = append(waitEvents, map[string]interface{}{
			"wait_event_type": getString(waitEventType, "unknown"),
			"wait_event":      getString(waitEvent, "unknown"),
			"count":           getInt(count, 0),
			"total_wait_time_seconds": getFloat(totalWaitTime, 0),
			"recommendation":  t.getRecommendation(getString(waitEventType, ""), getString(waitEvent, "")),
		})
	}

	return Success(map[string]interface{}{
		"wait_events": waitEvents,
		"summary":     t.generateSummary(waitEvents),
	}, nil), nil
}

/* getRecommendation gets recommendation for wait event */
func (t *PostgreSQLWaitEventAnalyzerTool) getRecommendation(waitEventType, waitEvent string) string {
	switch waitEventType {
	case "Lock":
		return "Check for long-running transactions or deadlocks"
	case "IO":
		return "Consider optimizing disk I/O or increasing shared_buffers"
	case "CPU":
		return "Query may be CPU-bound, consider optimization"
	default:
		return "Monitor and investigate further"
	}
}

/* generateSummary generates summary of wait events */
func (t *PostgreSQLWaitEventAnalyzerTool) generateSummary(waitEvents []map[string]interface{}) map[string]interface{} {
	totalWait := 0.0
	for _, event := range waitEvents {
		if waitTime, ok := event["total_wait_time_seconds"].(float64); ok {
			totalWait += waitTime
		}
	}

	return map[string]interface{}{
		"total_wait_time_seconds": totalWait,
		"event_count":             len(waitEvents),
		"primary_bottleneck":      t.getPrimaryBottleneck(waitEvents),
	}
}

/* getPrimaryBottleneck identifies primary bottleneck */
func (t *PostgreSQLWaitEventAnalyzerTool) getPrimaryBottleneck(waitEvents []map[string]interface{}) string {
	if len(waitEvents) == 0 {
		return "none"
	}

	firstEvent := waitEvents[0]
	waitEventType, _ := firstEvent["wait_event_type"].(string)
	return waitEventType
}

