/*-------------------------------------------------------------------------
 *
 * debug.go
 *    Enhanced debugging tools for NeuronMCP
 *
 * Provides tools for debugging tool calls, query plans, and request tracing.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/debugging/debug.go
 *
 *-------------------------------------------------------------------------
 */

package debugging

import (
	"context"
	"fmt"
	"time"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/pkg/mcp"
)

/* ToolRegistryInterface provides access to tools without importing tools package */
type ToolRegistryInterface interface {
	GetTool(name string) ToolInterface
}

/* ToolInterface represents a tool that can be executed */
type ToolInterface interface {
	Execute(ctx context.Context, arguments map[string]interface{}) (*ToolResult, error)
}

/* DebugToolCallTool executes a tool with detailed logging and tracing */
type DebugToolCallTool struct {
	baseTool *BaseToolWrapper
	toolRegistry ToolRegistryInterface
	logger       *logging.Logger
}

/* NewDebugToolCallTool creates a new debug tool call tool */
func NewDebugToolCallTool(toolRegistry ToolRegistryInterface, logger *logging.Logger) *DebugToolCallTool {
	return &DebugToolCallTool{
		baseTool: &BaseToolWrapper{
			name:        "debug_tool_call",
			description: "Execute a tool with detailed logging and debugging information",
			inputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"tool_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the tool to execute",
					},
					"arguments": map[string]interface{}{
						"type":        "object",
						"description": "Tool arguments",
					},
					"enable_trace": map[string]interface{}{
						"type":        "boolean",
						"description": "Enable detailed tracing",
						"default":     true,
					},
					"log_level": map[string]interface{}{
						"type":        "string",
						"description": "Log level: debug, info, warn, error",
						"default":     "debug",
					},
				},
				"required": []interface{}{"tool_name", "arguments"},
			},
			version: "2.0.0",
		},
		toolRegistry: toolRegistry,
		logger:       logger,
	}
}

/* Name returns the tool name */
func (t *DebugToolCallTool) Name() string { return t.baseTool.Name() }

/* Description returns the tool description */
func (t *DebugToolCallTool) Description() string { return t.baseTool.Description() }

/* InputSchema returns the input schema */
func (t *DebugToolCallTool) InputSchema() map[string]interface{} { return t.baseTool.InputSchema() }

/* Version returns the tool version */
func (t *DebugToolCallTool) Version() string { return t.baseTool.Version() }

/* OutputSchema returns the output schema */
func (t *DebugToolCallTool) OutputSchema() map[string]interface{} { return t.baseTool.OutputSchema() }

/* Deprecated returns whether the tool is deprecated */
func (t *DebugToolCallTool) Deprecated() bool { return t.baseTool.Deprecated() }

/* Deprecation returns deprecation information */
func (t *DebugToolCallTool) Deprecation() *mcp.DeprecationInfo { return t.baseTool.Deprecation() }

/* Execute executes a tool with detailed debugging */
func (t *DebugToolCallTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	toolName, _ := params["tool_name"].(string)
	arguments, _ := params["arguments"].(map[string]interface{})
	enableTrace, _ := params["enable_trace"].(bool)
	if !enableTrace {
		enableTrace = true /* Default to true */
	}
	logLevel, _ := params["log_level"].(string)
	if logLevel == "" {
		logLevel = "debug"
	}

	if toolName == "" {
		return errorResult("tool_name is required", "VALIDATION_ERROR", nil), nil
	}

	/* Get tool from registry */
	tool := t.toolRegistry.GetTool(toolName)
	if tool == nil {
		return errorResult(fmt.Sprintf("tool not found: %s", toolName), "TOOL_NOT_FOUND", nil), nil
	}

	/* Create debug context with tracing */
	debugCtx := context.WithValue(ctx, "debug_mode", true)
	debugCtx = context.WithValue(debugCtx, "trace_enabled", enableTrace)
	debugCtx = context.WithValue(debugCtx, "log_level", logLevel)

	/* Log execution start */
	startTime := time.Now()
	if t.logger != nil {
		t.logger.Info("Debug tool execution started", map[string]interface{}{
			"tool_name":    toolName,
			"arguments":    arguments,
			"trace_enabled": enableTrace,
			"log_level":    logLevel,
		})
	}

	/* Execute tool */
	result, err := tool.Execute(debugCtx, arguments)

	/* Calculate execution time */
	duration := time.Since(startTime)

	/* Build debug information */
	debugInfo := map[string]interface{}{
		"tool_name":     toolName,
		"arguments":     arguments,
		"execution_time_ms": duration.Milliseconds(),
		"success":       err == nil && (result == nil || result.Success),
		"timestamp":     startTime,
	}

	if err != nil {
		debugInfo["error"] = err.Error()
	}

	if result != nil {
		debugInfo["result_success"] = result.Success
		if result.Error != nil {
			debugInfo["result_error"] = map[string]interface{}{
				"message": result.Error.Message,
				"code":    result.Error.Code,
			}
		}
		if enableTrace {
			debugInfo["result_data"] = result.Data
			debugInfo["result_metadata"] = result.Metadata
		}
	}

	/* Log execution end */
	if t.logger != nil {
		t.logger.Info("Debug tool execution completed", debugInfo)
	}

	return successResult(map[string]interface{}{
		"debug_info": debugInfo,
		"result":     result,
	}), nil
}

/* DebugQueryPlanTool provides detailed query execution plans */
type DebugQueryPlanTool struct {
	baseTool *BaseToolWrapper
	db       *database.Database
	logger   *logging.Logger
}

/* NewDebugQueryPlanTool creates a new debug query plan tool */
func NewDebugQueryPlanTool(db *database.Database, logger *logging.Logger) *DebugQueryPlanTool {
	return &DebugQueryPlanTool{
		baseTool: &BaseToolWrapper{
			name:        "debug_query_plan",
			description: "Get detailed query execution plan with statistics",
			inputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]interface{}{
						"type":        "string",
						"description": "SQL query to analyze",
					},
					"analyze": map[string]interface{}{
						"type":        "boolean",
						"description": "Execute ANALYZE to get actual execution statistics",
						"default":     false,
					},
					"verbose": map[string]interface{}{
						"type":        "boolean",
						"description": "Include verbose output",
						"default":     false,
					},
					"buffers": map[string]interface{}{
						"type":        "boolean",
						"description": "Include buffer usage statistics",
						"default":     false,
					},
				},
				"required": []interface{}{"query"},
			},
			version: "2.0.0",
		},
		db:     db,
		logger: logger,
	}
}

/* Name returns the tool name */
func (t *DebugQueryPlanTool) Name() string { return t.baseTool.Name() }

/* Description returns the tool description */
func (t *DebugQueryPlanTool) Description() string { return t.baseTool.Description() }

/* InputSchema returns the input schema */
func (t *DebugQueryPlanTool) InputSchema() map[string]interface{} { return t.baseTool.InputSchema() }

/* Version returns the tool version */
func (t *DebugQueryPlanTool) Version() string { return t.baseTool.Version() }

/* OutputSchema returns the output schema */
func (t *DebugQueryPlanTool) OutputSchema() map[string]interface{} { return t.baseTool.OutputSchema() }

/* Deprecated returns whether the tool is deprecated */
func (t *DebugQueryPlanTool) Deprecated() bool { return t.baseTool.Deprecated() }

/* Deprecation returns deprecation information */
func (t *DebugQueryPlanTool) Deprecation() *mcp.DeprecationInfo { return t.baseTool.Deprecation() }

/* Execute gets detailed query plan */
func (t *DebugQueryPlanTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	query, _ := params["query"].(string)
	analyze, _ := params["analyze"].(bool)
	verbose, _ := params["verbose"].(bool)
	buffers, _ := params["buffers"].(bool)

	if query == "" {
		return errorResult("query is required", "VALIDATION_ERROR", nil), nil
	}

	/* Build EXPLAIN query */
	explainQuery := "EXPLAIN "
	if analyze {
		explainQuery += "ANALYZE "
	}
	if verbose {
		explainQuery += "VERBOSE "
	}
	if buffers {
		explainQuery += "BUFFERS "
	}
	explainQuery += query

	/* Execute EXPLAIN */
	rows, err := t.db.Query(ctx, explainQuery, nil)
	if err != nil {
		return errorResult(fmt.Sprintf("failed to get query plan: %v", err), "QUERY_ERROR", nil), nil
	}
	defer rows.Close()

	/* Read plan output */
	planLines := []string{}
	for rows.Next() {
		var line string
		if err := rows.Scan(&line); err == nil {
			planLines = append(planLines, line)
		}
	}

	/* Get additional statistics if ANALYZE was used */
	stats := map[string]interface{}{}
	if analyze {
		/* Get query statistics from pg_stat_statements if available */
		statsQuery := `
			SELECT 
				calls,
				total_exec_time,
				mean_exec_time,
				max_exec_time,
				stddev_exec_time
			FROM pg_stat_statements
			WHERE query = $1
			LIMIT 1
		`
		statsRows, err := t.db.Query(ctx, statsQuery, []interface{}{query})
		if err == nil {
			defer statsRows.Close()
			if statsRows.Next() {
				var calls, totalTime, meanTime, maxTime, stddevTime interface{}
				if err := statsRows.Scan(&calls, &totalTime, &meanTime, &maxTime, &stddevTime); err == nil {
					stats["calls"] = calls
					stats["total_exec_time_ms"] = totalTime
					stats["mean_exec_time_ms"] = meanTime
					stats["max_exec_time_ms"] = maxTime
					stats["stddev_exec_time_ms"] = stddevTime
				}
			}
		}
	}

	return successResult(map[string]interface{}{
		"query":      query,
		"plan":       planLines,
		"plan_text":  fmt.Sprintf("%s", planLines),
		"statistics": stats,
		"options": map[string]interface{}{
			"analyze":  analyze,
			"verbose":  verbose,
			"buffers":  buffers,
		},
	}), nil
}

/* MonitorActiveConnectionsTool monitors active database connections */
type MonitorActiveConnectionsTool struct {
	baseTool *BaseToolWrapper
	db     *database.Database
	logger *logging.Logger
}

/* NewMonitorActiveConnectionsTool creates a new monitor connections tool */
func NewMonitorActiveConnectionsTool(db *database.Database, logger *logging.Logger) *MonitorActiveConnectionsTool {
	return &MonitorActiveConnectionsTool{
		baseTool: &BaseToolWrapper{
			name:        "monitor_active_connections",
			description: "Monitor active database connections with detailed information",
			inputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"include_idle": map[string]interface{}{
						"type":        "boolean",
						"description": "Include idle connections",
						"default":     false,
					},
					"min_duration_seconds": map[string]interface{}{
						"type":        "number",
						"description": "Minimum connection duration to include",
						"default":     0,
					},
				},
			},
			version: "2.0.0",
		},
		db:     db,
		logger: logger,
	}
}

/* Name returns the tool name */
func (t *MonitorActiveConnectionsTool) Name() string { return t.baseTool.Name() }

/* Description returns the tool description */
func (t *MonitorActiveConnectionsTool) Description() string { return t.baseTool.Description() }

/* InputSchema returns the input schema */
func (t *MonitorActiveConnectionsTool) InputSchema() map[string]interface{} { return t.baseTool.InputSchema() }

/* Version returns the tool version */
func (t *MonitorActiveConnectionsTool) Version() string { return t.baseTool.Version() }

/* OutputSchema returns the output schema */
func (t *MonitorActiveConnectionsTool) OutputSchema() map[string]interface{} { return t.baseTool.OutputSchema() }

/* Deprecated returns whether the tool is deprecated */
func (t *MonitorActiveConnectionsTool) Deprecated() bool { return t.baseTool.Deprecated() }

/* Deprecation returns deprecation information */
func (t *MonitorActiveConnectionsTool) Deprecation() *mcp.DeprecationInfo { return t.baseTool.Deprecation() }

/* Execute monitors connections */
func (t *MonitorActiveConnectionsTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	includeIdle, _ := params["include_idle"].(bool)
	minDuration, _ := params["min_duration_seconds"].(float64)

	query := `
		SELECT 
			pid,
			usename,
			application_name,
			client_addr,
			state,
			query_start,
			state_change,
			EXTRACT(EPOCH FROM (NOW() - query_start)) as query_duration,
			EXTRACT(EPOCH FROM (NOW() - state_change)) as state_duration,
			LEFT(query, 100) as query_preview,
			wait_event_type,
			wait_event
		FROM pg_stat_activity
		WHERE datname = current_database()
	`

	conditions := []string{}
	if !includeIdle {
		conditions = append(conditions, "state != 'idle'")
	}
	if minDuration > 0 {
		conditions = append(conditions, fmt.Sprintf("EXTRACT(EPOCH FROM (NOW() - query_start)) >= %f", minDuration))
	}

	if len(conditions) > 0 {
		query += " AND " + fmt.Sprintf("(%s)", fmt.Sprintf("%s", conditions))
	}

	query += " ORDER BY query_start DESC"

	rows, err := t.db.Query(ctx, query, nil)
	if err != nil {
		return errorResult(fmt.Sprintf("failed to query connections: %v", err), "QUERY_ERROR", nil), nil
	}
	defer rows.Close()

	connections := []map[string]interface{}{}
	for rows.Next() {
		var pid, usename, appName, clientAddr, state, queryPreview, waitEventType, waitEvent interface{}
		var queryStart, stateChange, queryDuration, stateDuration interface{}

		if err := rows.Scan(&pid, &usename, &appName, &clientAddr, &state, &queryStart, &stateChange, &queryDuration, &stateDuration, &queryPreview, &waitEventType, &waitEvent); err == nil {
			connections = append(connections, map[string]interface{}{
				"pid":             pid,
				"username":        usename,
				"application_name": appName,
				"client_address":  clientAddr,
				"state":           state,
				"query_start":     queryStart,
				"state_change":    stateChange,
				"query_duration_seconds": queryDuration,
				"state_duration_seconds": stateDuration,
				"query_preview":   queryPreview,
				"wait_event_type": waitEventType,
				"wait_event":      waitEvent,
			})
		}
	}

	/* Get summary statistics */
	summaryQuery := `
		SELECT 
			COUNT(*) as total,
			COUNT(*) FILTER (WHERE state = 'active') as active,
			COUNT(*) FILTER (WHERE state = 'idle') as idle,
			COUNT(*) FILTER (WHERE state = 'idle in transaction') as idle_in_transaction,
			MAX(EXTRACT(EPOCH FROM (NOW() - query_start))) as max_query_duration
		FROM pg_stat_activity
		WHERE datname = current_database()
	`

	summaryRows, err := t.db.Query(ctx, summaryQuery, nil)
	summary := map[string]interface{}{}
	if err == nil {
		defer summaryRows.Close()
		if summaryRows.Next() {
			var total, active, idle, idleInTrans, maxDuration interface{}
			if err := summaryRows.Scan(&total, &active, &idle, &idleInTrans, &maxDuration); err == nil {
				summary = map[string]interface{}{
					"total":                total,
					"active":               active,
					"idle":                 idle,
					"idle_in_transaction":  idleInTrans,
					"max_query_duration_seconds": maxDuration,
				}
			}
		}
	}

	return successResult(map[string]interface{}{
		"connections": connections,
		"count":       len(connections),
		"summary":     summary,
		"timestamp":   time.Now(),
	}), nil
}

/* MonitorQueryPerformanceTool tracks slow queries */
type MonitorQueryPerformanceTool struct {
	baseTool *BaseToolWrapper
	db     *database.Database
	logger *logging.Logger
}

/* NewMonitorQueryPerformanceTool creates a new query performance monitor */
func NewMonitorQueryPerformanceTool(db *database.Database, logger *logging.Logger) *MonitorQueryPerformanceTool {
	return &MonitorQueryPerformanceTool{
		baseTool: &BaseToolWrapper{
			name: "monitor_query_performance",
			description: "Track and analyze slow queries and performance issues",
			inputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"min_duration_ms": map[string]interface{}{
						"type":        "number",
						"description": "Minimum query duration in milliseconds",
						"default":     1000,
					},
					"limit": map[string]interface{}{
						"type":        "number",
						"description": "Maximum number of queries to return",
						"default":     50,
					},
					"order_by": map[string]interface{}{
						"type":        "string",
						"description": "Order by: duration, calls, total_time",
						"default":     "duration",
					},
				},
			},
			version: "2.0.0",
		},
		db:     db,
		logger: logger,
	}
}

/* Name returns the tool name */
func (t *MonitorQueryPerformanceTool) Name() string { return t.baseTool.Name() }

/* Description returns the tool description */
func (t *MonitorQueryPerformanceTool) Description() string { return t.baseTool.Description() }

/* InputSchema returns the input schema */
func (t *MonitorQueryPerformanceTool) InputSchema() map[string]interface{} { return t.baseTool.InputSchema() }

/* Version returns the tool version */
func (t *MonitorQueryPerformanceTool) Version() string { return t.baseTool.Version() }

/* OutputSchema returns the output schema */
func (t *MonitorQueryPerformanceTool) OutputSchema() map[string]interface{} { return t.baseTool.OutputSchema() }

/* Deprecated returns whether the tool is deprecated */
func (t *MonitorQueryPerformanceTool) Deprecated() bool { return t.baseTool.Deprecated() }

/* Deprecation returns deprecation information */
func (t *MonitorQueryPerformanceTool) Deprecation() *mcp.DeprecationInfo { return t.baseTool.Deprecation() }

/* Execute monitors query performance */
func (t *MonitorQueryPerformanceTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	minDuration, _ := params["min_duration_ms"].(float64)
	if minDuration == 0 {
		minDuration = 1000
	}
	limit, _ := params["limit"].(float64)
	if limit == 0 {
		limit = 50
	}
	orderBy, _ := params["order_by"].(string)
	if orderBy == "" {
		orderBy = "duration"
	}

	/* Check if pg_stat_statements is available */
	checkQuery := `
		SELECT EXISTS (
			SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements'
		)
	`
	rows, err := t.db.Query(ctx, checkQuery, nil)
	hasStats := false
	if err == nil {
		defer rows.Close()
		if rows.Next() {
			rows.Scan(&hasStats)
		}
	}

	if !hasStats {
		return errorResult("pg_stat_statements extension is not installed. Install it to use query performance monitoring.", "EXTENSION_NOT_FOUND", map[string]interface{}{
			"suggestion": "Run: CREATE EXTENSION IF NOT EXISTS pg_stat_statements;",
		}), nil
	}

	/* Build order by clause */
	orderClause := "mean_exec_time DESC"
	switch orderBy {
	case "calls":
		orderClause = "calls DESC"
	case "total_time":
		orderClause = "total_exec_time DESC"
	}

	query := fmt.Sprintf(`
		SELECT 
			queryid,
			LEFT(query, 200) as query_preview,
			calls,
			total_exec_time,
			mean_exec_time,
			max_exec_time,
			stddev_exec_time,
			rows,
			100.0 * shared_blks_hit / NULLIF(shared_blks_hit + shared_blks_read, 0) AS hit_percent
		FROM pg_stat_statements
		WHERE mean_exec_time >= %f
		ORDER BY %s
		LIMIT %d
	`, minDuration, orderClause, int(limit))

	rows, err = t.db.Query(ctx, query, nil)
	if err != nil {
		return errorResult(fmt.Sprintf("failed to query performance stats: %v", err), "QUERY_ERROR", nil), nil
	}
	defer rows.Close()

	queries := []map[string]interface{}{}
	for rows.Next() {
		var queryID, queryPreview, calls, totalTime, meanTime, maxTime, stddevTime, rowsCount, hitPercent interface{}

		if err := rows.Scan(&queryID, &queryPreview, &calls, &totalTime, &meanTime, &maxTime, &stddevTime, &rowsCount, &hitPercent); err == nil {
			queries = append(queries, map[string]interface{}{
				"query_id":           queryID,
				"query_preview":      queryPreview,
				"calls":              calls,
				"total_exec_time_ms": totalTime,
				"mean_exec_time_ms":  meanTime,
				"max_exec_time_ms":   maxTime,
				"stddev_exec_time_ms": stddevTime,
				"rows":               rowsCount,
				"cache_hit_percent":  hitPercent,
			})
		}
	}

	return successResult(map[string]interface{}{
		"queries":    queries,
		"count":      len(queries),
		"min_duration_ms": minDuration,
		"timestamp":  time.Now(),
	}), nil
}

/* TraceRequestTool enables request tracing for debugging */
type TraceRequestTool struct {
	baseTool *BaseToolWrapper
	logger *logging.Logger
}

/* NewTraceRequestTool creates a new trace request tool */
func NewTraceRequestTool(logger *logging.Logger) *TraceRequestTool {
	return &TraceRequestTool{
		baseTool: &BaseToolWrapper{
			name: "trace_request",
			description: "Enable request tracing for debugging (returns trace ID for use in subsequent requests)",
			inputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"enable": map[string]interface{}{
						"type":        "boolean",
						"description": "Enable or disable tracing",
						"default":     true,
					},
					"trace_id": map[string]interface{}{
						"type":        "string",
						"description": "Optional trace ID (generated if not provided)",
					},
					"log_level": map[string]interface{}{
						"type":        "string",
						"description": "Log level for trace: debug, info, warn, error",
						"default":     "debug",
					},
				},
			},
			version: "2.0.0",
		},
		logger: logger,
	}
}

/* Name returns the tool name */
func (t *TraceRequestTool) Name() string { return t.baseTool.Name() }

/* Description returns the tool description */
func (t *TraceRequestTool) Description() string { return t.baseTool.Description() }

/* InputSchema returns the input schema */
func (t *TraceRequestTool) InputSchema() map[string]interface{} { return t.baseTool.InputSchema() }

/* Version returns the tool version */
func (t *TraceRequestTool) Version() string { return t.baseTool.Version() }

/* OutputSchema returns the output schema */
func (t *TraceRequestTool) OutputSchema() map[string]interface{} { return t.baseTool.OutputSchema() }

/* Deprecated returns whether the tool is deprecated */
func (t *TraceRequestTool) Deprecated() bool { return t.baseTool.Deprecated() }

/* Deprecation returns deprecation information */
func (t *TraceRequestTool) Deprecation() *mcp.DeprecationInfo { return t.baseTool.Deprecation() }

/* Execute enables tracing */
func (t *TraceRequestTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	enable, _ := params["enable"].(bool)
	if !enable {
		enable = true /* Default to true */
	}
	traceID, _ := params["trace_id"].(string)
	logLevel, _ := params["log_level"].(string)
	if logLevel == "" {
		logLevel = "debug"
	}

	if traceID == "" {
		traceID = fmt.Sprintf("trace-%d", time.Now().UnixNano())
	}

	if t.logger != nil {
		t.logger.Info("Request tracing enabled", map[string]interface{}{
			"trace_id":  traceID,
			"enabled":   enable,
			"log_level": logLevel,
		})
	}

	return successResult(map[string]interface{}{
		"trace_id":  traceID,
		"enabled":   enable,
		"log_level": logLevel,
		"message":   fmt.Sprintf("Use trace_id '%s' in subsequent requests to enable tracing", traceID),
		"usage":     "Include 'X-Trace-ID' header or 'trace_id' parameter in requests",
	}), nil
}
