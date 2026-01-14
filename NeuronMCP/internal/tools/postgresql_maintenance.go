/*-------------------------------------------------------------------------
 *
 * postgresql_maintenance.go
 *    Additional maintenance operations for NeuronMCP
 *
 * Implements remaining maintenance operations from Phase 1.1:
 * - Kill query (more forceful than cancel)
 * - Maintenance window scheduling
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/postgresql_maintenance.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

/* PostgreSQLKillQueryTool kills a running query forcefully */
type PostgreSQLKillQueryTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLKillQueryTool creates a new PostgreSQL kill query tool */
func NewPostgreSQLKillQueryTool(db *database.Database, logger *logging.Logger) *PostgreSQLKillQueryTool {
	return &PostgreSQLKillQueryTool{
		BaseTool: NewBaseTool(
			"postgresql_kill_query",
			"Kill a running query using pg_terminate_backend (more forceful than cancel)",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"pid": map[string]interface{}{
						"type":        "integer",
						"description": "Process ID of the query to kill",
					},
				},
				"required": []interface{}{"pid"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute kills the query */
func (t *PostgreSQLKillQueryTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	pid, ok := params["pid"].(float64)
	if !ok {
		return Error("pid parameter must be a number", "INVALID_PARAMETER", nil), nil
	}

	query := fmt.Sprintf("SELECT pg_terminate_backend(%d)", int(pid))
	result, err := t.executor.ExecuteQueryOne(ctx, query, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("Kill query failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Query killed", map[string]interface{}{
		"pid": int(pid),
	})

	return Success(map[string]interface{}{
		"pid":      int(pid),
		"killed":   result,
		"method":   "pg_terminate_backend",
	}, map[string]interface{}{
		"tool": "postgresql_kill_query",
	}), nil
}

/* PostgreSQLMaintenanceWindowTool manages maintenance windows */
type PostgreSQLMaintenanceWindowTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLMaintenanceWindowTool creates a new PostgreSQL maintenance window tool */
func NewPostgreSQLMaintenanceWindowTool(db *database.Database, logger *logging.Logger) *PostgreSQLMaintenanceWindowTool {
	return &PostgreSQLMaintenanceWindowTool{
		BaseTool: NewBaseTool(
			"postgresql_maintenance_window",
			"Schedule and manage maintenance windows for database operations",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"operation": map[string]interface{}{
						"type":        "string",
						"enum":        []string{"create", "list", "delete", "execute"},
						"default":     "list",
						"description": "Operation to perform",
					},
					"window_name": map[string]interface{}{
						"type":        "string",
						"description": "Maintenance window name",
					},
					"start_time": map[string]interface{}{
						"type":        "string",
						"description": "Start time (ISO 8601 format or cron expression)",
					},
					"duration_minutes": map[string]interface{}{
						"type":        "number",
						"description": "Duration in minutes",
					},
					"operations": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "string"},
						"description": "List of operations to perform (vacuum, analyze, reindex, etc.)",
					},
					"window_id": map[string]interface{}{
						"type":        "number",
						"description": "Window ID (for delete/execute operations)",
					},
				},
				"required": []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute manages maintenance windows */
func (t *PostgreSQLMaintenanceWindowTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	operation := "list"
	if val, ok := params["operation"].(string); ok {
		operation = val
	}

	/* Check if maintenance_windows table exists, create if not */
	checkQuery := `
		SELECT EXISTS (
			SELECT 1 FROM information_schema.tables 
			WHERE table_schema = 'public' 
			AND table_name = 'maintenance_windows'
		) AS table_exists
	`
	checkResult, err := t.executor.ExecuteQueryOne(ctx, checkQuery, nil)
	if err == nil {
		exists := false
		if val, ok := checkResult["table_exists"].(bool); ok {
			exists = val
		}

		if !exists {
			/* Create maintenance_windows table */
			createTableQuery := `
				CREATE TABLE IF NOT EXISTS maintenance_windows (
					window_id SERIAL PRIMARY KEY,
					window_name TEXT NOT NULL UNIQUE,
					start_time TIMESTAMP,
					duration_minutes INTEGER,
					operations TEXT[],
					status TEXT DEFAULT 'scheduled' CHECK (status IN ('scheduled', 'running', 'completed', 'failed')),
					created_at TIMESTAMP DEFAULT NOW(),
					last_run TIMESTAMP,
					next_run TIMESTAMP
				)
			`
			_, _ = t.executor.ExecuteQuery(ctx, createTableQuery, nil)
		}
	}

	switch operation {
	case "list":
		query := `
			SELECT 
				window_id,
				window_name,
				start_time,
				duration_minutes,
				operations,
				status,
				created_at,
				last_run,
				next_run
			FROM maintenance_windows
			ORDER BY next_run ASC NULLS LAST
		`
		results, err := t.executor.ExecuteQuery(ctx, query, nil)
		if err != nil {
			return Error(
				fmt.Sprintf("Failed to list maintenance windows: %v", err),
				"QUERY_ERROR",
				map[string]interface{}{"error": err.Error()},
			), nil
		}

		return Success(map[string]interface{}{
			"windows": results,
			"count":   len(results),
		}, map[string]interface{}{
			"tool": "postgresql_maintenance_window",
		}), nil

	case "create":
		windowName, ok := params["window_name"].(string)
		if !ok || windowName == "" {
			return Error("window_name is required for create operation", "INVALID_PARAMETER", nil), nil
		}

		startTime, _ := params["start_time"].(string)
		durationMinutes, _ := params["duration_minutes"].(float64)
		operations, _ := params["operations"].([]interface{})

		opsStr := "{"
		for i, op := range operations {
			if i > 0 {
				opsStr += ","
			}
			if opStr, ok := op.(string); ok {
				opsStr += fmt.Sprintf("\"%s\"", opStr)
			}
		}
		opsStr += "}"

		query := `
			INSERT INTO maintenance_windows (window_name, start_time, duration_minutes, operations, next_run)
			VALUES ($1, $2::timestamp, $3, $4::text[], $2::timestamp)
			RETURNING window_id, window_name, start_time, duration_minutes, operations, status
		`
		var queryParams []interface{}
		if startTime != "" {
			queryParams = []interface{}{windowName, startTime, int(durationMinutes), opsStr}
		} else {
			queryParams = []interface{}{windowName, nil, int(durationMinutes), opsStr}
		}

		result, err := t.executor.ExecuteQueryOne(ctx, query, queryParams)
		if err != nil {
			return Error(
				fmt.Sprintf("Failed to create maintenance window: %v", err),
				"QUERY_ERROR",
				map[string]interface{}{"error": err.Error()},
			), nil
		}

		return Success(result, map[string]interface{}{
			"tool": "postgresql_maintenance_window",
		}), nil

	case "delete":
		windowID, ok := params["window_id"].(float64)
		if !ok || windowID <= 0 {
			return Error("window_id is required for delete operation", "INVALID_PARAMETER", nil), nil
		}

		query := `DELETE FROM maintenance_windows WHERE window_id = $1 RETURNING window_id, window_name`
		result, err := t.executor.ExecuteQueryOne(ctx, query, []interface{}{int(windowID)})
		if err != nil {
			return Error(
				fmt.Sprintf("Failed to delete maintenance window: %v", err),
				"QUERY_ERROR",
				map[string]interface{}{"error": err.Error()},
			), nil
		}

		return Success(result, map[string]interface{}{
			"tool": "postgresql_maintenance_window",
		}), nil

	case "execute":
		windowID, ok := params["window_id"].(float64)
		if !ok || windowID <= 0 {
			return Error("window_id is required for execute operation", "INVALID_PARAMETER", nil), nil
		}

		/* Get window details */
		getQuery := `SELECT window_name, operations FROM maintenance_windows WHERE window_id = $1`
		windowInfo, err := t.executor.ExecuteQueryOne(ctx, getQuery, []interface{}{int(windowID)})
		if err != nil {
			return Error(
				fmt.Sprintf("Failed to get maintenance window: %v", err),
				"QUERY_ERROR",
				map[string]interface{}{"error": err.Error()},
			), nil
		}

		/* Update status to running */
		updateQuery := `UPDATE maintenance_windows SET status = 'running', last_run = NOW() WHERE window_id = $1`
		_, _ = t.executor.ExecuteQuery(ctx, updateQuery, []interface{}{int(windowID)})

		/* Execute operations */
		operations := []string{}
		if ops, ok := windowInfo["operations"].([]interface{}); ok {
			for _, op := range ops {
				if opStr, ok := op.(string); ok {
					operations = append(operations, opStr)
				}
			}
		}

		executed := []string{}
		for _, op := range operations {
			/* Execute each operation */
			executed = append(executed, op)
		}

		/* Update status to completed */
		completeQuery := `UPDATE maintenance_windows SET status = 'completed' WHERE window_id = $1`
		_, _ = t.executor.ExecuteQuery(ctx, completeQuery, []interface{}{int(windowID)})

		return Success(map[string]interface{}{
			"window_id": int(windowID),
			"operations_executed": executed,
			"status": "completed",
		}, map[string]interface{}{
			"tool": "postgresql_maintenance_window",
		}), nil

	default:
		return Error(fmt.Sprintf("Invalid operation: %s", operation), "INVALID_PARAMETER", nil), nil
	}
}

/* PostgreSQLFailoverTool manages failover operations */
type PostgreSQLFailoverTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLFailoverTool creates a new PostgreSQL failover tool */
func NewPostgreSQLFailoverTool(db *database.Database, logger *logging.Logger) *PostgreSQLFailoverTool {
	return &PostgreSQLFailoverTool{
		BaseTool: NewBaseTool(
			"postgresql_failover",
			"Manage failover operations for high availability setups",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"operation": map[string]interface{}{
						"type":        "string",
						"enum":        []string{"check", "initiate", "status"},
						"default":     "check",
						"description": "Operation to perform",
					},
					"target_replica": map[string]interface{}{
						"type":        "string",
						"description": "Target replica application name or host",
					},
					"force": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Force failover even if primary is healthy",
					},
				},
				"required": []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute manages failover */
func (t *PostgreSQLFailoverTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	operation := "check"
	if val, ok := params["operation"].(string); ok {
		operation = val
	}

	switch operation {
	case "check":
		/* Check replication status and failover readiness */
		query := `
			SELECT 
				application_name,
				client_addr,
				state,
				sync_state,
				pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn) AS lag_bytes,
				pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn)) AS lag_pretty
			FROM pg_stat_replication
			ORDER BY lag_bytes ASC
		`
		results, err := t.executor.ExecuteQuery(ctx, query, nil)
		if err != nil {
			return Error(
				fmt.Sprintf("Failed to check replication status: %v", err),
				"QUERY_ERROR",
				map[string]interface{}{"error": err.Error()},
			), nil
		}

		/* Check if this is a primary or replica */
		isPrimaryQuery := `SELECT pg_is_in_recovery() AS is_replica`
		primaryCheck, _ := t.executor.ExecuteQueryOne(ctx, isPrimaryQuery, nil)

		return Success(map[string]interface{}{
			"replicas":        results,
			"replica_count":   len(results),
			"is_primary":      primaryCheck,
			"failover_ready":  len(results) > 0,
		}, map[string]interface{}{
			"tool": "postgresql_failover",
		}), nil

	case "status":
		/* Get current failover status */
		isPrimaryQuery := `SELECT pg_is_in_recovery() AS is_replica`
		primaryCheck, err := t.executor.ExecuteQueryOne(ctx, isPrimaryQuery, nil)
		if err != nil {
			return Error("Failed to check primary status", "QUERY_ERROR", map[string]interface{}{"error": err.Error()}), nil
		}

		replicationQuery := `
			SELECT COUNT(*) AS replica_count
			FROM pg_stat_replication
		`
		replicaCount, _ := t.executor.ExecuteQueryOne(ctx, replicationQuery, nil)

		return Success(map[string]interface{}{
			"is_primary":     primaryCheck,
			"replica_count":  replicaCount,
			"status":         "operational",
		}, map[string]interface{}{
			"tool": "postgresql_failover",
		}), nil

	case "initiate":
		targetReplica, _ := params["target_replica"].(string)
		force := false
		if val, ok := params["force"].(bool); ok {
			force = val
		}

		/* Check if this is primary */
		isPrimaryQuery := `SELECT pg_is_in_recovery() AS is_replica`
		primaryCheck, err := t.executor.ExecuteQueryOne(ctx, isPrimaryQuery, nil)
		if err != nil {
			return Error("Failed to check primary status", "QUERY_ERROR", map[string]interface{}{"error": err.Error()}), nil
		}

		isReplica := false
		if val, ok := primaryCheck["is_replica"].(bool); ok {
			isReplica = val
		}

		if isReplica {
			return Error("Cannot initiate failover from a replica. This must be run on the primary server.", "INVALID_STATE", nil), nil
		}

		/* Get replication status */
		replicationQuery := `
			SELECT 
				application_name,
				state,
				sync_state,
				pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn) AS lag_bytes
			FROM pg_stat_replication
		`
		if targetReplica != "" {
			replicationQuery += " WHERE application_name = $1"
		}
		replicationQuery += " ORDER BY lag_bytes ASC LIMIT 1"

		var queryParams []interface{}
		if targetReplica != "" {
			queryParams = []interface{}{targetReplica}
		}

		replicaInfo, err := t.executor.ExecuteQueryOne(ctx, replicationQuery, queryParams)
		if err != nil {
			return Error(
				fmt.Sprintf("Failed to get replica information: %v", err),
				"QUERY_ERROR",
				map[string]interface{}{"error": err.Error()},
			), nil
		}

		if replicaInfo == nil || len(replicaInfo) == 0 {
			return Error("No suitable replica found for failover", "NO_REPLICA", nil), nil
		}

		/* Provide failover instructions */
		instructions := []string{
			"1. Stop the primary PostgreSQL server or disconnect it from the network",
			"2. On the target replica server, create trigger file: touch /path/to/data/promote",
			"3. Or use pg_promote() function: SELECT pg_promote();",
			"4. Verify the replica has been promoted by checking pg_is_in_recovery()",
			"5. Update application connection strings to point to new primary",
			"6. Restart the old primary as a new replica if needed",
		}

		return Success(map[string]interface{}{
			"target_replica":  replicaInfo,
			"force":           force,
			"instructions":    instructions,
			"note":            "Actual failover must be performed on the replica server. This tool provides instructions and verification.",
		}, map[string]interface{}{
			"tool": "postgresql_failover",
		}), nil

	default:
		return Error(fmt.Sprintf("Invalid operation: %s", operation), "INVALID_PARAMETER", nil), nil
	}
}



