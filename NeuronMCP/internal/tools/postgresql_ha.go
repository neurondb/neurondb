/*-------------------------------------------------------------------------
 *
 * postgresql_ha.go
 *    High Availability tools for NeuronMCP
 *
 * Implements high availability operations as specified in Phase 1.1
 * of the roadmap.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/postgresql_ha.go
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

/* PostgreSQLReplicationLagTool provides detailed replication lag analysis */
type PostgreSQLReplicationLagTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLReplicationLagTool creates a new PostgreSQL replication lag tool */
func NewPostgreSQLReplicationLagTool(db *database.Database, logger *logging.Logger) *PostgreSQLReplicationLagTool {
	return &PostgreSQLReplicationLagTool{
		BaseTool: NewBaseTool(
			"postgresql_replication_lag",
			"Get detailed replication lag analysis with byte and time lag",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"replica_name": map[string]interface{}{
						"type":        "string",
						"description": "Filter by replica application name (optional)",
					},
				},
				"required": []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute retrieves replication lag details */
func (t *PostgreSQLReplicationLagTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	replicaName, _ := params["replica_name"].(string)

	var query string
	var queryParams []interface{}

	if replicaName != "" {
		query = `
			SELECT 
				pid,
				usename,
				application_name,
				client_addr,
				state,
				sent_lsn,
				write_lsn,
				flush_lsn,
				replay_lsn,
				sync_priority,
				sync_state,
				pg_wal_lsn_diff(pg_current_wal_lsn(), sent_lsn) AS sent_lag_bytes,
				pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), sent_lsn)) AS sent_lag_pretty,
				pg_wal_lsn_diff(pg_current_wal_lsn(), write_lsn) AS write_lag_bytes,
				pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), write_lsn)) AS write_lag_pretty,
				pg_wal_lsn_diff(pg_current_wal_lsn(), flush_lsn) AS flush_lag_bytes,
				pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), flush_lsn)) AS flush_lag_pretty,
				pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn) AS replay_lag_bytes,
				pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn)) AS replay_lag_pretty
			FROM pg_stat_replication
			WHERE application_name = $1
		`
		queryParams = []interface{}{replicaName}
	} else {
		query = `
			SELECT 
				pid,
				usename,
				application_name,
				client_addr,
				state,
				sent_lsn,
				write_lsn,
				flush_lsn,
				replay_lsn,
				sync_priority,
				sync_state,
				pg_wal_lsn_diff(pg_current_wal_lsn(), sent_lsn) AS sent_lag_bytes,
				pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), sent_lsn)) AS sent_lag_pretty,
				pg_wal_lsn_diff(pg_current_wal_lsn(), write_lsn) AS write_lag_bytes,
				pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), write_lsn)) AS write_lag_pretty,
				pg_wal_lsn_diff(pg_current_wal_lsn(), flush_lsn) AS flush_lag_bytes,
				pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), flush_lsn)) AS flush_lag_pretty,
				pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn) AS replay_lag_bytes,
				pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn)) AS replay_lag_pretty
			FROM pg_stat_replication
		`
		queryParams = []interface{}{}
	}

	results, err := t.executor.ExecuteQuery(ctx, query, queryParams)
	if err != nil {
		return Error(
			fmt.Sprintf("Replication lag query failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	return Success(map[string]interface{}{
		"replication_lag": results,
		"count":          len(results),
	}, map[string]interface{}{
		"tool": "postgresql_replication_lag",
	}), nil
}

/* PostgreSQLPromoteReplicaTool provides instructions for promoting a replica */
type PostgreSQLPromoteReplicaTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLPromoteReplicaTool creates a new PostgreSQL promote replica tool */
func NewPostgreSQLPromoteReplicaTool(db *database.Database, logger *logging.Logger) *PostgreSQLPromoteReplicaTool {
	return &PostgreSQLPromoteReplicaTool{
		BaseTool: NewBaseTool(
			"postgresql_promote_replica",
			"Provide instructions and verify prerequisites for promoting a replica to primary",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"replica_name": map[string]interface{}{
						"type":        "string",
						"description": "Replica application name or host",
					},
					"verify_only": map[string]interface{}{
						"type":        "boolean",
						"default":     true,
						"description": "Only verify prerequisites, do not provide promotion commands",
					},
				},
				"required": []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute provides promotion instructions */
func (t *PostgreSQLPromoteReplicaTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	replicaName, _ := params["replica_name"].(string)
	verifyOnly := true
	if val, ok := params["verify_only"].(bool); ok {
		verifyOnly = val
	}

	/* Check replication status */
	query := `
		SELECT 
			application_name,
			state,
			sync_state,
			pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn) AS lag_bytes
		FROM pg_stat_replication
	`
	var queryParams []interface{}
	if replicaName != "" {
		query += " WHERE application_name = $1"
		queryParams = []interface{}{replicaName}
	}

	results, err := t.executor.ExecuteQuery(ctx, query, queryParams)
	if err != nil {
		return Error(
			fmt.Sprintf("Failed to check replication status: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	if len(results) == 0 {
		return Error(
			"No replication replicas found",
			"NO_REPLICAS",
			nil,
		), nil
	}

	instructions := []string{
		"1. Stop the primary server or disconnect it from the network",
		"2. On the replica server, create trigger file: touch /path/to/data/promote",
		"3. Or use pg_promote() function if available: SELECT pg_promote();",
		"4. Verify the replica has been promoted by checking pg_is_in_recovery()",
		"5. Update application connection strings to point to new primary",
	}

	if verifyOnly {
		return Success(map[string]interface{}{
			"replicas":     results,
			"count":        len(results),
			"instructions": instructions,
			"note":         "This tool provides instructions only. Actual promotion must be performed on the replica server.",
		}, map[string]interface{}{
			"tool": "postgresql_promote_replica",
		}), nil
	}

	return Success(map[string]interface{}{
		"replicas":     results,
		"count":        len(results),
		"instructions": instructions,
	}, map[string]interface{}{
		"tool": "postgresql_promote_replica",
	}), nil
}

/* PostgreSQLSyncStatusTool checks synchronization status */
type PostgreSQLSyncStatusTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLSyncStatusTool creates a new PostgreSQL sync status tool */
func NewPostgreSQLSyncStatusTool(db *database.Database, logger *logging.Logger) *PostgreSQLSyncStatusTool {
	return &PostgreSQLSyncStatusTool{
		BaseTool: NewBaseTool(
			"postgresql_sync_status",
			"Check synchronization status of replication replicas",
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

/* Execute checks sync status */
func (t *PostgreSQLSyncStatusTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	query := `
		SELECT 
			application_name,
			client_addr,
			state,
			sync_state,
			sync_priority,
			pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn) AS lag_bytes,
			pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn)) AS lag_pretty,
			CASE 
				WHEN sync_state = 'sync' THEN 'FULLY_SYNCED'
				WHEN sync_state = 'async' AND pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn) < 1048576 THEN 'NEARLY_SYNCED'
				ELSE 'LAGGING'
			END AS sync_status
		FROM pg_stat_replication
		ORDER BY sync_priority DESC, lag_bytes ASC
	`

	results, err := t.executor.ExecuteQuery(ctx, query, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("Sync status query failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	/* Calculate summary */
	syncedCount := 0
	laggingCount := 0
	for _, result := range results {
		if status, ok := result["sync_status"].(string); ok {
			if status == "FULLY_SYNCED" {
				syncedCount++
			} else {
				laggingCount++
			}
		}
	}

	return Success(map[string]interface{}{
		"replicas":      results,
		"count":         len(results),
		"synced_count":  syncedCount,
		"lagging_count": laggingCount,
	}, map[string]interface{}{
		"tool": "postgresql_sync_status",
	}), nil
}

/* PostgreSQLClusterTool runs CLUSTER operation */
type PostgreSQLClusterTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLClusterTool creates a new PostgreSQL cluster tool */
func NewPostgreSQLClusterTool(db *database.Database, logger *logging.Logger) *PostgreSQLClusterTool {
	return &PostgreSQLClusterTool{
		BaseTool: NewBaseTool(
			"postgresql_cluster",
			"Run CLUSTER operation to physically reorder table data according to an index",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table_name": map[string]interface{}{
						"type":        "string",
						"description": "Table name to cluster",
					},
					"index_name": map[string]interface{}{
						"type":        "string",
						"description": "Index name to cluster by (optional, uses primary key if not specified)",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"default":     "public",
						"description": "Schema name",
					},
				},
				"required": []interface{}{"table_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute runs CLUSTER operation */
func (t *PostgreSQLClusterTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	tableName, ok := params["table_name"].(string)
	if !ok || tableName == "" {
		return Error("table_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	indexName, _ := params["index_name"].(string)

	fullTableName := fmt.Sprintf("%s.%s", schema, tableName)
	var clusterQuery string

	if indexName != "" {
		fullIndexName := fmt.Sprintf("%s.%s", schema, indexName)
		clusterQuery = fmt.Sprintf("CLUSTER %s USING %s", fullTableName, fullIndexName)
	} else {
		clusterQuery = fmt.Sprintf("CLUSTER %s", fullTableName)
	}

	/* Execute CLUSTER */
	_, err := t.executor.ExecuteQuery(ctx, clusterQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("CLUSTER operation failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Table clustered", map[string]interface{}{
		"schema":     schema,
		"table":      tableName,
		"index":      indexName,
	})

	return Success(map[string]interface{}{
		"schema":     schema,
		"table":      tableName,
		"index":      indexName,
		"query":      clusterQuery,
	}, map[string]interface{}{
		"tool": "postgresql_cluster",
	}), nil
}



