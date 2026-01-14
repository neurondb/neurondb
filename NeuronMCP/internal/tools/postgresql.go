/*-------------------------------------------------------------------------
 *
 * postgresql.go
 *    Tool implementation for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/postgresql.go
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

/* PostgreSQLVersionTool retrieves PostgreSQL version information */
type PostgreSQLVersionTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLVersionTool creates a new PostgreSQL version tool */
func NewPostgreSQLVersionTool(db *database.Database, logger *logging.Logger) *PostgreSQLVersionTool {
	return &PostgreSQLVersionTool{
		BaseTool: NewBaseTool(
			"postgresql_version",
			"Get PostgreSQL server version and build information",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{},
				"required": []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the PostgreSQL version query */
func (t *PostgreSQLVersionTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
  /* Query PostgreSQL version information */
	versionQuery := `
		SELECT 
			version() AS version,
			current_setting('server_version') AS server_version,
			current_setting('server_version_num')::bigint AS server_version_num,
			current_setting('server_version_num')::bigint / 10000 AS major_version,
			(current_setting('server_version_num')::bigint / 100) % 100 AS minor_version,
			current_setting('server_version_num')::bigint % 100 AS patch_version
	`

	result, err := t.executor.ExecuteQueryOne(ctx, versionQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("PostgreSQL version query execution failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{
				"error": err.Error(),
			},
		), nil
	}

	t.logger.Info("PostgreSQL version retrieved", map[string]interface{}{
		"server_version": result["server_version"],
		"major_version":  result["major_version"],
		"minor_version":  result["minor_version"],
	})

	return Success(result, map[string]interface{}{
		"tool": "postgresql_version",
	}), nil
}

/* PostgreSQLStatsTool retrieves PostgreSQL statistics */
type PostgreSQLStatsTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLStatsTool creates a new PostgreSQL statistics tool */
func NewPostgreSQLStatsTool(db *database.Database, logger *logging.Logger) *PostgreSQLStatsTool {
	return &PostgreSQLStatsTool{
		BaseTool: NewBaseTool(
			"postgresql_stats",
			"Get comprehensive PostgreSQL server statistics including database size, connection info, table stats, and performance metrics",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"include_database_stats": map[string]interface{}{
						"type":        "boolean",
						"default":     true,
						"description": "Include database-level statistics",
					},
					"include_table_stats": map[string]interface{}{
						"type":        "boolean",
						"default":     true,
						"description": "Include table statistics",
					},
					"include_connection_stats": map[string]interface{}{
						"type":        "boolean",
						"default":     true,
						"description": "Include connection statistics",
					},
					"include_performance_stats": map[string]interface{}{
						"type":        "boolean",
						"default":     true,
						"description": "Include performance metrics",
					},
				},
				"required": []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the PostgreSQL statistics query */
func (t *PostgreSQLStatsTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	includeDBStats := true
	includeTableStats := true
	includeConnStats := true
	includePerfStats := true

	if val, ok := params["include_database_stats"].(bool); ok {
		includeDBStats = val
	}
	if val, ok := params["include_table_stats"].(bool); ok {
		includeTableStats = val
	}
	if val, ok := params["include_connection_stats"].(bool); ok {
		includeConnStats = val
	}
	if val, ok := params["include_performance_stats"].(bool); ok {
		includePerfStats = val
	}

	stats := make(map[string]interface{})

  /* Database statistics */
	if includeDBStats {
		dbStatsQuery := `
			SELECT 
				current_database() AS current_database,
				pg_database_size(current_database()) AS database_size_bytes,
				pg_size_pretty(pg_database_size(current_database())) AS database_size_pretty,
				(SELECT count(*) FROM pg_database) AS total_databases,
				(SELECT count(*) FROM pg_namespace WHERE nspname NOT LIKE 'pg_%' AND nspname != 'information_schema') AS user_schemas
		`
		dbStats, err := t.executor.ExecuteQueryOne(ctx, dbStatsQuery, nil)
		if err != nil {
			t.logger.Warn("Failed to get database stats", map[string]interface{}{
				"error": err.Error(),
			})
		} else {
			stats["database"] = dbStats
		}
	}

  /* Connection statistics */
	if includeConnStats {
		connStatsQuery := `
			SELECT 
				(SELECT count(*) FROM pg_stat_activity WHERE state = 'active') AS active_connections,
				(SELECT count(*) FROM pg_stat_activity WHERE state = 'idle') AS idle_connections,
				(SELECT count(*) FROM pg_stat_activity WHERE state = 'idle in transaction') AS idle_in_transaction,
				(SELECT count(*) FROM pg_stat_activity) AS total_connections,
				current_setting('max_connections')::int AS max_connections,
				(SELECT count(*) FROM pg_stat_activity)::float / NULLIF(current_setting('max_connections')::int, 0) * 100 AS connection_usage_percent
		`
		connStats, err := t.executor.ExecuteQueryOne(ctx, connStatsQuery, nil)
		if err != nil {
			t.logger.Warn("Failed to get connection stats", map[string]interface{}{
				"error": err.Error(),
			})
		} else {
			stats["connections"] = connStats
		}
	}

  /* Table statistics */
	if includeTableStats {
		tableStatsQuery := `
			SELECT 
				(SELECT count(*) FROM pg_tables WHERE schemaname NOT IN ('pg_catalog', 'information_schema')) AS user_tables,
				(SELECT count(*) FROM pg_tables) AS total_tables,
				(SELECT sum(pg_total_relation_size(schemaname||'.'||tablename)) FROM pg_tables WHERE schemaname NOT IN ('pg_catalog', 'information_schema')) AS total_user_tables_size_bytes,
				(SELECT pg_size_pretty(sum(pg_total_relation_size(schemaname||'.'||tablename))) FROM pg_tables WHERE schemaname NOT IN ('pg_catalog', 'information_schema')) AS total_user_tables_size_pretty,
				(SELECT count(*) FROM pg_indexes WHERE schemaname NOT IN ('pg_catalog', 'information_schema')) AS user_indexes,
				(SELECT count(*) FROM pg_indexes) AS total_indexes
		`
		tableStats, err := t.executor.ExecuteQueryOne(ctx, tableStatsQuery, nil)
		if err != nil {
			t.logger.Warn("Failed to get table stats", map[string]interface{}{
				"error": err.Error(),
			})
		} else {
			stats["tables"] = tableStats
		}
	}

  /* Performance statistics */
	if includePerfStats {
		perfStatsQuery := `
			SELECT 
				(SELECT sum(seq_scan) FROM pg_stat_user_tables) AS total_seq_scans,
				(SELECT sum(idx_scan) FROM pg_stat_user_tables) AS total_idx_scans,
				(SELECT sum(n_tup_ins) FROM pg_stat_user_tables) AS total_inserts,
				(SELECT sum(n_tup_upd) FROM pg_stat_user_tables) AS total_updates,
				(SELECT sum(n_tup_del) FROM pg_stat_user_tables) AS total_deletes,
				(SELECT sum(n_live_tup) FROM pg_stat_user_tables) AS total_live_tuples,
				(SELECT sum(n_dead_tup) FROM pg_stat_user_tables) AS total_dead_tuples,
				(SELECT sum(n_dead_tup)::float / NULLIF(sum(n_live_tup), 0) * 100 FROM pg_stat_user_tables) AS dead_tuple_percent,
				(SELECT count(*) FROM pg_stat_user_tables WHERE last_vacuum IS NOT NULL) AS tables_vacuumed,
				(SELECT count(*) FROM pg_stat_user_tables WHERE last_analyze IS NOT NULL) AS tables_analyzed
		`
		perfStats, err := t.executor.ExecuteQueryOne(ctx, perfStatsQuery, nil)
		if err != nil {
			t.logger.Warn("Failed to get performance stats", map[string]interface{}{
				"error": err.Error(),
			})
		} else {
			stats["performance"] = perfStats
		}

   /* Cache hit ratio */
		cacheQuery := `
			SELECT 
				sum(heap_blks_hit)::float / NULLIF(sum(heap_blks_hit) + sum(heap_blks_read), 0) * 100 AS heap_cache_hit_ratio,
				sum(idx_blks_hit)::float / NULLIF(sum(idx_blks_hit) + sum(idx_blks_read), 0) * 100 AS index_cache_hit_ratio
			FROM pg_statio_user_tables
		`
		cacheStats, err := t.executor.ExecuteQueryOne(ctx, cacheQuery, nil)
		if err == nil {
			if perfStatsMap, ok := stats["performance"].(map[string]interface{}); ok {
				if cacheStats != nil {
					for k, v := range cacheStats {
						perfStatsMap[k] = v
					}
					stats["performance"] = perfStatsMap
				}
			}
		}
	}

  /* Server information */
	serverInfoQuery := `
		SELECT 
			current_setting('server_version') AS server_version,
			current_setting('shared_buffers') AS shared_buffers,
			current_setting('effective_cache_size') AS effective_cache_size,
			current_setting('work_mem') AS work_mem,
			current_setting('maintenance_work_mem') AS maintenance_work_mem,
			current_setting('max_connections') AS max_connections,
			current_setting('checkpoint_timeout') AS checkpoint_timeout,
			pg_postmaster_start_time() AS server_start_time,
			now() - pg_postmaster_start_time() AS server_uptime
	`
	serverInfo, err := t.executor.ExecuteQueryOne(ctx, serverInfoQuery, nil)
	if err != nil {
		t.logger.Warn("Failed to get server info", map[string]interface{}{
			"error": err.Error(),
		})
	} else {
		stats["server"] = serverInfo
	}

	t.logger.Info("PostgreSQL statistics retrieved", map[string]interface{}{
		"include_database_stats":    includeDBStats,
		"include_table_stats":       includeTableStats,
		"include_connection_stats":  includeConnStats,
		"include_performance_stats": includePerfStats,
	})

	return Success(stats, map[string]interface{}{
		"tool": "postgresql_stats",
	}), nil
}

/* PostgreSQLDatabaseListTool lists all databases */
type PostgreSQLDatabaseListTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLDatabaseListTool creates a new PostgreSQL database list tool */
func NewPostgreSQLDatabaseListTool(db *database.Database, logger *logging.Logger) *PostgreSQLDatabaseListTool {
	return &PostgreSQLDatabaseListTool{
		BaseTool: NewBaseTool(
			"postgresql_databases",
			"List all PostgreSQL databases with their sizes and connection counts",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"include_system": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Include system databases (template0, template1, postgres)",
					},
				},
				"required": []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the database list query */
func (t *PostgreSQLDatabaseListTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	includeSystem := false
	if val, ok := params["include_system"].(bool); ok {
		includeSystem = val
	}

	var query string
	if includeSystem {
		query = `
			SELECT 
				d.datname AS name,
				pg_database_size(d.datname) AS size_bytes,
				pg_size_pretty(pg_database_size(d.datname)) AS size_pretty,
				COALESCE(s.numbackends, 0) AS connections,
				d.datcollate AS collation,
				d.datctype AS ctype
			FROM pg_database d
			LEFT JOIN pg_stat_database s ON d.oid = s.datid
			ORDER BY pg_database_size(d.datname) DESC
		`
	} else {
		query = `
			SELECT 
				d.datname AS name,
				pg_database_size(d.datname) AS size_bytes,
				pg_size_pretty(pg_database_size(d.datname)) AS size_pretty,
				COALESCE(s.numbackends, 0) AS connections,
				d.datcollate AS collation,
				d.datctype AS ctype
			FROM pg_database d
			LEFT JOIN pg_stat_database s ON d.oid = s.datid
			WHERE d.datname NOT IN ('template0', 'template1', 'postgres')
			ORDER BY pg_database_size(d.datname) DESC
		`
	}

	databases, err := t.executor.ExecuteQuery(ctx, query, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("PostgreSQL database list query execution failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{
				"error": err.Error(),
			},
		), nil
	}

	t.logger.Info("PostgreSQL databases listed", map[string]interface{}{
		"count":          len(databases),
		"include_system": includeSystem,
	})

	return Success(map[string]interface{}{
		"databases": databases,
		"count":     len(databases),
	}, map[string]interface{}{
		"tool": "postgresql_databases",
	}), nil
}

