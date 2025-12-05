package tools

import (
	"context"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

// PostgreSQLConnectionsTool retrieves detailed connection information
type PostgreSQLConnectionsTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewPostgreSQLConnectionsTool creates a new PostgreSQL connections tool
func NewPostgreSQLConnectionsTool(db *database.Database, logger *logging.Logger) *PostgreSQLConnectionsTool {
	return &PostgreSQLConnectionsTool{
		BaseTool: NewBaseTool(
			"postgresql_connections",
			"Get detailed PostgreSQL connection information",
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

// Execute executes the connections query
func (t *PostgreSQLConnectionsTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	query := `
		SELECT 
			pid,
			usename,
			application_name,
			client_addr,
			client_port,
			state,
			query_start,
			state_change,
			wait_event_type,
			wait_event,
			query
		FROM pg_stat_activity
		WHERE datname = current_database()
		ORDER BY query_start DESC
	`

	results, err := t.executor.ExecuteQuery(ctx, query, nil)
	if err != nil {
		t.logger.Error("PostgreSQL connections query failed", err, nil)
		return Error(fmt.Sprintf("PostgreSQL connections query failed: error=%v", err), "QUERY_ERROR", map[string]interface{}{
			"error": err.Error(),
		}), nil
	}

	return Success(map[string]interface{}{
		"connections": results,
		"count":       len(results),
	}, map[string]interface{}{
		"tool": "postgresql_connections",
		"count": len(results),
	}), nil
}

// PostgreSQLLocksTool retrieves lock information
type PostgreSQLLocksTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewPostgreSQLLocksTool creates a new PostgreSQL locks tool
func NewPostgreSQLLocksTool(db *database.Database, logger *logging.Logger) *PostgreSQLLocksTool {
	return &PostgreSQLLocksTool{
		BaseTool: NewBaseTool(
			"postgresql_locks",
			"Get PostgreSQL lock information",
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

// Execute executes the locks query
func (t *PostgreSQLLocksTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	query := `
		SELECT 
			locktype,
			database,
			relation,
			page,
			tuple,
			virtualxid,
			transactionid,
			classid,
			objid,
			objsubid,
			virtualtransaction,
			pid,
			mode,
			granted
		FROM pg_locks
		WHERE database = (SELECT oid FROM pg_database WHERE datname = current_database())
		ORDER BY pid, locktype
	`

	results, err := t.executor.ExecuteQuery(ctx, query, nil)
	if err != nil {
		t.logger.Error("PostgreSQL locks query failed", err, nil)
		return Error(fmt.Sprintf("PostgreSQL locks query failed: error=%v", err), "QUERY_ERROR", map[string]interface{}{
			"error": err.Error(),
		}), nil
	}

	return Success(map[string]interface{}{
		"locks": results,
		"count": len(results),
	}, map[string]interface{}{
		"tool": "postgresql_locks",
		"count": len(results),
	}), nil
}

// PostgreSQLReplicationTool retrieves replication status
type PostgreSQLReplicationTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewPostgreSQLReplicationTool creates a new PostgreSQL replication tool
func NewPostgreSQLReplicationTool(db *database.Database, logger *logging.Logger) *PostgreSQLReplicationTool {
	return &PostgreSQLReplicationTool{
		BaseTool: NewBaseTool(
			"postgresql_replication",
			"Get PostgreSQL replication status",
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

// Execute executes the replication query
func (t *PostgreSQLReplicationTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	query := `
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
			sync_state
		FROM pg_stat_replication
	`

	results, err := t.executor.ExecuteQuery(ctx, query, nil)
	if err != nil {
		t.logger.Error("PostgreSQL replication query failed", err, nil)
		return Error(fmt.Sprintf("PostgreSQL replication query failed: error=%v", err), "QUERY_ERROR", map[string]interface{}{
			"error": err.Error(),
		}), nil
	}

	return Success(map[string]interface{}{
		"replication": results,
		"count":       len(results),
	}, map[string]interface{}{
		"tool": "postgresql_replication",
		"count": len(results),
	}), nil
}

// PostgreSQLSettingsTool retrieves configuration settings
type PostgreSQLSettingsTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewPostgreSQLSettingsTool creates a new PostgreSQL settings tool
func NewPostgreSQLSettingsTool(db *database.Database, logger *logging.Logger) *PostgreSQLSettingsTool {
	return &PostgreSQLSettingsTool{
		BaseTool: NewBaseTool(
			"postgresql_settings",
			"Get PostgreSQL configuration settings",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"pattern": map[string]interface{}{
						"type":        "string",
						"description": "Optional pattern to filter settings (e.g., 'shared_buffers')",
					},
				},
				"required": []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes the settings query
func (t *PostgreSQLSettingsTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	pattern, _ := params["pattern"].(string)

	var query string
	var queryParams []interface{}

	if pattern != "" {
		query = `
			SELECT 
				name,
				setting,
				unit,
				category,
				short_desc,
				context,
				vartype,
				source,
				min_val,
				max_val,
				enumvals
			FROM pg_settings
			WHERE name LIKE $1
			ORDER BY name
		`
		queryParams = []interface{}{"%" + pattern + "%"}
	} else {
		query = `
			SELECT 
				name,
				setting,
				unit,
				category,
				short_desc,
				context,
				vartype,
				source,
				min_val,
				max_val,
				enumvals
			FROM pg_settings
			ORDER BY category, name
		`
		queryParams = []interface{}{}
	}

	results, err := t.executor.ExecuteQuery(ctx, query, queryParams)
	if err != nil {
		t.logger.Error("PostgreSQL settings query failed", err, nil)
		return Error(fmt.Sprintf("PostgreSQL settings query failed: error=%v", err), "QUERY_ERROR", map[string]interface{}{
			"error": err.Error(),
		}), nil
	}

	return Success(map[string]interface{}{
		"settings": results,
		"count":    len(results),
	}, map[string]interface{}{
		"tool": "postgresql_settings",
		"count": len(results),
	}), nil
}

// PostgreSQLExtensionsTool lists installed extensions
type PostgreSQLExtensionsTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewPostgreSQLExtensionsTool creates a new PostgreSQL extensions tool
func NewPostgreSQLExtensionsTool(db *database.Database, logger *logging.Logger) *PostgreSQLExtensionsTool {
	return &PostgreSQLExtensionsTool{
		BaseTool: NewBaseTool(
			"postgresql_extensions",
			"List installed PostgreSQL extensions",
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

// Execute executes the extensions query
func (t *PostgreSQLExtensionsTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	query := `
		SELECT 
			extname,
			extversion,
			nspname AS schema,
			extrelocatable,
			extconfig
		FROM pg_extension e
		JOIN pg_namespace n ON e.extnamespace = n.oid
		ORDER BY extname
	`

	results, err := t.executor.ExecuteQuery(ctx, query, nil)
	if err != nil {
		t.logger.Error("PostgreSQL extensions query failed", err, nil)
		return Error(fmt.Sprintf("PostgreSQL extensions query failed: error=%v", err), "QUERY_ERROR", map[string]interface{}{
			"error": err.Error(),
		}), nil
	}

	return Success(map[string]interface{}{
		"extensions": results,
		"count":     len(results),
	}, map[string]interface{}{
		"tool": "postgresql_extensions",
		"count": len(results),
	}), nil
}

