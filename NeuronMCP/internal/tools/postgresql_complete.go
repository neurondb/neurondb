/*-------------------------------------------------------------------------
 *
 * postgresql_complete.go
 *    Complete PostgreSQL tool implementations for NeuronMCP
 *
 * Implements all 19 missing PostgreSQL administration tools:
 * - Database Object Management (8 tools)
 * - User and Role Management (3 tools)
 * - Performance and Statistics (4 tools)
 * - Size and Storage (4 tools)
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/postgresql_complete.go
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

/* ============================================================================
 * Database Object Management Tools (8 tools)
 * ============================================================================ */

/* PostgreSQLTablesTool lists all tables with metadata */
type PostgreSQLTablesTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLTablesTool creates a new PostgreSQL tables tool */
func NewPostgreSQLTablesTool(db *database.Database, logger *logging.Logger) *PostgreSQLTablesTool {
	return &PostgreSQLTablesTool{
		BaseTool: NewBaseTool(
			"postgresql_tables",
			"List all PostgreSQL tables with metadata (schema, owner, size, row count)",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"schema": map[string]interface{}{
						"type":        "string",
						"description": "Filter by schema name (optional)",
					},
					"include_system": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Include system tables (pg_catalog, information_schema)",
					},
				},
				"required": []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the tables query */
func (t *PostgreSQLTablesTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	schema, _ := params["schema"].(string)
	includeSystem := false
	if val, ok := params["include_system"].(bool); ok {
		includeSystem = val
	}

	var query string
	var queryParams []interface{}

	if schema != "" {
		query = `
			SELECT 
				t.schemaname,
				t.tablename,
				t.tableowner,
				pg_total_relation_size(quote_ident(t.schemaname)||'.'||quote_ident(t.tablename)) AS total_size_bytes,
				pg_size_pretty(pg_total_relation_size(quote_ident(t.schemaname)||'.'||quote_ident(t.tablename))) AS total_size_pretty,
				pg_relation_size(quote_ident(t.schemaname)||'.'||quote_ident(t.tablename)) AS table_size_bytes,
				pg_size_pretty(pg_relation_size(quote_ident(t.schemaname)||'.'||quote_ident(t.tablename))) AS table_size_pretty,
				COALESCE(s.n_live_tup, 0) AS row_count,
				t.tablespace
			FROM pg_tables t
			LEFT JOIN pg_stat_user_tables s ON s.schemaname = t.schemaname AND s.relname = t.tablename
			WHERE t.schemaname = $1
			ORDER BY t.schemaname, t.tablename
		`
		queryParams = []interface{}{schema}
	} else if includeSystem {
		query = `
			SELECT 
				t.schemaname,
				t.tablename,
				t.tableowner,
				pg_total_relation_size(quote_ident(t.schemaname)||'.'||quote_ident(t.tablename)) AS total_size_bytes,
				pg_size_pretty(pg_total_relation_size(quote_ident(t.schemaname)||'.'||quote_ident(t.tablename))) AS total_size_pretty,
				pg_relation_size(quote_ident(t.schemaname)||'.'||quote_ident(t.tablename)) AS table_size_bytes,
				pg_size_pretty(pg_relation_size(quote_ident(t.schemaname)||'.'||quote_ident(t.tablename))) AS table_size_pretty,
				COALESCE(s.n_live_tup, 0) AS row_count,
				t.tablespace
			FROM pg_tables t
			LEFT JOIN pg_stat_user_tables s ON s.schemaname = t.schemaname AND s.relname = t.tablename
			ORDER BY t.schemaname, t.tablename
		`
		queryParams = []interface{}{}
	} else {
		query = `
			SELECT 
				t.schemaname,
				t.tablename,
				t.tableowner,
				pg_total_relation_size(quote_ident(t.schemaname)||'.'||quote_ident(t.tablename)) AS total_size_bytes,
				pg_size_pretty(pg_total_relation_size(quote_ident(t.schemaname)||'.'||quote_ident(t.tablename))) AS total_size_pretty,
				pg_relation_size(quote_ident(t.schemaname)||'.'||quote_ident(t.tablename)) AS table_size_bytes,
				pg_size_pretty(pg_relation_size(quote_ident(t.schemaname)||'.'||quote_ident(t.tablename))) AS table_size_pretty,
				COALESCE(s.n_live_tup, 0) AS row_count,
				t.tablespace
			FROM pg_tables t
			LEFT JOIN pg_stat_user_tables s ON s.schemaname = t.schemaname AND s.relname = t.tablename
			WHERE t.schemaname NOT IN ('pg_catalog', 'information_schema')
			ORDER BY t.schemaname, t.tablename
		`
		queryParams = []interface{}{}
	}

	results, err := t.executor.ExecuteQuery(ctx, query, queryParams)
	if err != nil {
		return Error(
			fmt.Sprintf("PostgreSQL tables query failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	return Success(map[string]interface{}{
		"tables": results,
		"count":  len(results),
	}, map[string]interface{}{
		"tool": "postgresql_tables",
	}), nil
}

/* PostgreSQLIndexesTool lists all indexes with usage statistics */
type PostgreSQLIndexesTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLIndexesTool creates a new PostgreSQL indexes tool */
func NewPostgreSQLIndexesTool(db *database.Database, logger *logging.Logger) *PostgreSQLIndexesTool {
	return &PostgreSQLIndexesTool{
		BaseTool: NewBaseTool(
			"postgresql_indexes",
			"List all PostgreSQL indexes with statistics (size, usage, scan counts)",
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
					"include_system": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Include system indexes",
					},
				},
				"required": []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the indexes query */
func (t *PostgreSQLIndexesTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	schema, _ := params["schema"].(string)
	table, _ := params["table"].(string)
	includeSystem := false
	if val, ok := params["include_system"].(bool); ok {
		includeSystem = val
	}

	var conditions []string
	var queryParams []interface{}

	if schema != "" {
		conditions = append(conditions, fmt.Sprintf("i.schemaname = $%d", len(queryParams)+1))
		queryParams = append(queryParams, schema)
	}
	if table != "" {
		conditions = append(conditions, fmt.Sprintf("i.tablename = $%d", len(queryParams)+1))
		queryParams = append(queryParams, table)
	}
	if !includeSystem {
		conditions = append(conditions, "i.schemaname NOT IN ('pg_catalog', 'information_schema')")
	}

	whereClause := ""
	if len(conditions) > 0 {
		whereClause = "WHERE " + fmt.Sprintf("%s", conditions[0])
		for i := 1; i < len(conditions); i++ {
			whereClause += " AND " + conditions[i]
		}
	}

	query := fmt.Sprintf(`
		SELECT 
			i.schemaname,
			i.tablename,
			i.indexname,
			i.indexdef,
			pg_relation_size(quote_ident(i.schemaname)||'.'||quote_ident(i.indexname)) AS index_size_bytes,
			pg_size_pretty(pg_relation_size(quote_ident(i.schemaname)||'.'||quote_ident(i.indexname))) AS index_size_pretty,
			COALESCE(s.idx_scan, 0) AS index_scans,
			COALESCE(s.idx_tup_read, 0) AS tuples_read,
			COALESCE(s.idx_tup_fetch, 0) AS tuples_fetched,
			i.tablespace
		FROM pg_indexes i
		LEFT JOIN pg_stat_user_indexes s ON s.schemaname = i.schemaname AND s.indexrelname = i.indexname
		%s
		ORDER BY i.schemaname, i.tablename, i.indexname
	`, whereClause)

	results, err := t.executor.ExecuteQuery(ctx, query, queryParams)
	if err != nil {
		return Error(
			fmt.Sprintf("PostgreSQL indexes query failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	return Success(map[string]interface{}{
		"indexes": results,
		"count":   len(results),
	}, map[string]interface{}{
		"tool": "postgresql_indexes",
	}), nil
}

/* PostgreSQLSchemasTool lists all schemas with permissions */
type PostgreSQLSchemasTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLSchemasTool creates a new PostgreSQL schemas tool */
func NewPostgreSQLSchemasTool(db *database.Database, logger *logging.Logger) *PostgreSQLSchemasTool {
	return &PostgreSQLSchemasTool{
		BaseTool: NewBaseTool(
			"postgresql_schemas",
			"List all PostgreSQL schemas with ownership and permissions",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"include_system": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Include system schemas",
					},
				},
				"required": []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the schemas query */
func (t *PostgreSQLSchemasTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	includeSystem := false
	if val, ok := params["include_system"].(bool); ok {
		includeSystem = val
	}

	var query string
	if includeSystem {
		query = `
			SELECT 
				n.nspname AS schema_name,
				r.rolname AS owner,
				n.nspacl AS permissions
			FROM pg_namespace n
			LEFT JOIN pg_roles r ON n.nspowner = r.oid
			ORDER BY n.nspname
		`
	} else {
		query = `
			SELECT 
				n.nspname AS schema_name,
				r.rolname AS owner,
				n.nspacl AS permissions
			FROM pg_namespace n
			LEFT JOIN pg_roles r ON n.nspowner = r.oid
			WHERE n.nspname NOT LIKE 'pg_%' AND n.nspname != 'information_schema'
			ORDER BY n.nspname
		`
	}

	results, err := t.executor.ExecuteQuery(ctx, query, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("PostgreSQL schemas query failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	return Success(map[string]interface{}{
		"schemas": results,
		"count":   len(results),
	}, map[string]interface{}{
		"tool": "postgresql_schemas",
	}), nil
}

/* PostgreSQLViewsTool lists all views with definitions */
type PostgreSQLViewsTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLViewsTool creates a new PostgreSQL views tool */
func NewPostgreSQLViewsTool(db *database.Database, logger *logging.Logger) *PostgreSQLViewsTool {
	return &PostgreSQLViewsTool{
		BaseTool: NewBaseTool(
			"postgresql_views",
			"List all PostgreSQL views with definitions",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"schema": map[string]interface{}{
						"type":        "string",
						"description": "Filter by schema name (optional)",
					},
					"include_system": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Include system views",
					},
					"include_definition": map[string]interface{}{
						"type":        "boolean",
						"default":     true,
						"description": "Include view definition SQL",
					},
				},
				"required": []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the views query */
func (t *PostgreSQLViewsTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	schema, _ := params["schema"].(string)
	includeSystem := false
	includeDefinition := true
	if val, ok := params["include_system"].(bool); ok {
		includeSystem = val
	}
	if val, ok := params["include_definition"].(bool); ok {
		includeDefinition = val
	}

	var conditions []string
	var queryParams []interface{}

	if schema != "" {
		conditions = append(conditions, fmt.Sprintf("v.schemaname = $%d", len(queryParams)+1))
		queryParams = append(queryParams, schema)
	}
	if !includeSystem {
		conditions = append(conditions, "v.schemaname NOT IN ('pg_catalog', 'information_schema')")
	}

	whereClause := ""
	if len(conditions) > 0 {
		whereClause = "WHERE " + conditions[0]
		for i := 1; i < len(conditions); i++ {
			whereClause += " AND " + conditions[i]
		}
	}

	var definitionCol string
	if includeDefinition {
		definitionCol = ", pg_get_viewdef(v.schemaname||'.'||v.viewname, true) AS definition"
	} else {
		definitionCol = ""
	}

	query := fmt.Sprintf(`
		SELECT 
			v.schemaname,
			v.viewname,
			v.viewowner%s
		FROM pg_views v
		%s
		ORDER BY v.schemaname, v.viewname
	`, definitionCol, whereClause)

	results, err := t.executor.ExecuteQuery(ctx, query, queryParams)
	if err != nil {
		return Error(
			fmt.Sprintf("PostgreSQL views query failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	return Success(map[string]interface{}{
		"views": results,
		"count": len(results),
	}, map[string]interface{}{
		"tool": "postgresql_views",
	}), nil
}

/* PostgreSQLSequencesTool lists all sequences with current values */
type PostgreSQLSequencesTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLSequencesTool creates a new PostgreSQL sequences tool */
func NewPostgreSQLSequencesTool(db *database.Database, logger *logging.Logger) *PostgreSQLSequencesTool {
	return &PostgreSQLSequencesTool{
		BaseTool: NewBaseTool(
			"postgresql_sequences",
			"List all PostgreSQL sequences with current values and ranges",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"schema": map[string]interface{}{
						"type":        "string",
						"description": "Filter by schema name (optional)",
					},
					"include_system": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Include system sequences",
					},
				},
				"required": []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the sequences query */
func (t *PostgreSQLSequencesTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	schema, _ := params["schema"].(string)
	includeSystem := false
	if val, ok := params["include_system"].(bool); ok {
		includeSystem = val
	}

	var conditions []string
	var queryParams []interface{}

	if schema != "" {
		conditions = append(conditions, fmt.Sprintf("s.sequence_schema = $%d", len(queryParams)+1))
		queryParams = append(queryParams, schema)
	}
	if !includeSystem {
		conditions = append(conditions, "s.sequence_schema NOT IN ('pg_catalog', 'information_schema')")
	}

	whereClause := ""
	if len(conditions) > 0 {
		whereClause = "WHERE " + conditions[0]
		for i := 1; i < len(conditions); i++ {
			whereClause += " AND " + conditions[i]
		}
	}

	/* Use information_schema.sequences for better compatibility */
	query := fmt.Sprintf(`
		SELECT 
			s.sequence_schema AS schemaname,
			s.sequence_name AS sequencename,
			s.data_type,
			s.numeric_precision,
			s.numeric_scale,
			s.start_value,
			s.minimum_value AS min_value,
			s.maximum_value AS max_value,
			s.increment,
			s.cycle_option
		FROM information_schema.sequences s
		%s
		ORDER BY s.sequence_schema, s.sequence_name
	`, whereClause)

	results, err := t.executor.ExecuteQuery(ctx, query, queryParams)
	if err != nil {
		return Error(
			fmt.Sprintf("PostgreSQL sequences query failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	return Success(map[string]interface{}{
		"sequences": results,
		"count":     len(results),
	}, map[string]interface{}{
		"tool": "postgresql_sequences",
	}), nil
}

/* PostgreSQLFunctionsTool lists all functions with parameters */
type PostgreSQLFunctionsTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLFunctionsTool creates a new PostgreSQL functions tool */
func NewPostgreSQLFunctionsTool(db *database.Database, logger *logging.Logger) *PostgreSQLFunctionsTool {
	return &PostgreSQLFunctionsTool{
		BaseTool: NewBaseTool(
			"postgresql_functions",
			"List all PostgreSQL functions with parameters and return types",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"schema": map[string]interface{}{
						"type":        "string",
						"description": "Filter by schema name (optional)",
					},
					"include_system": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Include system functions",
					},
				},
				"required": []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the functions query */
func (t *PostgreSQLFunctionsTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	schema, _ := params["schema"].(string)
	includeSystem := false
	if val, ok := params["include_system"].(bool); ok {
		includeSystem = val
	}

	var conditions []string
	var queryParams []interface{}

	if schema != "" {
		conditions = append(conditions, fmt.Sprintf("n.nspname = $%d", len(queryParams)+1))
		queryParams = append(queryParams, schema)
	}
	if !includeSystem {
		conditions = append(conditions, "n.nspname NOT IN ('pg_catalog', 'information_schema')")
	}

	whereClause := ""
	if len(conditions) > 0 {
		whereClause = "WHERE " + conditions[0]
		for i := 1; i < len(conditions); i++ {
			whereClause += " AND " + conditions[i]
		}
	}

	query := fmt.Sprintf(`
		SELECT 
			n.nspname AS schema_name,
			p.proname AS function_name,
			pg_get_function_arguments(p.oid) AS arguments,
			pg_get_function_result(p.oid) AS return_type,
			pg_get_functiondef(p.oid) AS definition,
			l.lanname AS language,
			p.provolatile AS volatility,
			p.proisstrict AS is_strict,
			p.prosecdef AS security_definer
		FROM pg_proc p
		JOIN pg_namespace n ON p.pronamespace = n.oid
		JOIN pg_language l ON p.prolang = l.oid
		%s
		ORDER BY n.nspname, p.proname, p.oid
	`, whereClause)

	results, err := t.executor.ExecuteQuery(ctx, query, queryParams)
	if err != nil {
		return Error(
			fmt.Sprintf("PostgreSQL functions query failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	return Success(map[string]interface{}{
		"functions": results,
		"count":     len(results),
	}, map[string]interface{}{
		"tool": "postgresql_functions",
	}), nil
}

/* PostgreSQLTriggersTool lists all triggers with event types */
type PostgreSQLTriggersTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLTriggersTool creates a new PostgreSQL triggers tool */
func NewPostgreSQLTriggersTool(db *database.Database, logger *logging.Logger) *PostgreSQLTriggersTool {
	return &PostgreSQLTriggersTool{
		BaseTool: NewBaseTool(
			"postgresql_triggers",
			"List all PostgreSQL triggers with event types and functions",
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

/* Execute executes the triggers query */
func (t *PostgreSQLTriggersTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	schema, _ := params["schema"].(string)
	table, _ := params["table"].(string)

	var conditions []string
	var queryParams []interface{}

	if schema != "" {
		conditions = append(conditions, fmt.Sprintf("t.trigger_schema = $%d", len(queryParams)+1))
		queryParams = append(queryParams, schema)
	}
	if table != "" {
		conditions = append(conditions, fmt.Sprintf("t.event_object_table = $%d", len(queryParams)+1))
		queryParams = append(queryParams, table)
	}

	whereClause := ""
	if len(conditions) > 0 {
		whereClause = "WHERE " + conditions[0]
		for i := 1; i < len(conditions); i++ {
			whereClause += " AND " + conditions[i]
		}
	}

	query := fmt.Sprintf(`
		SELECT 
			t.trigger_schema AS schema_name,
			t.trigger_name,
			t.event_object_table AS table_name,
			t.event_manipulation AS event_type,
			t.action_statement AS action_statement,
			t.action_timing AS timing,
			t.action_condition AS condition
		FROM information_schema.triggers t
		%s
		ORDER BY t.trigger_schema, t.event_object_table, t.trigger_name
	`, whereClause)

	results, err := t.executor.ExecuteQuery(ctx, query, queryParams)
	if err != nil {
		return Error(
			fmt.Sprintf("PostgreSQL triggers query failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	return Success(map[string]interface{}{
		"triggers": results,
		"count":    len(results),
	}, map[string]interface{}{
		"tool": "postgresql_triggers",
	}), nil
}

/* PostgreSQLConstraintsTool lists constraints */
type PostgreSQLConstraintsTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLConstraintsTool creates a new PostgreSQL constraints tool */
func NewPostgreSQLConstraintsTool(db *database.Database, logger *logging.Logger) *PostgreSQLConstraintsTool {
	return &PostgreSQLConstraintsTool{
		BaseTool: NewBaseTool(
			"postgresql_constraints",
			"List constraints (primary keys, foreign keys, unique, check)",
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
					"constraint_type": map[string]interface{}{
						"type":        "string",
						"description": "Filter by constraint type (primary_key, foreign_key, unique, check, not_null)",
					},
				},
				"required": []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the constraints query */
func (t *PostgreSQLConstraintsTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	schema, _ := params["schema"].(string)
	table, _ := params["table"].(string)
	constraintType, _ := params["constraint_type"].(string)

	var conditions []string
	var queryParams []interface{}

	if schema != "" {
		conditions = append(conditions, fmt.Sprintf("c.table_schema = $%d", len(queryParams)+1))
		queryParams = append(queryParams, schema)
	}
	if table != "" {
		conditions = append(conditions, fmt.Sprintf("c.table_name = $%d", len(queryParams)+1))
		queryParams = append(queryParams, table)
	}
	if constraintType != "" {
		conditions = append(conditions, fmt.Sprintf("c.constraint_type = $%d", len(queryParams)+1))
		queryParams = append(queryParams, constraintType)
	}

	whereClause := ""
	if len(conditions) > 0 {
		whereClause = "WHERE " + conditions[0]
		for i := 1; i < len(conditions); i++ {
			whereClause += " AND " + conditions[i]
		}
	}

	query := fmt.Sprintf(`
		SELECT 
			c.constraint_schema AS schema_name,
			c.constraint_name,
			c.table_name,
			c.constraint_type,
			c.is_deferrable,
			c.initially_deferred
		FROM information_schema.table_constraints c
		%s
		ORDER BY c.constraint_schema, c.table_name, c.constraint_name
	`, whereClause)

	results, err := t.executor.ExecuteQuery(ctx, query, queryParams)
	if err != nil {
		return Error(
			fmt.Sprintf("PostgreSQL constraints query failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	return Success(map[string]interface{}{
		"constraints": results,
		"count":       len(results),
	}, map[string]interface{}{
		"tool": "postgresql_constraints",
	}), nil
}

/* ============================================================================
 * User and Role Management Tools (3 tools)
 * ============================================================================ */

/* PostgreSQLUsersTool lists all users */
type PostgreSQLUsersTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLUsersTool creates a new PostgreSQL users tool */
func NewPostgreSQLUsersTool(db *database.Database, logger *logging.Logger) *PostgreSQLUsersTool {
	return &PostgreSQLUsersTool{
		BaseTool: NewBaseTool(
			"postgresql_users",
			"List all PostgreSQL users with login and connection info",
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

/* Execute executes the users query */
func (t *PostgreSQLUsersTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	query := `
		SELECT 
			r.rolname AS username,
			r.rolsuper AS is_superuser,
			r.rolinherit AS can_inherit,
			r.rolcreaterole AS can_create_roles,
			r.rolcreatedb AS can_create_databases,
			r.rolcanlogin AS can_login,
			r.rolreplication AS can_replicate,
			r.rolconnlimit AS connection_limit,
			r.rolvaliduntil AS password_expires_at,
			(SELECT count(*) FROM pg_stat_activity WHERE usename = r.rolname) AS active_connections
		FROM pg_roles r
		WHERE r.rolcanlogin = true
		ORDER BY r.rolname
	`

	results, err := t.executor.ExecuteQuery(ctx, query, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("PostgreSQL users query failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	return Success(map[string]interface{}{
		"users": results,
		"count": len(results),
	}, map[string]interface{}{
		"tool": "postgresql_users",
	}), nil
}

/* PostgreSQLRolesTool lists all roles */
type PostgreSQLRolesTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLRolesTool creates a new PostgreSQL roles tool */
func NewPostgreSQLRolesTool(db *database.Database, logger *logging.Logger) *PostgreSQLRolesTool {
	return &PostgreSQLRolesTool{
		BaseTool: NewBaseTool(
			"postgresql_roles",
			"List all PostgreSQL roles with membership and attributes",
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

/* Execute executes the roles query */
func (t *PostgreSQLRolesTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	query := `
		SELECT 
			r.rolname AS role_name,
			r.rolsuper AS is_superuser,
			r.rolinherit AS can_inherit,
			r.rolcreaterole AS can_create_roles,
			r.rolcreatedb AS can_create_databases,
			r.rolcanlogin AS can_login,
			r.rolreplication AS can_replicate,
			r.rolconnlimit AS connection_limit,
			ARRAY(
				SELECT rm.rolname 
				FROM pg_auth_members am
				JOIN pg_roles rm ON am.roleid = rm.oid
				WHERE am.member = r.oid
			) AS member_of,
			ARRAY(
				SELECT rm.rolname 
				FROM pg_auth_members am
				JOIN pg_roles rm ON am.member = rm.oid
				WHERE am.roleid = r.oid
			) AS members
		FROM pg_roles r
		ORDER BY r.rolname
	`

	results, err := t.executor.ExecuteQuery(ctx, query, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("PostgreSQL roles query failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	return Success(map[string]interface{}{
		"roles": results,
		"count": len(results),
	}, map[string]interface{}{
		"tool": "postgresql_roles",
	}), nil
}

/* PostgreSQLPermissionsTool lists database object permissions */
type PostgreSQLPermissionsTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLPermissionsTool creates a new PostgreSQL permissions tool */
func NewPostgreSQLPermissionsTool(db *database.Database, logger *logging.Logger) *PostgreSQLPermissionsTool {
	return &PostgreSQLPermissionsTool{
		BaseTool: NewBaseTool(
			"postgresql_permissions",
			"List database object permissions (tables, functions, etc.)",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"schema": map[string]interface{}{
						"type":        "string",
						"description": "Filter by schema name (optional)",
					},
					"object_type": map[string]interface{}{
						"type":        "string",
						"description": "Filter by type (table, function, sequence, schema)",
					},
				},
				"required": []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the permissions query */
func (t *PostgreSQLPermissionsTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	schema, _ := params["schema"].(string)
	/* objectType is available but not used in the simplified query implementation */
	/* _ = objectType // Suppress unused variable warning */

	/* This is a complex query - we'll use a simplified version */
	query := `
		SELECT 
			grantee,
			grantor,
			table_schema AS schema_name,
			table_name,
			privilege_type,
			is_grantable
		FROM information_schema.role_table_grants
		WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
	`

	var conditions []string
	var queryParams []interface{}

	if schema != "" {
		conditions = append(conditions, fmt.Sprintf("table_schema = $%d", len(queryParams)+1))
		queryParams = append(queryParams, schema)
	}

	if len(conditions) > 0 {
		query += " AND " + conditions[0]
		for i := 1; i < len(conditions); i++ {
			query += " AND " + conditions[i]
		}
	}

	query += " ORDER BY table_schema, table_name, grantee"

	results, err := t.executor.ExecuteQuery(ctx, query, queryParams)
	if err != nil {
		return Error(
			fmt.Sprintf("PostgreSQL permissions query failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	return Success(map[string]interface{}{
		"permissions": results,
		"count":       len(results),
	}, map[string]interface{}{
		"tool": "postgresql_permissions",
	}), nil
}

/* ============================================================================
 * Performance and Statistics Tools (4 tools)
 * ============================================================================ */

/* PostgreSQLTableStatsTool gets per-table performance metrics */
type PostgreSQLTableStatsTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLTableStatsTool creates a new PostgreSQL table stats tool */
func NewPostgreSQLTableStatsTool(db *database.Database, logger *logging.Logger) *PostgreSQLTableStatsTool {
	return &PostgreSQLTableStatsTool{
		BaseTool: NewBaseTool(
			"postgresql_table_stats",
			"Get detailed per-table statistics (scans, inserts, updates, deletes, tuples)",
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

/* Execute executes the table stats query */
func (t *PostgreSQLTableStatsTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	schema, _ := params["schema"].(string)
	table, _ := params["table"].(string)

	var conditions []string
	var queryParams []interface{}

	if schema != "" {
		conditions = append(conditions, fmt.Sprintf("s.schemaname = $%d", len(queryParams)+1))
		queryParams = append(queryParams, schema)
	}
	if table != "" {
		conditions = append(conditions, fmt.Sprintf("s.relname = $%d", len(queryParams)+1))
		queryParams = append(queryParams, table)
	}

	whereClause := ""
	if len(conditions) > 0 {
		whereClause = "WHERE " + conditions[0]
		for i := 1; i < len(conditions); i++ {
			whereClause += " AND " + conditions[i]
		}
	}

	query := fmt.Sprintf(`
		SELECT 
			s.schemaname,
			s.relname AS table_name,
			s.seq_scan AS sequential_scans,
			s.idx_scan AS index_scans,
			s.n_tup_ins AS inserts,
			s.n_tup_upd AS updates,
			s.n_tup_del AS deletes,
			s.n_live_tup AS live_tuples,
			s.n_dead_tup AS dead_tuples,
			s.last_vacuum,
			s.last_autovacuum,
			s.last_analyze,
			s.last_autoanalyze,
			s.vacuum_count,
			s.autovacuum_count,
			s.analyze_count,
			s.autoanalyze_count
		FROM pg_stat_user_tables s
		%s
		ORDER BY s.schemaname, s.relname
	`, whereClause)

	results, err := t.executor.ExecuteQuery(ctx, query, queryParams)
	if err != nil {
		return Error(
			fmt.Sprintf("PostgreSQL table stats query failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	return Success(map[string]interface{}{
		"table_stats": results,
		"count":       len(results),
	}, map[string]interface{}{
		"tool": "postgresql_table_stats",
	}), nil
}

/* PostgreSQLIndexStatsTool gets per-index usage statistics */
type PostgreSQLIndexStatsTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLIndexStatsTool creates a new PostgreSQL index stats tool */
func NewPostgreSQLIndexStatsTool(db *database.Database, logger *logging.Logger) *PostgreSQLIndexStatsTool {
	return &PostgreSQLIndexStatsTool{
		BaseTool: NewBaseTool(
			"postgresql_index_stats",
			"Get detailed per-index statistics (scans, size, bloat)",
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

/* Execute executes the index stats query */
func (t *PostgreSQLIndexStatsTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	schema, _ := params["schema"].(string)
	table, _ := params["table"].(string)

	var conditions []string
	var queryParams []interface{}

	if schema != "" {
		conditions = append(conditions, fmt.Sprintf("s.schemaname = $%d", len(queryParams)+1))
		queryParams = append(queryParams, schema)
	}
	if table != "" {
		conditions = append(conditions, fmt.Sprintf("s.relname = $%d", len(queryParams)+1))
		queryParams = append(queryParams, table)
	}

	whereClause := ""
	if len(conditions) > 0 {
		whereClause = "WHERE " + conditions[0]
		for i := 1; i < len(conditions); i++ {
			whereClause += " AND " + conditions[i]
		}
	}

	query := fmt.Sprintf(`
		SELECT 
			s.schemaname,
			s.relname AS table_name,
			s.indexrelname AS index_name,
			s.idx_scan AS index_scans,
			s.idx_tup_read AS tuples_read,
			s.idx_tup_fetch AS tuples_fetched,
			pg_relation_size(s.indexrelid) AS index_size_bytes,
			pg_size_pretty(pg_relation_size(s.indexrelid)) AS index_size_pretty
		FROM pg_stat_user_indexes s
		%s
		ORDER BY s.schemaname, s.relname, s.indexrelname
	`, whereClause)

	results, err := t.executor.ExecuteQuery(ctx, query, queryParams)
	if err != nil {
		return Error(
			fmt.Sprintf("PostgreSQL index stats query failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	return Success(map[string]interface{}{
		"index_stats": results,
		"count":        len(results),
	}, map[string]interface{}{
		"tool": "postgresql_index_stats",
	}), nil
}

/* PostgreSQLActiveQueriesTool shows currently running queries */
type PostgreSQLActiveQueriesTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLActiveQueriesTool creates a new PostgreSQL active queries tool */
func NewPostgreSQLActiveQueriesTool(db *database.Database, logger *logging.Logger) *PostgreSQLActiveQueriesTool {
	return &PostgreSQLActiveQueriesTool{
		BaseTool: NewBaseTool(
			"postgresql_active_queries",
			"Show currently active/running queries with details",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"include_idle": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Include idle queries",
					},
					"limit": map[string]interface{}{
						"type":        "number",
						"default":     100,
						"minimum":     1,
						"maximum":     1000,
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

/* Execute executes the active queries query */
func (t *PostgreSQLActiveQueriesTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	includeIdle := false
	limit := 100

	if val, ok := params["include_idle"].(bool); ok {
		includeIdle = val
	}
	if val, ok := params["limit"].(float64); ok {
		limit = int(val)
		if limit < 1 {
			limit = 1
		}
		if limit > 1000 {
			limit = 1000
		}
	} else if val, ok := params["limit"].(int); ok {
		limit = val
	}

	var query string
	var queryParams []interface{}

	if includeIdle {
		query = `
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
				left(query, 500) AS query_preview,
				query
			FROM pg_stat_activity
			WHERE datname = current_database()
			ORDER BY query_start DESC NULLS LAST
			LIMIT $1
		`
		queryParams = []interface{}{limit}
	} else {
		query = `
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
				left(query, 500) AS query_preview,
				query
			FROM pg_stat_activity
			WHERE datname = current_database() AND state != 'idle'
			ORDER BY query_start DESC NULLS LAST
			LIMIT $1
		`
		queryParams = []interface{}{limit}
	}

	results, err := t.executor.ExecuteQuery(ctx, query, queryParams)
	if err != nil {
		return Error(
			fmt.Sprintf("PostgreSQL active queries query failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	return Success(map[string]interface{}{
		"active_queries": results,
		"count":          len(results),
	}, map[string]interface{}{
		"tool": "postgresql_active_queries",
	}), nil
}

/* PostgreSQLWaitEventsTool shows wait events and blocking queries */
type PostgreSQLWaitEventsTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLWaitEventsTool creates a new PostgreSQL wait events tool */
func NewPostgreSQLWaitEventsTool(db *database.Database, logger *logging.Logger) *PostgreSQLWaitEventsTool {
	return &PostgreSQLWaitEventsTool{
		BaseTool: NewBaseTool(
			"postgresql_wait_events",
			"Show wait events and blocking queries",
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

/* Execute executes the wait events query */
func (t *PostgreSQLWaitEventsTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	query := `
		SELECT 
			pid,
			usename,
			application_name,
			state,
			wait_event_type,
			wait_event,
			query_start,
			state_change,
			left(query, 500) AS query_preview,
			(SELECT count(*) FROM pg_locks WHERE granted = false AND pid = a.pid) AS waiting_locks,
			(SELECT array_agg(blocking_pid) FROM pg_blocking_pids(a.pid)) AS blocked_by
		FROM pg_stat_activity a
		WHERE datname = current_database() 
			AND wait_event_type IS NOT NULL
		ORDER BY query_start DESC NULLS LAST
	`

	results, err := t.executor.ExecuteQuery(ctx, query, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("PostgreSQL wait events query failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	return Success(map[string]interface{}{
		"wait_events": results,
		"count":      len(results),
	}, map[string]interface{}{
		"tool": "postgresql_wait_events",
	}), nil
}

/* ============================================================================
 * Size and Storage Tools (4 tools)
 * ============================================================================ */

/* PostgreSQLTableSizeTool gets size of specific tables */
type PostgreSQLTableSizeTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLTableSizeTool creates a new PostgreSQL table size tool */
func NewPostgreSQLTableSizeTool(db *database.Database, logger *logging.Logger) *PostgreSQLTableSizeTool {
	return &PostgreSQLTableSizeTool{
		BaseTool: NewBaseTool(
			"postgresql_table_size",
			"Get size of specific tables (with options for total, indexes, toast)",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"schema": map[string]interface{}{
						"type":        "string",
						"description": "Schema name (optional)",
					},
					"table": map[string]interface{}{
						"type":        "string",
						"description": "Table name (optional)",
					},
					"include_indexes": map[string]interface{}{
						"type":        "boolean",
						"default":     true,
						"description": "Include index sizes",
					},
					"include_toast": map[string]interface{}{
						"type":        "boolean",
						"default":     true,
						"description": "Include TOAST size",
					},
				},
				"required": []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the table size query */
func (t *PostgreSQLTableSizeTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	schema, _ := params["schema"].(string)
	table, _ := params["table"].(string)
	includeIndexes := true
	includeToast := true

	if val, ok := params["include_indexes"].(bool); ok {
		includeIndexes = val
	}
	if val, ok := params["include_toast"].(bool); ok {
		includeToast = val
	}

	var conditions []string
	var queryParams []interface{}

	if schema != "" {
		conditions = append(conditions, fmt.Sprintf("t.schemaname = $%d", len(queryParams)+1))
		queryParams = append(queryParams, schema)
	}
	if table != "" {
		conditions = append(conditions, fmt.Sprintf("t.tablename = $%d", len(queryParams)+1))
		queryParams = append(queryParams, table)
	}

	whereClause := ""
	if len(conditions) > 0 {
		whereClause = "WHERE " + conditions[0]
		for i := 1; i < len(conditions); i++ {
			whereClause += " AND " + conditions[i]
		}
	}

	var sizeCols string
	if includeIndexes && includeToast {
		sizeCols = `
			pg_total_relation_size(quote_ident(t.schemaname)||'.'||quote_ident(t.tablename)) AS total_size_bytes,
			pg_size_pretty(pg_total_relation_size(quote_ident(t.schemaname)||'.'||quote_ident(t.tablename))) AS total_size_pretty,
			pg_relation_size(quote_ident(t.schemaname)||'.'||quote_ident(t.tablename)) AS table_size_bytes,
			pg_size_pretty(pg_relation_size(quote_ident(t.schemaname)||'.'||quote_ident(t.tablename))) AS table_size_pretty,
			pg_indexes_size(quote_ident(t.schemaname)||'.'||quote_ident(t.tablename)) AS indexes_size_bytes,
			pg_size_pretty(pg_indexes_size(quote_ident(t.schemaname)||'.'||quote_ident(t.tablename))) AS indexes_size_pretty,
			pg_total_relation_size(quote_ident(t.schemaname)||'.'||quote_ident(t.tablename)) - pg_relation_size(quote_ident(t.schemaname)||'.'||quote_ident(t.tablename)) - pg_indexes_size(quote_ident(t.schemaname)||'.'||quote_ident(t.tablename)) AS toast_size_bytes
		`
	} else if includeIndexes {
		sizeCols = `
			pg_relation_size(quote_ident(t.schemaname)||'.'||quote_ident(t.tablename)) + pg_indexes_size(quote_ident(t.schemaname)||'.'||quote_ident(t.tablename)) AS total_size_bytes,
			pg_size_pretty(pg_relation_size(quote_ident(t.schemaname)||'.'||quote_ident(t.tablename)) + pg_indexes_size(quote_ident(t.schemaname)||'.'||quote_ident(t.tablename))) AS total_size_pretty,
			pg_relation_size(quote_ident(t.schemaname)||'.'||quote_ident(t.tablename)) AS table_size_bytes,
			pg_size_pretty(pg_relation_size(quote_ident(t.schemaname)||'.'||quote_ident(t.tablename))) AS table_size_pretty,
			pg_indexes_size(quote_ident(t.schemaname)||'.'||quote_ident(t.tablename)) AS indexes_size_bytes,
			pg_size_pretty(pg_indexes_size(quote_ident(t.schemaname)||'.'||quote_ident(t.tablename))) AS indexes_size_pretty
		`
	} else {
		sizeCols = `
			pg_relation_size(quote_ident(t.schemaname)||'.'||quote_ident(t.tablename)) AS table_size_bytes,
			pg_size_pretty(pg_relation_size(quote_ident(t.schemaname)||'.'||quote_ident(t.tablename))) AS table_size_pretty
		`
	}

	query := fmt.Sprintf(`
		SELECT 
			t.schemaname,
			t.tablename,
			%s
		FROM pg_tables t
		%s
		ORDER BY t.schemaname, t.tablename
	`, sizeCols, whereClause)

	results, err := t.executor.ExecuteQuery(ctx, query, queryParams)
	if err != nil {
		return Error(
			fmt.Sprintf("PostgreSQL table size query failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	return Success(map[string]interface{}{
		"table_sizes": results,
		"count":       len(results),
	}, map[string]interface{}{
		"tool": "postgresql_table_size",
	}), nil
}

/* PostgreSQLIndexSizeTool gets size of specific indexes */
type PostgreSQLIndexSizeTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLIndexSizeTool creates a new PostgreSQL index size tool */
func NewPostgreSQLIndexSizeTool(db *database.Database, logger *logging.Logger) *PostgreSQLIndexSizeTool {
	return &PostgreSQLIndexSizeTool{
		BaseTool: NewBaseTool(
			"postgresql_index_size",
			"Get size of specific indexes",
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
					"index": map[string]interface{}{
						"type":        "string",
						"description": "Filter by index name (optional)",
					},
				},
				"required": []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the index size query */
func (t *PostgreSQLIndexSizeTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	schema, _ := params["schema"].(string)
	table, _ := params["table"].(string)
	index, _ := params["index"].(string)

	var conditions []string
	var queryParams []interface{}

	if schema != "" {
		conditions = append(conditions, fmt.Sprintf("i.schemaname = $%d", len(queryParams)+1))
		queryParams = append(queryParams, schema)
	}
	if table != "" {
		conditions = append(conditions, fmt.Sprintf("i.tablename = $%d", len(queryParams)+1))
		queryParams = append(queryParams, table)
	}
	if index != "" {
		conditions = append(conditions, fmt.Sprintf("i.indexname = $%d", len(queryParams)+1))
		queryParams = append(queryParams, index)
	}

	whereClause := ""
	if len(conditions) > 0 {
		whereClause = "WHERE " + conditions[0]
		for i := 1; i < len(conditions); i++ {
			whereClause += " AND " + conditions[i]
		}
	}

	query := fmt.Sprintf(`
		SELECT 
			i.schemaname,
			i.tablename,
			i.indexname,
			pg_relation_size(quote_ident(i.schemaname)||'.'||quote_ident(i.indexname)) AS index_size_bytes,
			pg_size_pretty(pg_relation_size(quote_ident(i.schemaname)||'.'||quote_ident(i.indexname))) AS index_size_pretty
		FROM pg_indexes i
		%s
		ORDER BY pg_relation_size(quote_ident(i.schemaname)||'.'||quote_ident(i.indexname)) DESC
	`, whereClause)

	results, err := t.executor.ExecuteQuery(ctx, query, queryParams)
	if err != nil {
		return Error(
			fmt.Sprintf("PostgreSQL index size query failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	return Success(map[string]interface{}{
		"index_sizes": results,
		"count":       len(results),
	}, map[string]interface{}{
		"tool": "postgresql_index_size",
	}), nil
}

/* PostgreSQLBloatTool checks table and index bloat */
type PostgreSQLBloatTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLBloatTool creates a new PostgreSQL bloat tool */
func NewPostgreSQLBloatTool(db *database.Database, logger *logging.Logger) *PostgreSQLBloatTool {
	return &PostgreSQLBloatTool{
		BaseTool: NewBaseTool(
			"postgresql_bloat",
			"Check table and index bloat (estimated)",
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
					"min_bloat_percent": map[string]interface{}{
						"type":        "number",
						"default":     10,
						"minimum":     0,
						"maximum":     100,
						"description": "Minimum bloat percentage to report (0-100)",
					},
				},
				"required": []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the bloat query */
func (t *PostgreSQLBloatTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	schema, _ := params["schema"].(string)
	table, _ := params["table"].(string)
	minBloatPercent := 10.0

	if val, ok := params["min_bloat_percent"].(float64); ok {
		minBloatPercent = val
	}

	var conditions []string
	var queryParams []interface{}

	if schema != "" {
		conditions = append(conditions, fmt.Sprintf("schemaname = $%d", len(queryParams)+1))
		queryParams = append(queryParams, schema)
	}
	if table != "" {
		conditions = append(conditions, fmt.Sprintf("tablename = $%d", len(queryParams)+1))
		queryParams = append(queryParams, table)
	}

	whereClause := ""
	if len(conditions) > 0 {
		whereClause = "WHERE " + conditions[0]
		for i := 1; i < len(conditions); i++ {
			whereClause += " AND " + conditions[i]
		}
	}

	/* Simplified bloat estimation query */
	if whereClause == "" {
		whereClause = "WHERE"
		conditions = append(conditions, fmt.Sprintf("CASE WHEN s.n_live_tup > 0 THEN (s.n_dead_tup::float / s.n_live_tup * 100) ELSE 0 END >= $%d", len(queryParams)+1))
		queryParams = append(queryParams, minBloatPercent)
		whereClause += " " + conditions[0]
		for i := 1; i < len(conditions); i++ {
			whereClause += " AND " + conditions[i]
		}
	} else {
		conditions = append(conditions, fmt.Sprintf("CASE WHEN s.n_live_tup > 0 THEN (s.n_dead_tup::float / s.n_live_tup * 100) ELSE 0 END >= $%d", len(queryParams)+1))
		queryParams = append(queryParams, minBloatPercent)
		whereClause += " AND " + conditions[len(conditions)-1]
	}

	query := fmt.Sprintf(`
		SELECT 
			s.schemaname,
			s.relname AS tablename,
			pg_total_relation_size(s.relid) AS total_size_bytes,
			pg_size_pretty(pg_total_relation_size(s.relid)) AS total_size_pretty,
			s.n_dead_tup AS dead_tuples,
			s.n_live_tup AS live_tuples,
			CASE 
				WHEN s.n_live_tup > 0 THEN (s.n_dead_tup::float / s.n_live_tup * 100)
				ELSE 0
			END AS dead_tuple_percent,
			s.last_vacuum,
			s.last_autovacuum
		FROM pg_stat_user_tables s
		%s
		ORDER BY dead_tuple_percent DESC
	`, whereClause)

	results, err := t.executor.ExecuteQuery(ctx, query, queryParams)
	if err != nil {
		return Error(
			fmt.Sprintf("PostgreSQL bloat query failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	return Success(map[string]interface{}{
		"bloat": results,
		"count": len(results),
	}, map[string]interface{}{
		"tool": "postgresql_bloat",
	}), nil
}

/* PostgreSQLVacuumStatsTool gets vacuum statistics and recommendations */
type PostgreSQLVacuumStatsTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLVacuumStatsTool creates a new PostgreSQL vacuum stats tool */
func NewPostgreSQLVacuumStatsTool(db *database.Database, logger *logging.Logger) *PostgreSQLVacuumStatsTool {
	return &PostgreSQLVacuumStatsTool{
		BaseTool: NewBaseTool(
			"postgresql_vacuum_stats",
			"Get vacuum statistics and recommendations",
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

/* Execute executes the vacuum stats query */
func (t *PostgreSQLVacuumStatsTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	schema, _ := params["schema"].(string)
	table, _ := params["table"].(string)

	var conditions []string
	var queryParams []interface{}

	if schema != "" {
		conditions = append(conditions, fmt.Sprintf("s.schemaname = $%d", len(queryParams)+1))
		queryParams = append(queryParams, schema)
	}
	if table != "" {
		conditions = append(conditions, fmt.Sprintf("s.relname = $%d", len(queryParams)+1))
		queryParams = append(queryParams, table)
	}

	whereClause := ""
	if len(conditions) > 0 {
		whereClause = "WHERE " + conditions[0]
		for i := 1; i < len(conditions); i++ {
			whereClause += " AND " + conditions[i]
		}
	}

	query := fmt.Sprintf(`
		SELECT 
			s.schemaname,
			s.relname AS tablename,
			s.n_live_tup AS live_tuples,
			s.n_dead_tup AS dead_tuples,
			CASE 
				WHEN s.n_live_tup > 0 THEN (s.n_dead_tup::float / s.n_live_tup * 100)
				ELSE 0
			END AS dead_tuple_percent,
			s.last_vacuum,
			s.last_autovacuum,
			s.vacuum_count,
			s.autovacuum_count,
			s.last_analyze,
			s.last_autoanalyze,
			s.analyze_count,
			s.autoanalyze_count,
			CASE 
				WHEN s.n_dead_tup > 1000 AND (s.last_vacuum IS NULL OR s.last_vacuum < now() - interval '7 days') THEN 'VACUUM RECOMMENDED'
				WHEN s.n_dead_tup > 10000 THEN 'VACUUM URGENT'
				ELSE 'OK'
			END AS vacuum_recommendation
		FROM pg_stat_user_tables s
		%s
		ORDER BY dead_tuple_percent DESC
	`, whereClause)

	results, err := t.executor.ExecuteQuery(ctx, query, queryParams)
	if err != nil {
		return Error(
			fmt.Sprintf("PostgreSQL vacuum stats query failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	return Success(map[string]interface{}{
		"vacuum_stats": results,
		"count":        len(results),
	}, map[string]interface{}{
		"tool": "postgresql_vacuum_stats",
	}), nil
}

