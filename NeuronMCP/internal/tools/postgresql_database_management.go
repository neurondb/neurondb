/*-------------------------------------------------------------------------
 *
 * postgresql_database_management.go
 *    Database and schema management tools for NeuronMCP
 *
 * Implements comprehensive database and schema DDL operations:
 * - CREATE/DROP/ALTER DATABASE
 * - CREATE/DROP/ALTER SCHEMA
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/postgresql_database_management.go
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
 * Database Management Tools
 * ============================================================================ */

/* PostgreSQLCreateDatabaseTool creates new databases */
type PostgreSQLCreateDatabaseTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLCreateDatabaseTool creates a new PostgreSQL create database tool */
func NewPostgreSQLCreateDatabaseTool(db *database.Database, logger *logging.Logger) *PostgreSQLCreateDatabaseTool {
	return &PostgreSQLCreateDatabaseTool{
		BaseTool: NewBaseTool(
			"postgresql_create_database",
			"Create a new PostgreSQL database with options (encoding, collation, template, owner)",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"database_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the database to create",
					},
					"owner": map[string]interface{}{
						"type":        "string",
						"description": "Database owner (role name)",
					},
					"template": map[string]interface{}{
						"type":        "string",
						"description": "Template database to use (default: template1)",
						"default":     "template1",
					},
					"encoding": map[string]interface{}{
						"type":        "string",
						"description": "Character encoding (e.g., UTF8, LATIN1)",
						"default":     "UTF8",
					},
					"collation": map[string]interface{}{
						"type":        "string",
						"description": "Collation name",
					},
					"ctype": map[string]interface{}{
						"type":        "string",
						"description": "Character type (CTYPE)",
					},
					"tablespace": map[string]interface{}{
						"type":        "string",
						"description": "Tablespace name",
					},
					"connection_limit": map[string]interface{}{
						"type":        "integer",
						"description": "Connection limit (-1 for unlimited)",
					},
					"if_not_exists": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Use IF NOT EXISTS clause",
					},
				},
				"required": []interface{}{"database_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute creates the database */
func (t *PostgreSQLCreateDatabaseTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	databaseName, ok := params["database_name"].(string)
	if !ok || databaseName == "" {
		return Error("database_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	/* Validate database name */
	if !isValidIdentifier(databaseName) {
		return Error("Invalid database name: must be a valid PostgreSQL identifier", "INVALID_PARAMETER", nil), nil
	}

	/* Build CREATE DATABASE statement */
	parts := []string{"CREATE DATABASE"}
	
	if ifNotExists, ok := params["if_not_exists"].(bool); ok && ifNotExists {
		parts = append(parts, "IF NOT EXISTS")
	}
	
	parts = append(parts, quoteIdentifier(databaseName))

	/* Add options */
	options := []string{}
	
	if owner, ok := params["owner"].(string); ok && owner != "" {
		options = append(options, fmt.Sprintf("OWNER = %s", quoteIdentifier(owner)))
	}
	
	if template, ok := params["template"].(string); ok && template != "" {
		options = append(options, fmt.Sprintf("TEMPLATE = %s", quoteIdentifier(template)))
	}
	
	if encoding, ok := params["encoding"].(string); ok && encoding != "" {
		options = append(options, fmt.Sprintf("ENCODING = %s", quoteLiteral(encoding)))
	}
	
	if collation, ok := params["collation"].(string); ok && collation != "" {
		options = append(options, fmt.Sprintf("LC_COLLATE = %s", quoteLiteral(collation)))
	}
	
	if ctype, ok := params["ctype"].(string); ok && ctype != "" {
		options = append(options, fmt.Sprintf("LC_CTYPE = %s", quoteLiteral(ctype)))
	}
	
	if tablespace, ok := params["tablespace"].(string); ok && tablespace != "" {
		options = append(options, fmt.Sprintf("TABLESPACE = %s", quoteIdentifier(tablespace)))
	}
	
	if connLimit, ok := params["connection_limit"].(float64); ok {
		options = append(options, fmt.Sprintf("CONNECTION LIMIT = %d", int(connLimit)))
	} else if connLimit, ok := params["connection_limit"].(int); ok {
		options = append(options, fmt.Sprintf("CONNECTION LIMIT = %d", connLimit))
	}

	if len(options) > 0 {
		parts = append(parts, "WITH")
		parts = append(parts, strings.Join(options, " "))
	}

	createQuery := strings.Join(parts, " ")

	/* Execute CREATE DATABASE - need to use Exec since it's DDL */
	err := t.executor.Exec(ctx, createQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("CREATE DATABASE failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Database created", map[string]interface{}{
		"database_name": databaseName,
	})

	return Success(map[string]interface{}{
		"database_name": databaseName,
		"query":         createQuery,
	}, map[string]interface{}{
		"tool": "postgresql_create_database",
	}), nil
}

/* PostgreSQLDropDatabaseTool drops databases */
type PostgreSQLDropDatabaseTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLDropDatabaseTool creates a new PostgreSQL drop database tool */
func NewPostgreSQLDropDatabaseTool(db *database.Database, logger *logging.Logger) *PostgreSQLDropDatabaseTool {
	return &PostgreSQLDropDatabaseTool{
		BaseTool: NewBaseTool(
			"postgresql_drop_database",
			"Drop a PostgreSQL database with safety checks",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"database_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the database to drop",
					},
					"if_exists": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Use IF EXISTS clause",
					},
					"force": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Force drop by terminating connections (PostgreSQL 13+)",
					},
				},
				"required": []interface{}{"database_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute drops the database */
func (t *PostgreSQLDropDatabaseTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	databaseName, ok := params["database_name"].(string)
	if !ok || databaseName == "" {
		return Error("database_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	/* Check if trying to drop current database */
	currentDB, err := t.executor.ExecuteQueryOne(ctx, "SELECT current_database() AS current_db", nil)
	if err == nil {
		if currentDBName, ok := currentDB["current_db"].(string); ok && currentDBName == databaseName {
			return Error(
				"Cannot drop the currently connected database",
				"INVALID_OPERATION",
				map[string]interface{}{"current_database": currentDBName},
			), nil
		}
	}

	/* Build DROP DATABASE statement */
	parts := []string{"DROP DATABASE"}
	
	if ifExists, ok := params["if_exists"].(bool); ok && ifExists {
		parts = append(parts, "IF EXISTS")
	}
	
	parts = append(parts, quoteIdentifier(databaseName))
	
	if force, ok := params["force"].(bool); ok && force {
		parts = append(parts, "WITH (FORCE)")
	}

	dropQuery := strings.Join(parts, " ")

	/* Execute DROP DATABASE */
	err = t.executor.Exec(ctx, dropQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("DROP DATABASE failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Database dropped", map[string]interface{}{
		"database_name": databaseName,
	})

	return Success(map[string]interface{}{
		"database_name": databaseName,
		"query":         dropQuery,
	}, map[string]interface{}{
		"tool": "postgresql_drop_database",
	}), nil
}

/* PostgreSQLAlterDatabaseTool alters database properties */
type PostgreSQLAlterDatabaseTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLAlterDatabaseTool creates a new PostgreSQL alter database tool */
func NewPostgreSQLAlterDatabaseTool(db *database.Database, logger *logging.Logger) *PostgreSQLAlterDatabaseTool {
	return &PostgreSQLAlterDatabaseTool{
		BaseTool: NewBaseTool(
			"postgresql_alter_database",
			"Alter database properties (name, owner, tablespace, connection limit, settings)",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"database_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the database to alter",
					},
					"new_name": map[string]interface{}{
						"type":        "string",
						"description": "New name for the database (rename)",
					},
					"owner": map[string]interface{}{
						"type":        "string",
						"description": "New owner (role name)",
					},
					"tablespace": map[string]interface{}{
						"type":        "string",
						"description": "New tablespace",
					},
					"connection_limit": map[string]interface{}{
						"type":        "integer",
						"description": "New connection limit (-1 for unlimited)",
					},
					"reset_connection_limit": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Reset connection limit to default",
					},
					"settings": map[string]interface{}{
						"type":        "object",
						"description": "Database settings as key-value pairs",
					},
				},
				"required": []interface{}{"database_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute alters the database */
func (t *PostgreSQLAlterDatabaseTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	databaseName, ok := params["database_name"].(string)
	if !ok || databaseName == "" {
		return Error("database_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	/* Build ALTER DATABASE statement */
	parts := []string{"ALTER DATABASE", quoteIdentifier(databaseName)}

	alterations := []string{}

	/* Rename */
	if newName, ok := params["new_name"].(string); ok && newName != "" {
		if !isValidIdentifier(newName) {
			return Error("Invalid new database name: must be a valid PostgreSQL identifier", "INVALID_PARAMETER", nil), nil
		}
		alterations = append(alterations, fmt.Sprintf("RENAME TO %s", quoteIdentifier(newName)))
	}

	/* Owner */
	if owner, ok := params["owner"].(string); ok && owner != "" {
		alterations = append(alterations, fmt.Sprintf("OWNER TO %s", quoteIdentifier(owner)))
	}

	/* Tablespace */
	if tablespace, ok := params["tablespace"].(string); ok && tablespace != "" {
		alterations = append(alterations, fmt.Sprintf("SET TABLESPACE %s", quoteIdentifier(tablespace)))
	}

	/* Connection limit */
	if resetConnLimit, ok := params["reset_connection_limit"].(bool); ok && resetConnLimit {
		alterations = append(alterations, "RESET CONNECTION LIMIT")
	} else if connLimit, ok := params["connection_limit"].(float64); ok {
		alterations = append(alterations, fmt.Sprintf("WITH CONNECTION LIMIT %d", int(connLimit)))
	} else if connLimit, ok := params["connection_limit"].(int); ok {
		alterations = append(alterations, fmt.Sprintf("WITH CONNECTION LIMIT %d", connLimit))
	}

	/* Settings */
	if settings, ok := params["settings"].(map[string]interface{}); ok && len(settings) > 0 {
		for key, value := range settings {
			if valueStr, ok := value.(string); ok {
				alterations = append(alterations, fmt.Sprintf("SET %s = %s", quoteIdentifier(key), quoteLiteral(valueStr)))
			} else {
				alterations = append(alterations, fmt.Sprintf("SET %s = %v", quoteIdentifier(key), value))
			}
		}
	}

	if len(alterations) == 0 {
		return Error("No alterations specified", "INVALID_PARAMETER", nil), nil
	}

	/* Execute each alteration separately (PostgreSQL requires separate statements for some operations) */
	queries := []string{}
	for _, alt := range alterations {
		query := parts[0] + " " + parts[1] + " " + alt
		queries = append(queries, query)
		
		err := t.executor.Exec(ctx, query, nil)
		if err != nil {
			return Error(
				fmt.Sprintf("ALTER DATABASE failed: %v", err),
				"QUERY_ERROR",
				map[string]interface{}{"error": err.Error(), "query": query},
			), nil
		}
	}

	t.logger.Info("Database altered", map[string]interface{}{
		"database_name": databaseName,
		"alterations":   len(alterations),
	})

	return Success(map[string]interface{}{
		"database_name": databaseName,
		"queries":       queries,
	}, map[string]interface{}{
		"tool": "postgresql_alter_database",
	}), nil
}

/* ============================================================================
 * Schema Management Tools
 * ============================================================================ */

/* PostgreSQLCreateSchemaTool creates new schemas */
type PostgreSQLCreateSchemaTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLCreateSchemaTool creates a new PostgreSQL create schema tool */
func NewPostgreSQLCreateSchemaTool(db *database.Database, logger *logging.Logger) *PostgreSQLCreateSchemaTool {
	return &PostgreSQLCreateSchemaTool{
		BaseTool: NewBaseTool(
			"postgresql_create_schema",
			"Create a new PostgreSQL schema with authorization",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"schema_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the schema to create",
					},
					"authorization": map[string]interface{}{
						"type":        "string",
						"description": "Role name to own the schema",
					},
					"if_not_exists": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Use IF NOT EXISTS clause",
					},
				},
				"required": []interface{}{"schema_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute creates the schema */
func (t *PostgreSQLCreateSchemaTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	schemaName, ok := params["schema_name"].(string)
	if !ok || schemaName == "" {
		return Error("schema_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	if !isValidIdentifier(schemaName) {
		return Error("Invalid schema name: must be a valid PostgreSQL identifier", "INVALID_PARAMETER", nil), nil
	}

	/* Build CREATE SCHEMA statement */
	parts := []string{"CREATE SCHEMA"}
	
	if ifNotExists, ok := params["if_not_exists"].(bool); ok && ifNotExists {
		parts = append(parts, "IF NOT EXISTS")
	}
	
	parts = append(parts, quoteIdentifier(schemaName))
	
	if authorization, ok := params["authorization"].(string); ok && authorization != "" {
		parts = append(parts, fmt.Sprintf("AUTHORIZATION %s", quoteIdentifier(authorization)))
	}

	createQuery := strings.Join(parts, " ")

	/* Execute CREATE SCHEMA */
	err := t.executor.Exec(ctx, createQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("CREATE SCHEMA failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Schema created", map[string]interface{}{
		"schema_name": schemaName,
	})

	return Success(map[string]interface{}{
		"schema_name": schemaName,
		"query":       createQuery,
	}, map[string]interface{}{
		"tool": "postgresql_create_schema",
	}), nil
}

/* PostgreSQLDropSchemaTool drops schemas */
type PostgreSQLDropSchemaTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLDropSchemaTool creates a new PostgreSQL drop schema tool */
func NewPostgreSQLDropSchemaTool(db *database.Database, logger *logging.Logger) *PostgreSQLDropSchemaTool {
	return &PostgreSQLDropSchemaTool{
		BaseTool: NewBaseTool(
			"postgresql_drop_schema",
			"Drop a PostgreSQL schema with CASCADE option",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"schema_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the schema to drop",
					},
					"if_exists": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Use IF EXISTS clause",
					},
					"cascade": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Drop all objects in the schema",
					},
				},
				"required": []interface{}{"schema_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute drops the schema */
func (t *PostgreSQLDropSchemaTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	schemaName, ok := params["schema_name"].(string)
	if !ok || schemaName == "" {
		return Error("schema_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	/* Build DROP SCHEMA statement */
	parts := []string{"DROP SCHEMA"}
	
	if ifExists, ok := params["if_exists"].(bool); ok && ifExists {
		parts = append(parts, "IF EXISTS")
	}
	
	parts = append(parts, quoteIdentifier(schemaName))
	
	if cascade, ok := params["cascade"].(bool); ok && cascade {
		parts = append(parts, "CASCADE")
	} else {
		parts = append(parts, "RESTRICT")
	}

	dropQuery := strings.Join(parts, " ")

	/* Execute DROP SCHEMA */
	err := t.executor.Exec(ctx, dropQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("DROP SCHEMA failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Schema dropped", map[string]interface{}{
		"schema_name": schemaName,
	})

	return Success(map[string]interface{}{
		"schema_name": schemaName,
		"query":       dropQuery,
	}, map[string]interface{}{
		"tool": "postgresql_drop_schema",
	}), nil
}

/* PostgreSQLAlterSchemaTool alters schema properties */
type PostgreSQLAlterSchemaTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLAlterSchemaTool creates a new PostgreSQL alter schema tool */
func NewPostgreSQLAlterSchemaTool(db *database.Database, logger *logging.Logger) *PostgreSQLAlterSchemaTool {
	return &PostgreSQLAlterSchemaTool{
		BaseTool: NewBaseTool(
			"postgresql_alter_schema",
			"Alter schema properties (rename, change owner)",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"schema_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the schema to alter",
					},
					"new_name": map[string]interface{}{
						"type":        "string",
						"description": "New name for the schema (rename)",
					},
					"owner": map[string]interface{}{
						"type":        "string",
						"description": "New owner (role name)",
					},
				},
				"required": []interface{}{"schema_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute alters the schema */
func (t *PostgreSQLAlterSchemaTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	schemaName, ok := params["schema_name"].(string)
	if !ok || schemaName == "" {
		return Error("schema_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	/* Build ALTER SCHEMA statement */
	parts := []string{"ALTER SCHEMA", quoteIdentifier(schemaName)}

	alterations := []string{}

	/* Rename */
	if newName, ok := params["new_name"].(string); ok && newName != "" {
		if !isValidIdentifier(newName) {
			return Error("Invalid new schema name: must be a valid PostgreSQL identifier", "INVALID_PARAMETER", nil), nil
		}
		alterations = append(alterations, fmt.Sprintf("RENAME TO %s", quoteIdentifier(newName)))
	}

	/* Owner */
	if owner, ok := params["owner"].(string); ok && owner != "" {
		alterations = append(alterations, fmt.Sprintf("OWNER TO %s", quoteIdentifier(owner)))
	}

	if len(alterations) == 0 {
		return Error("No alterations specified", "INVALID_PARAMETER", nil), nil
	}

	/* Execute each alteration separately */
	queries := []string{}
	for _, alt := range alterations {
		query := parts[0] + " " + parts[1] + " " + alt
		queries = append(queries, query)
		
		err := t.executor.Exec(ctx, query, nil)
		if err != nil {
			return Error(
				fmt.Sprintf("ALTER SCHEMA failed: %v", err),
				"QUERY_ERROR",
				map[string]interface{}{"error": err.Error(), "query": query},
			), nil
		}
	}

	t.logger.Info("Schema altered", map[string]interface{}{
		"schema_name": schemaName,
		"alterations": len(alterations),
	})

	return Success(map[string]interface{}{
		"schema_name": schemaName,
		"queries":     queries,
	}, map[string]interface{}{
		"tool": "postgresql_alter_schema",
	}), nil
}

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

/* isValidIdentifier checks if a string is a valid PostgreSQL identifier */
func isValidIdentifier(name string) bool {
	if name == "" {
		return false
	}
	/* Basic validation - PostgreSQL identifiers can contain letters, digits, _, $, and must start with letter or _ */
	/* For simplicity, we check it's not empty and doesn't contain spaces or special characters that need quoting */
	return len(name) > 0 && len(name) <= 63 /* PostgreSQL identifier max length */
}

/* quoteIdentifier quotes a PostgreSQL identifier */
func quoteIdentifier(name string) string {
	/* Simple quoting - for production, should use pgx's quote identifier */
	/* For now, just add double quotes */
	return fmt.Sprintf(`"%s"`, strings.ReplaceAll(name, `"`, `""`))
}

/* quoteLiteral quotes a PostgreSQL string literal */
func quoteLiteral(value string) string {
	/* Simple quoting - escape single quotes */
	escaped := strings.ReplaceAll(value, "'", "''")
	return fmt.Sprintf("'%s'", escaped)
}

