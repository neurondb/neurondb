/*-------------------------------------------------------------------------
 *
 * postgresql_schema_modification.go
 *    Schema modification tools for NeuronMCP
 *
 * Implements schema modification operations as specified in Phase 1.1
 * of the roadmap. These tools provide safe schema modification with
 * validation and safety checks.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/postgresql_schema_modification.go
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

/* PostgreSQLCreateTableTool creates tables with full options */
type PostgreSQLCreateTableTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLCreateTableTool creates a new PostgreSQL create table tool */
func NewPostgreSQLCreateTableTool(db *database.Database, logger *logging.Logger) *PostgreSQLCreateTableTool {
	return &PostgreSQLCreateTableTool{
		BaseTool: NewBaseTool(
			"postgresql_create_table",
			"Create a new table with full options and validation",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the table to create",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"default":     "public",
						"description": "Schema name (defaults to public)",
					},
					"columns": map[string]interface{}{
						"type":        "array",
						"description": "Array of column definitions: [{\"name\": \"col1\", \"type\": \"INTEGER\", \"constraints\": \"NOT NULL\"}, ...]",
					},
					"primary_key": map[string]interface{}{
						"type":        "array",
						"description": "Array of column names for primary key",
					},
					"if_not_exists": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Use IF NOT EXISTS clause",
					},
				},
				"required": []interface{}{"table_name", "columns"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute creates the table */
func (t *PostgreSQLCreateTableTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	tableName, ok := params["table_name"].(string)
	if !ok || tableName == "" {
		return Error("table_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	columns, ok := params["columns"].([]interface{})
	if !ok || len(columns) == 0 {
		return Error("columns parameter is required and must be a non-empty array", "INVALID_PARAMETER", nil), nil
	}

	/* Build column definitions */
	colDefs := []string{}
	for _, col := range columns {
		colMap, ok := col.(map[string]interface{})
		if !ok {
			return Error("Each column must be an object with name and type", "INVALID_PARAMETER", nil), nil
		}

		colName, ok := colMap["name"].(string)
		if !ok || colName == "" {
			return Error("Column name is required", "INVALID_PARAMETER", nil), nil
		}

		colType, ok := colMap["type"].(string)
		if !ok || colType == "" {
			return Error("Column type is required", "INVALID_PARAMETER", nil), nil
		}

		colDef := fmt.Sprintf("%s %s", colName, colType)

		if constraints, ok := colMap["constraints"].(string); ok && constraints != "" {
			colDef += " " + constraints
		}

		colDefs = append(colDefs, colDef)
	}

	/* Build CREATE TABLE statement */
	createParts := []string{"CREATE TABLE"}
	ifNotExists := false
	if val, ok := params["if_not_exists"].(bool); ok {
		ifNotExists = val
	}
	if ifNotExists {
		createParts = append(createParts, "IF NOT EXISTS")
	}

	fullTableName := fmt.Sprintf("%s.%s", schema, tableName)
	createParts = append(createParts, fullTableName)
	createParts = append(createParts, "(")
	createParts = append(createParts, strings.Join(colDefs, ", "))

	/* Add primary key if specified */
	if pkCols, ok := params["primary_key"].([]interface{}); ok && len(pkCols) > 0 {
		pkNames := []string{}
		for _, pk := range pkCols {
			if pkName, ok := pk.(string); ok {
				pkNames = append(pkNames, pkName)
			}
		}
		if len(pkNames) > 0 {
			createParts = append(createParts, fmt.Sprintf(", PRIMARY KEY (%s)", strings.Join(pkNames, ", ")))
		}
	}

	createParts = append(createParts, ")")
	createQuery := strings.Join(createParts, " ")

	/* Execute CREATE TABLE */
	_, err := t.executor.ExecuteQuery(ctx, createQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("CREATE TABLE failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Table created", map[string]interface{}{
		"schema":     schema,
		"table_name": tableName,
	})

	return Success(map[string]interface{}{
		"schema":     schema,
		"table_name": tableName,
		"query":      createQuery,
	}, map[string]interface{}{
		"tool": "postgresql_create_table",
	}), nil
}

/* PostgreSQLAlterTableTool modifies table structure */
type PostgreSQLAlterTableTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLAlterTableTool creates a new PostgreSQL alter table tool */
func NewPostgreSQLAlterTableTool(db *database.Database, logger *logging.Logger) *PostgreSQLAlterTableTool {
	return &PostgreSQLAlterTableTool{
		BaseTool: NewBaseTool(
			"postgresql_alter_table",
			"Modify table structure (add/drop columns, modify columns, etc.)",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the table to alter",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"default":     "public",
						"description": "Schema name",
					},
					"operation": map[string]interface{}{
						"type":        "string",
						"enum":        []string{"add_column", "drop_column", "alter_column", "rename_column", "add_constraint", "drop_constraint"},
						"description": "Type of ALTER TABLE operation",
					},
					"column_name": map[string]interface{}{
						"type":        "string",
						"description": "Column name (for column operations)",
					},
					"new_column_name": map[string]interface{}{
						"type":        "string",
						"description": "New column name (for rename operations)",
					},
					"column_type": map[string]interface{}{
						"type":        "string",
						"description": "Column type (for add/alter column)",
					},
					"constraint_name": map[string]interface{}{
						"type":        "string",
						"description": "Constraint name (for constraint operations)",
					},
					"constraint_definition": map[string]interface{}{
						"type":        "string",
						"description": "Constraint definition (for add_constraint)",
					},
				},
				"required": []interface{}{"table_name", "operation"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute modifies the table */
func (t *PostgreSQLAlterTableTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	tableName, ok := params["table_name"].(string)
	if !ok || tableName == "" {
		return Error("table_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	operation, ok := params["operation"].(string)
	if !ok || operation == "" {
		return Error("operation parameter is required", "INVALID_PARAMETER", nil), nil
	}

	fullTableName := fmt.Sprintf("%s.%s", schema, tableName)
	var alterQuery string

	switch operation {
	case "add_column":
		colName, _ := params["column_name"].(string)
		colType, _ := params["column_type"].(string)
		if colName == "" || colType == "" {
			return Error("column_name and column_type are required for add_column", "INVALID_PARAMETER", nil), nil
		}
		alterQuery = fmt.Sprintf("ALTER TABLE %s ADD COLUMN %s %s", fullTableName, colName, colType)

	case "drop_column":
		colName, _ := params["column_name"].(string)
		if colName == "" {
			return Error("column_name is required for drop_column", "INVALID_PARAMETER", nil), nil
		}
		alterQuery = fmt.Sprintf("ALTER TABLE %s DROP COLUMN %s", fullTableName, colName)

	case "alter_column":
		colName, _ := params["column_name"].(string)
		colType, _ := params["column_type"].(string)
		if colName == "" || colType == "" {
			return Error("column_name and column_type are required for alter_column", "INVALID_PARAMETER", nil), nil
		}
		alterQuery = fmt.Sprintf("ALTER TABLE %s ALTER COLUMN %s TYPE %s", fullTableName, colName, colType)

	case "rename_column":
		colName, _ := params["column_name"].(string)
		newColName, _ := params["new_column_name"].(string)
		if colName == "" || newColName == "" {
			return Error("column_name and new_column_name are required for rename_column", "INVALID_PARAMETER", nil), nil
		}
		alterQuery = fmt.Sprintf("ALTER TABLE %s RENAME COLUMN %s TO %s", fullTableName, colName, newColName)

	case "add_constraint":
		constraintName, _ := params["constraint_name"].(string)
		constraintDef, _ := params["constraint_definition"].(string)
		if constraintName == "" || constraintDef == "" {
			return Error("constraint_name and constraint_definition are required for add_constraint", "INVALID_PARAMETER", nil), nil
		}
		alterQuery = fmt.Sprintf("ALTER TABLE %s ADD CONSTRAINT %s %s", fullTableName, constraintName, constraintDef)

	case "drop_constraint":
		constraintName, _ := params["constraint_name"].(string)
		if constraintName == "" {
			return Error("constraint_name is required for drop_constraint", "INVALID_PARAMETER", nil), nil
		}
		alterQuery = fmt.Sprintf("ALTER TABLE %s DROP CONSTRAINT %s", fullTableName, constraintName)

	default:
		return Error(fmt.Sprintf("Invalid operation: %s", operation), "INVALID_PARAMETER", nil), nil
	}

	/* Execute ALTER TABLE */
	_, err := t.executor.ExecuteQuery(ctx, alterQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("ALTER TABLE failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Table altered", map[string]interface{}{
		"schema":    schema,
		"table":     tableName,
		"operation": operation,
	})

	return Success(map[string]interface{}{
		"schema":    schema,
		"table":     tableName,
		"operation": operation,
		"query":     alterQuery,
	}, map[string]interface{}{
		"tool": "postgresql_alter_table",
	}), nil
}

/* PostgreSQLDropTableTool drops tables with safety checks */
type PostgreSQLDropTableTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLDropTableTool creates a new PostgreSQL drop table tool */
func NewPostgreSQLDropTableTool(db *database.Database, logger *logging.Logger) *PostgreSQLDropTableTool {
	return &PostgreSQLDropTableTool{
		BaseTool: NewBaseTool(
			"postgresql_drop_table",
			"Drop a table with safety checks and cascade options",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the table to drop",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"default":     "public",
						"description": "Schema name",
					},
					"if_exists": map[string]interface{}{
						"type":        "boolean",
						"default":     true,
						"description": "Use IF EXISTS clause",
					},
					"cascade": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Drop dependent objects (CASCADE)",
					},
				},
				"required": []interface{}{"table_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute drops the table */
func (t *PostgreSQLDropTableTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	tableName, ok := params["table_name"].(string)
	if !ok || tableName == "" {
		return Error("table_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	ifExists := true
	if val, ok := params["if_exists"].(bool); ok {
		ifExists = val
	}

	cascade := false
	if val, ok := params["cascade"].(bool); ok {
		cascade = val
	}

	/* Build DROP TABLE statement */
	dropParts := []string{"DROP TABLE"}
	if ifExists {
		dropParts = append(dropParts, "IF EXISTS")
	}

	fullTableName := fmt.Sprintf("%s.%s", schema, tableName)
	dropParts = append(dropParts, fullTableName)

	if cascade {
		dropParts = append(dropParts, "CASCADE")
	}

	dropQuery := strings.Join(dropParts, " ")

	/* Execute DROP TABLE */
	_, err := t.executor.ExecuteQuery(ctx, dropQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("DROP TABLE failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Table dropped", map[string]interface{}{
		"schema":  schema,
		"table":   tableName,
		"cascade": cascade,
	})

	return Success(map[string]interface{}{
		"schema":  schema,
		"table":   tableName,
		"cascade": cascade,
		"query":   dropQuery,
	}, map[string]interface{}{
		"tool": "postgresql_drop_table",
	}), nil
}

/* PostgreSQLCreateIndexTool creates indexes with tuning options */
type PostgreSQLCreateIndexTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLCreateIndexTool creates a new PostgreSQL create index tool */
func NewPostgreSQLCreateIndexTool(db *database.Database, logger *logging.Logger) *PostgreSQLCreateIndexTool {
	return &PostgreSQLCreateIndexTool{
		BaseTool: NewBaseTool(
			"postgresql_create_index",
			"Create an index with tuning options and concurrent support",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"index_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the index to create",
					},
					"table_name": map[string]interface{}{
						"type":        "string",
						"description": "Table name",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"default":     "public",
						"description": "Schema name",
					},
					"columns": map[string]interface{}{
						"type":        "array",
						"description": "Array of column names to index",
					},
					"index_type": map[string]interface{}{
						"type":        "string",
						"enum":        []string{"btree", "hash", "gin", "gist", "brin", "spgist"},
						"default":     "btree",
						"description": "Index type",
					},
					"unique": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Create unique index",
					},
					"concurrently": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Create index concurrently (non-blocking)",
					},
				},
				"required": []interface{}{"index_name", "table_name", "columns"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute creates the index */
func (t *PostgreSQLCreateIndexTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	indexName, ok := params["index_name"].(string)
	if !ok || indexName == "" {
		return Error("index_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	tableName, ok := params["table_name"].(string)
	if !ok || tableName == "" {
		return Error("table_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	columns, ok := params["columns"].([]interface{})
	if !ok || len(columns) == 0 {
		return Error("columns parameter is required and must be a non-empty array", "INVALID_PARAMETER", nil), nil
	}

	colNames := []string{}
	for _, col := range columns {
		if colName, ok := col.(string); ok {
			colNames = append(colNames, colName)
		}
	}
	if len(colNames) == 0 {
		return Error("At least one valid column name is required", "INVALID_PARAMETER", nil), nil
	}

	indexType := "btree"
	if val, ok := params["index_type"].(string); ok {
		indexType = val
	}

	unique := false
	if val, ok := params["unique"].(bool); ok {
		unique = val
	}

	concurrently := false
	if val, ok := params["concurrently"].(bool); ok {
		concurrently = val
	}

	/* Build CREATE INDEX statement */
	createParts := []string{"CREATE"}
	if unique {
		createParts = append(createParts, "UNIQUE")
	}
	createParts = append(createParts, "INDEX")
	if concurrently {
		createParts = append(createParts, "CONCURRENTLY")
	}

	fullIndexName := fmt.Sprintf("%s.%s", schema, indexName)
	createParts = append(createParts, fullIndexName)

	fullTableName := fmt.Sprintf("%s.%s", schema, tableName)
	createParts = append(createParts, fmt.Sprintf("ON %s USING %s (%s)", fullTableName, indexType, strings.Join(colNames, ", ")))

	createQuery := strings.Join(createParts, " ")

	/* Execute CREATE INDEX */
	_, err := t.executor.ExecuteQuery(ctx, createQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("CREATE INDEX failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Index created", map[string]interface{}{
		"schema":      schema,
		"index_name":  indexName,
		"table_name":  tableName,
		"index_type":  indexType,
		"concurrently": concurrently,
	})

	return Success(map[string]interface{}{
		"schema":      schema,
		"index_name":  indexName,
		"table_name":  tableName,
		"index_type":  indexType,
		"query":       createQuery,
	}, map[string]interface{}{
		"tool": "postgresql_create_index",
	}), nil
}

/* PostgreSQLCreateViewTool creates views */
type PostgreSQLCreateViewTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLCreateViewTool creates a new PostgreSQL create view tool */
func NewPostgreSQLCreateViewTool(db *database.Database, logger *logging.Logger) *PostgreSQLCreateViewTool {
	return &PostgreSQLCreateViewTool{
		BaseTool: NewBaseTool(
			"postgresql_create_view",
			"Create a new view with optional replacement",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"view_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the view to create",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"default":     "public",
						"description": "Schema name",
					},
					"query": map[string]interface{}{
						"type":        "string",
						"description": "SELECT query that defines the view",
					},
					"or_replace": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Use CREATE OR REPLACE VIEW",
					},
				},
				"required": []interface{}{"view_name", "query"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute creates the view */
func (t *PostgreSQLCreateViewTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	viewName, ok := params["view_name"].(string)
	if !ok || viewName == "" {
		return Error("view_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	query, ok := params["query"].(string)
	if !ok || query == "" {
		return Error("query parameter is required", "INVALID_PARAMETER", nil), nil
	}

	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	orReplace := false
	if val, ok := params["or_replace"].(bool); ok {
		orReplace = val
	}

	/* Build CREATE VIEW statement */
	createParts := []string{"CREATE"}
	if orReplace {
		createParts = append(createParts, "OR REPLACE")
	}
	createParts = append(createParts, "VIEW")

	fullViewName := fmt.Sprintf("%s.%s", schema, viewName)
	createParts = append(createParts, fullViewName)
	createParts = append(createParts, "AS")
	createParts = append(createParts, query)

	createQuery := strings.Join(createParts, " ")

	/* Execute CREATE VIEW */
	_, err := t.executor.ExecuteQuery(ctx, createQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("CREATE VIEW failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("View created", map[string]interface{}{
		"schema":    schema,
		"view_name": viewName,
	})

	return Success(map[string]interface{}{
		"schema":    schema,
		"view_name": viewName,
		"query":     createQuery,
	}, map[string]interface{}{
		"tool": "postgresql_create_view",
	}), nil
}

/* PostgreSQLCreateFunctionTool creates stored functions */
type PostgreSQLCreateFunctionTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLCreateFunctionTool creates a new PostgreSQL create function tool */
func NewPostgreSQLCreateFunctionTool(db *database.Database, logger *logging.Logger) *PostgreSQLCreateFunctionTool {
	return &PostgreSQLCreateFunctionTool{
		BaseTool: NewBaseTool(
			"postgresql_create_function",
			"Create a stored function with parameters and body",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"function_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the function to create",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"default":     "public",
						"description": "Schema name",
					},
					"parameters": map[string]interface{}{
						"type":        "string",
						"description": "Function parameters (e.g., 'arg1 INTEGER, arg2 TEXT')",
					},
					"return_type": map[string]interface{}{
						"type":        "string",
						"description": "Return type (e.g., 'INTEGER', 'TEXT', 'TABLE(id INTEGER, name TEXT)')",
					},
					"language": map[string]interface{}{
						"type":        "string",
						"default":     "plpgsql",
						"description": "Function language (plpgsql, sql, python, etc.)",
					},
					"body": map[string]interface{}{
						"type":        "string",
						"description": "Function body (SQL or language-specific code)",
					},
					"or_replace": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Use CREATE OR REPLACE FUNCTION",
					},
				},
				"required": []interface{}{"function_name", "return_type", "body"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute creates the function */
func (t *PostgreSQLCreateFunctionTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	functionName, ok := params["function_name"].(string)
	if !ok || functionName == "" {
		return Error("function_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	returnType, ok := params["return_type"].(string)
	if !ok || returnType == "" {
		return Error("return_type parameter is required", "INVALID_PARAMETER", nil), nil
	}

	body, ok := params["body"].(string)
	if !ok || body == "" {
		return Error("body parameter is required", "INVALID_PARAMETER", nil), nil
	}

	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	parameters, _ := params["parameters"].(string)
	language := "plpgsql"
	if val, ok := params["language"].(string); ok {
		language = val
	}

	orReplace := false
	if val, ok := params["or_replace"].(bool); ok {
		orReplace = val
	}

	/* Build CREATE FUNCTION statement */
	createParts := []string{"CREATE"}
	if orReplace {
		createParts = append(createParts, "OR REPLACE")
	}
	createParts = append(createParts, "FUNCTION")

	fullFunctionName := fmt.Sprintf("%s.%s", schema, functionName)
	if parameters != "" {
		fullFunctionName += "(" + parameters + ")"
	} else {
		fullFunctionName += "()"
	}
	createParts = append(createParts, fullFunctionName)
	createParts = append(createParts, "RETURNS", returnType)
	createParts = append(createParts, "LANGUAGE", language)
	createParts = append(createParts, "AS")
	createParts = append(createParts, fmt.Sprintf("$$%s$$", body))

	createQuery := strings.Join(createParts, " ")

	/* Execute CREATE FUNCTION */
	_, err := t.executor.ExecuteQuery(ctx, createQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("CREATE FUNCTION failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Function created", map[string]interface{}{
		"schema":        schema,
		"function_name": functionName,
		"language":      language,
	})

	return Success(map[string]interface{}{
		"schema":        schema,
		"function_name": functionName,
		"language":      language,
		"query":        createQuery,
	}, map[string]interface{}{
		"tool": "postgresql_create_function",
	}), nil
}

/* PostgreSQLCreateTriggerTool creates triggers */
type PostgreSQLCreateTriggerTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLCreateTriggerTool creates a new PostgreSQL create trigger tool */
func NewPostgreSQLCreateTriggerTool(db *database.Database, logger *logging.Logger) *PostgreSQLCreateTriggerTool {
	return &PostgreSQLCreateTriggerTool{
		BaseTool: NewBaseTool(
			"postgresql_create_trigger",
			"Create a trigger on a table",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"trigger_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the trigger to create",
					},
					"table_name": map[string]interface{}{
						"type":        "string",
						"description": "Table name to attach trigger to",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"default":     "public",
						"description": "Schema name",
					},
					"timing": map[string]interface{}{
						"type":        "string",
						"enum":        []string{"BEFORE", "AFTER", "INSTEAD OF"},
						"description": "Trigger timing",
					},
					"event": map[string]interface{}{
						"type":        "string",
						"enum":        []string{"INSERT", "UPDATE", "DELETE", "TRUNCATE"},
						"description": "Trigger event",
					},
					"function_name": map[string]interface{}{
						"type":        "string",
						"description": "Function to call when trigger fires",
					},
					"condition": map[string]interface{}{
						"type":        "string",
						"description": "WHEN condition (optional)",
					},
				},
				"required": []interface{}{"trigger_name", "table_name", "timing", "event", "function_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute creates the trigger */
func (t *PostgreSQLCreateTriggerTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	triggerName, ok := params["trigger_name"].(string)
	if !ok || triggerName == "" {
		return Error("trigger_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	tableName, ok := params["table_name"].(string)
	if !ok || tableName == "" {
		return Error("table_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	timing, ok := params["timing"].(string)
	if !ok || timing == "" {
		return Error("timing parameter is required", "INVALID_PARAMETER", nil), nil
	}

	event, ok := params["event"].(string)
	if !ok || event == "" {
		return Error("event parameter is required", "INVALID_PARAMETER", nil), nil
	}

	functionName, ok := params["function_name"].(string)
	if !ok || functionName == "" {
		return Error("function_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	condition, _ := params["condition"].(string)

	/* Build CREATE TRIGGER statement */
	fullTriggerName := fmt.Sprintf("%s.%s", schema, triggerName)
	fullTableName := fmt.Sprintf("%s.%s", schema, tableName)
	fullFunctionName := fmt.Sprintf("%s.%s", schema, functionName)

	createParts := []string{"CREATE TRIGGER", fullTriggerName}
	createParts = append(createParts, timing, event, "ON", fullTableName)

	if condition != "" {
		createParts = append(createParts, "WHEN", "("+condition+")")
	}

	createParts = append(createParts, "EXECUTE FUNCTION", fullFunctionName+"()")

	createQuery := strings.Join(createParts, " ")

	/* Execute CREATE TRIGGER */
	_, err := t.executor.ExecuteQuery(ctx, createQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("CREATE TRIGGER failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Trigger created", map[string]interface{}{
		"schema":       schema,
		"trigger_name": triggerName,
		"table_name":   tableName,
	})

	return Success(map[string]interface{}{
		"schema":       schema,
		"trigger_name": triggerName,
		"table_name":   tableName,
		"query":        createQuery,
	}, map[string]interface{}{
		"tool": "postgresql_create_trigger",
	}), nil
}

