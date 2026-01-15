/*-------------------------------------------------------------------------
 *
 * postgresql_object_management.go
 *    Complete object DDL management tools for NeuronMCP
 *
 * Implements comprehensive object DDL operations:
 * - ALTER/DROP INDEX, VIEW, FUNCTION, TRIGGER
 * - CREATE/ALTER/DROP SEQUENCE, TYPE, DOMAIN
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/postgresql_object_management.go
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
 * Index Management Tools
 * ============================================================================ */

/* PostgreSQLAlterIndexTool alters index properties */
type PostgreSQLAlterIndexTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLAlterIndexTool creates a new PostgreSQL alter index tool */
func NewPostgreSQLAlterIndexTool(db *database.Database, logger *logging.Logger) *PostgreSQLAlterIndexTool {
	return &PostgreSQLAlterIndexTool{
		BaseTool: NewBaseTool(
			"postgresql_alter_index",
			"Alter index properties (rename, set tablespace, alter properties)",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"index_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the index to alter",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"default":     "public",
						"description": "Schema name",
					},
					"new_name": map[string]interface{}{
						"type":        "string",
						"description": "New name for the index (rename)",
					},
					"tablespace": map[string]interface{}{
						"type":        "string",
						"description": "New tablespace for the index",
					},
					"set_storage_parameter": map[string]interface{}{
						"type":        "object",
						"description": "Storage parameters as key-value pairs",
					},
				},
				"required": []interface{}{"index_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute alters the index */
func (t *PostgreSQLAlterIndexTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	indexName, ok := params["index_name"].(string)
	if !ok || indexName == "" {
		return Error("index_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	fullIndexName := fmt.Sprintf("%s.%s", quoteIdentifier(schema), quoteIdentifier(indexName))
	parts := []string{"ALTER INDEX", fullIndexName}

	alterations := []string{}

	/* Rename */
	if newName, ok := params["new_name"].(string); ok && newName != "" {
		if !isValidIdentifier(newName) {
			return Error("Invalid new index name: must be a valid PostgreSQL identifier", "INVALID_PARAMETER", nil), nil
		}
		alterations = append(alterations, fmt.Sprintf("RENAME TO %s", quoteIdentifier(newName)))
	}

	/* Tablespace */
	if tablespace, ok := params["tablespace"].(string); ok && tablespace != "" {
		alterations = append(alterations, fmt.Sprintf("SET TABLESPACE %s", quoteIdentifier(tablespace)))
	}

	/* Storage parameters */
	if storageParams, ok := params["set_storage_parameter"].(map[string]interface{}); ok && len(storageParams) > 0 {
		for key, value := range storageParams {
			if valueStr, ok := value.(string); ok {
				alterations = append(alterations, fmt.Sprintf("SET (%s = %s)", quoteIdentifier(key), quoteLiteral(valueStr)))
			} else {
				alterations = append(alterations, fmt.Sprintf("SET (%s = %v)", quoteIdentifier(key), value))
			}
		}
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
				fmt.Sprintf("ALTER INDEX failed: %v", err),
				"QUERY_ERROR",
				map[string]interface{}{"error": err.Error(), "query": query},
			), nil
		}
	}

	t.logger.Info("Index altered", map[string]interface{}{
		"index_name": indexName,
		"schema":     schema,
	})

	return Success(map[string]interface{}{
		"index_name": indexName,
		"schema":     schema,
		"queries":     queries,
	}, map[string]interface{}{
		"tool": "postgresql_alter_index",
	}), nil
}

/* PostgreSQLDropIndexTool drops indexes */
type PostgreSQLDropIndexTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLDropIndexTool creates a new PostgreSQL drop index tool */
func NewPostgreSQLDropIndexTool(db *database.Database, logger *logging.Logger) *PostgreSQLDropIndexTool {
	return &PostgreSQLDropIndexTool{
		BaseTool: NewBaseTool(
			"postgresql_drop_index",
			"Drop an index with CONCURRENTLY option",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"index_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the index to drop",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"default":     "public",
						"description": "Schema name",
					},
					"if_exists": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Use IF EXISTS clause",
					},
					"concurrently": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Drop index concurrently (non-blocking)",
					},
				},
				"required": []interface{}{"index_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute drops the index */
func (t *PostgreSQLDropIndexTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	indexName, ok := params["index_name"].(string)
	if !ok || indexName == "" {
		return Error("index_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	parts := []string{"DROP INDEX"}
	
	if ifExists, ok := params["if_exists"].(bool); ok && ifExists {
		parts = append(parts, "IF EXISTS")
	}
	
	if concurrently, ok := params["concurrently"].(bool); ok && concurrently {
		parts = append(parts, "CONCURRENTLY")
	}
	
	fullIndexName := fmt.Sprintf("%s.%s", quoteIdentifier(schema), quoteIdentifier(indexName))
	parts = append(parts, fullIndexName)

	dropQuery := strings.Join(parts, " ")

	/* Execute DROP INDEX */
	err := t.executor.Exec(ctx, dropQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("DROP INDEX failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Index dropped", map[string]interface{}{
		"index_name": indexName,
		"schema":     schema,
	})

	return Success(map[string]interface{}{
		"index_name": indexName,
		"schema":     schema,
		"query":      dropQuery,
	}, map[string]interface{}{
		"tool": "postgresql_drop_index",
	}), nil
}

/* ============================================================================
 * View Management Tools
 * ============================================================================ */

/* PostgreSQLAlterViewTool alters view properties */
type PostgreSQLAlterViewTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLAlterViewTool creates a new PostgreSQL alter view tool */
func NewPostgreSQLAlterViewTool(db *database.Database, logger *logging.Logger) *PostgreSQLAlterViewTool {
	return &PostgreSQLAlterViewTool{
		BaseTool: NewBaseTool(
			"postgresql_alter_view",
			"Alter view properties (rename, change owner, set options)",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"view_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the view to alter",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"default":     "public",
						"description": "Schema name",
					},
					"new_name": map[string]interface{}{
						"type":        "string",
						"description": "New name for the view (rename)",
					},
					"owner": map[string]interface{}{
						"type":        "string",
						"description": "New owner (role name)",
					},
					"set_options": map[string]interface{}{
						"type":        "object",
						"description": "View options as key-value pairs",
					},
				},
				"required": []interface{}{"view_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute alters the view */
func (t *PostgreSQLAlterViewTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	viewName, ok := params["view_name"].(string)
	if !ok || viewName == "" {
		return Error("view_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	fullViewName := fmt.Sprintf("%s.%s", quoteIdentifier(schema), quoteIdentifier(viewName))
	parts := []string{"ALTER VIEW", fullViewName}

	alterations := []string{}

	/* Rename */
	if newName, ok := params["new_name"].(string); ok && newName != "" {
		if !isValidIdentifier(newName) {
			return Error("Invalid new view name: must be a valid PostgreSQL identifier", "INVALID_PARAMETER", nil), nil
		}
		alterations = append(alterations, fmt.Sprintf("RENAME TO %s", quoteIdentifier(newName)))
	}

	/* Owner */
	if owner, ok := params["owner"].(string); ok && owner != "" {
		alterations = append(alterations, fmt.Sprintf("OWNER TO %s", quoteIdentifier(owner)))
	}

	/* Options */
	if options, ok := params["set_options"].(map[string]interface{}); ok && len(options) > 0 {
		for key, value := range options {
			if valueStr, ok := value.(string); ok {
				alterations = append(alterations, fmt.Sprintf("SET (%s = %s)", quoteIdentifier(key), quoteLiteral(valueStr)))
			} else {
				alterations = append(alterations, fmt.Sprintf("SET (%s = %v)", quoteIdentifier(key), value))
			}
		}
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
				fmt.Sprintf("ALTER VIEW failed: %v", err),
				"QUERY_ERROR",
				map[string]interface{}{"error": err.Error(), "query": query},
			), nil
		}
	}

	t.logger.Info("View altered", map[string]interface{}{
		"view_name": viewName,
		"schema":    schema,
	})

	return Success(map[string]interface{}{
		"view_name": viewName,
		"schema":    schema,
		"queries":   queries,
	}, map[string]interface{}{
		"tool": "postgresql_alter_view",
	}), nil
}

/* PostgreSQLDropViewTool drops views */
type PostgreSQLDropViewTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLDropViewTool creates a new PostgreSQL drop view tool */
func NewPostgreSQLDropViewTool(db *database.Database, logger *logging.Logger) *PostgreSQLDropViewTool {
	return &PostgreSQLDropViewTool{
		BaseTool: NewBaseTool(
			"postgresql_drop_view",
			"Drop a view with CASCADE option",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"view_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the view to drop",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"default":     "public",
						"description": "Schema name",
					},
					"if_exists": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Use IF EXISTS clause",
					},
					"cascade": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Drop dependent objects (CASCADE)",
					},
				},
				"required": []interface{}{"view_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute drops the view */
func (t *PostgreSQLDropViewTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	viewName, ok := params["view_name"].(string)
	if !ok || viewName == "" {
		return Error("view_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	parts := []string{"DROP VIEW"}
	
	if ifExists, ok := params["if_exists"].(bool); ok && ifExists {
		parts = append(parts, "IF EXISTS")
	}
	
	fullViewName := fmt.Sprintf("%s.%s", quoteIdentifier(schema), quoteIdentifier(viewName))
	parts = append(parts, fullViewName)
	
	if cascade, ok := params["cascade"].(bool); ok && cascade {
		parts = append(parts, "CASCADE")
	} else {
		parts = append(parts, "RESTRICT")
	}

	dropQuery := strings.Join(parts, " ")

	/* Execute DROP VIEW */
	err := t.executor.Exec(ctx, dropQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("DROP VIEW failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("View dropped", map[string]interface{}{
		"view_name": viewName,
		"schema":    schema,
	})

	return Success(map[string]interface{}{
		"view_name": viewName,
		"schema":    schema,
		"query":     dropQuery,
	}, map[string]interface{}{
		"tool": "postgresql_drop_view",
	}), nil
}

/* ============================================================================
 * Function Management Tools
 * ============================================================================ */

/* PostgreSQLAlterFunctionTool alters function properties */
type PostgreSQLAlterFunctionTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLAlterFunctionTool creates a new PostgreSQL alter function tool */
func NewPostgreSQLAlterFunctionTool(db *database.Database, logger *logging.Logger) *PostgreSQLAlterFunctionTool {
	return &PostgreSQLAlterFunctionTool{
		BaseTool: NewBaseTool(
			"postgresql_alter_function",
			"Alter function properties (name, owner, schema, volatility, etc.)",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"function_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the function to alter",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"default":     "public",
						"description": "Schema name",
					},
					"arg_types": map[string]interface{}{
						"type":        "array",
						"description": "Array of argument types (for function signature)",
					},
					"new_name": map[string]interface{}{
						"type":        "string",
						"description": "New name for the function (rename)",
					},
					"owner": map[string]interface{}{
						"type":        "string",
						"description": "New owner (role name)",
					},
					"new_schema": map[string]interface{}{
						"type":        "string",
						"description": "New schema for the function",
					},
					"volatility": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"IMMUTABLE", "STABLE", "VOLATILE"},
						"description": "Function volatility",
					},
					"leakproof": map[string]interface{}{
						"type":        "boolean",
						"description": "Set LEAKPROOF property",
					},
					"parallel": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"SAFE", "UNSAFE", "RESTRICTED"},
						"description": "Parallel safety",
					},
				},
				"required": []interface{}{"function_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute alters the function */
func (t *PostgreSQLAlterFunctionTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	functionName, ok := params["function_name"].(string)
	if !ok || functionName == "" {
		return Error("function_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	/* Build function signature */
	argTypes, _ := params["arg_types"].([]interface{})
	signature := quoteIdentifier(functionName)
	if len(argTypes) > 0 {
		typeList := []string{}
		for _, argType := range argTypes {
			if typeStr, ok := argType.(string); ok {
				typeList = append(typeList, typeStr)
			}
		}
		if len(typeList) > 0 {
			signature += "(" + strings.Join(typeList, ", ") + ")"
		}
	}

	fullFunctionName := fmt.Sprintf("%s.%s", quoteIdentifier(schema), signature)
	parts := []string{"ALTER FUNCTION", fullFunctionName}

	alterations := []string{}

	/* Rename */
	if newName, ok := params["new_name"].(string); ok && newName != "" {
		if !isValidIdentifier(newName) {
			return Error("Invalid new function name: must be a valid PostgreSQL identifier", "INVALID_PARAMETER", nil), nil
		}
		alterations = append(alterations, fmt.Sprintf("RENAME TO %s", quoteIdentifier(newName)))
	}

	/* Owner */
	if owner, ok := params["owner"].(string); ok && owner != "" {
		alterations = append(alterations, fmt.Sprintf("OWNER TO %s", quoteIdentifier(owner)))
	}

	/* Schema */
	if newSchema, ok := params["new_schema"].(string); ok && newSchema != "" {
		alterations = append(alterations, fmt.Sprintf("SET SCHEMA %s", quoteIdentifier(newSchema)))
	}

	/* Volatility */
	if volatility, ok := params["volatility"].(string); ok && volatility != "" {
		alterations = append(alterations, volatility)
	}

	/* Leakproof */
	if leakproof, ok := params["leakproof"].(bool); ok {
		if leakproof {
			alterations = append(alterations, "LEAKPROOF")
		} else {
			alterations = append(alterations, "NOT LEAKPROOF")
		}
	}

	/* Parallel */
	if parallel, ok := params["parallel"].(string); ok && parallel != "" {
		alterations = append(alterations, fmt.Sprintf("PARALLEL %s", parallel))
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
				fmt.Sprintf("ALTER FUNCTION failed: %v", err),
				"QUERY_ERROR",
				map[string]interface{}{"error": err.Error(), "query": query},
			), nil
		}
	}

	t.logger.Info("Function altered", map[string]interface{}{
		"function_name": functionName,
		"schema":        schema,
	})

	return Success(map[string]interface{}{
		"function_name": functionName,
		"schema":        schema,
		"queries":        queries,
	}, map[string]interface{}{
		"tool": "postgresql_alter_function",
	}), nil
}

/* PostgreSQLDropFunctionTool drops functions */
type PostgreSQLDropFunctionTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLDropFunctionTool creates a new PostgreSQL drop function tool */
func NewPostgreSQLDropFunctionTool(db *database.Database, logger *logging.Logger) *PostgreSQLDropFunctionTool {
	return &PostgreSQLDropFunctionTool{
		BaseTool: NewBaseTool(
			"postgresql_drop_function",
			"Drop a function with signature matching",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"function_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the function to drop",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"default":     "public",
						"description": "Schema name",
					},
					"arg_types": map[string]interface{}{
						"type":        "array",
						"description": "Array of argument types (for function signature)",
					},
					"if_exists": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Use IF EXISTS clause",
					},
					"cascade": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Drop dependent objects (CASCADE)",
					},
				},
				"required": []interface{}{"function_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute drops the function */
func (t *PostgreSQLDropFunctionTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	functionName, ok := params["function_name"].(string)
	if !ok || functionName == "" {
		return Error("function_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	/* Build function signature */
	argTypes, _ := params["arg_types"].([]interface{})
	signature := quoteIdentifier(functionName)
	if len(argTypes) > 0 {
		typeList := []string{}
		for _, argType := range argTypes {
			if typeStr, ok := argType.(string); ok {
				typeList = append(typeList, typeStr)
			}
		}
		if len(typeList) > 0 {
			signature += "(" + strings.Join(typeList, ", ") + ")"
		}
	}

	parts := []string{"DROP FUNCTION"}
	
	if ifExists, ok := params["if_exists"].(bool); ok && ifExists {
		parts = append(parts, "IF EXISTS")
	}
	
	fullFunctionName := fmt.Sprintf("%s.%s", quoteIdentifier(schema), signature)
	parts = append(parts, fullFunctionName)
	
	if cascade, ok := params["cascade"].(bool); ok && cascade {
		parts = append(parts, "CASCADE")
	} else {
		parts = append(parts, "RESTRICT")
	}

	dropQuery := strings.Join(parts, " ")

	/* Execute DROP FUNCTION */
	err := t.executor.Exec(ctx, dropQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("DROP FUNCTION failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Function dropped", map[string]interface{}{
		"function_name": functionName,
		"schema":        schema,
	})

	return Success(map[string]interface{}{
		"function_name": functionName,
		"schema":        schema,
		"query":         dropQuery,
	}, map[string]interface{}{
		"tool": "postgresql_drop_function",
	}), nil
}

/* ============================================================================
 * Trigger Management Tools
 * ============================================================================ */

/* PostgreSQLAlterTriggerTool alters trigger properties */
type PostgreSQLAlterTriggerTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLAlterTriggerTool creates a new PostgreSQL alter trigger tool */
func NewPostgreSQLAlterTriggerTool(db *database.Database, logger *logging.Logger) *PostgreSQLAlterTriggerTool {
	return &PostgreSQLAlterTriggerTool{
		BaseTool: NewBaseTool(
			"postgresql_alter_trigger",
			"Enable/disable triggers or rename triggers",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"trigger_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the trigger to alter",
					},
					"table_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the table the trigger is on",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"default":     "public",
						"description": "Schema name",
					},
					"new_name": map[string]interface{}{
						"type":        "string",
						"description": "New name for the trigger (rename)",
					},
					"enable": map[string]interface{}{
						"type":        "boolean",
						"description": "Enable the trigger (true) or disable (false)",
					},
				},
				"required": []interface{}{"trigger_name", "table_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute alters the trigger */
func (t *PostgreSQLAlterTriggerTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	triggerName, ok := params["trigger_name"].(string)
	if !ok || triggerName == "" {
		return Error("trigger_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	tableName, ok := params["table_name"].(string)
	if !ok || tableName == "" {
		return Error("table_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	fullTableName := fmt.Sprintf("%s.%s", quoteIdentifier(schema), quoteIdentifier(tableName))
	parts := []string{"ALTER TABLE", fullTableName}

	alterations := []string{}

	/* Rename */
	if newName, ok := params["new_name"].(string); ok && newName != "" {
		if !isValidIdentifier(newName) {
			return Error("Invalid new trigger name: must be a valid PostgreSQL identifier", "INVALID_PARAMETER", nil), nil
		}
		alterations = append(alterations, fmt.Sprintf("RENAME TRIGGER %s TO %s", quoteIdentifier(triggerName), quoteIdentifier(newName)))
	}

	/* Enable/Disable */
	if enable, ok := params["enable"].(bool); ok {
		if enable {
			alterations = append(alterations, fmt.Sprintf("ENABLE TRIGGER %s", quoteIdentifier(triggerName)))
		} else {
			alterations = append(alterations, fmt.Sprintf("DISABLE TRIGGER %s", quoteIdentifier(triggerName)))
		}
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
				fmt.Sprintf("ALTER TRIGGER failed: %v", err),
				"QUERY_ERROR",
				map[string]interface{}{"error": err.Error(), "query": query},
			), nil
		}
	}

	t.logger.Info("Trigger altered", map[string]interface{}{
		"trigger_name": triggerName,
		"table_name":   tableName,
		"schema":       schema,
	})

	return Success(map[string]interface{}{
		"trigger_name": triggerName,
		"table_name":   tableName,
		"schema":       schema,
		"queries":      queries,
	}, map[string]interface{}{
		"tool": "postgresql_alter_trigger",
	}), nil
}

/* PostgreSQLDropTriggerTool drops triggers */
type PostgreSQLDropTriggerTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLDropTriggerTool creates a new PostgreSQL drop trigger tool */
func NewPostgreSQLDropTriggerTool(db *database.Database, logger *logging.Logger) *PostgreSQLDropTriggerTool {
	return &PostgreSQLDropTriggerTool{
		BaseTool: NewBaseTool(
			"postgresql_drop_trigger",
			"Drop a trigger",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"trigger_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the trigger to drop",
					},
					"table_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the table the trigger is on",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"default":     "public",
						"description": "Schema name",
					},
					"if_exists": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Use IF EXISTS clause",
					},
				},
				"required": []interface{}{"trigger_name", "table_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute drops the trigger */
func (t *PostgreSQLDropTriggerTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	triggerName, ok := params["trigger_name"].(string)
	if !ok || triggerName == "" {
		return Error("trigger_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	tableName, ok := params["table_name"].(string)
	if !ok || tableName == "" {
		return Error("table_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	fullTableName := fmt.Sprintf("%s.%s", quoteIdentifier(schema), quoteIdentifier(tableName))
	parts := []string{"DROP TRIGGER"}
	
	if ifExists, ok := params["if_exists"].(bool); ok && ifExists {
		parts = append(parts, "IF EXISTS")
	}
	
	parts = append(parts, quoteIdentifier(triggerName), "ON", fullTableName)

	dropQuery := strings.Join(parts, " ")

	/* Execute DROP TRIGGER */
	err := t.executor.Exec(ctx, dropQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("DROP TRIGGER failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Trigger dropped", map[string]interface{}{
		"trigger_name": triggerName,
		"table_name":   tableName,
		"schema":       schema,
	})

	return Success(map[string]interface{}{
		"trigger_name": triggerName,
		"table_name":   tableName,
		"schema":       schema,
		"query":        dropQuery,
	}, map[string]interface{}{
		"tool": "postgresql_drop_trigger",
	}), nil
}

/* ============================================================================
 * Sequence Management Tools
 * ============================================================================ */

/* PostgreSQLCreateSequenceTool creates sequences */
type PostgreSQLCreateSequenceTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLCreateSequenceTool creates a new PostgreSQL create sequence tool */
func NewPostgreSQLCreateSequenceTool(db *database.Database, logger *logging.Logger) *PostgreSQLCreateSequenceTool {
	return &PostgreSQLCreateSequenceTool{
		BaseTool: NewBaseTool(
			"postgresql_create_sequence",
			"Create a sequence with full options (INCREMENT, MINVALUE, MAXVALUE, START, CACHE)",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"sequence_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the sequence to create",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"default":     "public",
						"description": "Schema name",
					},
					"increment": map[string]interface{}{
						"type":        "integer",
						"default":     1,
						"description": "Increment value",
					},
					"minvalue": map[string]interface{}{
						"type":        "integer",
						"description": "Minimum value (default: 1)",
					},
					"maxvalue": map[string]interface{}{
						"type":        "integer",
						"description": "Maximum value",
					},
					"start": map[string]interface{}{
						"type":        "integer",
						"description": "Start value",
					},
					"cache": map[string]interface{}{
						"type":        "integer",
						"default":     1,
						"description": "Cache size",
					},
					"cycle": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Allow sequence to cycle",
					},
					"owned_by": map[string]interface{}{
						"type":        "string",
						"description": "Table and column that owns the sequence (format: table.column)",
					},
					"if_not_exists": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Use IF NOT EXISTS clause",
					},
				},
				"required": []interface{}{"sequence_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute creates the sequence */
func (t *PostgreSQLCreateSequenceTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	sequenceName, ok := params["sequence_name"].(string)
	if !ok || sequenceName == "" {
		return Error("sequence_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	parts := []string{"CREATE SEQUENCE"}
	
	if ifNotExists, ok := params["if_not_exists"].(bool); ok && ifNotExists {
		parts = append(parts, "IF NOT EXISTS")
	}
	
	fullSequenceName := fmt.Sprintf("%s.%s", quoteIdentifier(schema), quoteIdentifier(sequenceName))
	parts = append(parts, fullSequenceName)

	options := []string{}

	/* Increment */
	if increment, ok := params["increment"].(float64); ok {
		options = append(options, fmt.Sprintf("INCREMENT %d", int(increment)))
	} else if increment, ok := params["increment"].(int); ok {
		options = append(options, fmt.Sprintf("INCREMENT %d", increment))
	}

	/* Minvalue */
	if minvalue, ok := params["minvalue"].(float64); ok {
		options = append(options, fmt.Sprintf("MINVALUE %d", int(minvalue)))
	} else if minvalue, ok := params["minvalue"].(int); ok {
		options = append(options, fmt.Sprintf("MINVALUE %d", minvalue))
	} else {
		options = append(options, "NO MINVALUE")
	}

	/* Maxvalue */
	if maxvalue, ok := params["maxvalue"].(float64); ok {
		options = append(options, fmt.Sprintf("MAXVALUE %d", int(maxvalue)))
	} else if maxvalue, ok := params["maxvalue"].(int); ok {
		options = append(options, fmt.Sprintf("MAXVALUE %d", maxvalue))
	} else {
		options = append(options, "NO MAXVALUE")
	}

	/* Start */
	if start, ok := params["start"].(float64); ok {
		options = append(options, fmt.Sprintf("START %d", int(start)))
	} else if start, ok := params["start"].(int); ok {
		options = append(options, fmt.Sprintf("START %d", start))
	}

	/* Cache */
	if cache, ok := params["cache"].(float64); ok {
		options = append(options, fmt.Sprintf("CACHE %d", int(cache)))
	} else if cache, ok := params["cache"].(int); ok {
		options = append(options, fmt.Sprintf("CACHE %d", cache))
	}

	/* Cycle */
	if cycle, ok := params["cycle"].(bool); ok && cycle {
		options = append(options, "CYCLE")
	} else {
		options = append(options, "NO CYCLE")
	}

	/* Owned by */
	if ownedBy, ok := params["owned_by"].(string); ok && ownedBy != "" {
		options = append(options, fmt.Sprintf("OWNED BY %s", ownedBy))
	}

	if len(options) > 0 {
		parts = append(parts, strings.Join(options, " "))
	}

	createQuery := strings.Join(parts, " ")

	/* Execute CREATE SEQUENCE */
	err := t.executor.Exec(ctx, createQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("CREATE SEQUENCE failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Sequence created", map[string]interface{}{
		"sequence_name": sequenceName,
		"schema":        schema,
	})

	return Success(map[string]interface{}{
		"sequence_name": sequenceName,
		"schema":        schema,
		"query":         createQuery,
	}, map[string]interface{}{
		"tool": "postgresql_create_sequence",
	}), nil
}

/* PostgreSQLAlterSequenceTool alters sequence properties */
type PostgreSQLAlterSequenceTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLAlterSequenceTool creates a new PostgreSQL alter sequence tool */
func NewPostgreSQLAlterSequenceTool(db *database.Database, logger *logging.Logger) *PostgreSQLAlterSequenceTool {
	return &PostgreSQLAlterSequenceTool{
		BaseTool: NewBaseTool(
			"postgresql_alter_sequence",
			"Modify sequence properties or set sequence value",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"sequence_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the sequence to alter",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"default":     "public",
						"description": "Schema name",
					},
					"increment": map[string]interface{}{
						"type":        "integer",
						"description": "New increment value",
					},
					"minvalue": map[string]interface{}{
						"type":        "integer",
						"description": "New minimum value",
					},
					"maxvalue": map[string]interface{}{
						"type":        "integer",
						"description": "New maximum value",
					},
					"start": map[string]interface{}{
						"type":        "integer",
						"description": "New start value",
					},
					"restart": map[string]interface{}{
						"type":        "integer",
						"description": "Restart sequence at this value",
					},
					"cache": map[string]interface{}{
						"type":        "integer",
						"description": "New cache size",
					},
					"cycle": map[string]interface{}{
						"type":        "boolean",
						"description": "Set cycle property",
					},
					"owned_by": map[string]interface{}{
						"type":        "string",
						"description": "Table and column that owns the sequence (format: table.column) or NONE",
					},
				},
				"required": []interface{}{"sequence_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute alters the sequence */
func (t *PostgreSQLAlterSequenceTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	sequenceName, ok := params["sequence_name"].(string)
	if !ok || sequenceName == "" {
		return Error("sequence_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	fullSequenceName := fmt.Sprintf("%s.%s", quoteIdentifier(schema), quoteIdentifier(sequenceName))
	parts := []string{"ALTER SEQUENCE", fullSequenceName}

	alterations := []string{}

	/* Increment */
	if increment, ok := params["increment"].(float64); ok {
		alterations = append(alterations, fmt.Sprintf("INCREMENT BY %d", int(increment)))
	} else if increment, ok := params["increment"].(int); ok {
		alterations = append(alterations, fmt.Sprintf("INCREMENT BY %d", increment))
	}

	/* Minvalue */
	if minvalue, ok := params["minvalue"].(float64); ok {
		alterations = append(alterations, fmt.Sprintf("MINVALUE %d", int(minvalue)))
	} else if minvalue, ok := params["minvalue"].(int); ok {
		alterations = append(alterations, fmt.Sprintf("MINVALUE %d", minvalue))
	} else if _, ok := params["minvalue"]; ok {
		alterations = append(alterations, "NO MINVALUE")
	}

	/* Maxvalue */
	if maxvalue, ok := params["maxvalue"].(float64); ok {
		alterations = append(alterations, fmt.Sprintf("MAXVALUE %d", int(maxvalue)))
	} else if maxvalue, ok := params["maxvalue"].(int); ok {
		alterations = append(alterations, fmt.Sprintf("MAXVALUE %d", maxvalue))
	} else if _, ok := params["maxvalue"]; ok {
		alterations = append(alterations, "NO MAXVALUE")
	}

	/* Start */
	if start, ok := params["start"].(float64); ok {
		alterations = append(alterations, fmt.Sprintf("START WITH %d", int(start)))
	} else if start, ok := params["start"].(int); ok {
		alterations = append(alterations, fmt.Sprintf("START WITH %d", start))
	}

	/* Restart */
	if restart, ok := params["restart"].(float64); ok {
		alterations = append(alterations, fmt.Sprintf("RESTART WITH %d", int(restart)))
	} else if restart, ok := params["restart"].(int); ok {
		alterations = append(alterations, fmt.Sprintf("RESTART WITH %d", restart))
	} else if _, ok := params["restart"]; ok {
		alterations = append(alterations, "RESTART")
	}

	/* Cache */
	if cache, ok := params["cache"].(float64); ok {
		alterations = append(alterations, fmt.Sprintf("CACHE %d", int(cache)))
	} else if cache, ok := params["cache"].(int); ok {
		alterations = append(alterations, fmt.Sprintf("CACHE %d", cache))
	}

	/* Cycle */
	if cycle, ok := params["cycle"].(bool); ok {
		if cycle {
			alterations = append(alterations, "CYCLE")
		} else {
			alterations = append(alterations, "NO CYCLE")
		}
	}

	/* Owned by */
	if ownedBy, ok := params["owned_by"].(string); ok {
		if ownedBy == "NONE" || ownedBy == "" {
			alterations = append(alterations, "OWNED BY NONE")
		} else {
			alterations = append(alterations, fmt.Sprintf("OWNED BY %s", ownedBy))
		}
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
				fmt.Sprintf("ALTER SEQUENCE failed: %v", err),
				"QUERY_ERROR",
				map[string]interface{}{"error": err.Error(), "query": query},
			), nil
		}
	}

	t.logger.Info("Sequence altered", map[string]interface{}{
		"sequence_name": sequenceName,
		"schema":        schema,
	})

	return Success(map[string]interface{}{
		"sequence_name": sequenceName,
		"schema":        schema,
		"queries":        queries,
	}, map[string]interface{}{
		"tool": "postgresql_alter_sequence",
	}), nil
}

/* PostgreSQLDropSequenceTool drops sequences */
type PostgreSQLDropSequenceTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLDropSequenceTool creates a new PostgreSQL drop sequence tool */
func NewPostgreSQLDropSequenceTool(db *database.Database, logger *logging.Logger) *PostgreSQLDropSequenceTool {
	return &PostgreSQLDropSequenceTool{
		BaseTool: NewBaseTool(
			"postgresql_drop_sequence",
			"Drop a sequence with CASCADE option",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"sequence_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the sequence to drop",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"default":     "public",
						"description": "Schema name",
					},
					"if_exists": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Use IF EXISTS clause",
					},
					"cascade": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Drop dependent objects (CASCADE)",
					},
				},
				"required": []interface{}{"sequence_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute drops the sequence */
func (t *PostgreSQLDropSequenceTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	sequenceName, ok := params["sequence_name"].(string)
	if !ok || sequenceName == "" {
		return Error("sequence_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	parts := []string{"DROP SEQUENCE"}
	
	if ifExists, ok := params["if_exists"].(bool); ok && ifExists {
		parts = append(parts, "IF EXISTS")
	}
	
	fullSequenceName := fmt.Sprintf("%s.%s", quoteIdentifier(schema), quoteIdentifier(sequenceName))
	parts = append(parts, fullSequenceName)
	
	if cascade, ok := params["cascade"].(bool); ok && cascade {
		parts = append(parts, "CASCADE")
	} else {
		parts = append(parts, "RESTRICT")
	}

	dropQuery := strings.Join(parts, " ")

	/* Execute DROP SEQUENCE */
	err := t.executor.Exec(ctx, dropQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("DROP SEQUENCE failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Sequence dropped", map[string]interface{}{
		"sequence_name": sequenceName,
		"schema":        schema,
	})

	return Success(map[string]interface{}{
		"sequence_name": sequenceName,
		"schema":        schema,
		"query":         dropQuery,
	}, map[string]interface{}{
		"tool": "postgresql_drop_sequence",
	}), nil
}

/* ============================================================================
 * Type Management Tools
 * ============================================================================ */

/* PostgreSQLCreateTypeTool creates types */
type PostgreSQLCreateTypeTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLCreateTypeTool creates a new PostgreSQL create type tool */
func NewPostgreSQLCreateTypeTool(db *database.Database, logger *logging.Logger) *PostgreSQLCreateTypeTool {
	return &PostgreSQLCreateTypeTool{
		BaseTool: NewBaseTool(
			"postgresql_create_type",
			"Create composite types, enum types, or range types",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"type_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the type to create",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"default":     "public",
						"description": "Schema name",
					},
					"type_kind": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"composite", "enum", "range"},
						"description": "Type of type to create",
					},
					"attributes": map[string]interface{}{
						"type":        "array",
						"description": "For composite types: array of {name, type} objects",
					},
					"enum_values": map[string]interface{}{
						"type":        "array",
						"description": "For enum types: array of enum value strings",
					},
					"subtype": map[string]interface{}{
						"type":        "string",
						"description": "For range types: subtype name",
					},
					"subtype_opclass": map[string]interface{}{
						"type":        "string",
						"description": "For range types: subtype operator class",
					},
					"canonical": map[string]interface{}{
						"type":        "string",
						"description": "For range types: canonical function name",
					},
					"subtype_diff": map[string]interface{}{
						"type":        "string",
						"description": "For range types: subtype difference function name",
					},
					"if_not_exists": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Use IF NOT EXISTS clause",
					},
				},
				"required": []interface{}{"type_name", "type_kind"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute creates the type */
func (t *PostgreSQLCreateTypeTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	typeName, ok := params["type_name"].(string)
	if !ok || typeName == "" {
		return Error("type_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	typeKind, ok := params["type_kind"].(string)
	if !ok || typeKind == "" {
		return Error("type_kind parameter is required", "INVALID_PARAMETER", nil), nil
	}

	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	parts := []string{"CREATE TYPE"}
	
	if ifNotExists, ok := params["if_not_exists"].(bool); ok && ifNotExists {
		parts = append(parts, "IF NOT EXISTS")
	}
	
	fullTypeName := fmt.Sprintf("%s.%s", quoteIdentifier(schema), quoteIdentifier(typeName))
	parts = append(parts, fullTypeName, "AS")

	var typeDef string
	switch typeKind {
	case "composite":
		attributes, ok := params["attributes"].([]interface{})
		if !ok || len(attributes) == 0 {
			return Error("attributes parameter is required for composite types", "INVALID_PARAMETER", nil), nil
		}
		attrList := []string{}
		for _, attr := range attributes {
			if attrMap, ok := attr.(map[string]interface{}); ok {
				name, _ := attrMap["name"].(string)
				attrType, _ := attrMap["type"].(string)
				if name != "" && attrType != "" {
					attrList = append(attrList, fmt.Sprintf("%s %s", quoteIdentifier(name), attrType))
				}
			}
		}
		if len(attrList) == 0 {
			return Error("No valid attributes specified", "INVALID_PARAMETER", nil), nil
		}
		typeDef = "(" + strings.Join(attrList, ", ") + ")"

	case "enum":
		enumValues, ok := params["enum_values"].([]interface{})
		if !ok || len(enumValues) == 0 {
			return Error("enum_values parameter is required for enum types", "INVALID_PARAMETER", nil), nil
		}
		valueList := []string{}
		for _, val := range enumValues {
			if valStr, ok := val.(string); ok {
				valueList = append(valueList, quoteLiteral(valStr))
			}
		}
		if len(valueList) == 0 {
			return Error("No valid enum values specified", "INVALID_PARAMETER", nil), nil
		}
		typeDef = "(" + strings.Join(valueList, ", ") + ")"

	case "range":
		subtype, ok := params["subtype"].(string)
		if !ok || subtype == "" {
			return Error("subtype parameter is required for range types", "INVALID_PARAMETER", nil), nil
		}
		typeDef = subtype
		if subtypeOpclass, ok := params["subtype_opclass"].(string); ok && subtypeOpclass != "" {
			typeDef += " " + quoteIdentifier(subtypeOpclass)
		}
		if canonical, ok := params["canonical"].(string); ok && canonical != "" {
			typeDef += " CANONICAL " + quoteIdentifier(canonical)
		}
		if subtypeDiff, ok := params["subtype_diff"].(string); ok && subtypeDiff != "" {
			typeDef += " SUBTYPE_DIFF " + quoteIdentifier(subtypeDiff)
		}

	default:
		return Error(fmt.Sprintf("Invalid type_kind: %s", typeKind), "INVALID_PARAMETER", nil), nil
	}

	parts = append(parts, typeDef)
	createQuery := strings.Join(parts, " ")

	/* Execute CREATE TYPE */
	err := t.executor.Exec(ctx, createQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("CREATE TYPE failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Type created", map[string]interface{}{
		"type_name": typeName,
		"type_kind": typeKind,
		"schema":    schema,
	})

	return Success(map[string]interface{}{
		"type_name": typeName,
		"type_kind": typeKind,
		"schema":    schema,
		"query":     createQuery,
	}, map[string]interface{}{
		"tool": "postgresql_create_type",
	}), nil
}

/* PostgreSQLAlterTypeTool alters type properties */
type PostgreSQLAlterTypeTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLAlterTypeTool creates a new PostgreSQL alter type tool */
func NewPostgreSQLAlterTypeTool(db *database.Database, logger *logging.Logger) *PostgreSQLAlterTypeTool {
	return &PostgreSQLAlterTypeTool{
		BaseTool: NewBaseTool(
			"postgresql_alter_type",
			"Add attributes to composite types or rename types and attributes",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"type_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the type to alter",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"default":     "public",
						"description": "Schema name",
					},
					"new_name": map[string]interface{}{
						"type":        "string",
						"description": "New name for the type (rename)",
					},
					"add_attribute": map[string]interface{}{
						"type":        "object",
						"description": "Add attribute: {name: string, type: string}",
					},
					"drop_attribute": map[string]interface{}{
						"type":        "string",
						"description": "Name of attribute to drop",
					},
					"rename_attribute": map[string]interface{}{
						"type":        "object",
						"description": "Rename attribute: {old_name: string, new_name: string}",
					},
					"add_value": map[string]interface{}{
						"type":        "string",
						"description": "Add value to enum type (BEFORE or AFTER another value)",
					},
					"add_value_before": map[string]interface{}{
						"type":        "string",
						"description": "Add enum value before this value",
					},
					"add_value_after": map[string]interface{}{
						"type":        "string",
						"description": "Add enum value after this value",
					},
				},
				"required": []interface{}{"type_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute alters the type */
func (t *PostgreSQLAlterTypeTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	typeName, ok := params["type_name"].(string)
	if !ok || typeName == "" {
		return Error("type_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	fullTypeName := fmt.Sprintf("%s.%s", quoteIdentifier(schema), quoteIdentifier(typeName))
	parts := []string{"ALTER TYPE", fullTypeName}

	alterations := []string{}

	/* Rename */
	if newName, ok := params["new_name"].(string); ok && newName != "" {
		if !isValidIdentifier(newName) {
			return Error("Invalid new type name: must be a valid PostgreSQL identifier", "INVALID_PARAMETER", nil), nil
		}
		alterations = append(alterations, fmt.Sprintf("RENAME TO %s", quoteIdentifier(newName)))
	}

	/* Add attribute */
	if addAttr, ok := params["add_attribute"].(map[string]interface{}); ok && len(addAttr) > 0 {
		name, _ := addAttr["name"].(string)
		attrType, _ := addAttr["type"].(string)
		if name != "" && attrType != "" {
			alterations = append(alterations, fmt.Sprintf("ADD ATTRIBUTE %s %s", quoteIdentifier(name), attrType))
		}
	}

	/* Drop attribute */
	if dropAttr, ok := params["drop_attribute"].(string); ok && dropAttr != "" {
		alterations = append(alterations, fmt.Sprintf("DROP ATTRIBUTE %s", quoteIdentifier(dropAttr)))
	}

	/* Rename attribute */
	if renameAttr, ok := params["rename_attribute"].(map[string]interface{}); ok && len(renameAttr) > 0 {
		oldName, _ := renameAttr["old_name"].(string)
		newName, _ := renameAttr["new_name"].(string)
		if oldName != "" && newName != "" {
			alterations = append(alterations, fmt.Sprintf("RENAME ATTRIBUTE %s TO %s", quoteIdentifier(oldName), quoteIdentifier(newName)))
		}
	}

	/* Add enum value */
	if addValue, ok := params["add_value"].(string); ok && addValue != "" {
		if before, ok := params["add_value_before"].(string); ok && before != "" {
			alterations = append(alterations, fmt.Sprintf("ADD VALUE %s BEFORE %s", quoteLiteral(addValue), quoteLiteral(before)))
		} else if after, ok := params["add_value_after"].(string); ok && after != "" {
			alterations = append(alterations, fmt.Sprintf("ADD VALUE %s AFTER %s", quoteLiteral(addValue), quoteLiteral(after)))
		} else {
			alterations = append(alterations, fmt.Sprintf("ADD VALUE %s", quoteLiteral(addValue)))
		}
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
				fmt.Sprintf("ALTER TYPE failed: %v", err),
				"QUERY_ERROR",
				map[string]interface{}{"error": err.Error(), "query": query},
			), nil
		}
	}

	t.logger.Info("Type altered", map[string]interface{}{
		"type_name": typeName,
		"schema":    schema,
	})

	return Success(map[string]interface{}{
		"type_name": typeName,
		"schema":    schema,
		"queries":   queries,
	}, map[string]interface{}{
		"tool": "postgresql_alter_type",
	}), nil
}

/* PostgreSQLDropTypeTool drops types */
type PostgreSQLDropTypeTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLDropTypeTool creates a new PostgreSQL drop type tool */
func NewPostgreSQLDropTypeTool(db *database.Database, logger *logging.Logger) *PostgreSQLDropTypeTool {
	return &PostgreSQLDropTypeTool{
		BaseTool: NewBaseTool(
			"postgresql_drop_type",
			"Drop a type with CASCADE option",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"type_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the type to drop",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"default":     "public",
						"description": "Schema name",
					},
					"if_exists": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Use IF EXISTS clause",
					},
					"cascade": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Drop dependent objects (CASCADE)",
					},
				},
				"required": []interface{}{"type_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute drops the type */
func (t *PostgreSQLDropTypeTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	typeName, ok := params["type_name"].(string)
	if !ok || typeName == "" {
		return Error("type_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	parts := []string{"DROP TYPE"}
	
	if ifExists, ok := params["if_exists"].(bool); ok && ifExists {
		parts = append(parts, "IF EXISTS")
	}
	
	fullTypeName := fmt.Sprintf("%s.%s", quoteIdentifier(schema), quoteIdentifier(typeName))
	parts = append(parts, fullTypeName)
	
	if cascade, ok := params["cascade"].(bool); ok && cascade {
		parts = append(parts, "CASCADE")
	} else {
		parts = append(parts, "RESTRICT")
	}

	dropQuery := strings.Join(parts, " ")

	/* Execute DROP TYPE */
	err := t.executor.Exec(ctx, dropQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("DROP TYPE failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Type dropped", map[string]interface{}{
		"type_name": typeName,
		"schema":    schema,
	})

	return Success(map[string]interface{}{
		"type_name": typeName,
		"schema":    schema,
		"query":     dropQuery,
	}, map[string]interface{}{
		"tool": "postgresql_drop_type",
	}), nil
}

/* ============================================================================
 * Domain Management Tools
 * ============================================================================ */

/* PostgreSQLCreateDomainTool creates domains */
type PostgreSQLCreateDomainTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLCreateDomainTool creates a new PostgreSQL create domain tool */
func NewPostgreSQLCreateDomainTool(db *database.Database, logger *logging.Logger) *PostgreSQLCreateDomainTool {
	return &PostgreSQLCreateDomainTool{
		BaseTool: NewBaseTool(
			"postgresql_create_domain",
			"Create a domain with constraints",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"domain_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the domain to create",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"default":     "public",
						"description": "Schema name",
					},
					"base_type": map[string]interface{}{
						"type":        "string",
						"description": "Base type for the domain",
					},
					"constraint_name": map[string]interface{}{
						"type":        "string",
						"description": "Constraint name",
					},
					"constraint_definition": map[string]interface{}{
						"type":        "string",
						"description": "Constraint definition (e.g., CHECK (value > 0))",
					},
					"collation": map[string]interface{}{
						"type":        "string",
						"description": "Collation name",
					},
					"default": map[string]interface{}{
						"type":        "string",
						"description": "Default value",
					},
					"not_null": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Set NOT NULL constraint",
					},
					"if_not_exists": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Use IF NOT EXISTS clause",
					},
				},
				"required": []interface{}{"domain_name", "base_type"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute creates the domain */
func (t *PostgreSQLCreateDomainTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	domainName, ok := params["domain_name"].(string)
	if !ok || domainName == "" {
		return Error("domain_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	baseType, ok := params["base_type"].(string)
	if !ok || baseType == "" {
		return Error("base_type parameter is required", "INVALID_PARAMETER", nil), nil
	}

	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	parts := []string{"CREATE DOMAIN"}
	
	if ifNotExists, ok := params["if_not_exists"].(bool); ok && ifNotExists {
		parts = append(parts, "IF NOT EXISTS")
	}
	
	fullDomainName := fmt.Sprintf("%s.%s", quoteIdentifier(schema), quoteIdentifier(domainName))
	parts = append(parts, fullDomainName, "AS", baseType)

	/* Constraints and options */
	if constraintName, ok := params["constraint_name"].(string); ok && constraintName != "" {
		constraintDef, _ := params["constraint_definition"].(string)
		if constraintDef != "" {
			parts = append(parts, fmt.Sprintf("CONSTRAINT %s %s", quoteIdentifier(constraintName), constraintDef))
		}
	} else if constraintDef, ok := params["constraint_definition"].(string); ok && constraintDef != "" {
		parts = append(parts, constraintDef)
	}

	if collation, ok := params["collation"].(string); ok && collation != "" {
		parts = append(parts, fmt.Sprintf("COLLATE %s", quoteIdentifier(collation)))
	}

	if defaultValue, ok := params["default"].(string); ok && defaultValue != "" {
		parts = append(parts, fmt.Sprintf("DEFAULT %s", defaultValue))
	}

	if notNull, ok := params["not_null"].(bool); ok && notNull {
		parts = append(parts, "NOT NULL")
	}

	createQuery := strings.Join(parts, " ")

	/* Execute CREATE DOMAIN */
	err := t.executor.Exec(ctx, createQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("CREATE DOMAIN failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Domain created", map[string]interface{}{
		"domain_name": domainName,
		"schema":      schema,
	})

	return Success(map[string]interface{}{
		"domain_name": domainName,
		"schema":      schema,
		"query":       createQuery,
	}, map[string]interface{}{
		"tool": "postgresql_create_domain",
	}), nil
}

/* PostgreSQLAlterDomainTool alters domain properties */
type PostgreSQLAlterDomainTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLAlterDomainTool creates a new PostgreSQL alter domain tool */
func NewPostgreSQLAlterDomainTool(db *database.Database, logger *logging.Logger) *PostgreSQLAlterDomainTool {
	return &PostgreSQLAlterDomainTool{
		BaseTool: NewBaseTool(
			"postgresql_alter_domain",
			"Modify domain constraints or rename domains",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"domain_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the domain to alter",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"default":     "public",
						"description": "Schema name",
					},
					"new_name": map[string]interface{}{
						"type":        "string",
						"description": "New name for the domain (rename)",
					},
					"set_default": map[string]interface{}{
						"type":        "string",
						"description": "Set default value",
					},
					"drop_default": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Drop default value",
					},
					"set_not_null": map[string]interface{}{
						"type":        "boolean",
						"description": "Set NOT NULL constraint",
					},
					"drop_not_null": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Drop NOT NULL constraint",
					},
					"add_constraint": map[string]interface{}{
						"type":        "object",
						"description": "Add constraint: {name: string, definition: string}",
					},
					"drop_constraint": map[string]interface{}{
						"type":        "string",
						"description": "Name of constraint to drop",
					},
					"validate_constraint": map[string]interface{}{
						"type":        "string",
						"description": "Name of constraint to validate",
					},
				},
				"required": []interface{}{"domain_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute alters the domain */
func (t *PostgreSQLAlterDomainTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	domainName, ok := params["domain_name"].(string)
	if !ok || domainName == "" {
		return Error("domain_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	fullDomainName := fmt.Sprintf("%s.%s", quoteIdentifier(schema), quoteIdentifier(domainName))
	parts := []string{"ALTER DOMAIN", fullDomainName}

	alterations := []string{}

	/* Rename */
	if newName, ok := params["new_name"].(string); ok && newName != "" {
		if !isValidIdentifier(newName) {
			return Error("Invalid new domain name: must be a valid PostgreSQL identifier", "INVALID_PARAMETER", nil), nil
		}
		alterations = append(alterations, fmt.Sprintf("RENAME TO %s", quoteIdentifier(newName)))
	}

	/* Default */
	if setDefault, ok := params["set_default"].(string); ok && setDefault != "" {
		alterations = append(alterations, fmt.Sprintf("SET DEFAULT %s", setDefault))
	}
	if dropDefault, ok := params["drop_default"].(bool); ok && dropDefault {
		alterations = append(alterations, "DROP DEFAULT")
	}

	/* NOT NULL */
	if setNotNull, ok := params["set_not_null"].(bool); ok {
		if setNotNull {
			alterations = append(alterations, "SET NOT NULL")
		} else {
			alterations = append(alterations, "DROP NOT NULL")
		}
	}
	if dropNotNull, ok := params["drop_not_null"].(bool); ok && dropNotNull {
		alterations = append(alterations, "DROP NOT NULL")
	}

	/* Constraints */
	if addConstraint, ok := params["add_constraint"].(map[string]interface{}); ok && len(addConstraint) > 0 {
		name, _ := addConstraint["name"].(string)
		def, _ := addConstraint["definition"].(string)
		if name != "" && def != "" {
			alterations = append(alterations, fmt.Sprintf("ADD CONSTRAINT %s %s", quoteIdentifier(name), def))
		}
	}
	if dropConstraint, ok := params["drop_constraint"].(string); ok && dropConstraint != "" {
		alterations = append(alterations, fmt.Sprintf("DROP CONSTRAINT %s", quoteIdentifier(dropConstraint)))
	}
	if validateConstraint, ok := params["validate_constraint"].(string); ok && validateConstraint != "" {
		alterations = append(alterations, fmt.Sprintf("VALIDATE CONSTRAINT %s", quoteIdentifier(validateConstraint)))
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
				fmt.Sprintf("ALTER DOMAIN failed: %v", err),
				"QUERY_ERROR",
				map[string]interface{}{"error": err.Error(), "query": query},
			), nil
		}
	}

	t.logger.Info("Domain altered", map[string]interface{}{
		"domain_name": domainName,
		"schema":      schema,
	})

	return Success(map[string]interface{}{
		"domain_name": domainName,
		"schema":      schema,
		"queries":     queries,
	}, map[string]interface{}{
		"tool": "postgresql_alter_domain",
	}), nil
}

/* PostgreSQLDropDomainTool drops domains */
type PostgreSQLDropDomainTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLDropDomainTool creates a new PostgreSQL drop domain tool */
func NewPostgreSQLDropDomainTool(db *database.Database, logger *logging.Logger) *PostgreSQLDropDomainTool {
	return &PostgreSQLDropDomainTool{
		BaseTool: NewBaseTool(
			"postgresql_drop_domain",
			"Drop a domain",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"domain_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the domain to drop",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"default":     "public",
						"description": "Schema name",
					},
					"if_exists": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Use IF EXISTS clause",
					},
					"cascade": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Drop dependent objects (CASCADE)",
					},
				},
				"required": []interface{}{"domain_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute drops the domain */
func (t *PostgreSQLDropDomainTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	domainName, ok := params["domain_name"].(string)
	if !ok || domainName == "" {
		return Error("domain_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	parts := []string{"DROP DOMAIN"}
	
	if ifExists, ok := params["if_exists"].(bool); ok && ifExists {
		parts = append(parts, "IF EXISTS")
	}
	
	fullDomainName := fmt.Sprintf("%s.%s", quoteIdentifier(schema), quoteIdentifier(domainName))
	parts = append(parts, fullDomainName)
	
	if cascade, ok := params["cascade"].(bool); ok && cascade {
		parts = append(parts, "CASCADE")
	} else {
		parts = append(parts, "RESTRICT")
	}

	dropQuery := strings.Join(parts, " ")

	/* Execute DROP DOMAIN */
	err := t.executor.Exec(ctx, dropQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("DROP DOMAIN failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Domain dropped", map[string]interface{}{
		"domain_name": domainName,
		"schema":      schema,
	})

	return Success(map[string]interface{}{
		"domain_name": domainName,
		"schema":      schema,
		"query":       dropQuery,
	}, map[string]interface{}{
		"tool": "postgresql_drop_domain",
	}), nil
}

