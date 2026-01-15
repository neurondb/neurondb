/*-------------------------------------------------------------------------
 *
 * postgresql_dml.go
 *    Data Manipulation Language (DML) tools for NeuronMCP
 *
 * Implements comprehensive DML operations:
 * - INSERT, UPDATE, DELETE, TRUNCATE, COPY
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/postgresql_dml.go
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
 * INSERT Operations
 * ============================================================================ */

/* PostgreSQLInsertTool inserts rows into tables */
type PostgreSQLInsertTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLInsertTool creates a new PostgreSQL insert tool */
func NewPostgreSQLInsertTool(db *database.Database, logger *logging.Logger) *PostgreSQLInsertTool {
	return &PostgreSQLInsertTool{
		BaseTool: NewBaseTool(
			"postgresql_insert",
			"Insert single or multiple rows into a table with RETURNING clause and ON CONFLICT handling",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the table to insert into",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"default":     "public",
						"description": "Schema name",
					},
					"columns": map[string]interface{}{
						"type":        "array",
						"description": "Array of column names",
					},
					"values": map[string]interface{}{
						"type":        "array",
						"description": "Array of value arrays (for multiple rows) or single value array",
					},
					"returning": map[string]interface{}{
						"type":        "array",
						"description": "Array of column names to return",
					},
					"on_conflict": map[string]interface{}{
						"type":        "object",
						"description": "ON CONFLICT clause: {target: array, action: string, update_columns: array}",
					},
				},
				"required": []interface{}{"table_name", "values"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute inserts the rows */
func (t *PostgreSQLInsertTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	tableName, ok := params["table_name"].(string)
	if !ok || tableName == "" {
		return Error("table_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	values, ok := params["values"].([]interface{})
	if !ok || len(values) == 0 {
		return Error("values parameter is required and must be a non-empty array", "INVALID_PARAMETER", nil), nil
	}

	fullTableName := fmt.Sprintf("%s.%s", quoteIdentifier(schema), quoteIdentifier(tableName))

	/* Build INSERT statement */
	parts := []string{"INSERT INTO", fullTableName}

	/* Columns */
	columns, _ := params["columns"].([]interface{})
	if len(columns) > 0 {
		colList := []string{}
		for _, col := range columns {
			if colStr, ok := col.(string); ok {
				colList = append(colList, quoteIdentifier(colStr))
			}
		}
		if len(colList) > 0 {
			parts = append(parts, "("+strings.Join(colList, ", ")+")")
		}
	}

	/* Values */
	parts = append(parts, "VALUES")
	valueList := []string{}
	paramValues := []interface{}{}
	paramIndex := 1

	for _, row := range values {
		if rowArray, ok := row.([]interface{}); ok {
			rowValues := []string{}
			for _, val := range rowArray {
				rowValues = append(rowValues, fmt.Sprintf("$%d", paramIndex))
				paramValues = append(paramValues, val)
				paramIndex++
			}
			valueList = append(valueList, "("+strings.Join(rowValues, ", ")+")")
		} else {
			/* Single value */
			valueList = append(valueList, fmt.Sprintf("($%d)", paramIndex))
			paramValues = append(paramValues, row)
			paramIndex++
		}
	}
	parts = append(parts, strings.Join(valueList, ", "))

	/* ON CONFLICT */
	if onConflict, ok := params["on_conflict"].(map[string]interface{}); ok && len(onConflict) > 0 {
		parts = append(parts, "ON CONFLICT")
		
		if target, ok := onConflict["target"].([]interface{}); ok && len(target) > 0 {
			targetList := []string{}
			for _, t := range target {
				if tStr, ok := t.(string); ok {
					targetList = append(targetList, quoteIdentifier(tStr))
				}
			}
			if len(targetList) > 0 {
				parts = append(parts, "("+strings.Join(targetList, ", ")+")")
			}
		}
		
		if action, ok := onConflict["action"].(string); ok {
			switch strings.ToUpper(action) {
			case "DO NOTHING":
				parts = append(parts, "DO NOTHING")
			case "DO UPDATE":
				parts = append(parts, "DO UPDATE SET")
				if updateCols, ok := onConflict["update_columns"].([]interface{}); ok && len(updateCols) > 0 {
					updateList := []string{}
					for _, col := range updateCols {
						if colStr, ok := col.(string); ok {
							updateList = append(updateList, fmt.Sprintf("%s = EXCLUDED.%s", quoteIdentifier(colStr), quoteIdentifier(colStr)))
						}
					}
					if len(updateList) > 0 {
						parts = append(parts, strings.Join(updateList, ", "))
					}
				}
			}
		}
	}

	/* RETURNING */
	if returning, ok := params["returning"].([]interface{}); ok && len(returning) > 0 {
		returnList := []string{}
		for _, col := range returning {
			if colStr, ok := col.(string); ok {
				returnList = append(returnList, quoteIdentifier(colStr))
			}
		}
		if len(returnList) > 0 {
			parts = append(parts, "RETURNING", strings.Join(returnList, ", "))
		}
	}

	insertQuery := strings.Join(parts, " ")

	/* Execute INSERT */
	var results []map[string]interface{}
	var err error
	
	if len(paramValues) > 0 {
		/* Use parameterized query */
		if returning, ok := params["returning"].([]interface{}); ok && len(returning) > 0 {
			results, err = t.executor.ExecuteQuery(ctx, insertQuery, paramValues)
		} else {
			err = t.executor.Exec(ctx, insertQuery, paramValues)
		}
	} else {
		/* Direct execution without parameters */
		if returning, ok := params["returning"].([]interface{}); ok && len(returning) > 0 {
			results, err = t.executor.ExecuteQuery(ctx, insertQuery, nil)
		} else {
			err = t.executor.Exec(ctx, insertQuery, nil)
		}
	}

	if err != nil {
		return Error(
			fmt.Sprintf("INSERT failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	rowsAffected := len(valueList)
	if results != nil {
		rowsAffected = len(results)
	}

	t.logger.Info("Rows inserted", map[string]interface{}{
		"table_name":   tableName,
		"schema":       schema,
		"rows_affected": rowsAffected,
	})

	resultData := map[string]interface{}{
		"table_name":    tableName,
		"schema":        schema,
		"rows_affected": rowsAffected,
		"query":         insertQuery,
	}
	if results != nil && len(results) > 0 {
		resultData["returned_rows"] = results
	}

	return Success(resultData, map[string]interface{}{
		"tool": "postgresql_insert",
	}), nil
}

/* ============================================================================
 * UPDATE Operations
 * ============================================================================ */

/* PostgreSQLUpdateTool updates rows in tables */
type PostgreSQLUpdateTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLUpdateTool creates a new PostgreSQL update tool */
func NewPostgreSQLUpdateTool(db *database.Database, logger *logging.Logger) *PostgreSQLUpdateTool {
	return &PostgreSQLUpdateTool{
		BaseTool: NewBaseTool(
			"postgresql_update",
			"Update rows in a table with WHERE clause and RETURNING clause",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the table to update",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"default":     "public",
						"description": "Schema name",
					},
					"set": map[string]interface{}{
						"type":        "object",
						"description": "Column-value pairs to set",
					},
					"where": map[string]interface{}{
						"type":        "string",
						"description": "WHERE clause (can use $1, $2, etc. for parameters)",
					},
					"where_params": map[string]interface{}{
						"type":        "array",
						"description": "Parameters for WHERE clause",
					},
					"returning": map[string]interface{}{
						"type":        "array",
						"description": "Array of column names to return",
					},
				},
				"required": []interface{}{"table_name", "set"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute updates the rows */
func (t *PostgreSQLUpdateTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	tableName, ok := params["table_name"].(string)
	if !ok || tableName == "" {
		return Error("table_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	set, ok := params["set"].(map[string]interface{})
	if !ok || len(set) == 0 {
		return Error("set parameter is required and must be a non-empty object", "INVALID_PARAMETER", nil), nil
	}

	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	fullTableName := fmt.Sprintf("%s.%s", quoteIdentifier(schema), quoteIdentifier(tableName))

	/* Build UPDATE statement */
	parts := []string{"UPDATE", fullTableName, "SET"}

	setList := []string{}
	paramValues := []interface{}{}
	paramIndex := 1

	for col, val := range set {
		setList = append(setList, fmt.Sprintf("%s = $%d", quoteIdentifier(col), paramIndex))
		paramValues = append(paramValues, val)
		paramIndex++
	}
	parts = append(parts, strings.Join(setList, ", "))

	/* WHERE clause */
	where, _ := params["where"].(string)
	if where != "" {
		parts = append(parts, "WHERE", where)
		
		if whereParams, ok := params["where_params"].([]interface{}); ok && len(whereParams) > 0 {
			/* Replace $1, $2, etc. in WHERE clause with parameterized values */
			for _, p := range whereParams {
				paramValues = append(paramValues, p)
			}
		}
	}

	/* RETURNING */
	if returning, ok := params["returning"].([]interface{}); ok && len(returning) > 0 {
		returnList := []string{}
		for _, col := range returning {
			if colStr, ok := col.(string); ok {
				returnList = append(returnList, quoteIdentifier(colStr))
			}
		}
		if len(returnList) > 0 {
			parts = append(parts, "RETURNING", strings.Join(returnList, ", "))
		}
	}

	updateQuery := strings.Join(parts, " ")

	/* Execute UPDATE */
	var results []map[string]interface{}
	var err error
	
	if returning, ok := params["returning"].([]interface{}); ok && len(returning) > 0 {
		results, err = t.executor.ExecuteQuery(ctx, updateQuery, paramValues)
	} else {
		err = t.executor.Exec(ctx, updateQuery, paramValues)
	}

	if err != nil {
		return Error(
			fmt.Sprintf("UPDATE failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	rowsAffected := 0
	if results != nil {
		rowsAffected = len(results)
	}

	t.logger.Info("Rows updated", map[string]interface{}{
		"table_name":   tableName,
		"schema":       schema,
		"rows_affected": rowsAffected,
	})

	resultData := map[string]interface{}{
		"table_name":    tableName,
		"schema":        schema,
		"rows_affected": rowsAffected,
		"query":         updateQuery,
	}
	if results != nil && len(results) > 0 {
		resultData["returned_rows"] = results
	}

	return Success(resultData, map[string]interface{}{
		"tool": "postgresql_update",
	}), nil
}

/* ============================================================================
 * DELETE Operations
 * ============================================================================ */

/* PostgreSQLDeleteTool deletes rows from tables */
type PostgreSQLDeleteTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLDeleteTool creates a new PostgreSQL delete tool */
func NewPostgreSQLDeleteTool(db *database.Database, logger *logging.Logger) *PostgreSQLDeleteTool {
	return &PostgreSQLDeleteTool{
		BaseTool: NewBaseTool(
			"postgresql_delete",
			"Delete rows from a table with WHERE clause and RETURNING clause",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the table to delete from",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"default":     "public",
						"description": "Schema name",
					},
					"where": map[string]interface{}{
						"type":        "string",
						"description": "WHERE clause (can use $1, $2, etc. for parameters)",
					},
					"where_params": map[string]interface{}{
						"type":        "array",
						"description": "Parameters for WHERE clause",
					},
					"returning": map[string]interface{}{
						"type":        "array",
						"description": "Array of column names to return",
					},
				},
				"required": []interface{}{"table_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute deletes the rows */
func (t *PostgreSQLDeleteTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	tableName, ok := params["table_name"].(string)
	if !ok || tableName == "" {
		return Error("table_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	fullTableName := fmt.Sprintf("%s.%s", quoteIdentifier(schema), quoteIdentifier(tableName))

	/* Build DELETE statement */
	parts := []string{"DELETE FROM", fullTableName}

	paramValues := []interface{}{}

	/* WHERE clause */
	where, _ := params["where"].(string)
	if where != "" {
		parts = append(parts, "WHERE", where)
		
		if whereParams, ok := params["where_params"].([]interface{}); ok && len(whereParams) > 0 {
			for _, p := range whereParams {
				paramValues = append(paramValues, p)
			}
		}
	}

	/* RETURNING */
	if returning, ok := params["returning"].([]interface{}); ok && len(returning) > 0 {
		returnList := []string{}
		for _, col := range returning {
			if colStr, ok := col.(string); ok {
				returnList = append(returnList, quoteIdentifier(colStr))
			}
		}
		if len(returnList) > 0 {
			parts = append(parts, "RETURNING", strings.Join(returnList, ", "))
		}
	}

	deleteQuery := strings.Join(parts, " ")

	/* Execute DELETE */
	var results []map[string]interface{}
	var err error
	
	if returning, ok := params["returning"].([]interface{}); ok && len(returning) > 0 {
		results, err = t.executor.ExecuteQuery(ctx, deleteQuery, paramValues)
	} else {
		err = t.executor.Exec(ctx, deleteQuery, paramValues)
	}

	if err != nil {
		return Error(
			fmt.Sprintf("DELETE failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	rowsAffected := 0
	if results != nil {
		rowsAffected = len(results)
	}

	t.logger.Info("Rows deleted", map[string]interface{}{
		"table_name":   tableName,
		"schema":       schema,
		"rows_affected": rowsAffected,
	})

	resultData := map[string]interface{}{
		"table_name":    tableName,
		"schema":        schema,
		"rows_affected": rowsAffected,
		"query":         deleteQuery,
	}
	if results != nil && len(results) > 0 {
		resultData["returned_rows"] = results
	}

	return Success(resultData, map[string]interface{}{
		"tool": "postgresql_delete",
	}), nil
}

/* ============================================================================
 * TRUNCATE Operations
 * ============================================================================ */

/* PostgreSQLTruncateTool truncates tables */
type PostgreSQLTruncateTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLTruncateTool creates a new PostgreSQL truncate tool */
func NewPostgreSQLTruncateTool(db *database.Database, logger *logging.Logger) *PostgreSQLTruncateTool {
	return &PostgreSQLTruncateTool{
		BaseTool: NewBaseTool(
			"postgresql_truncate",
			"Truncate tables with CASCADE and RESTART IDENTITY options",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the table to truncate (or array of table names)",
					},
					"table_names": map[string]interface{}{
						"type":        "array",
						"description": "Array of table names to truncate",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"default":     "public",
						"description": "Schema name",
					},
					"cascade": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Truncate dependent tables (CASCADE)",
					},
					"restart_identity": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Restart sequences owned by columns",
					},
					"only": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Truncate only this table (not child tables)",
					},
				},
				"required": []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute truncates the tables */
func (t *PostgreSQLTruncateTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	/* Get table names */
	tableNames := []string{}
	
	if tableNamesArray, ok := params["table_names"].([]interface{}); ok && len(tableNamesArray) > 0 {
		for _, tn := range tableNamesArray {
			if tnStr, ok := tn.(string); ok {
				tableNames = append(tableNames, fmt.Sprintf("%s.%s", quoteIdentifier(schema), quoteIdentifier(tnStr)))
			}
		}
	} else if tableName, ok := params["table_name"].(string); ok && tableName != "" {
		tableNames = append(tableNames, fmt.Sprintf("%s.%s", quoteIdentifier(schema), quoteIdentifier(tableName)))
	}

	if len(tableNames) == 0 {
		return Error("table_name or table_names parameter is required", "INVALID_PARAMETER", nil), nil
	}

	/* Build TRUNCATE statement */
	parts := []string{"TRUNCATE"}
	
	if only, ok := params["only"].(bool); ok && only {
		parts = append(parts, "ONLY")
	}
	
	parts = append(parts, strings.Join(tableNames, ", "))
	
	if restartIdentity, ok := params["restart_identity"].(bool); ok && restartIdentity {
		parts = append(parts, "RESTART IDENTITY")
	}
	
	if cascade, ok := params["cascade"].(bool); ok && cascade {
		parts = append(parts, "CASCADE")
	} else {
		parts = append(parts, "RESTRICT")
	}

	truncateQuery := strings.Join(parts, " ")

	/* Execute TRUNCATE */
	err := t.executor.Exec(ctx, truncateQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("TRUNCATE failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Tables truncated", map[string]interface{}{
		"table_names": tableNames,
		"schema":      schema,
	})

	return Success(map[string]interface{}{
		"table_names": tableNames,
		"schema":      schema,
		"query":        truncateQuery,
	}, map[string]interface{}{
		"tool": "postgresql_truncate",
	}), nil
}

/* ============================================================================
 * COPY Operations
 * ============================================================================ */

/* PostgreSQLCopyTool performs COPY operations */
type PostgreSQLCopyTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLCopyTool creates a new PostgreSQL copy tool */
func NewPostgreSQLCopyTool(db *database.Database, logger *logging.Logger) *PostgreSQLCopyTool {
	return &PostgreSQLCopyTool{
		BaseTool: NewBaseTool(
			"postgresql_copy",
			"COPY FROM (import from file/stdin) or COPY TO (export to file/stdout) with CSV, binary, text formats",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"direction": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"FROM", "TO"},
						"description": "COPY direction: FROM (import) or TO (export)",
					},
					"table_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the table",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"default":     "public",
						"description": "Schema name",
					},
					"columns": map[string]interface{}{
						"type":        "array",
						"description": "Array of column names",
					},
					"source": map[string]interface{}{
						"type":        "string",
						"description": "Source file path (for FROM) or destination file path (for TO). Use 'stdin' or 'stdout' for standard streams",
					},
					"format": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"csv", "text", "binary"},
						"default":     "csv",
						"description": "Data format",
					},
					"delimiter": map[string]interface{}{
						"type":        "string",
						"default":     ",",
						"description": "Delimiter character (for CSV/TEXT)",
					},
					"null_string": map[string]interface{}{
						"type":        "string",
						"default":     "\\N",
						"description": "String representing NULL",
					},
					"header": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Include header row (for CSV)",
					},
					"quote": map[string]interface{}{
						"type":        "string",
						"default":     "\"",
						"description": "Quote character (for CSV)",
					},
					"escape": map[string]interface{}{
						"type":        "string",
						"description": "Escape character (for CSV)",
					},
				},
				"required": []interface{}{"direction", "table_name", "source"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute performs the COPY operation */
func (t *PostgreSQLCopyTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	direction, ok := params["direction"].(string)
	if !ok || direction == "" {
		return Error("direction parameter is required", "INVALID_PARAMETER", nil), nil
	}
	direction = strings.ToUpper(direction)

	tableName, ok := params["table_name"].(string)
	if !ok || tableName == "" {
		return Error("table_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	source, ok := params["source"].(string)
	if !ok || source == "" {
		return Error("source parameter is required", "INVALID_PARAMETER", nil), nil
	}

	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	fullTableName := fmt.Sprintf("%s.%s", quoteIdentifier(schema), quoteIdentifier(tableName))

	/* Build COPY statement */
	parts := []string{"COPY"}

	/* Columns */
	columns, _ := params["columns"].([]interface{})
	if len(columns) > 0 {
		colList := []string{}
		for _, col := range columns {
			if colStr, ok := col.(string); ok {
				colList = append(colList, quoteIdentifier(colStr))
			}
		}
		if len(colList) > 0 {
			parts = append(parts, "("+strings.Join(colList, ", ")+")")
		}
	}

	parts = append(parts, fullTableName)
	parts = append(parts, direction)

	/* Source/destination */
	if source == "stdin" || source == "stdout" {
		parts = append(parts, strings.ToUpper(source))
	} else {
		parts = append(parts, quoteLiteral(source))
	}

	/* Format and options */
	format, _ := params["format"].(string)
	if format == "" {
		format = "csv"
	}
	parts = append(parts, fmt.Sprintf("WITH (%s", strings.ToUpper(format)))

	options := []string{}

	if delimiter, ok := params["delimiter"].(string); ok && delimiter != "" {
		options = append(options, fmt.Sprintf("DELIMITER %s", quoteLiteral(delimiter)))
	}

	if nullString, ok := params["null_string"].(string); ok && nullString != "" {
		options = append(options, fmt.Sprintf("NULL %s", quoteLiteral(nullString)))
	}

	if header, ok := params["header"].(bool); ok && header {
		options = append(options, "HEADER")
	}

	if quote, ok := params["quote"].(string); ok && quote != "" {
		options = append(options, fmt.Sprintf("QUOTE %s", quoteLiteral(quote)))
	}

	if escape, ok := params["escape"].(string); ok && escape != "" {
		options = append(options, fmt.Sprintf("ESCAPE %s", quoteLiteral(escape)))
	}

	if len(options) > 0 {
		parts[len(parts)-1] = parts[len(parts)-1] + " " + strings.Join(options, ", ")
	}
	parts[len(parts)-1] = parts[len(parts)-1] + ")"

	copyQuery := strings.Join(parts, " ")

	/* Execute COPY */
	/* Note: COPY requires special handling - it may need to be executed differently
	 * depending on whether it's FROM stdin or TO stdout. For now, we'll use Exec.
	 * In production, this might need to use COPY protocol directly.
	 */
	err := t.executor.Exec(ctx, copyQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("COPY failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("COPY operation completed", map[string]interface{}{
		"direction": direction,
		"table_name": tableName,
		"schema":     schema,
		"source":     source,
	})

	return Success(map[string]interface{}{
		"direction":  direction,
		"table_name": tableName,
		"schema":     schema,
		"source":     source,
		"query":      copyQuery,
	}, map[string]interface{}{
		"tool": "postgresql_copy",
	}), nil
}


