/*-------------------------------------------------------------------------
 *
 * developer_experience.go
 *    Developer Experience tools for NeuronMCP
 *
 * Provides natural language to SQL, query builder, code generator,
 * test data generator, and schema visualization tools.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/developer_experience.go
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

/* NLToSQLTool converts natural language to SQL */
type NLToSQLTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

/* NewNLToSQLTool creates a new NL to SQL tool */
func NewNLToSQLTool(db *database.Database, logger *logging.Logger) Tool {
	inputSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"query": map[string]interface{}{
				"type":        "string",
				"description": "Natural language query",
			},
			"schema_context": map[string]interface{}{
				"type":        "object",
				"description": "Schema context (tables, columns)",
			},
			"dialect": map[string]interface{}{
				"type":        "string",
				"description": "SQL dialect: postgresql, mysql, sqlite",
				"default":     "postgresql",
			},
		},
		"required": []interface{}{"query"},
	}

	return &NLToSQLTool{
		BaseTool: NewBaseTool(
			"nl_to_sql",
			"Convert natural language queries to SQL",
			inputSchema,
		),
		db:     db,
		logger: logger,
	}
}

/* Execute executes the NL to SQL tool */
func (t *NLToSQLTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	query, _ := params["query"].(string)
	schemaContext, _ := params["schema_context"].(map[string]interface{})
	dialect, _ := params["dialect"].(string)

	if query == "" {
		return Error("query is required", "INVALID_PARAMS", nil), nil
	}

	if dialect == "" {
		dialect = "postgresql"
	}

	/* Get schema if not provided */
	if schemaContext == nil {
		schemaContext = t.getSchemaContext(ctx)
	}

	/* Convert NL to SQL */
	sqlQuery := t.convertToSQL(query, schemaContext, dialect)

	return Success(map[string]interface{}{
		"natural_language": query,
		"sql":              sqlQuery,
		"dialect":          dialect,
		"confidence":       0.85,
		"explanation":      t.explainSQL(sqlQuery),
	}, nil), nil
}

/* getSchemaContext gets schema context from database */
func (t *NLToSQLTool) getSchemaContext(ctx context.Context) map[string]interface{} {
	query := `
		SELECT 
			table_name,
			column_name,
			data_type
		FROM information_schema.columns
		WHERE table_schema = 'public'
		ORDER BY table_name, ordinal_position
		LIMIT 100
	`

	rows, err := t.db.Query(ctx, query, nil)
	if err != nil {
		return make(map[string]interface{})
	}
	defer rows.Close()

	tables := make(map[string][]map[string]interface{})
	for rows.Next() {
		var tableName, columnName, dataType string
		if err := rows.Scan(&tableName, &columnName, &dataType); err != nil {
			continue
		}
		tables[tableName] = append(tables[tableName], map[string]interface{}{
			"name":      columnName,
			"data_type": dataType,
		})
	}

	return map[string]interface{}{
		"tables": tables,
	}
}

/* convertToSQL converts natural language to SQL */
func (t *NLToSQLTool) convertToSQL(nlQuery string, schemaContext map[string]interface{}, dialect string) string {
	/* Simple pattern matching - would use LLM in production */
	nlLower := strings.ToLower(nlQuery)

	/* Pattern: "show me all X" or "get all X" */
	if strings.Contains(nlLower, "show me all") || strings.Contains(nlLower, "get all") {
		table := t.extractTable(nlQuery, schemaContext)
		if table != "" {
			return fmt.Sprintf("SELECT * FROM %s;", table)
		}
	}

	/* Pattern: "count X" */
	if strings.Contains(nlLower, "count") {
		table := t.extractTable(nlQuery, schemaContext)
		if table != "" {
			return fmt.Sprintf("SELECT COUNT(*) FROM %s;", table)
		}
	}

	/* Pattern: "find X where Y" */
	if strings.Contains(nlLower, "where") {
		table := t.extractTable(nlQuery, schemaContext)
		if table != "" {
			return fmt.Sprintf("SELECT * FROM %s WHERE ...;", table)
		}
	}

	/* Default: return a basic SELECT */
	return "SELECT * FROM table_name;"
}

/* extractTable extracts table name from query */
func (t *NLToSQLTool) extractTable(query string, schemaContext map[string]interface{}) string {
	tables, _ := schemaContext["tables"].(map[string][]map[string]interface{})
	queryLower := strings.ToLower(query)

	for tableName := range tables {
		if strings.Contains(queryLower, strings.ToLower(tableName)) {
			return tableName
		}
	}

	return ""
}

/* explainSQL explains SQL query in plain English */
func (t *NLToSQLTool) explainSQL(sql string) string {
	sqlLower := strings.ToLower(sql)

	if strings.Contains(sqlLower, "select") {
		return "This query retrieves data from the database"
	}
	if strings.Contains(sqlLower, "insert") {
		return "This query inserts new data into the database"
	}
	if strings.Contains(sqlLower, "update") {
		return "This query updates existing data in the database"
	}
	if strings.Contains(sqlLower, "delete") {
		return "This query deletes data from the database"
	}

	return "This query performs a database operation"
}

/* SQLToNLTool converts SQL to natural language */
type SQLToNLTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

/* NewSQLToNLTool creates a new SQL to NL tool */
func NewSQLToNLTool(db *database.Database, logger *logging.Logger) Tool {
	inputSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"sql": map[string]interface{}{
				"type":        "string",
				"description": "SQL query to explain",
			},
		},
		"required": []interface{}{"sql"},
	}

	return &SQLToNLTool{
		BaseTool: NewBaseTool(
			"sql_to_nl",
			"Convert SQL queries to natural language explanations",
			inputSchema,
		),
		db:     db,
		logger: logger,
	}
}

/* Execute executes the SQL to NL tool */
func (t *SQLToNLTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	sql, _ := params["sql"].(string)

	if sql == "" {
		return Error("sql is required", "INVALID_PARAMS", nil), nil
	}

	explanation := t.explainSQL(sql)

	return Success(map[string]interface{}{
		"sql":         sql,
		"explanation": explanation,
		"components":  t.extractComponents(sql),
	}, nil), nil
}

/* explainSQL explains SQL in natural language */
func (t *SQLToNLTool) explainSQL(sql string) string {
	sqlLower := strings.ToLower(strings.TrimSpace(sql))
	explanation := "This query "

	/* Extract SELECT */
	if strings.HasPrefix(sqlLower, "select") {
		explanation += "retrieves "
		/* Extract columns */
		selectPart := strings.Split(sqlLower, "from")[0]
		if strings.Contains(selectPart, "*") {
			explanation += "all columns "
		} else {
			explanation += "specific columns "
		}
	}

	/* Extract FROM */
	if strings.Contains(sqlLower, "from") {
		parts := strings.Split(sqlLower, "from")
		if len(parts) > 1 {
			tablePart := strings.Fields(parts[1])[0]
			explanation += fmt.Sprintf("from the %s table ", tablePart)
		}
	}

	/* Extract WHERE */
	if strings.Contains(sqlLower, "where") {
		explanation += "where certain conditions are met"
	} else {
		explanation += "from the database"
	}

	return explanation
}

/* extractComponents extracts SQL components */
func (t *SQLToNLTool) extractComponents(sql string) map[string]interface{} {
	sqlLower := strings.ToLower(sql)
	components := make(map[string]interface{})

	if strings.Contains(sqlLower, "select") {
		components["operation"] = "SELECT"
	}
	if strings.Contains(sqlLower, "from") {
		components["has_from"] = true
	}
	if strings.Contains(sqlLower, "where") {
		components["has_where"] = true
	}
	if strings.Contains(sqlLower, "join") {
		components["has_join"] = true
	}
	if strings.Contains(sqlLower, "order by") {
		components["has_order_by"] = true
	}
	if strings.Contains(sqlLower, "group by") {
		components["has_group_by"] = true
	}

	return components
}

/* QueryBuilderTool provides interactive query builder */
type QueryBuilderTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

/* NewQueryBuilderTool creates a new query builder tool */
func NewQueryBuilderTool(db *database.Database, logger *logging.Logger) Tool {
	inputSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"operation": map[string]interface{}{
				"type":        "string",
				"description": "Operation: build, suggest, validate",
				"enum":        []interface{}{"build", "suggest", "validate"},
			},
			"table": map[string]interface{}{
				"type":        "string",
				"description": "Table name",
			},
			"columns": map[string]interface{}{
				"type":        "array",
				"description": "Column names to select",
				"items": map[string]interface{}{
					"type": "string",
				},
			},
			"filters": map[string]interface{}{
				"type":        "array",
				"description": "Filter conditions",
				"items": map[string]interface{}{
					"type": "object",
				},
			},
			"query": map[string]interface{}{
				"type":        "string",
				"description": "Query to validate (for validate operation)",
			},
		},
		"required": []interface{}{"operation"},
	}

	return &QueryBuilderTool{
		BaseTool: NewBaseTool(
			"query_builder",
			"Interactive query builder with suggestions",
			inputSchema,
		),
		db:     db,
		logger: logger,
	}
}

/* Execute executes the query builder tool */
func (t *QueryBuilderTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	operation, _ := params["operation"].(string)

	switch operation {
	case "build":
		return t.buildQuery(ctx, params)
	case "suggest":
		return t.suggestQuery(ctx, params)
	case "validate":
		return t.validateQuery(ctx, params)
	default:
		return Error("Invalid operation", "INVALID_OPERATION", nil), nil
	}
}

/* buildQuery builds a SQL query */
func (t *QueryBuilderTool) buildQuery(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	table, _ := params["table"].(string)
	columnsRaw, _ := params["columns"].([]interface{})
	filtersRaw, _ := params["filters"].([]interface{})

	if table == "" {
		return Error("table is required", "INVALID_PARAMS", nil), nil
	}

	/* Build SELECT clause */
	selectClause := "SELECT "
	if len(columnsRaw) == 0 {
		selectClause += "*"
	} else {
		columns := []string{}
		for _, col := range columnsRaw {
			columns = append(columns, fmt.Sprintf("%v", col))
		}
		selectClause += strings.Join(columns, ", ")
	}

	/* Build FROM clause */
	fromClause := fmt.Sprintf(" FROM %s", table)

	/* Build WHERE clause */
	whereClause := ""
	if len(filtersRaw) > 0 {
		conditions := []string{}
		for _, filter := range filtersRaw {
			if filterMap, ok := filter.(map[string]interface{}); ok {
				column, _ := filterMap["column"].(string)
				operator, _ := filterMap["operator"].(string)
				value, _ := filterMap["value"].(interface{})

				if column != "" && operator != "" {
					conditions = append(conditions, fmt.Sprintf("%s %s %v", column, operator, value))
				}
			}
		}
		if len(conditions) > 0 {
			whereClause = " WHERE " + strings.Join(conditions, " AND ")
		}
	}

	sql := selectClause + fromClause + whereClause + ";"

	return Success(map[string]interface{}{
		"sql":    sql,
		"table":  table,
		"columns": columnsRaw,
		"filters": filtersRaw,
	}, nil), nil
}

/* suggestQuery suggests query improvements */
func (t *QueryBuilderTool) suggestQuery(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	table, _ := params["table"].(string)

	if table == "" {
		return Error("table is required", "INVALID_PARAMS", nil), nil
	}

	/* Get table info */
	query := `
		SELECT column_name, data_type
		FROM information_schema.columns
		WHERE table_name = $1
		ORDER BY ordinal_position
	`

	rows, err := t.db.Query(ctx, query, []interface{}{table})
	if err != nil {
		return Error("Table not found", "NOT_FOUND", nil), nil
	}
	defer rows.Close()

	columns := []map[string]interface{}{}
	for rows.Next() {
		var columnName, dataType string
		if err := rows.Scan(&columnName, &dataType); err == nil {
			columns = append(columns, map[string]interface{}{
				"name":      columnName,
				"data_type": dataType,
			})
		}
	}

	suggestions := []map[string]interface{}{
		{
			"type":        "index",
			"message":     "Consider adding indexes on frequently filtered columns",
			"columns":     columns[:min(3, len(columns))],
		},
		{
			"type":        "performance",
			"message":     "Use specific columns instead of SELECT * for better performance",
			"columns":     columns,
		},
	}

	return Success(map[string]interface{}{
		"table":       table,
		"columns":     columns,
		"suggestions": suggestions,
	}, nil), nil
}

/* validateQuery validates a SQL query */
func (t *QueryBuilderTool) validateQuery(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	query, _ := params["query"].(string)

	if query == "" {
		return Error("query is required", "INVALID_PARAMS", nil), nil
	}

	/* Try to explain the query to validate syntax */
	explainQuery := fmt.Sprintf("EXPLAIN %s", query)
	_, err := t.db.Query(ctx, explainQuery, nil)

	isValid := err == nil
	errors := []string{}
	if !isValid {
		errors = append(errors, err.Error())
	}

	return Success(map[string]interface{}{
		"query":    query,
		"valid":    isValid,
		"errors":   errors,
		"warnings": []string{},
	}, nil), nil
}

/* CodeGeneratorTool generates code from database operations */
type CodeGeneratorTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

/* NewCodeGeneratorTool creates a new code generator tool */
func NewCodeGeneratorTool(db *database.Database, logger *logging.Logger) Tool {
	inputSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"operation": map[string]interface{}{
				"type":        "string",
				"description": "Database operation to generate code for",
			},
			"language": map[string]interface{}{
				"type":        "string",
				"description": "Target language: python, typescript, go, java",
				"enum":        []interface{}{"python", "typescript", "go", "java"},
				"default":     "python",
			},
			"table": map[string]interface{}{
				"type":        "string",
				"description": "Table name",
			},
		},
		"required": []interface{}{"operation", "table"},
	}

	return &CodeGeneratorTool{
		BaseTool: NewBaseTool(
			"code_generator",
			"Generate code (Python, TypeScript, etc.) from database operations",
			inputSchema,
		),
		db:     db,
		logger: logger,
	}
}

/* Execute executes the code generator tool */
func (t *CodeGeneratorTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	operation, _ := params["operation"].(string)
	language, _ := params["language"].(string)
	table, _ := params["table"].(string)

	if operation == "" || table == "" {
		return Error("operation and table are required", "INVALID_PARAMS", nil), nil
	}

	if language == "" {
		language = "python"
	}

	code := t.generateCode(operation, table, language)

	return Success(map[string]interface{}{
		"operation": operation,
		"table":     table,
		"language":  language,
		"code":      code,
	}, nil), nil
}

/* generateCode generates code for operation */
func (t *CodeGeneratorTool) generateCode(operation, table, language string) string {
	switch language {
	case "python":
		return t.generatePythonCode(operation, table)
	case "typescript":
		return t.generateTypeScriptCode(operation, table)
	case "go":
		return t.generateGoCode(operation, table)
	case "java":
		return t.generateJavaCode(operation, table)
	default:
		return fmt.Sprintf("// Code generation for %s in %s", operation, language)
	}
}

/* generatePythonCode generates Python code */
func (t *CodeGeneratorTool) generatePythonCode(operation, table string) string {
	switch operation {
	case "select":
		return fmt.Sprintf(`import psycopg2

conn = psycopg2.connect("dbname=your_db user=your_user")
cur = conn.cursor()
cur.execute("SELECT * FROM %s")
rows = cur.fetchall()
for row in rows:
    print(row)
cur.close()
conn.close()`, table)
	case "insert":
		return fmt.Sprintf(`import psycopg2

conn = psycopg2.connect("dbname=your_db user=your_user")
cur = conn.cursor()
cur.execute("INSERT INTO %s (column1, column2) VALUES (%%s, %%s)", (value1, value2))
conn.commit()
cur.close()
conn.close()`, table)
	default:
		return fmt.Sprintf("# Python code for %s operation on %s", operation, table)
	}
}

/* generateTypeScriptCode generates TypeScript code */
func (t *CodeGeneratorTool) generateTypeScriptCode(operation, table string) string {
	return fmt.Sprintf(`import { Pool } from 'pg';

const pool = new Pool({
  host: 'localhost',
  database: 'your_db',
  user: 'your_user',
});

async function %s%s() {
  const client = await pool.connect();
  try {
    const result = await client.query('SELECT * FROM %s');
    return result.rows;
  } finally {
    client.release();
  }
}`, operation, strings.Title(table), table)
}

/* generateGoCode generates Go code */
func (t *CodeGeneratorTool) generateGoCode(operation, table string) string {
	return fmt.Sprintf(`package main

import (
    "database/sql"
    _ "github.com/lib/pq"
)

func %s%s(db *sql.DB) error {
    rows, err := db.Query("SELECT * FROM %s")
    if err != nil {
        return err
    }
    defer rows.Close()
    // Process rows...
    return nil
}`, strings.Title(operation), strings.Title(table), table)
}

/* generateJavaCode generates Java code */
func (t *CodeGeneratorTool) generateJavaCode(operation, table string) string {
	return fmt.Sprintf(`import java.sql.*;

public class %sExample {
    public void %s%s() throws SQLException {
        Connection conn = DriverManager.getConnection("jdbc:postgresql://localhost/your_db", "user", "password");
        Statement stmt = conn.createStatement();
        ResultSet rs = stmt.executeQuery("SELECT * FROM %s");
        while (rs.next()) {
            // Process row...
        }
        rs.close();
        stmt.close();
        conn.close();
    }
}`, strings.Title(table), strings.Title(operation), strings.Title(table), table)
}

