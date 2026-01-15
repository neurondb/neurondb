/*-------------------------------------------------------------------------
 *
 * developer_experience_additional.go
 *    Additional Developer Experience tools for NeuronMCP
 *
 * Provides test data generation, schema visualization, query explanation,
 * schema documentation, and migration generation.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/developer_experience_additional.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"strings"
	"time"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

/* TestDataGeneratorTool generates realistic test data */
type TestDataGeneratorTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

/* NewTestDataGeneratorTool creates a new test data generator tool */
func NewTestDataGeneratorTool(db *database.Database, logger *logging.Logger) Tool {
	inputSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"table": map[string]interface{}{
				"type":        "string",
				"description": "Table name",
			},
			"row_count": map[string]interface{}{
				"type":        "integer",
				"description": "Number of rows to generate",
				"default":     100,
			},
			"seed": map[string]interface{}{
				"type":        "integer",
				"description": "Random seed for reproducible data",
			},
		},
		"required": []interface{}{"table"},
	}

	return &TestDataGeneratorTool{
		BaseTool: NewBaseTool(
			"test_data_generator",
			"Generate realistic test data for tables",
			inputSchema,
		),
		db:     db,
		logger: logger,
	}
}

/* Execute executes the test data generator tool */
func (t *TestDataGeneratorTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	table, _ := params["table"].(string)
	rowCount, _ := params["row_count"].(float64)
	seed, _ := params["seed"].(float64)

	if table == "" {
		return Error("table is required", "INVALID_PARAMS", nil), nil
	}

	if rowCount == 0 {
		rowCount = 100
	}

	/* Set seed if provided */
	if seed > 0 {
		rand.Seed(int64(seed))
	} else {
		rand.Seed(time.Now().UnixNano())
	}

	/* Get table schema */
	schema := t.getTableSchema(ctx, table)
	if len(schema) == 0 {
		return Error("Table not found", "NOT_FOUND", nil), nil
	}

	/* Generate INSERT statements */
	insertStatements := t.generateInsertStatements(table, schema, int(rowCount))

	return Success(map[string]interface{}{
		"table":            table,
		"row_count":        int(rowCount),
		"insert_statements": insertStatements,
		"preview":          insertStatements[:min(5, len(insertStatements))],
	}, nil), nil
}

/* getTableSchema gets table schema */
func (t *TestDataGeneratorTool) getTableSchema(ctx context.Context, table string) []map[string]interface{} {
	query := `
		SELECT 
			column_name,
			data_type,
			is_nullable,
			column_default
		FROM information_schema.columns
		WHERE table_name = $1
		ORDER BY ordinal_position
	`

	rows, err := t.db.Query(ctx, query, []interface{}{table})
	if err != nil {
		return []map[string]interface{}{}
	}
	defer rows.Close()

	schema := []map[string]interface{}{}
	for rows.Next() {
		var columnName, dataType, isNullable string
		var columnDefault *string

		if err := rows.Scan(&columnName, &dataType, &isNullable, &columnDefault); err != nil {
			continue
		}

		schema = append(schema, map[string]interface{}{
			"name":       columnName,
			"data_type":  dataType,
			"nullable":   isNullable == "YES",
			"default":    getString(columnDefault, ""),
		})
	}

	return schema
}

/* generateInsertStatements generates INSERT statements */
func (t *TestDataGeneratorTool) generateInsertStatements(table string, schema []map[string]interface{}, rowCount int) []string {
	statements := []string{}

	/* Get column names */
	columns := []string{}
	for _, col := range schema {
		columns = append(columns, col["name"].(string))
	}

	columnsStr := strings.Join(columns, ", ")

	for i := 0; i < rowCount; i++ {
		values := []string{}
		for _, col := range schema {
			value := t.generateValue(col)
			values = append(values, value)
		}

		valuesStr := strings.Join(values, ", ")
		stmt := fmt.Sprintf("INSERT INTO %s (%s) VALUES (%s);", table, columnsStr, valuesStr)
		statements = append(statements, stmt)
	}

	return statements
}

/* generateValue generates a test value for a column */
func (t *TestDataGeneratorTool) generateValue(column map[string]interface{}) string {
	dataType, _ := column["data_type"].(string)
	nullable, _ := column["nullable"].(bool)

	if nullable && rand.Float32() < 0.1 {
		return "NULL"
	}

	switch strings.ToLower(dataType) {
	case "integer", "int", "bigint", "smallint":
		return fmt.Sprintf("%d", rand.Intn(1000000))
	case "text", "varchar", "character varying":
		return fmt.Sprintf("'Test Data %d'", rand.Intn(10000))
	case "boolean", "bool":
		if rand.Float32() < 0.5 {
			return "true"
		}
		return "false"
	case "timestamp", "timestamptz":
		return fmt.Sprintf("'%s'", time.Now().Add(time.Duration(rand.Intn(365*24))*time.Hour).Format(time.RFC3339))
	case "date":
		return fmt.Sprintf("'%s'", time.Now().AddDate(0, 0, rand.Intn(365)).Format("2006-01-02"))
	case "numeric", "decimal", "real", "double precision":
		return fmt.Sprintf("%.2f", rand.Float64()*1000)
	default:
		return fmt.Sprintf("'Value %d'", rand.Intn(1000))
	}
}

/* SchemaVisualizerTool generates ER diagrams and schema visualizations */
type SchemaVisualizerTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

/* NewSchemaVisualizerTool creates a new schema visualizer tool */
func NewSchemaVisualizerTool(db *database.Database, logger *logging.Logger) Tool {
	inputSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"schema_name": map[string]interface{}{
				"type":        "string",
				"description": "Schema name (optional, uses public if not provided)",
			},
			"format": map[string]interface{}{
				"type":        "string",
				"description": "Output format: mermaid, dot, json",
				"enum":        []interface{}{"mermaid", "dot", "json"},
				"default":     "mermaid",
			},
		},
	}

	return &SchemaVisualizerTool{
		BaseTool: NewBaseTool(
			"schema_visualizer",
			"Generate ER diagrams and schema visualizations",
			inputSchema,
		),
		db:     db,
		logger: logger,
	}
}

/* Execute executes the schema visualizer tool */
func (t *SchemaVisualizerTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	schemaName, _ := params["schema_name"].(string)
	format, _ := params["format"].(string)

	if schemaName == "" {
		schemaName = "public"
	}
	if format == "" {
		format = "mermaid"
	}

	/* Get schema information */
	schema := t.getSchemaInfo(ctx, schemaName)

	/* Generate visualization */
	visualization := t.generateVisualization(schema, format)

	return Success(map[string]interface{}{
		"schema_name":   schemaName,
		"format":        format,
		"visualization": visualization,
		"schema":       schema,
	}, nil), nil
}

/* getSchemaInfo gets schema information */
func (t *SchemaVisualizerTool) getSchemaInfo(ctx context.Context, schemaName string) map[string]interface{} {
	query := `
		SELECT 
			t.table_name,
			c.column_name,
			c.data_type,
			c.is_nullable,
			tc.constraint_type
		FROM information_schema.tables t
		LEFT JOIN information_schema.columns c ON t.table_name = c.table_name AND t.table_schema = c.table_schema
		LEFT JOIN information_schema.table_constraints tc ON t.table_name = tc.table_name AND t.table_schema = tc.table_schema
		WHERE t.table_schema = $1 AND t.table_type = 'BASE TABLE'
		ORDER BY t.table_name, c.ordinal_position
	`

	rows, err := t.db.Query(ctx, query, []interface{}{schemaName})
	if err != nil {
		return make(map[string]interface{})
	}
	defer rows.Close()

	tables := make(map[string]map[string]interface{})
	for rows.Next() {
		var tableName, columnName, dataType, isNullable, constraintType *string

		if err := rows.Scan(&tableName, &columnName, &dataType, &isNullable, &constraintType); err != nil {
			continue
		}

		if tableName == nil {
			continue
		}

		if _, exists := tables[*tableName]; !exists {
			tables[*tableName] = map[string]interface{}{
				"columns": []map[string]interface{}{},
			}
		}

		if columnName != nil {
			column := map[string]interface{}{
				"name":      *columnName,
				"data_type": getString(dataType, ""),
				"nullable":  getString(isNullable, "") == "YES",
			}
			if constraintType != nil {
				column["constraint"] = *constraintType
			}

			columns := tables[*tableName]["columns"].([]map[string]interface{})
			columns = append(columns, column)
			tables[*tableName]["columns"] = columns
		}
	}

	return map[string]interface{}{
		"tables": tables,
	}
}

/* generateVisualization generates visualization in specified format */
func (t *SchemaVisualizerTool) generateVisualization(schema map[string]interface{}, format string) string {
	tables, _ := schema["tables"].(map[string]map[string]interface{})

	switch format {
	case "mermaid":
		return t.generateMermaid(tables)
	case "dot":
		return t.generateDot(tables)
	case "json":
		jsonBytes, _ := json.MarshalIndent(schema, "", "  ")
		return string(jsonBytes)
	default:
		return fmt.Sprintf("Schema visualization in %s format", format)
	}
}

/* generateMermaid generates Mermaid ER diagram */
func (t *SchemaVisualizerTool) generateMermaid(tables map[string]map[string]interface{}) string {
	var sb strings.Builder
	sb.WriteString("erDiagram\n")

	for tableName, tableInfo := range tables {
		sb.WriteString(fmt.Sprintf("    %s {\n", tableName))
		columns, _ := tableInfo["columns"].([]map[string]interface{})
		for _, col := range columns {
			name, _ := col["name"].(string)
			dataType, _ := col["data_type"].(string)
			nullable, _ := col["nullable"].(bool)
			nullableStr := ""
			if !nullable {
				nullableStr = " PK"
			}
			sb.WriteString(fmt.Sprintf("        %s %s%s\n", dataType, name, nullableStr))
		}
		sb.WriteString("    }\n")
	}

	return sb.String()
}

/* generateDot generates Graphviz DOT format */
func (t *SchemaVisualizerTool) generateDot(tables map[string]map[string]interface{}) string {
	var sb strings.Builder
	sb.WriteString("digraph schema {\n")
	sb.WriteString("    rankdir=LR;\n")

	for tableName := range tables {
		sb.WriteString(fmt.Sprintf("    %s [shape=box];\n", tableName))
	}

	sb.WriteString("}\n")
	return sb.String()
}

/* QueryExplainerTool explains SQL queries in plain English */
type QueryExplainerTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

/* NewQueryExplainerTool creates a new query explainer tool */
func NewQueryExplainerTool(db *database.Database, logger *logging.Logger) Tool {
	inputSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"query": map[string]interface{}{
				"type":        "string",
				"description": "SQL query to explain",
			},
			"detailed": map[string]interface{}{
				"type":        "boolean",
				"description": "Provide detailed explanation",
				"default":     false,
			},
		},
		"required": []interface{}{"query"},
	}

	return &QueryExplainerTool{
		BaseTool: NewBaseTool(
			"query_explainer",
			"Explain SQL queries in plain English",
			inputSchema,
		),
		db:     db,
		logger: logger,
	}
}

/* Execute executes the query explainer tool */
func (t *QueryExplainerTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	query, _ := params["query"].(string)
	detailed, _ := params["detailed"].(bool)

	if query == "" {
		return Error("query is required", "INVALID_PARAMS", nil), nil
	}

	explanation := t.explainQuery(query, detailed)

	return Success(map[string]interface{}{
		"query":       query,
		"explanation": explanation,
		"components":  t.extractQueryComponents(query),
	}, nil), nil
}

/* explainQuery explains query in plain English */
func (t *QueryExplainerTool) explainQuery(query string, detailed bool) string {
	queryLower := strings.ToLower(strings.TrimSpace(query))
	explanation := ""

	/* Identify query type */
	if strings.HasPrefix(queryLower, "select") {
		explanation = "This SELECT query retrieves data from the database. "
		if detailed {
			explanation += t.explainSelectDetails(query)
		}
	} else if strings.HasPrefix(queryLower, "insert") {
		explanation = "This INSERT query adds new rows to a table. "
	} else if strings.HasPrefix(queryLower, "update") {
		explanation = "This UPDATE query modifies existing rows in a table. "
	} else if strings.HasPrefix(queryLower, "delete") {
		explanation = "This DELETE query removes rows from a table. "
	} else {
		explanation = "This query performs a database operation. "
	}

	return explanation
}

/* explainSelectDetails explains SELECT query details */
func (t *QueryExplainerTool) explainSelectDetails(query string) string {
	details := []string{}

	if strings.Contains(strings.ToLower(query), "join") {
		details = append(details, "It joins multiple tables together")
	}
	if strings.Contains(strings.ToLower(query), "where") {
		details = append(details, "It filters rows based on conditions")
	}
	if strings.Contains(strings.ToLower(query), "group by") {
		details = append(details, "It groups rows together")
	}
	if strings.Contains(strings.ToLower(query), "order by") {
		details = append(details, "It sorts the results")
	}

	if len(details) > 0 {
		return strings.Join(details, ". ") + ". "
	}

	return ""
}

/* extractQueryComponents extracts query components */
func (t *QueryExplainerTool) extractQueryComponents(query string) map[string]interface{} {
	queryLower := strings.ToLower(query)
	components := make(map[string]interface{})

	if strings.Contains(queryLower, "select") {
		components["has_select"] = true
	}
	if strings.Contains(queryLower, "from") {
		components["has_from"] = true
	}
	if strings.Contains(queryLower, "where") {
		components["has_where"] = true
	}
	if strings.Contains(queryLower, "join") {
		components["has_join"] = true
	}
	if strings.Contains(queryLower, "group by") {
		components["has_group_by"] = true
	}
	if strings.Contains(queryLower, "order by") {
		components["has_order_by"] = true
	}
	if strings.Contains(queryLower, "limit") {
		components["has_limit"] = true
	}

	return components
}

/* SchemaDocumentationTool auto-generates schema documentation */
type SchemaDocumentationTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

/* NewSchemaDocumentationTool creates a new schema documentation tool */
func NewSchemaDocumentationTool(db *database.Database, logger *logging.Logger) Tool {
	inputSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"schema_name": map[string]interface{}{
				"type":        "string",
				"description": "Schema name",
			},
			"format": map[string]interface{}{
				"type":        "string",
				"description": "Documentation format: markdown, html, json",
				"enum":        []interface{}{"markdown", "html", "json"},
				"default":     "markdown",
			},
		},
	}

	return &SchemaDocumentationTool{
		BaseTool: NewBaseTool(
			"schema_documentation",
			"Auto-generate schema documentation",
			inputSchema,
		),
		db:     db,
		logger: logger,
	}
}

/* Execute executes the schema documentation tool */
func (t *SchemaDocumentationTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	schemaName, _ := params["schema_name"].(string)
	format, _ := params["format"].(string)

	if schemaName == "" {
		schemaName = "public"
	}
	if format == "" {
		format = "markdown"
	}

	/* Get schema information */
	schema := t.getSchemaDocumentation(ctx, schemaName)

	/* Generate documentation */
	documentation := t.generateDocumentation(schema, format)

	return Success(map[string]interface{}{
		"schema_name":  schemaName,
		"format":       format,
		"documentation": documentation,
	}, nil), nil
}

/* getSchemaDocumentation gets schema documentation data */
func (t *SchemaDocumentationTool) getSchemaDocumentation(ctx context.Context, schemaName string) map[string]interface{} {
	/* Similar to schema visualizer but with more details */
	return t.getSchemaInfo(ctx, schemaName)
}

/* generateDocumentation generates documentation in specified format */
func (t *SchemaDocumentationTool) generateDocumentation(schema map[string]interface{}, format string) string {
	tables, _ := schema["tables"].(map[string]map[string]interface{})

	switch format {
	case "markdown":
		return t.generateMarkdown(tables)
	case "html":
		return t.generateHTML(tables)
	case "json":
		jsonBytes, _ := json.MarshalIndent(schema, "", "  ")
		return string(jsonBytes)
	default:
		return "Schema documentation"
	}
}

/* generateMarkdown generates Markdown documentation */
func (t *SchemaDocumentationTool) generateMarkdown(tables map[string]map[string]interface{}) string {
	var sb strings.Builder
	sb.WriteString("# Database Schema Documentation\n\n")

	for tableName, tableInfo := range tables {
		sb.WriteString(fmt.Sprintf("## Table: %s\n\n", tableName))
		columns, _ := tableInfo["columns"].([]map[string]interface{})
		sb.WriteString("| Column | Type | Nullable |\n")
		sb.WriteString("|--------|------|----------|\n")
		for _, col := range columns {
			name, _ := col["name"].(string)
			dataType, _ := col["data_type"].(string)
			nullable, _ := col["nullable"].(bool)
			nullableStr := "Yes"
			if !nullable {
				nullableStr = "No"
			}
			sb.WriteString(fmt.Sprintf("| %s | %s | %s |\n", name, dataType, nullableStr))
		}
		sb.WriteString("\n")
	}

	return sb.String()
}

/* generateHTML generates HTML documentation */
func (t *SchemaDocumentationTool) generateHTML(tables map[string]map[string]interface{}) string {
	var sb strings.Builder
	sb.WriteString("<html><head><title>Schema Documentation</title></head><body>")
	sb.WriteString("<h1>Database Schema Documentation</h1>")

	for tableName, tableInfo := range tables {
		sb.WriteString(fmt.Sprintf("<h2>Table: %s</h2>", tableName))
		sb.WriteString("<table border='1'><tr><th>Column</th><th>Type</th><th>Nullable</th></tr>")
		columns, _ := tableInfo["columns"].([]map[string]interface{})
		for _, col := range columns {
			name, _ := col["name"].(string)
			dataType, _ := col["data_type"].(string)
			nullable, _ := col["nullable"].(bool)
			nullableStr := "Yes"
			if !nullable {
				nullableStr = "No"
			}
			sb.WriteString(fmt.Sprintf("<tr><td>%s</td><td>%s</td><td>%s</td></tr>", name, dataType, nullableStr))
		}
		sb.WriteString("</table>")
	}

	sb.WriteString("</body></html>")
	return sb.String()
}

/* getSchemaInfo gets schema information (reused from visualizer) */
func (t *SchemaDocumentationTool) getSchemaInfo(ctx context.Context, schemaName string) map[string]interface{} {
	/* Reuse logic from SchemaVisualizerTool */
	visualizer := NewSchemaVisualizerTool(t.db, t.logger)
	params := map[string]interface{}{
		"schema_name": schemaName,
		"format":      "json",
	}
	result, _ := visualizer.Execute(ctx, params)
	if result != nil && result.Success {
		if data, ok := result.Data.(map[string]interface{}); ok {
			if schema, ok := data["schema"].(map[string]interface{}); ok {
				return schema
			}
		}
	}
	return make(map[string]interface{})
}

/* MigrationGeneratorTool generates migrations from schema diffs */
type MigrationGeneratorTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

/* NewMigrationGeneratorTool creates a new migration generator tool */
func NewMigrationGeneratorTool(db *database.Database, logger *logging.Logger) Tool {
	inputSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"from_schema": map[string]interface{}{
				"type":        "string",
				"description": "Source schema",
			},
			"to_schema": map[string]interface{}{
				"type":        "string",
				"description": "Target schema",
			},
			"migration_name": map[string]interface{}{
				"type":        "string",
				"description": "Migration name",
			},
		},
		"required": []interface{}{"from_schema", "to_schema"},
	}

	return &MigrationGeneratorTool{
		BaseTool: NewBaseTool(
			"migration_generator",
			"Generate migrations from schema differences",
			inputSchema,
		),
		db:     db,
		logger: logger,
	}
}

/* Execute executes the migration generator tool */
func (t *MigrationGeneratorTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	fromSchema, _ := params["from_schema"].(string)
	toSchema, _ := params["to_schema"].(string)
	migrationName, _ := params["migration_name"].(string)

	if fromSchema == "" || toSchema == "" {
		return Error("from_schema and to_schema are required", "INVALID_PARAMS", nil), nil
	}

	/* Use migration tool to generate migration */
	migrationTool := NewPostgreSQLMigrationTool(t.db, t.logger)
	migrationParams := map[string]interface{}{
		"operation":   "generate",
		"from_schema": fromSchema,
		"to_schema":   toSchema,
	}

	if migrationName != "" {
		migrationParams["migration_name"] = migrationName
	}

	return migrationTool.Execute(ctx, migrationParams)
}

