/*-------------------------------------------------------------------------
 *
 * tables.go
 *    Tables resource for NeuronMCP
 *
 * Provides table information including schemas, tables, columns, constraints,
 * and row counts for easy browsing without SQL.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/resources/tables.go
 *
 *-------------------------------------------------------------------------
 */

package resources

import (
	"context"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/database"
)

/* TablesResource provides table information */
type TablesResource struct {
	*BaseResource
}

/* NewTablesResource creates a new tables resource */
func NewTablesResource(db *database.Database) *TablesResource {
	return &TablesResource{BaseResource: NewBaseResource(db)}
}

/* URI returns the resource URI */
func (r *TablesResource) URI() string {
	return "neurondb://tables"
}

/* Name returns the resource name */
func (r *TablesResource) Name() string {
	return "Database Tables"
}

/* Description returns the resource description */
func (r *TablesResource) Description() string {
	return "Database tables with columns, constraints, and row counts"
}

/* MimeType returns the MIME type */
func (r *TablesResource) MimeType() string {
	return "application/json"
}

/* GetContent returns the tables content */
func (r *TablesResource) GetContent(ctx context.Context) (interface{}, error) {
	/* Get all tables with their schemas */
	tablesQuery := `
		SELECT 
			t.table_schema,
			t.table_name,
			t.table_type,
			COALESCE(obj_description(c.oid, 'pg_class'), '') AS table_comment
		FROM information_schema.tables t
		LEFT JOIN pg_class c ON c.relname = t.table_name
		LEFT JOIN pg_namespace n ON n.oid = c.relnamespace AND n.nspname = t.table_schema
		WHERE t.table_schema NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
		ORDER BY t.table_schema, t.table_name
	`
	tables, err := r.executeQuery(ctx, tablesQuery, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to query tables: %w", err)
	}

	/* Get columns for each table */
	columnsQuery := `
		SELECT 
			table_schema,
			table_name,
			column_name,
			ordinal_position,
			column_default,
			is_nullable,
			data_type,
			udt_name,
			character_maximum_length,
			numeric_precision,
			numeric_scale,
			COALESCE(col_description(c.oid, a.attnum), '') AS column_comment
		FROM information_schema.columns c
		LEFT JOIN pg_class cl ON cl.relname = c.table_name
		LEFT JOIN pg_namespace n ON n.oid = cl.relnamespace AND n.nspname = c.table_schema
		LEFT JOIN pg_attribute a ON a.attrelid = cl.oid AND a.attname = c.column_name
		WHERE c.table_schema NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
		ORDER BY c.table_schema, c.table_name, c.ordinal_position
	`
	columns, err := r.executeQuery(ctx, columnsQuery, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to query columns: %w", err)
	}

	/* Get constraints */
	constraintsQuery := `
		SELECT 
			tc.table_schema,
			tc.table_name,
			tc.constraint_name,
			tc.constraint_type,
			kcu.column_name,
			cc.check_clause
		FROM information_schema.table_constraints tc
		LEFT JOIN information_schema.key_column_usage kcu 
			ON tc.constraint_name = kcu.constraint_name 
			AND tc.table_schema = kcu.table_schema
		LEFT JOIN information_schema.check_constraints cc
			ON tc.constraint_name = cc.constraint_name
			AND tc.table_schema = cc.constraint_schema
		WHERE tc.table_schema NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
		ORDER BY tc.table_schema, tc.table_name, tc.constraint_name
	`
	constraints, err := r.executeQuery(ctx, constraintsQuery, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to query constraints: %w", err)
	}

	/* Build result structure */
	result := make(map[string]interface{})
	tablesMap := make(map[string]map[string]interface{})

	/* Process tables */
	for _, table := range tables {
		schema, _ := table["table_schema"].(string)
		tableName, _ := table["table_name"].(string)
		key := fmt.Sprintf("%s.%s", schema, tableName)

		tablesMap[key] = map[string]interface{}{
			"schema":       schema,
			"name":         tableName,
			"type":         table["table_type"],
			"comment":      table["table_comment"],
			"columns":      []interface{}{},
			"constraints":  []interface{}{},
			"row_count":    nil, /* Will be filled later */
		}
	}

	/* Process columns */
	for _, col := range columns {
		schema, _ := col["table_schema"].(string)
		tableName, _ := col["table_name"].(string)
		key := fmt.Sprintf("%s.%s", schema, tableName)

		if table, exists := tablesMap[key]; exists {
			columnsList := table["columns"].([]interface{})
			columnsList = append(columnsList, map[string]interface{}{
				"name":         col["column_name"],
				"position":     col["ordinal_position"],
				"default":      col["column_default"],
				"nullable":     col["is_nullable"] == "YES",
				"data_type":    col["data_type"],
				"udt_name":     col["udt_name"],
				"max_length":   col["character_maximum_length"],
				"precision":    col["numeric_precision"],
				"scale":        col["numeric_scale"],
				"comment":      col["column_comment"],
			})
			table["columns"] = columnsList
		}
	}

	/* Process constraints */
	for _, constraint := range constraints {
		schema, _ := constraint["table_schema"].(string)
		tableName, _ := constraint["table_name"].(string)
		key := fmt.Sprintf("%s.%s", schema, tableName)

		if table, exists := tablesMap[key]; exists {
			constraintsList := table["constraints"].([]interface{})
			constraintData := map[string]interface{}{
				"name": constraint["constraint_name"],
				"type": constraint["constraint_type"],
			}
			if colName := constraint["column_name"]; colName != nil {
				constraintData["column"] = colName
			}
			if checkClause := constraint["check_clause"]; checkClause != nil {
				constraintData["check"] = checkClause
			}
			constraintsList = append(constraintsList, constraintData)
			table["constraints"] = constraintsList
		}
	}

	/* Get row counts for each table (this might be slow for large databases) */
	for _, table := range tablesMap {
		schema := table["schema"].(string)
		tableName := table["name"].(string)
		
		/* Use pg_class for faster row count estimation */
		rowCountQuery := fmt.Sprintf(`
			SELECT 
				COALESCE(n_live_tup, 0) AS row_count
			FROM pg_stat_user_tables
			WHERE schemaname = $1 AND relname = $2
		`)
		rowCount, err := r.executeQueryOne(ctx, rowCountQuery, []interface{}{schema, tableName})
		if err == nil {
			table["row_count"] = rowCount["row_count"]
		}
	}

	/* Convert map to list */
	tablesList := make([]interface{}, 0, len(tablesMap))
	for _, table := range tablesMap {
		tablesList = append(tablesList, table)
	}

	result["tables"] = tablesList
	result["count"] = len(tablesList)

	return result, nil
}

/* TablesBySchemaResource provides tables filtered by schema */
type TablesBySchemaResource struct {
	*BaseResource
	schema string
}

/* NewTablesBySchemaResource creates a new tables by schema resource */
func NewTablesBySchemaResource(db *database.Database, schema string) *TablesBySchemaResource {
	return &TablesBySchemaResource{
		BaseResource: NewBaseResource(db),
		schema:       schema,
	}
}

/* URI returns the resource URI */
func (r *TablesBySchemaResource) URI() string {
	return fmt.Sprintf("neurondb://tables/%s", r.schema)
}

/* Name returns the resource name */
func (r *TablesBySchemaResource) Name() string {
	return fmt.Sprintf("Tables in schema: %s", r.schema)
}

/* Description returns the resource description */
func (r *TablesBySchemaResource) Description() string {
	return fmt.Sprintf("Tables in schema %s", r.schema)
}

/* MimeType returns the MIME type */
func (r *TablesBySchemaResource) MimeType() string {
	return "application/json"
}

/* GetContent returns the tables content filtered by schema */
func (r *TablesBySchemaResource) GetContent(ctx context.Context) (interface{}, error) {
	/* Validate schema name to prevent SQL injection */
	if !isValidIdentifier(r.schema) {
		return nil, fmt.Errorf("invalid schema name: %s", r.schema)
	}

	query := fmt.Sprintf(`
		SELECT 
			table_schema,
			table_name,
			table_type
		FROM information_schema.tables
		WHERE table_schema = $1
		ORDER BY table_name
	`)
	tables, err := r.executeQuery(ctx, query, []interface{}{r.schema})
	if err != nil {
		return nil, fmt.Errorf("failed to query tables: %w", err)
	}

	return map[string]interface{}{
		"schema": r.schema,
		"tables": tables,
		"count":  len(tables),
	}, nil
}

/* isValidIdentifier checks if a string is a valid SQL identifier */
func isValidIdentifier(s string) bool {
	if len(s) == 0 || len(s) > 63 {
		return false
	}
	/* Simple check - should start with letter or underscore, contain only alphanumeric and underscore */
	for i, r := range s {
		if i == 0 {
			if !((r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || r == '_') {
				return false
			}
		} else {
			if !((r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_') {
				return false
			}
		}
	}
	return true
}

