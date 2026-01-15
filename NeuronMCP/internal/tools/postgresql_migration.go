/*-------------------------------------------------------------------------
 *
 * postgresql_migration.go
 *    PostgreSQL Migration and Schema Evolution tools for NeuronMCP
 *
 * Provides schema evolution tracking, migration generation, and execution.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/postgresql_migration.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

/* PostgreSQLSchemaEvolutionTool tracks and manages schema changes */
type PostgreSQLSchemaEvolutionTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

/* NewPostgreSQLSchemaEvolutionTool creates a new schema evolution tool */
func NewPostgreSQLSchemaEvolutionTool(db *database.Database, logger *logging.Logger) Tool {
	inputSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"operation": map[string]interface{}{
				"type":        "string",
				"description": "Operation: track_change, get_history, compare_schemas",
				"enum":        []interface{}{"track_change", "get_history", "compare_schemas"},
			},
			"schema_name": map[string]interface{}{
				"type":        "string",
				"description": "Schema name",
			},
			"change_type": map[string]interface{}{
				"type":        "string",
				"description": "Type of change: create_table, alter_table, drop_table, add_column, etc.",
			},
			"change_description": map[string]interface{}{
				"type":        "string",
				"description": "Description of the change",
			},
			"sql_statement": map[string]interface{}{
				"type":        "string",
				"description": "SQL statement that caused the change",
			},
			"target_schema": map[string]interface{}{
				"type":        "string",
				"description": "Target schema for comparison",
			},
		},
		"required": []interface{}{"operation"},
	}

	return &PostgreSQLSchemaEvolutionTool{
		BaseTool: NewBaseTool(
			"postgresql_schema_evolution",
			"Track and manage schema changes over time",
			inputSchema,
		),
		db:     db,
		logger: logger,
	}
}

/* Execute executes the schema evolution tool */
func (t *PostgreSQLSchemaEvolutionTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	operation, _ := params["operation"].(string)

	switch operation {
	case "track_change":
		return t.trackChange(ctx, params)
	case "get_history":
		return t.getHistory(ctx, params)
	case "compare_schemas":
		return t.compareSchemas(ctx, params)
	default:
		return Error("Invalid operation", "INVALID_OPERATION", nil), nil
	}
}

/* trackChange tracks a schema change */
func (t *PostgreSQLSchemaEvolutionTool) trackChange(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	schemaName, _ := params["schema_name"].(string)
	changeType, _ := params["change_type"].(string)
	description, _ := params["change_description"].(string)
	sqlStatement, _ := params["sql_statement"].(string)

	if schemaName == "" {
		schemaName = "public"
	}

	query := `
		INSERT INTO neurondb.schema_changes 
		(schema_name, change_type, description, sql_statement, created_at)
		VALUES ($1, $2, $3, $4, NOW())
		RETURNING id, version
	`

	_, err := t.db.Query(ctx, query, []interface{}{schemaName, changeType, description, sqlStatement})
	if err != nil {
		/* Create table if needed */
		createTable := `
			CREATE TABLE IF NOT EXISTS neurondb.schema_changes (
				id SERIAL PRIMARY KEY,
				schema_name VARCHAR(200) NOT NULL,
				change_type VARCHAR(100) NOT NULL,
				description TEXT,
				sql_statement TEXT,
				version VARCHAR(50),
				created_at TIMESTAMP NOT NULL DEFAULT NOW(),
				created_by VARCHAR(200),
				INDEX idx_schema_created (schema_name, created_at)
			)
		`
		_, _ = t.db.Query(ctx, createTable, nil)
		_, err = t.db.Query(ctx, query, []interface{}{schemaName, changeType, description, sqlStatement})
		if err != nil {
			return Error(fmt.Sprintf("Failed to track change: %v", err), "TRACK_ERROR", nil), nil
		}
	}

	return Success(map[string]interface{}{
		"schema_name": schemaName,
		"change_type": changeType,
		"message":     "Schema change tracked",
	}, nil), nil
}

/* getHistory gets schema change history */
func (t *PostgreSQLSchemaEvolutionTool) getHistory(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	schemaName, _ := params["schema_name"].(string)

	if schemaName == "" {
		schemaName = "public"
	}

	query := `
		SELECT 
			id,
			change_type,
			description,
			sql_statement,
			version,
			created_at,
			created_by
		FROM neurondb.schema_changes
		WHERE schema_name = $1
		ORDER BY created_at DESC
		LIMIT 100
	`

	rows, err := t.db.Query(ctx, query, []interface{}{schemaName})
	if err != nil {
		return Success(map[string]interface{}{
			"schema_name": schemaName,
			"history":     []interface{}{},
		}, nil), nil
	}
	defer rows.Close()

	history := []map[string]interface{}{}
	for rows.Next() {
		var id int64
		var changeType, description, sqlStatement, version, createdBy *string
		var createdAt time.Time

		if err := rows.Scan(&id, &changeType, &description, &sqlStatement, &version, &createdAt, &createdBy); err != nil {
			continue
		}

		entry := map[string]interface{}{
			"id":         id,
			"change_type": getString(changeType, ""),
			"created_at": createdAt,
		}

		if description != nil {
			entry["description"] = *description
		}
		if sqlStatement != nil {
			entry["sql_statement"] = *sqlStatement
		}
		if version != nil {
			entry["version"] = *version
		}
		if createdBy != nil {
			entry["created_by"] = *createdBy
		}

		history = append(history, entry)
	}

	return Success(map[string]interface{}{
		"schema_name": schemaName,
		"history":     history,
	}, nil), nil
}

/* compareSchemas compares two schemas */
func (t *PostgreSQLSchemaEvolutionTool) compareSchemas(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	schemaName, _ := params["schema_name"].(string)
	targetSchema, _ := params["target_schema"].(string)

	if schemaName == "" {
		schemaName = "public"
	}
	if targetSchema == "" {
		return Error("target_schema is required", "INVALID_PARAMS", nil), nil
	}

	/* Get current schema */
	currentSchema := t.getSchema(ctx, schemaName)
	targetSchemaData := t.getSchema(ctx, targetSchema)

	/* Compare schemas */
	differences := t.compareSchemaData(currentSchema, targetSchemaData)

	return Success(map[string]interface{}{
		"source_schema": schemaName,
		"target_schema": targetSchema,
		"differences":   differences,
	}, nil), nil
}

/* getSchema gets schema definition */
func (t *PostgreSQLSchemaEvolutionTool) getSchema(ctx context.Context, schemaName string) map[string]interface{} {
	query := `
		SELECT 
			table_name,
			column_name,
			data_type,
			is_nullable,
			column_default
		FROM information_schema.columns
		WHERE table_schema = $1
		ORDER BY table_name, ordinal_position
	`

	rows, err := t.db.Query(ctx, query, []interface{}{schemaName})
	if err != nil {
		return make(map[string]interface{})
	}
	defer rows.Close()

	schema := make(map[string]interface{})
	tables := make(map[string][]map[string]interface{})

	for rows.Next() {
		var tableName, columnName, dataType, isNullable string
		var columnDefault *string

		if err := rows.Scan(&tableName, &columnName, &dataType, &isNullable, &columnDefault); err != nil {
			continue
		}

		column := map[string]interface{}{
			"name":       columnName,
			"data_type":  dataType,
			"nullable":   isNullable == "YES",
		}

		if columnDefault != nil {
			column["default"] = *columnDefault
		}

		tables[tableName] = append(tables[tableName], column)
	}

	schema["tables"] = tables
	return schema
}

/* compareSchemaData compares two schema definitions */
func (t *PostgreSQLSchemaEvolutionTool) compareSchemaData(current, target map[string]interface{}) []map[string]interface{} {
	differences := []map[string]interface{}{}

	currentTables, _ := current["tables"].(map[string][]map[string]interface{})
	targetTables, _ := target["tables"].(map[string][]map[string]interface{})

	/* Find tables in current but not in target */
	for tableName := range currentTables {
		if _, exists := targetTables[tableName]; !exists {
			differences = append(differences, map[string]interface{}{
				"type":        "table_missing",
				"table":       tableName,
				"description": fmt.Sprintf("Table %s exists in source but not in target", tableName),
			})
		}
	}

	/* Find tables in target but not in current */
	for tableName := range targetTables {
		if _, exists := currentTables[tableName]; !exists {
			differences = append(differences, map[string]interface{}{
				"type":        "table_extra",
				"table":       tableName,
				"description": fmt.Sprintf("Table %s exists in target but not in source", tableName),
			})
		}
	}

	return differences
}

func getString(s *string, defaultVal string) string {
	if s == nil {
		return defaultVal
	}
	return *s
}

/* PostgreSQLMigrationTool generates and executes database migrations */
type PostgreSQLMigrationTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

/* NewPostgreSQLMigrationTool creates a new migration tool */
func NewPostgreSQLMigrationTool(db *database.Database, logger *logging.Logger) Tool {
	inputSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"operation": map[string]interface{}{
				"type":        "string",
				"description": "Operation: generate, execute, list, rollback",
				"enum":        []interface{}{"generate", "execute", "list", "rollback"},
			},
			"migration_name": map[string]interface{}{
				"type":        "string",
				"description": "Migration name",
			},
			"from_schema": map[string]interface{}{
				"type":        "string",
				"description": "Source schema for migration generation",
			},
			"to_schema": map[string]interface{}{
				"type":        "string",
				"description": "Target schema for migration generation",
			},
			"sql_statements": map[string]interface{}{
				"type":        "array",
				"description": "SQL statements for migration",
				"items": map[string]interface{}{
					"type": "string",
				},
			},
			"migration_id": map[string]interface{}{
				"type":        "string",
				"description": "Migration ID for execute/rollback",
			},
		},
		"required": []interface{}{"operation"},
	}

	return &PostgreSQLMigrationTool{
		BaseTool: NewBaseTool(
			"postgresql_migration_tool",
			"Generate and execute database migrations",
			inputSchema,
		),
		db:     db,
		logger: logger,
	}
}

/* Execute executes the migration tool */
func (t *PostgreSQLMigrationTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	operation, _ := params["operation"].(string)

	switch operation {
	case "generate":
		return t.generateMigration(ctx, params)
	case "execute":
		return t.executeMigration(ctx, params)
	case "list":
		return t.listMigrations(ctx, params)
	case "rollback":
		return t.rollbackMigration(ctx, params)
	default:
		return Error("Invalid operation", "INVALID_OPERATION", nil), nil
	}
}

/* generateMigration generates a migration from schema differences */
func (t *PostgreSQLMigrationTool) generateMigration(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	migrationName, _ := params["migration_name"].(string)
	fromSchema, _ := params["from_schema"].(string)
	toSchema, _ := params["to_schema"].(string)

	if migrationName == "" {
		migrationName = fmt.Sprintf("migration_%d", time.Now().Unix())
	}

	if fromSchema == "" {
		fromSchema = "public"
	}
	if toSchema == "" {
		return Error("to_schema is required", "INVALID_PARAMS", nil), nil
	}

	/* Get schema differences */
	schemaTool := NewPostgreSQLSchemaEvolutionTool(t.db, t.logger)
	compareParams := map[string]interface{}{
		"operation":     "compare_schemas",
		"schema_name":   fromSchema,
		"target_schema": toSchema,
	}

	compareResult, err := schemaTool.Execute(ctx, compareParams)
	if err != nil {
		return Error("Failed to compare schemas", "COMPARE_ERROR", nil), nil
	}

	compareData, _ := compareResult.Data.(map[string]interface{})
	differences, _ := compareData["differences"].([]map[string]interface{})

	/* Generate migration SQL */
	migrationSQL := t.generateMigrationSQL(differences, fromSchema, toSchema)

	/* Store migration */
	migrationID := fmt.Sprintf("mig_%d", time.Now().UnixNano())
	query := `
		INSERT INTO neurondb.migrations 
		(migration_id, migration_name, from_schema, to_schema, sql_statements, status, created_at)
		VALUES ($1, $2, $3, $4, $5, 'pending', NOW())
	`

	sqlJSON, _ := json.Marshal(migrationSQL)
	_, err = t.db.Query(ctx, query, []interface{}{migrationID, migrationName, fromSchema, toSchema, string(sqlJSON)})
	if err != nil {
		/* Create table if needed */
		createTable := `
			CREATE TABLE IF NOT EXISTS neurondb.migrations (
				migration_id VARCHAR(200) PRIMARY KEY,
				migration_name VARCHAR(200) NOT NULL,
				from_schema VARCHAR(200),
				to_schema VARCHAR(200),
				sql_statements JSONB,
				rollback_sql JSONB,
				status VARCHAR(50) NOT NULL,
				created_at TIMESTAMP NOT NULL,
				executed_at TIMESTAMP,
				executed_by VARCHAR(200)
			)
		`
		_, _ = t.db.Query(ctx, createTable, nil)
		_, err = t.db.Query(ctx, query, []interface{}{migrationID, migrationName, fromSchema, toSchema, string(sqlJSON)})
		if err != nil {
			return Error(fmt.Sprintf("Failed to store migration: %v", err), "STORE_ERROR", nil), nil
		}
	}

	return Success(map[string]interface{}{
		"migration_id":   migrationID,
		"migration_name":  migrationName,
		"sql_statements": migrationSQL,
		"status":         "pending",
	}, nil), nil
}

/* generateMigrationSQL generates SQL statements from differences */
func (t *PostgreSQLMigrationTool) generateMigrationSQL(differences []map[string]interface{}, fromSchema, toSchema string) []string {
	sqlStatements := []string{}

	for _, diff := range differences {
		diffType, _ := diff["type"].(string)
		table, _ := diff["table"].(string)

		switch diffType {
		case "table_missing":
			/* Generate CREATE TABLE statement */
			sqlStatements = append(sqlStatements, fmt.Sprintf("CREATE TABLE %s (...);", table))
		case "table_extra":
			/* Generate DROP TABLE statement */
			sqlStatements = append(sqlStatements, fmt.Sprintf("DROP TABLE IF EXISTS %s;", table))
		}
	}

	return sqlStatements
}

/* executeMigration executes a migration */
func (t *PostgreSQLMigrationTool) executeMigration(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	migrationID, _ := params["migration_id"].(string)

	if migrationID == "" {
		return Error("migration_id is required", "INVALID_PARAMS", nil), nil
	}

	/* Get migration */
	query := `
		SELECT sql_statements, status
		FROM neurondb.migrations
		WHERE migration_id = $1
	`

	rows, err := t.db.Query(ctx, query, []interface{}{migrationID})
	if err != nil {
		return Error("Migration not found", "NOT_FOUND", nil), nil
	}
	defer rows.Close()

	if rows.Next() {
		var sqlJSON *string
		var status string

		if err := rows.Scan(&sqlJSON, &status); err != nil {
			return Error("Failed to read migration", "READ_ERROR", nil), nil
		}

		if status == "executed" {
			return Error("Migration already executed", "ALREADY_EXECUTED", nil), nil
		}

		if sqlJSON == nil {
			return Error("Migration has no SQL statements", "NO_SQL", nil), nil
		}

		var sqlStatements []string
		if err := json.Unmarshal([]byte(*sqlJSON), &sqlStatements); err != nil {
			return Error("Failed to parse SQL statements", "PARSE_ERROR", nil), nil
		}

		/* Execute SQL statements */
		for _, stmt := range sqlStatements {
			_, err := t.db.Query(ctx, stmt, nil)
			if err != nil {
				return Error(fmt.Sprintf("Failed to execute migration: %v", err), "EXECUTION_ERROR", nil), nil
			}
		}

		/* Update migration status */
		updateQuery := `
			UPDATE neurondb.migrations
			SET status = 'executed', executed_at = NOW()
			WHERE migration_id = $1
		`
		_, _ = t.db.Query(ctx, updateQuery, []interface{}{migrationID})

		return Success(map[string]interface{}{
			"migration_id": migrationID,
			"status":       "executed",
			"message":      "Migration executed successfully",
		}, nil), nil
	}

	return Error("Migration not found", "NOT_FOUND", nil), nil
}

/* listMigrations lists all migrations */
func (t *PostgreSQLMigrationTool) listMigrations(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	query := `
		SELECT 
			migration_id,
			migration_name,
			from_schema,
			to_schema,
			status,
			created_at,
			executed_at
		FROM neurondb.migrations
		ORDER BY created_at DESC
		LIMIT 100
	`

	rows, err := t.db.Query(ctx, query, nil)
	if err != nil {
		return Success(map[string]interface{}{
			"migrations": []interface{}{},
		}, nil), nil
	}
	defer rows.Close()

	migrations := []map[string]interface{}{}
	for rows.Next() {
		var migrationID, migrationName, fromSchema, toSchema, status string
		var createdAt time.Time
		var executedAt *time.Time

		if err := rows.Scan(&migrationID, &migrationName, &fromSchema, &toSchema, &status, &createdAt, &executedAt); err != nil {
			continue
		}

		migration := map[string]interface{}{
			"migration_id":   migrationID,
			"migration_name": migrationName,
			"from_schema":    fromSchema,
			"to_schema":      toSchema,
			"status":         status,
			"created_at":     createdAt,
		}

		if executedAt != nil {
			migration["executed_at"] = *executedAt
		}

		migrations = append(migrations, migration)
	}

	return Success(map[string]interface{}{
		"migrations": migrations,
	}, nil), nil
}

/* rollbackMigration rolls back a migration */
func (t *PostgreSQLMigrationTool) rollbackMigration(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	migrationID, _ := params["migration_id"].(string)

	if migrationID == "" {
		return Error("migration_id is required", "INVALID_PARAMS", nil), nil
	}

	/* Get rollback SQL */
	query := `
		SELECT rollback_sql, status
		FROM neurondb.migrations
		WHERE migration_id = $1
	`

	rows, err := t.db.Query(ctx, query, []interface{}{migrationID})
	if err != nil {
		return Error("Migration not found", "NOT_FOUND", nil), nil
	}
	defer rows.Close()

	if rows.Next() {
		var rollbackJSON *string
		var status string

		if err := rows.Scan(&rollbackJSON, &status); err != nil {
			return Error("Failed to read migration", "READ_ERROR", nil), nil
		}

		if status != "executed" {
			return Error("Migration not executed, cannot rollback", "NOT_EXECUTED", nil), nil
		}

		if rollbackJSON == nil {
			return Error("Migration has no rollback SQL", "NO_ROLLBACK", nil), nil
		}

		var rollbackSQL []string
		if err := json.Unmarshal([]byte(*rollbackJSON), &rollbackSQL); err != nil {
			return Error("Failed to parse rollback SQL", "PARSE_ERROR", nil), nil
		}

		/* Execute rollback SQL */
		for _, stmt := range rollbackSQL {
			_, err := t.db.Query(ctx, stmt, nil)
			if err != nil {
				return Error(fmt.Sprintf("Failed to rollback migration: %v", err), "ROLLBACK_ERROR", nil), nil
			}
		}

		/* Update migration status */
		updateQuery := `
			UPDATE neurondb.migrations
			SET status = 'rolled_back'
			WHERE migration_id = $1
		`
		_, _ = t.db.Query(ctx, updateQuery, []interface{}{migrationID})

		return Success(map[string]interface{}{
			"migration_id": migrationID,
			"status":       "rolled_back",
			"message":      "Migration rolled back successfully",
		}, nil), nil
	}

	return Error("Migration not found", "NOT_FOUND", nil), nil
}

