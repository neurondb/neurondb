/*-------------------------------------------------------------------------
 *
 * postgresql_backup.go
 *    Backup and recovery tools for NeuronMCP
 *
 * Implements backup and recovery operations using pg_dump and pg_restore:
 * - Database backup/restore
 * - Table backup/restore
 * - Backup listing and verification
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/postgresql_backup.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

/* PostgreSQLBackupDatabaseTool backs up databases using pg_dump */
type PostgreSQLBackupDatabaseTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLBackupDatabaseTool creates a new PostgreSQL backup database tool */
func NewPostgreSQLBackupDatabaseTool(db *database.Database, logger *logging.Logger) *PostgreSQLBackupDatabaseTool {
	return &PostgreSQLBackupDatabaseTool{
		BaseTool: NewBaseTool(
			"postgresql_backup_database",
			"Backup a database using pg_dump with custom, directory, or tar formats",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"database_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the database to backup",
					},
					"output_file": map[string]interface{}{
						"type":        "string",
						"description": "Output file path",
					},
					"format": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"custom", "directory", "tar", "plain"},
						"default":     "custom",
						"description": "Backup format",
					},
					"schema": map[string]interface{}{
						"type":        "array",
						"description": "Include only these schemas",
					},
					"exclude_schema": map[string]interface{}{
						"type":        "array",
						"description": "Exclude these schemas",
					},
					"table": map[string]interface{}{
						"type":        "array",
						"description": "Include only these tables",
					},
					"exclude_table": map[string]interface{}{
						"type":        "array",
						"description": "Exclude these tables",
					},
					"data_only": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Backup data only (no schema)",
					},
					"schema_only": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Backup schema only (no data)",
					},
					"compress": map[string]interface{}{
						"type":        "integer",
						"description": "Compression level (0-9)",
					},
					"jobs": map[string]interface{}{
						"type":        "integer",
						"description": "Number of parallel jobs",
					},
				},
				"required": []interface{}{"database_name", "output_file"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute backs up the database */
func (t *PostgreSQLBackupDatabaseTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	databaseName, ok := params["database_name"].(string)
	if !ok || databaseName == "" {
		return Error("database_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	outputFile, ok := params["output_file"].(string)
	if !ok || outputFile == "" {
		return Error("output_file parameter is required", "INVALID_PARAMETER", nil), nil
	}

	/* Get database connection info from executor */
	/* Note: This is a simplified implementation. In production, you'd get connection details from the database instance */
	
	format, _ := params["format"].(string)
	if format == "" {
		format = "custom"
	}

	/* Build pg_dump command */
	args := []string{}

	/* Format */
	switch format {
	case "custom":
		args = append(args, "-Fc")
	case "directory":
		args = append(args, "-Fd")
	case "tar":
		args = append(args, "-Ft")
	case "plain":
		args = append(args, "-Fp")
	}

	/* Database name */
	args = append(args, "-d", databaseName)

	/* Output file */
	args = append(args, "-f", outputFile)

	/* Schema filters */
	if schemas, ok := params["schema"].([]interface{}); ok && len(schemas) > 0 {
		for _, schema := range schemas {
			if schemaStr, ok := schema.(string); ok {
				args = append(args, "-n", schemaStr)
			}
		}
	}

	if excludeSchemas, ok := params["exclude_schema"].([]interface{}); ok && len(excludeSchemas) > 0 {
		for _, schema := range excludeSchemas {
			if schemaStr, ok := schema.(string); ok {
				args = append(args, "-N", schemaStr)
			}
		}
	}

	/* Table filters */
	if tables, ok := params["table"].([]interface{}); ok && len(tables) > 0 {
		for _, table := range tables {
			if tableStr, ok := table.(string); ok {
				args = append(args, "-t", tableStr)
			}
		}
	}

	if excludeTables, ok := params["exclude_table"].([]interface{}); ok && len(excludeTables) > 0 {
		for _, table := range excludeTables {
			if tableStr, ok := table.(string); ok {
				args = append(args, "-T", tableStr)
			}
		}
	}

	/* Data/schema only */
	if dataOnly, ok := params["data_only"].(bool); ok && dataOnly {
		args = append(args, "-a")
	}

	if schemaOnly, ok := params["schema_only"].(bool); ok && schemaOnly {
		args = append(args, "-s")
	}

	/* Compression */
	if compress, ok := params["compress"].(float64); ok {
		args = append(args, "-Z", fmt.Sprintf("%d", int(compress)))
	} else if compress, ok := params["compress"].(int); ok {
		args = append(args, "-Z", fmt.Sprintf("%d", compress))
	}

	/* Parallel jobs */
	if jobs, ok := params["jobs"].(float64); ok {
		args = append(args, "-j", fmt.Sprintf("%d", int(jobs)))
	} else if jobs, ok := params["jobs"].(int); ok {
		args = append(args, "-j", fmt.Sprintf("%d", jobs))
	}

	/* Execute pg_dump */
	cmd := exec.CommandContext(ctx, "pg_dump", args...)
	
	/* Set environment variables from database connection if available */
	/* In production, you'd get these from the database connection */
	
	output, err := cmd.CombinedOutput()
	if err != nil {
		return Error(
			fmt.Sprintf("pg_dump failed: %v", err),
			"BACKUP_ERROR",
			map[string]interface{}{
				"error":  err.Error(),
				"output": string(output),
			},
		), nil
	}

	/* Get file size */
	fileInfo, err := os.Stat(outputFile)
	fileSize := int64(0)
	if err == nil {
		fileSize = fileInfo.Size()
	}

	t.logger.Info("Database backed up", map[string]interface{}{
		"database_name": databaseName,
		"output_file":   outputFile,
		"format":        format,
		"file_size":     fileSize,
	})

	return Success(map[string]interface{}{
		"database_name": databaseName,
		"output_file":   outputFile,
		"format":        format,
		"file_size":     fileSize,
		"command":       "pg_dump " + strings.Join(args, " "),
	}, map[string]interface{}{
		"tool": "postgresql_backup_database",
	}), nil
}

/* PostgreSQLRestoreDatabaseTool restores databases using pg_restore */
type PostgreSQLRestoreDatabaseTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLRestoreDatabaseTool creates a new PostgreSQL restore database tool */
func NewPostgreSQLRestoreDatabaseTool(db *database.Database, logger *logging.Logger) *PostgreSQLRestoreDatabaseTool {
	return &PostgreSQLRestoreDatabaseTool{
		BaseTool: NewBaseTool(
			"postgresql_restore_database",
			"Restore a database using pg_restore with support for all dump formats",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"input_file": map[string]interface{}{
						"type":        "string",
						"description": "Input backup file path",
					},
					"database_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the database to restore to",
					},
					"create_database": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Create database if it doesn't exist",
					},
					"clean": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Clean (drop) database objects before recreating",
					},
					"if_exists": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Use IF EXISTS clause",
					},
					"data_only": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Restore data only (no schema)",
					},
					"schema_only": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Restore schema only (no data)",
					},
					"jobs": map[string]interface{}{
						"type":        "integer",
						"description": "Number of parallel jobs",
					},
					"verbose": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Verbose output",
					},
				},
				"required": []interface{}{"input_file", "database_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute restores the database */
func (t *PostgreSQLRestoreDatabaseTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	inputFile, ok := params["input_file"].(string)
	if !ok || inputFile == "" {
		return Error("input_file parameter is required", "INVALID_PARAMETER", nil), nil
	}

	databaseName, ok := params["database_name"].(string)
	if !ok || databaseName == "" {
		return Error("database_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	/* Check if file exists */
	if _, err := os.Stat(inputFile); os.IsNotExist(err) {
		return Error(
			fmt.Sprintf("Backup file does not exist: %s", inputFile),
			"FILE_NOT_FOUND",
			nil,
		), nil
	}

	/* Create database if needed */
	if createDB, ok := params["create_database"].(bool); ok && createDB {
		createQuery := fmt.Sprintf("CREATE DATABASE %s", quoteIdentifier(databaseName))
		if err := t.executor.Exec(ctx, createQuery, nil); err != nil {
			t.logger.Warn("Failed to create database (may already exist)", map[string]interface{}{
				"database_name": databaseName,
				"error":         err.Error(),
			})
		}
	}

	/* Build pg_restore command */
	args := []string{}

	/* Database name */
	args = append(args, "-d", databaseName)

	/* Input file */
	args = append(args, inputFile)

	/* Options */
	if clean, ok := params["clean"].(bool); ok && clean {
		args = append(args, "-c")
	}

	if ifExists, ok := params["if_exists"].(bool); ok && ifExists {
		args = append(args, "--if-exists")
	}

	if dataOnly, ok := params["data_only"].(bool); ok && dataOnly {
		args = append(args, "-a")
	}

	if schemaOnly, ok := params["schema_only"].(bool); ok && schemaOnly {
		args = append(args, "-s")
	}

	if jobs, ok := params["jobs"].(float64); ok {
		args = append(args, "-j", fmt.Sprintf("%d", int(jobs)))
	} else if jobs, ok := params["jobs"].(int); ok {
		args = append(args, "-j", fmt.Sprintf("%d", jobs))
	}

	if verbose, ok := params["verbose"].(bool); ok && verbose {
		args = append(args, "-v")
	}

	/* Execute pg_restore */
	cmd := exec.CommandContext(ctx, "pg_restore", args...)
	
	output, err := cmd.CombinedOutput()
	if err != nil {
		return Error(
			fmt.Sprintf("pg_restore failed: %v", err),
			"RESTORE_ERROR",
			map[string]interface{}{
				"error":  err.Error(),
				"output": string(output),
			},
		), nil
	}

	t.logger.Info("Database restored", map[string]interface{}{
		"database_name": databaseName,
		"input_file":    inputFile,
	})

	return Success(map[string]interface{}{
		"database_name": databaseName,
		"input_file":    inputFile,
		"command":       "pg_restore " + strings.Join(args, " "),
	}, map[string]interface{}{
		"tool": "postgresql_restore_database",
	}), nil
}

/* PostgreSQLBackupTableTool backs up individual tables */
type PostgreSQLBackupTableTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLBackupTableTool creates a new PostgreSQL backup table tool */
func NewPostgreSQLBackupTableTool(db *database.Database, logger *logging.Logger) *PostgreSQLBackupTableTool {
	return &PostgreSQLBackupTableTool{
		BaseTool: NewBaseTool(
			"postgresql_backup_table",
			"Backup a specific table with data-only and schema-only options",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the table to backup",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"default":     "public",
						"description": "Schema name",
					},
					"output_file": map[string]interface{}{
						"type":        "string",
						"description": "Output file path",
					},
					"data_only": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Backup data only (no schema)",
					},
					"schema_only": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Backup schema only (no data)",
					},
				},
				"required": []interface{}{"table_name", "output_file"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute backs up the table */
func (t *PostgreSQLBackupTableTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	tableName, ok := params["table_name"].(string)
	if !ok || tableName == "" {
		return Error("table_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	outputFile, ok := params["output_file"].(string)
	if !ok || outputFile == "" {
		return Error("output_file parameter is required", "INVALID_PARAMETER", nil), nil
	}

	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	/* Get current database name */
	currentDB, err := t.executor.ExecuteQueryOne(ctx, "SELECT current_database() AS current_db", nil)
	if err != nil {
		return Error("Failed to get current database name", "QUERY_ERROR", nil), nil
	}
	databaseName, _ := currentDB["current_db"].(string)

	/* Build pg_dump command for table */
	args := []string{"-Fc", "-d", databaseName, "-t", fmt.Sprintf("%s.%s", schema, tableName), "-f", outputFile}

	if dataOnly, ok := params["data_only"].(bool); ok && dataOnly {
		args = append(args, "-a")
	}

	if schemaOnly, ok := params["schema_only"].(bool); ok && schemaOnly {
		args = append(args, "-s")
	}

	/* Execute pg_dump */
	cmd := exec.CommandContext(ctx, "pg_dump", args...)
	
	output, err := cmd.CombinedOutput()
	if err != nil {
		return Error(
			fmt.Sprintf("pg_dump failed: %v", err),
			"BACKUP_ERROR",
			map[string]interface{}{
				"error":  err.Error(),
				"output": string(output),
			},
		), nil
	}

	/* Get file size */
	fileInfo, err := os.Stat(outputFile)
	fileSize := int64(0)
	if err == nil {
		fileSize = fileInfo.Size()
	}

	t.logger.Info("Table backed up", map[string]interface{}{
		"table_name": tableName,
		"schema":     schema,
		"output_file": outputFile,
		"file_size":  fileSize,
	})

	return Success(map[string]interface{}{
		"table_name":  tableName,
		"schema":      schema,
		"output_file": outputFile,
		"file_size":   fileSize,
		"command":     "pg_dump " + strings.Join(args, " "),
	}, map[string]interface{}{
		"tool": "postgresql_backup_table",
	}), nil
}

/* PostgreSQLListBackupsTool lists available backups */
type PostgreSQLListBackupsTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLListBackupsTool creates a new PostgreSQL list backups tool */
func NewPostgreSQLListBackupsTool(db *database.Database, logger *logging.Logger) *PostgreSQLListBackupsTool {
	return &PostgreSQLListBackupsTool{
		BaseTool: NewBaseTool(
			"postgresql_list_backups",
			"List available backup files in a directory with metadata",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"backup_directory": map[string]interface{}{
						"type":        "string",
						"description": "Directory path to search for backups",
					},
					"pattern": map[string]interface{}{
						"type":        "string",
						"description": "File pattern to match (e.g., *.dump)",
					},
				},
				"required": []interface{}{"backup_directory"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute lists the backups */
func (t *PostgreSQLListBackupsTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	backupDir, ok := params["backup_directory"].(string)
	if !ok || backupDir == "" {
		return Error("backup_directory parameter is required", "INVALID_PARAMETER", nil), nil
	}

	pattern, _ := params["pattern"].(string)
	if pattern == "" {
		pattern = "*"
	}

	/* List files in directory */
	files, err := filepath.Glob(filepath.Join(backupDir, pattern))
	if err != nil {
		return Error(
			fmt.Sprintf("Failed to list backups: %v", err),
			"FILE_ERROR",
			nil,
		), nil
	}

	backups := []map[string]interface{}{}
	for _, file := range files {
		fileInfo, err := os.Stat(file)
		if err != nil {
			continue
		}

		backups = append(backups, map[string]interface{}{
			"file_name":    filepath.Base(file),
			"file_path":    file,
			"size_bytes":   fileInfo.Size(),
			"modified_time": fileInfo.ModTime().Format(time.RFC3339),
		})
	}

	t.logger.Info("Backups listed", map[string]interface{}{
		"backup_directory": backupDir,
		"count":            len(backups),
	})

	return Success(map[string]interface{}{
		"backup_directory": backupDir,
		"backups":          backups,
		"count":            len(backups),
	}, map[string]interface{}{
		"tool": "postgresql_list_backups",
	}), nil
}

/* PostgreSQLVerifyBackupTool verifies backup integrity */
type PostgreSQLVerifyBackupTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLVerifyBackupTool creates a new PostgreSQL verify backup tool */
func NewPostgreSQLVerifyBackupTool(db *database.Database, logger *logging.Logger) *PostgreSQLVerifyBackupTool {
	return &PostgreSQLVerifyBackupTool{
		BaseTool: NewBaseTool(
			"postgresql_verify_backup",
			"Verify backup file integrity",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"backup_file": map[string]interface{}{
						"type":        "string",
						"description": "Path to backup file to verify",
					},
				},
				"required": []interface{}{"backup_file"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute verifies the backup */
func (t *PostgreSQLVerifyBackupTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	backupFile, ok := params["backup_file"].(string)
	if !ok || backupFile == "" {
		return Error("backup_file parameter is required", "INVALID_PARAMETER", nil), nil
	}

	/* Check if file exists */
	fileInfo, err := os.Stat(backupFile)
	if os.IsNotExist(err) {
		return Error(
			fmt.Sprintf("Backup file does not exist: %s", backupFile),
			"FILE_NOT_FOUND",
			nil,
		), nil
	}

	/* Try to list contents using pg_restore --list */
	cmd := exec.CommandContext(ctx, "pg_restore", "--list", backupFile)
	output, err := cmd.CombinedOutput()
	
	isValid := err == nil
	errorMsg := ""
	if err != nil {
		errorMsg = err.Error()
	}

	t.logger.Info("Backup verified", map[string]interface{}{
		"backup_file": backupFile,
		"is_valid":    isValid,
		"file_size":   fileInfo.Size(),
	})

	return Success(map[string]interface{}{
		"backup_file": backupFile,
		"is_valid":    isValid,
		"file_size":   fileInfo.Size(),
		"error":       errorMsg,
		"output":      string(output),
	}, map[string]interface{}{
		"tool": "postgresql_verify_backup",
	}), nil
}

/* PostgreSQLBackupScheduleTool schedules automated backups */
type PostgreSQLBackupScheduleTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLBackupScheduleTool creates a new PostgreSQL backup schedule tool */
func NewPostgreSQLBackupScheduleTool(db *database.Database, logger *logging.Logger) *PostgreSQLBackupScheduleTool {
	return &PostgreSQLBackupScheduleTool{
		BaseTool: NewBaseTool(
			"postgresql_backup_schedule",
			"Schedule automated database backups using cron-like syntax",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"database_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the database to backup",
					},
					"schedule": map[string]interface{}{
						"type":        "string",
						"description": "Cron-like schedule (e.g., '0 2 * * *' for daily at 2 AM)",
					},
					"output_directory": map[string]interface{}{
						"type":        "string",
						"description": "Directory to store backups",
					},
					"format": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"custom", "directory", "tar", "plain"},
						"default":     "custom",
						"description": "Backup format",
					},
					"retention_days": map[string]interface{}{
						"type":        "integer",
						"default":     7,
						"description": "Number of days to retain backups",
					},
					"enabled": map[string]interface{}{
						"type":        "boolean",
						"default":     true,
						"description": "Enable or disable the schedule",
					},
				},
				"required": []interface{}{"database_name", "schedule", "output_directory"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute schedules the backup */
func (t *PostgreSQLBackupScheduleTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	databaseName, ok := params["database_name"].(string)
	if !ok || databaseName == "" {
		return Error("database_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	schedule, ok := params["schedule"].(string)
	if !ok || schedule == "" {
		return Error("schedule parameter is required", "INVALID_PARAMETER", nil), nil
	}

	outputDir, ok := params["output_directory"].(string)
	if !ok || outputDir == "" {
		return Error("output_directory parameter is required", "INVALID_PARAMETER", nil), nil
	}

	format, _ := params["format"].(string)
	if format == "" {
		format = "custom"
	}

	retentionDays := 7
	if rd, ok := params["retention_days"].(float64); ok {
		retentionDays = int(rd)
	} else if rd, ok := params["retention_days"].(int); ok {
		retentionDays = rd
	}

	enabled := true
	if e, ok := params["enabled"].(bool); ok {
		enabled = e
	}

	/* Create output directory if it doesn't exist */
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return Error(
			fmt.Sprintf("Failed to create output directory: %v", err),
			"FILE_ERROR",
			nil,
		), nil
	}

	/* Store schedule in a metadata file (in production, this would use a proper scheduler) */
	scheduleFile := filepath.Join(outputDir, fmt.Sprintf(".schedule_%s.json", databaseName))
	scheduleData := map[string]interface{}{
		"database_name":    databaseName,
		"schedule":         schedule,
		"output_directory": outputDir,
		"format":           format,
		"retention_days":   retentionDays,
		"enabled":          enabled,
		"created_at":       time.Now().Format(time.RFC3339),
	}

	/* In production, this would integrate with a cron scheduler or job queue */
	/* For now, we'll just store the schedule configuration */
	scheduleJSON, err := json.Marshal(scheduleData)
	if err != nil {
		return Error(
			fmt.Sprintf("Failed to serialize schedule: %v", err),
			"SERIALIZATION_ERROR",
			nil,
		), nil
	}

	if err := os.WriteFile(scheduleFile, scheduleJSON, 0644); err != nil {
		return Error(
			fmt.Sprintf("Failed to write schedule file: %v", err),
			"FILE_ERROR",
			nil,
		), nil
	}

	t.logger.Info("Backup schedule created", map[string]interface{}{
		"database_name": databaseName,
		"schedule":      schedule,
		"enabled":       enabled,
	})

	return Success(map[string]interface{}{
		"database_name":    databaseName,
		"schedule":         schedule,
		"output_directory": outputDir,
		"format":           format,
		"retention_days":   retentionDays,
		"enabled":          enabled,
		"schedule_file":    scheduleFile,
		"note":             "Schedule stored. In production, integrate with cron or job scheduler.",
	}, map[string]interface{}{
		"tool": "postgresql_backup_schedule",
	}), nil
}

