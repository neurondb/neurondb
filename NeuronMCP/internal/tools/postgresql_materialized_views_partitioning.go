/*-------------------------------------------------------------------------
 *
 * postgresql_materialized_views_partitioning.go
 *    Materialized views, partitioning, and foreign tables for NeuronMCP
 *
 * Implements PostgreSQL DDL operations for:
 * - Materialized views
 * - Table partitioning
 * - Foreign tables
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/postgresql_materialized_views_partitioning.go
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
 * Materialized View Management
 * ============================================================================ */

/* PostgreSQLCreateMaterializedViewTool creates materialized views */
type PostgreSQLCreateMaterializedViewTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLCreateMaterializedViewTool creates a new PostgreSQL create materialized view tool */
func NewPostgreSQLCreateMaterializedViewTool(db *database.Database, logger *logging.Logger) *PostgreSQLCreateMaterializedViewTool {
	return &PostgreSQLCreateMaterializedViewTool{
		BaseTool: NewBaseTool(
			"postgresql_create_materialized_view",
			"Create a materialized view with WITH DATA or WITH NO DATA",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"view_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the materialized view to create",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"default":     "public",
						"description": "Schema name",
					},
					"query": map[string]interface{}{
						"type":        "string",
						"description": "SELECT query that defines the materialized view",
					},
					"with_data": map[string]interface{}{
						"type":        "boolean",
						"default":     true,
						"description": "Populate the materialized view with data",
					},
					"tablespace": map[string]interface{}{
						"type":        "string",
						"description": "Tablespace name",
					},
					"storage_parameters": map[string]interface{}{
						"type":        "object",
						"description": "Storage parameters as key-value pairs",
					},
				},
				"required": []interface{}{"view_name", "query"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute creates the materialized view */
func (t *PostgreSQLCreateMaterializedViewTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
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

	parts := []string{"CREATE MATERIALIZED VIEW"}
	
	fullViewName := fmt.Sprintf("%s.%s", quoteIdentifier(schema), quoteIdentifier(viewName))
	parts = append(parts, fullViewName)

	/* Tablespace */
	if tablespace, ok := params["tablespace"].(string); ok && tablespace != "" {
		parts = append(parts, fmt.Sprintf("TABLESPACE %s", quoteIdentifier(tablespace)))
	}

	/* Storage parameters */
	if storageParams, ok := params["storage_parameters"].(map[string]interface{}); ok && len(storageParams) > 0 {
		paramList := []string{}
		for key, value := range storageParams {
			if valueStr, ok := value.(string); ok {
				paramList = append(paramList, fmt.Sprintf("%s = %s", quoteIdentifier(key), quoteLiteral(valueStr)))
			} else {
				paramList = append(paramList, fmt.Sprintf("%s = %v", quoteIdentifier(key), value))
			}
		}
		if len(paramList) > 0 {
			parts = append(parts, "WITH ("+strings.Join(paramList, ", ")+")")
		}
	}

	parts = append(parts, "AS", query)

	/* WITH DATA or WITH NO DATA */
	withData := true
	if val, ok := params["with_data"].(bool); ok {
		withData = val
	}
	if withData {
		parts = append(parts, "WITH DATA")
	} else {
		parts = append(parts, "WITH NO DATA")
	}

	createQuery := strings.Join(parts, " ")

	/* Execute CREATE MATERIALIZED VIEW */
	err := t.executor.Exec(ctx, createQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("CREATE MATERIALIZED VIEW failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Materialized view created", map[string]interface{}{
		"view_name": viewName,
		"schema":    schema,
	})

	return Success(map[string]interface{}{
		"view_name": viewName,
		"schema":    schema,
		"query":     createQuery,
	}, map[string]interface{}{
		"tool": "postgresql_create_materialized_view",
	}), nil
}

/* PostgreSQLRefreshMaterializedViewTool refreshes materialized views */
type PostgreSQLRefreshMaterializedViewTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLRefreshMaterializedViewTool creates a new PostgreSQL refresh materialized view tool */
func NewPostgreSQLRefreshMaterializedViewTool(db *database.Database, logger *logging.Logger) *PostgreSQLRefreshMaterializedViewTool {
	return &PostgreSQLRefreshMaterializedViewTool{
		BaseTool: NewBaseTool(
			"postgresql_refresh_materialized_view",
			"Refresh a materialized view with CONCURRENTLY option",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"view_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the materialized view to refresh",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"default":     "public",
						"description": "Schema name",
					},
					"concurrently": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Refresh concurrently (non-blocking)",
					},
					"with_data": map[string]interface{}{
						"type":        "boolean",
						"default":     true,
						"description": "Refresh with data",
					},
				},
				"required": []interface{}{"view_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute refreshes the materialized view */
func (t *PostgreSQLRefreshMaterializedViewTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	viewName, ok := params["view_name"].(string)
	if !ok || viewName == "" {
		return Error("view_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	fullViewName := fmt.Sprintf("%s.%s", quoteIdentifier(schema), quoteIdentifier(viewName))
	parts := []string{"REFRESH MATERIALIZED VIEW"}

	if concurrently, ok := params["concurrently"].(bool); ok && concurrently {
		parts = append(parts, "CONCURRENTLY")
	}

	parts = append(parts, fullViewName)

	withData := true
	if val, ok := params["with_data"].(bool); ok {
		withData = val
	}
	if withData {
		parts = append(parts, "WITH DATA")
	} else {
		parts = append(parts, "WITH NO DATA")
	}

	refreshQuery := strings.Join(parts, " ")

	/* Execute REFRESH MATERIALIZED VIEW */
	err := t.executor.Exec(ctx, refreshQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("REFRESH MATERIALIZED VIEW failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Materialized view refreshed", map[string]interface{}{
		"view_name": viewName,
		"schema":    schema,
	})

	return Success(map[string]interface{}{
		"view_name": viewName,
		"schema":    schema,
		"query":     refreshQuery,
	}, map[string]interface{}{
		"tool": "postgresql_refresh_materialized_view",
	}), nil
}

/* PostgreSQLDropMaterializedViewTool drops materialized views */
type PostgreSQLDropMaterializedViewTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLDropMaterializedViewTool creates a new PostgreSQL drop materialized view tool */
func NewPostgreSQLDropMaterializedViewTool(db *database.Database, logger *logging.Logger) *PostgreSQLDropMaterializedViewTool {
	return &PostgreSQLDropMaterializedViewTool{
		BaseTool: NewBaseTool(
			"postgresql_drop_materialized_view",
			"Drop a materialized view",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"view_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the materialized view to drop",
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

/* Execute drops the materialized view */
func (t *PostgreSQLDropMaterializedViewTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	viewName, ok := params["view_name"].(string)
	if !ok || viewName == "" {
		return Error("view_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	parts := []string{"DROP MATERIALIZED VIEW"}
	
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

	/* Execute DROP MATERIALIZED VIEW */
	err := t.executor.Exec(ctx, dropQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("DROP MATERIALIZED VIEW failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Materialized view dropped", map[string]interface{}{
		"view_name": viewName,
		"schema":    schema,
	})

	return Success(map[string]interface{}{
		"view_name": viewName,
		"schema":    schema,
		"query":     dropQuery,
	}, map[string]interface{}{
		"tool": "postgresql_drop_materialized_view",
	}), nil
}

/* ============================================================================
 * Partitioning Management
 * ============================================================================ */

/* PostgreSQLCreatePartitionTool creates partitioned tables */
type PostgreSQLCreatePartitionTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLCreatePartitionTool creates a new PostgreSQL create partition tool */
func NewPostgreSQLCreatePartitionTool(db *database.Database, logger *logging.Logger) *PostgreSQLCreatePartitionTool {
	return &PostgreSQLCreatePartitionTool{
		BaseTool: NewBaseTool(
			"postgresql_create_partition",
			"Create a partitioned table with RANGE, LIST, or HASH partitioning",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the partitioned table to create",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"default":     "public",
						"description": "Schema name",
					},
					"columns": map[string]interface{}{
						"type":        "array",
						"description": "Array of column definitions",
					},
					"partition_method": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"RANGE", "LIST", "HASH"},
						"description": "Partitioning method",
					},
					"partition_key": map[string]interface{}{
						"type":        "array",
						"description": "Array of column names for partition key",
					},
					"primary_key": map[string]interface{}{
						"type":        "array",
						"description": "Array of column names for primary key",
					},
				},
				"required": []interface{}{"table_name", "columns", "partition_method", "partition_key"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute creates the partitioned table */
func (t *PostgreSQLCreatePartitionTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	tableName, ok := params["table_name"].(string)
	if !ok || tableName == "" {
		return Error("table_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	columns, ok := params["columns"].([]interface{})
	if !ok || len(columns) == 0 {
		return Error("columns parameter is required and must be a non-empty array", "INVALID_PARAMETER", nil), nil
	}

	partitionMethod, ok := params["partition_method"].(string)
	if !ok || partitionMethod == "" {
		return Error("partition_method parameter is required", "INVALID_PARAMETER", nil), nil
	}
	partitionMethod = strings.ToUpper(partitionMethod)

	partitionKey, ok := params["partition_key"].([]interface{})
	if !ok || len(partitionKey) == 0 {
		return Error("partition_key parameter is required and must be a non-empty array", "INVALID_PARAMETER", nil), nil
	}

	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	/* Build column definitions */
	colDefs := []string{}
	for _, col := range columns {
		colMap, ok := col.(map[string]interface{})
		if !ok {
			return Error("Each column must be an object with name and type", "INVALID_PARAMETER", nil), nil
		}

		colName, _ := colMap["name"].(string)
		colType, _ := colMap["type"].(string)
		if colName == "" || colType == "" {
			return Error("Column name and type are required", "INVALID_PARAMETER", nil), nil
		}

		colDef := fmt.Sprintf("%s %s", quoteIdentifier(colName), colType)
		if constraints, ok := colMap["constraints"].(string); ok && constraints != "" {
			colDef += " " + constraints
		}
		colDefs = append(colDefs, colDef)
	}

	/* Build partition key */
	keyList := []string{}
	for _, key := range partitionKey {
		if keyStr, ok := key.(string); ok {
			keyList = append(keyList, quoteIdentifier(keyStr))
		}
	}

	/* Build CREATE TABLE statement */
	parts := []string{"CREATE TABLE"}
	fullTableName := fmt.Sprintf("%s.%s", quoteIdentifier(schema), quoteIdentifier(tableName))
	parts = append(parts, fullTableName)
	parts = append(parts, "(")
	parts = append(parts, strings.Join(colDefs, ", "))

	/* Primary key */
	if pkCols, ok := params["primary_key"].([]interface{}); ok && len(pkCols) > 0 {
		pkNames := []string{}
		for _, pk := range pkCols {
			if pkName, ok := pk.(string); ok {
				pkNames = append(pkNames, quoteIdentifier(pkName))
			}
		}
		if len(pkNames) > 0 {
			parts = append(parts, fmt.Sprintf(", PRIMARY KEY (%s)", strings.Join(pkNames, ", ")))
		}
	}

	parts = append(parts, ")")
	parts = append(parts, fmt.Sprintf("PARTITION BY %s (%s)", partitionMethod, strings.Join(keyList, ", ")))

	createQuery := strings.Join(parts, " ")

	/* Execute CREATE TABLE */
	err := t.executor.Exec(ctx, createQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("CREATE PARTITIONED TABLE failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Partitioned table created", map[string]interface{}{
		"table_name":      tableName,
		"schema":          schema,
		"partition_method": partitionMethod,
	})

	return Success(map[string]interface{}{
		"table_name":      tableName,
		"schema":          schema,
		"partition_method": partitionMethod,
		"query":           createQuery,
	}, map[string]interface{}{
		"tool": "postgresql_create_partition",
	}), nil
}

/* PostgreSQLAttachPartitionTool attaches partitions to partitioned tables */
type PostgreSQLAttachPartitionTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLAttachPartitionTool creates a new PostgreSQL attach partition tool */
func NewPostgreSQLAttachPartitionTool(db *database.Database, logger *logging.Logger) *PostgreSQLAttachPartitionTool {
	return &PostgreSQLAttachPartitionTool{
		BaseTool: NewBaseTool(
			"postgresql_attach_partition",
			"Attach a table as a partition to a partitioned table",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"partitioned_table": map[string]interface{}{
						"type":        "string",
						"description": "Name of the partitioned table",
					},
					"partition_table": map[string]interface{}{
						"type":        "string",
						"description": "Name of the table to attach as partition",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"default":     "public",
						"description": "Schema name",
					},
					"partition_bound": map[string]interface{}{
						"type":        "string",
						"description": "Partition bound specification (e.g., FOR VALUES FROM (1) TO (10))",
					},
				},
				"required": []interface{}{"partitioned_table", "partition_table", "partition_bound"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute attaches the partition */
func (t *PostgreSQLAttachPartitionTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	partitionedTable, ok := params["partitioned_table"].(string)
	if !ok || partitionedTable == "" {
		return Error("partitioned_table parameter is required", "INVALID_PARAMETER", nil), nil
	}

	partitionTable, ok := params["partition_table"].(string)
	if !ok || partitionTable == "" {
		return Error("partition_table parameter is required", "INVALID_PARAMETER", nil), nil
	}

	partitionBound, ok := params["partition_bound"].(string)
	if !ok || partitionBound == "" {
		return Error("partition_bound parameter is required", "INVALID_PARAMETER", nil), nil
	}

	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	fullPartitionedTable := fmt.Sprintf("%s.%s", quoteIdentifier(schema), quoteIdentifier(partitionedTable))
	fullPartitionTable := fmt.Sprintf("%s.%s", quoteIdentifier(schema), quoteIdentifier(partitionTable))

	attachQuery := fmt.Sprintf("ALTER TABLE %s ATTACH PARTITION %s %s", fullPartitionedTable, fullPartitionTable, partitionBound)

	/* Execute ATTACH PARTITION */
	err := t.executor.Exec(ctx, attachQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("ATTACH PARTITION failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Partition attached", map[string]interface{}{
		"partitioned_table": partitionedTable,
		"partition_table":   partitionTable,
		"schema":            schema,
	})

	return Success(map[string]interface{}{
		"partitioned_table": partitionedTable,
		"partition_table":   partitionTable,
		"schema":            schema,
		"query":             attachQuery,
	}, map[string]interface{}{
		"tool": "postgresql_attach_partition",
	}), nil
}

/* PostgreSQLDetachPartitionTool detaches partitions */
type PostgreSQLDetachPartitionTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLDetachPartitionTool creates a new PostgreSQL detach partition tool */
func NewPostgreSQLDetachPartitionTool(db *database.Database, logger *logging.Logger) *PostgreSQLDetachPartitionTool {
	return &PostgreSQLDetachPartitionTool{
		BaseTool: NewBaseTool(
			"postgresql_detach_partition",
			"Detach a partition from a partitioned table",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"partitioned_table": map[string]interface{}{
						"type":        "string",
						"description": "Name of the partitioned table",
					},
					"partition_table": map[string]interface{}{
						"type":        "string",
						"description": "Name of the partition to detach",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"default":     "public",
						"description": "Schema name",
					},
					"concurrently": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Detach concurrently (non-blocking)",
					},
					"finalize": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Finalize concurrent detach",
					},
				},
				"required": []interface{}{"partitioned_table", "partition_table"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute detaches the partition */
func (t *PostgreSQLDetachPartitionTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	partitionedTable, ok := params["partitioned_table"].(string)
	if !ok || partitionedTable == "" {
		return Error("partitioned_table parameter is required", "INVALID_PARAMETER", nil), nil
	}

	partitionTable, ok := params["partition_table"].(string)
	if !ok || partitionTable == "" {
		return Error("partition_table parameter is required", "INVALID_PARAMETER", nil), nil
	}

	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	fullPartitionedTable := fmt.Sprintf("%s.%s", quoteIdentifier(schema), quoteIdentifier(partitionedTable))
	fullPartitionTable := fmt.Sprintf("%s.%s", quoteIdentifier(schema), quoteIdentifier(partitionTable))

	parts := []string{"ALTER TABLE", fullPartitionedTable, "DETACH PARTITION"}

	if concurrently, ok := params["concurrently"].(bool); ok && concurrently {
		parts = append(parts, "CONCURRENTLY")
	}

	parts = append(parts, fullPartitionTable)

	if finalize, ok := params["finalize"].(bool); ok && finalize {
		parts = append(parts, "FINALIZE")
	}

	detachQuery := strings.Join(parts, " ")

	/* Execute DETACH PARTITION */
	err := t.executor.Exec(ctx, detachQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("DETACH PARTITION failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Partition detached", map[string]interface{}{
		"partitioned_table": partitionedTable,
		"partition_table":   partitionTable,
		"schema":            schema,
	})

	return Success(map[string]interface{}{
		"partitioned_table": partitionedTable,
		"partition_table":   partitionTable,
		"schema":            schema,
		"query":             detachQuery,
	}, map[string]interface{}{
		"tool": "postgresql_detach_partition",
	}), nil
}

/* ============================================================================
 * Foreign Table Management
 * ============================================================================ */

/* PostgreSQLCreateForeignTableTool creates foreign tables */
type PostgreSQLCreateForeignTableTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLCreateForeignTableTool creates a new PostgreSQL create foreign table tool */
func NewPostgreSQLCreateForeignTableTool(db *database.Database, logger *logging.Logger) *PostgreSQLCreateForeignTableTool {
	return &PostgreSQLCreateForeignTableTool{
		BaseTool: NewBaseTool(
			"postgresql_create_foreign_table",
			"Create a foreign table via Foreign Data Wrapper (FDW)",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the foreign table to create",
					},
					"schema": map[string]interface{}{
						"type":        "string",
						"default":     "public",
						"description": "Schema name",
					},
					"columns": map[string]interface{}{
						"type":        "array",
						"description": "Array of column definitions",
					},
					"server_name": map[string]interface{}{
						"type":        "string",
						"description": "Foreign server name",
					},
					"options": map[string]interface{}{
						"type":        "object",
						"description": "Foreign table options as key-value pairs",
					},
				},
				"required": []interface{}{"table_name", "columns", "server_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute creates the foreign table */
func (t *PostgreSQLCreateForeignTableTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	tableName, ok := params["table_name"].(string)
	if !ok || tableName == "" {
		return Error("table_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	columns, ok := params["columns"].([]interface{})
	if !ok || len(columns) == 0 {
		return Error("columns parameter is required and must be a non-empty array", "INVALID_PARAMETER", nil), nil
	}

	serverName, ok := params["server_name"].(string)
	if !ok || serverName == "" {
		return Error("server_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	/* Build column definitions */
	colDefs := []string{}
	for _, col := range columns {
		colMap, ok := col.(map[string]interface{})
		if !ok {
			return Error("Each column must be an object with name and type", "INVALID_PARAMETER", nil), nil
		}

		colName, _ := colMap["name"].(string)
		colType, _ := colMap["type"].(string)
		if colName == "" || colType == "" {
			return Error("Column name and type are required", "INVALID_PARAMETER", nil), nil
		}

		colDef := fmt.Sprintf("%s %s", quoteIdentifier(colName), colType)
		colDefs = append(colDefs, colDef)
	}

	/* Build CREATE FOREIGN TABLE statement */
	parts := []string{"CREATE FOREIGN TABLE"}
	fullTableName := fmt.Sprintf("%s.%s", quoteIdentifier(schema), quoteIdentifier(tableName))
	parts = append(parts, fullTableName)
	parts = append(parts, "(")
	parts = append(parts, strings.Join(colDefs, ", "))
	parts = append(parts, ")")
	parts = append(parts, "SERVER", quoteIdentifier(serverName))

	/* Options */
	if options, ok := params["options"].(map[string]interface{}); ok && len(options) > 0 {
		optionList := []string{}
		for key, value := range options {
			if valueStr, ok := value.(string); ok {
				optionList = append(optionList, fmt.Sprintf("%s %s", quoteIdentifier(key), quoteLiteral(valueStr)))
			} else {
				optionList = append(optionList, fmt.Sprintf("%s %v", quoteIdentifier(key), value))
			}
		}
		if len(optionList) > 0 {
			parts = append(parts, "OPTIONS ("+strings.Join(optionList, ", ")+")")
		}
	}

	createQuery := strings.Join(parts, " ")

	/* Execute CREATE FOREIGN TABLE */
	err := t.executor.Exec(ctx, createQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("CREATE FOREIGN TABLE failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Foreign table created", map[string]interface{}{
		"table_name": tableName,
		"schema":     schema,
		"server":     serverName,
	})

	return Success(map[string]interface{}{
		"table_name": tableName,
		"schema":     schema,
		"server":     serverName,
		"query":      createQuery,
	}, map[string]interface{}{
		"tool": "postgresql_create_foreign_table",
	}), nil
}

/* PostgreSQLDropPartitionTool drops a partition from a partitioned table */
type PostgreSQLDropPartitionTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLDropPartitionTool creates a new PostgreSQL drop partition tool */
func NewPostgreSQLDropPartitionTool(db *database.Database, logger *logging.Logger) *PostgreSQLDropPartitionTool {
	return &PostgreSQLDropPartitionTool{
		BaseTool: NewBaseTool(
			"postgresql_drop_partition",
			"Drop a partition from a partitioned table",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"partition_table": map[string]interface{}{
						"type":        "string",
						"description": "Name of the partition table to drop",
					},
					"partitioned_table": map[string]interface{}{
						"type":        "string",
						"description": "Name of the parent partitioned table",
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
				"required": []interface{}{"partition_table"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute drops the partition */
func (t *PostgreSQLDropPartitionTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	partitionTable, ok := params["partition_table"].(string)
	if !ok || partitionTable == "" {
		return Error("partition_table parameter is required", "INVALID_PARAMETER", nil), nil
	}

	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	ifExists, _ := params["if_exists"].(bool)

	/* Build DROP TABLE query for the partition */
	parts := []string{"DROP TABLE"}
	if ifExists {
		parts = append(parts, "IF EXISTS")
	}
	if schema != "" && schema != "public" {
		parts = append(parts, fmt.Sprintf(`"%s"."%s"`, schema, partitionTable))
	} else {
		parts = append(parts, fmt.Sprintf(`"%s"`, partitionTable))
	}

	dropQuery := strings.Join(parts, " ")

	/* Execute DROP TABLE */
	err := t.executor.Exec(ctx, dropQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("DROP PARTITION failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Partition dropped", map[string]interface{}{
		"partition_table": partitionTable,
		"schema":          schema,
	})

	return Success(map[string]interface{}{
		"partition_table": partitionTable,
		"schema":          schema,
		"query":           dropQuery,
	}, map[string]interface{}{
		"tool": "postgresql_drop_partition",
	}), nil
}

/* PostgreSQLDropForeignTableTool drops a foreign table */
type PostgreSQLDropForeignTableTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLDropForeignTableTool creates a new PostgreSQL drop foreign table tool */
func NewPostgreSQLDropForeignTableTool(db *database.Database, logger *logging.Logger) *PostgreSQLDropForeignTableTool {
	return &PostgreSQLDropForeignTableTool{
		BaseTool: NewBaseTool(
			"postgresql_drop_foreign_table",
			"Drop a foreign table",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the foreign table to drop",
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
				"required": []interface{}{"table_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute drops the foreign table */
func (t *PostgreSQLDropForeignTableTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	tableName, ok := params["table_name"].(string)
	if !ok || tableName == "" {
		return Error("table_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	ifExists, _ := params["if_exists"].(bool)
	cascade, _ := params["cascade"].(bool)

	/* Build DROP FOREIGN TABLE query */
	parts := []string{"DROP FOREIGN TABLE"}
	if ifExists {
		parts = append(parts, "IF EXISTS")
	}
	if schema != "" && schema != "public" {
		parts = append(parts, fmt.Sprintf(`"%s"."%s"`, schema, tableName))
	} else {
		parts = append(parts, fmt.Sprintf(`"%s"`, tableName))
	}
	if cascade {
		parts = append(parts, "CASCADE")
	}

	dropQuery := strings.Join(parts, " ")

	/* Execute DROP FOREIGN TABLE */
	err := t.executor.Exec(ctx, dropQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("DROP FOREIGN TABLE failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Foreign table dropped", map[string]interface{}{
		"table_name": tableName,
		"schema":     schema,
	})

	return Success(map[string]interface{}{
		"table_name": tableName,
		"schema":     schema,
		"query":      dropQuery,
	}, map[string]interface{}{
		"tool": "postgresql_drop_foreign_table",
	}), nil
}

/* PostgreSQLAlterTableAdvancedTool provides advanced ALTER TABLE operations */
type PostgreSQLAlterTableAdvancedTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewPostgreSQLAlterTableAdvancedTool creates a new PostgreSQL advanced alter table tool */
func NewPostgreSQLAlterTableAdvancedTool(db *database.Database, logger *logging.Logger) *PostgreSQLAlterTableAdvancedTool {
	return &PostgreSQLAlterTableAdvancedTool{
		BaseTool: NewBaseTool(
			"postgresql_alter_table_advanced",
			"Advanced ALTER TABLE operations: inheritance, constraints, storage parameters, and more",
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
					"operations": map[string]interface{}{
						"type":        "array",
						"description": "Array of ALTER TABLE operations",
					},
					"if_exists": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Use IF EXISTS clause",
					},
				},
				"required": []interface{}{"table_name", "operations"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute performs advanced ALTER TABLE operations */
func (t *PostgreSQLAlterTableAdvancedTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	tableName, ok := params["table_name"].(string)
	if !ok || tableName == "" {
		return Error("table_name parameter is required", "INVALID_PARAMETER", nil), nil
	}

	operations, ok := params["operations"].([]interface{})
	if !ok || len(operations) == 0 {
		return Error("operations parameter is required and must be a non-empty array", "INVALID_PARAMETER", nil), nil
	}

	schema, _ := params["schema"].(string)
	if schema == "" {
		schema = "public"
	}

	ifExists, _ := params["if_exists"].(bool)

	/* Build table reference */
	tableRef := fmt.Sprintf(`"%s"`, tableName)
	if schema != "" && schema != "public" {
		tableRef = fmt.Sprintf(`"%s"."%s"`, schema, tableName)
	}

	/* Build ALTER TABLE query */
	parts := []string{"ALTER TABLE"}
	if ifExists {
		parts = append(parts, "IF EXISTS")
	}
	parts = append(parts, tableRef)

	/* Process operations */
	operationStrings := []string{}
	for i, op := range operations {
		opMap, ok := op.(map[string]interface{})
		if !ok {
			return Error(fmt.Sprintf("Operation %d must be an object", i), "INVALID_PARAMETER", nil), nil
		}

		opType, ok := opMap["type"].(string)
		if !ok || opType == "" {
			return Error(fmt.Sprintf("Operation %d must have a 'type' field", i), "INVALID_PARAMETER", nil), nil
		}

		/* Build operation string based on type */
		var opStr string
		switch opType {
		case "SET_STORAGE_PARAMETER":
			param, _ := opMap["parameter"].(string)
			value, _ := opMap["value"].(string)
			if param == "" || value == "" {
				return Error(fmt.Sprintf("Operation %d: SET_STORAGE_PARAMETER requires 'parameter' and 'value'", i), "INVALID_PARAMETER", nil), nil
			}
			opStr = fmt.Sprintf("SET (%s = %s)", param, value)
		case "RESET_STORAGE_PARAMETER":
			param, _ := opMap["parameter"].(string)
			if param == "" {
				return Error(fmt.Sprintf("Operation %d: RESET_STORAGE_PARAMETER requires 'parameter'", i), "INVALID_PARAMETER", nil), nil
			}
			opStr = fmt.Sprintf("RESET (%s)", param)
		case "INHERIT":
			parentTable, _ := opMap["parent_table"].(string)
			if parentTable == "" {
				return Error(fmt.Sprintf("Operation %d: INHERIT requires 'parent_table'", i), "INVALID_PARAMETER", nil), nil
			}
			opStr = fmt.Sprintf("INHERIT \"%s\"", parentTable)
		case "NO_INHERIT":
			parentTable, _ := opMap["parent_table"].(string)
			if parentTable == "" {
				return Error(fmt.Sprintf("Operation %d: NO_INHERIT requires 'parent_table'", i), "INVALID_PARAMETER", nil), nil
			}
			opStr = fmt.Sprintf("NO INHERIT \"%s\"", parentTable)
		case "OF":
			typeName, _ := opMap["type_name"].(string)
			if typeName == "" {
				return Error(fmt.Sprintf("Operation %d: OF requires 'type_name'", i), "INVALID_PARAMETER", nil), nil
			}
			opStr = fmt.Sprintf("OF \"%s\"", typeName)
		case "NOT_OF":
			opStr = "NOT OF"
		case "ENABLE_ROW_LEVEL_SECURITY":
			opStr = "ENABLE ROW LEVEL SECURITY"
		case "DISABLE_ROW_LEVEL_SECURITY":
			opStr = "DISABLE ROW LEVEL SECURITY"
		case "FORCE_ROW_LEVEL_SECURITY":
			opStr = "FORCE ROW LEVEL SECURITY"
		case "NO_FORCE_ROW_LEVEL_SECURITY":
			opStr = "NO FORCE ROW LEVEL SECURITY"
		case "ENABLE_REPLICA_TRIGGER":
			trigger, _ := opMap["trigger"].(string)
			if trigger == "" {
				return Error(fmt.Sprintf("Operation %d: ENABLE_REPLICA_TRIGGER requires 'trigger'", i), "INVALID_PARAMETER", nil), nil
			}
			opStr = fmt.Sprintf("ENABLE REPLICA TRIGGER \"%s\"", trigger)
		case "DISABLE_TRIGGER":
			trigger, _ := opMap["trigger"].(string)
			if trigger == "" {
				opStr = "DISABLE TRIGGER ALL"
			} else {
				opStr = fmt.Sprintf("DISABLE TRIGGER \"%s\"", trigger)
			}
		case "ENABLE_TRIGGER":
			trigger, _ := opMap["trigger"].(string)
			if trigger == "" {
				opStr = "ENABLE TRIGGER ALL"
			} else {
				opStr = fmt.Sprintf("ENABLE TRIGGER \"%s\"", trigger)
			}
		case "SET_TABLESPACE":
			tablespace, _ := opMap["tablespace"].(string)
			if tablespace == "" {
				return Error(fmt.Sprintf("Operation %d: SET_TABLESPACE requires 'tablespace'", i), "INVALID_PARAMETER", nil), nil
			}
			opStr = fmt.Sprintf("SET TABLESPACE \"%s\"", tablespace)
		case "SET_LOGGED":
			opStr = "SET LOGGED"
		case "SET_UNLOGGED":
			opStr = "SET UNLOGGED"
		default:
			return Error(fmt.Sprintf("Unsupported operation type: %s", opType), "INVALID_PARAMETER", nil), nil
		}

		if i == 0 {
			parts = append(parts, opStr)
		} else {
			operationStrings = append(operationStrings, opStr)
		}
	}

	/* Combine all operations */
	alterQuery := strings.Join(parts, " ")
	if len(operationStrings) > 0 {
		alterQuery += ", " + strings.Join(operationStrings, ", ")
	}

	/* Execute ALTER TABLE */
	err := t.executor.Exec(ctx, alterQuery, nil)
	if err != nil {
		return Error(
			fmt.Sprintf("ALTER TABLE failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	t.logger.Info("Table altered (advanced)", map[string]interface{}{
		"table_name": tableName,
		"schema":     schema,
		"operations": len(operations),
	})

	return Success(map[string]interface{}{
		"table_name": tableName,
		"schema":     schema,
		"operations": operations,
		"query":      alterQuery,
	}, map[string]interface{}{
		"tool": "postgresql_alter_table_advanced",
	}), nil
}

