/*-------------------------------------------------------------------------
 *
 * datasets.go
 *    Dataset resources for NeuronMCP
 *
 * Exposes datasets (tables, views) as first-class resources.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/resources/datasets.go
 *
 *-------------------------------------------------------------------------
 */

package resources

import (
	"context"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/database"
)

/* DatasetsResource provides dataset resource */
type DatasetsResource struct {
	*BaseResource
}

/* NewDatasetsResource creates a new datasets resource */
func NewDatasetsResource(db *database.Database) Resource {
	return &DatasetsResource{BaseResource: NewBaseResource(db)}
}

/* URI returns the resource URI */
func (r *DatasetsResource) URI() string {
	return "neurondb://datasets"
}

/* Name returns the resource name */
func (r *DatasetsResource) Name() string {
	return "Datasets"
}

/* Description returns the resource description */
func (r *DatasetsResource) Description() string {
	return "Catalog of available datasets (tables and views)"
}

/* MimeType returns the MIME type */
func (r *DatasetsResource) MimeType() string {
	return "application/json"
}

/* GetContent returns the resource content */
func (r *DatasetsResource) GetContent(ctx context.Context) (interface{}, error) {
	/* Query for all user tables and views */
	query := `
		SELECT 
			schema_name,
			table_name,
			table_type,
			(SELECT COUNT(*) FROM information_schema.columns 
			 WHERE table_schema = t.schema_name AND table_name = t.table_name) as column_count
		FROM (
			SELECT table_schema as schema_name, table_name, 'table' as table_type
			FROM information_schema.tables
			WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
			UNION ALL
			SELECT table_schema as schema_name, table_name, 'view' as table_type
			FROM information_schema.views
			WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
		) t
		ORDER BY schema_name, table_name
	`

	results, err := r.executeQuery(ctx, query, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to query datasets: %w", err)
	}

	datasets := make([]map[string]interface{}, 0)
	for _, row := range results {
		schemaStr, _ := row["schema_name"].(string)
		tableStr, _ := row["table_name"].(string)
		typeStr, _ := row["table_type"].(string)
		countInt := 0
		if cnt, ok := row["column_count"].(int64); ok {
			countInt = int(cnt)
		} else if cnt, ok := row["column_count"].(int); ok {
			countInt = cnt
		}

		datasets = append(datasets, map[string]interface{}{
			"schema":       schemaStr,
			"name":         tableStr,
			"type":         typeStr,
			"column_count": countInt,
			"uri":          fmt.Sprintf("neurondb://dataset/%s.%s", schemaStr, tableStr),
		})
	}

	return map[string]interface{}{
		"datasets": datasets,
		"count":    len(datasets),
	}, nil
}

/* DatasetResource provides a specific dataset resource */
type DatasetResource struct {
	*BaseResource
	schemaName string
	tableName  string
}

/* NewDatasetResource creates a new dataset resource */
func NewDatasetResource(db *database.Database, schemaName, tableName string) Resource {
	return &DatasetResource{
		BaseResource: NewBaseResource(db),
		schemaName:   schemaName,
		tableName:    tableName,
	}
}

/* URI returns the resource URI */
func (r *DatasetResource) URI() string {
	return fmt.Sprintf("neurondb://dataset/%s.%s", r.schemaName, r.tableName)
}

/* Name returns the resource name */
func (r *DatasetResource) Name() string {
	return fmt.Sprintf("Dataset: %s.%s", r.schemaName, r.tableName)
}

/* Description returns the resource description */
func (r *DatasetResource) Description() string {
	return fmt.Sprintf("Dataset details for %s.%s", r.schemaName, r.tableName)
}

/* MimeType returns the MIME type */
func (r *DatasetResource) MimeType() string {
	return "application/json"
}

/* GetContent returns the resource content */
func (r *DatasetResource) GetContent(ctx context.Context) (interface{}, error) {
	/* Get table/view information */
	query := `
		SELECT 
			column_name,
			data_type,
			is_nullable,
			column_default
		FROM information_schema.columns
		WHERE table_schema = $1 AND table_name = $2
		ORDER BY ordinal_position
	`

	columnResults, err := r.executeQuery(ctx, query, []interface{}{r.schemaName, r.tableName})
	if err != nil {
		return nil, fmt.Errorf("failed to query dataset schema: %w", err)
	}

	columns := make([]map[string]interface{}, 0)
	for _, row := range columnResults {
		nameStr, _ := row["column_name"].(string)
		typeStr, _ := row["data_type"].(string)
		nullableStr, _ := row["is_nullable"].(string)

		col := map[string]interface{}{
			"name":     nameStr,
			"type":     typeStr,
			"nullable": nullableStr == "YES",
		}
		if def, ok := row["column_default"]; ok && def != nil {
			if defStr, ok := def.(string); ok {
				col["default"] = defStr
			}
		}
		columns = append(columns, col)
	}

	/* Get row count */
	rowCount := int64(-1) // Unknown
	countQuery := fmt.Sprintf("SELECT COUNT(*) as cnt FROM %s.%s", r.schemaName, r.tableName)
	countResults, err := r.executeQuery(ctx, countQuery, nil)
	if err == nil && len(countResults) > 0 {
		cntVal := countResults[0]["cnt"]
		if cnt, ok := cntVal.(int64); ok {
			rowCount = cnt
		} else if cnt, ok := cntVal.(int); ok {
			rowCount = int64(cnt)
		} else if cnt, ok := cntVal.(int32); ok {
			rowCount = int64(cnt)
		}
	}

	return map[string]interface{}{
		"schema":    r.schemaName,
		"name":      r.tableName,
		"columns":   columns,
		"row_count": rowCount,
	}, nil
}

