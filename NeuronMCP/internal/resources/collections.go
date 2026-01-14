/*-------------------------------------------------------------------------
 *
 * collections.go
 *    Document collection resources for NeuronMCP
 *
 * Exposes document collections as first-class resources.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/resources/collections.go
 *
 *-------------------------------------------------------------------------
 */

package resources

import (
	"context"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/database"
)

/* CollectionsResource provides document collections resource */
type CollectionsResource struct {
	*BaseResource
}

/* NewCollectionsResource creates a new collections resource */
func NewCollectionsResource(db *database.Database) Resource {
	return &CollectionsResource{BaseResource: NewBaseResource(db)}
}

/* URI returns the resource URI */
func (r *CollectionsResource) URI() string {
	return "neurondb://collections"
}

/* Name returns the resource name */
func (r *CollectionsResource) Name() string {
	return "Document Collections"
}

/* Description returns the resource description */
func (r *CollectionsResource) Description() string {
	return "Catalog of document collections used for RAG operations"
}

/* MimeType returns the MIME type */
func (r *CollectionsResource) MimeType() string {
	return "application/json"
}

/* GetContent returns the resource content */
func (r *CollectionsResource) GetContent(ctx context.Context) (interface{}, error) {
	/* Query for tables that likely contain document collections (have text and vector columns) */
	query := `
		SELECT DISTINCT
			t.table_schema,
			t.table_name,
			(SELECT COUNT(*) FROM information_schema.columns c1 
			 WHERE c1.table_schema = t.table_schema 
			 AND c1.table_name = t.table_name 
			 AND c1.data_type = 'text') as text_columns,
			(SELECT COUNT(*) FROM information_schema.columns c2 
			 WHERE c2.table_schema = t.table_schema 
			 AND c2.table_name = t.table_name 
			 AND c2.udt_name = 'vector') as vector_columns
		FROM information_schema.tables t
		WHERE t.table_schema NOT IN ('pg_catalog', 'information_schema')
		AND EXISTS (
			SELECT 1 FROM information_schema.columns c
			WHERE c.table_schema = t.table_schema
			AND c.table_name = t.table_name
			AND (c.data_type = 'text' OR c.udt_name = 'vector')
		)
		ORDER BY t.table_schema, t.table_name
	`

	results, err := r.executeQuery(ctx, query, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to query collections: %w", err)
	}

	collections := make([]map[string]interface{}, 0)
	for _, row := range results {
		schemaStr, _ := row["table_schema"].(string)
		tableStr, _ := row["table_name"].(string)
		textCnt := 0
		vecCnt := 0
		if tc, ok := row["text_columns"].(int64); ok {
			textCnt = int(tc)
		} else if tc, ok := row["text_columns"].(int); ok {
			textCnt = tc
		}
		if vc, ok := row["vector_columns"].(int64); ok {
			vecCnt = int(vc)
		} else if vc, ok := row["vector_columns"].(int); ok {
			vecCnt = vc
		}

		collections = append(collections, map[string]interface{}{
			"schema":         schemaStr,
			"name":           tableStr,
			"text_columns":   textCnt,
			"vector_columns": vecCnt,
			"uri":            fmt.Sprintf("neurondb://collection/%s.%s", schemaStr, tableStr),
		})
	}

	return map[string]interface{}{
		"collections": collections,
		"count":       len(collections),
	}, nil
}

