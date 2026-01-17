/*-------------------------------------------------------------------------
 *
 * indexes.go
 *    Database operations
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/resources/indexes.go
 *
 *-------------------------------------------------------------------------
 */

package resources

import (
	"context"

	"github.com/neurondb/NeuronMCP/internal/database"
)

/* IndexesResource provides vector indexes information */
type IndexesResource struct {
	*BaseResource
}

/* NewIndexesResource creates a new indexes resource */
func NewIndexesResource(db *database.Database) *IndexesResource {
	return &IndexesResource{BaseResource: NewBaseResource(db)}
}

/* URI returns the resource URI */
func (r *IndexesResource) URI() string {
	return "neurondb://indexes"
}

/* Name returns the resource name */
func (r *IndexesResource) Name() string {
	return "Vector Indexes"
}

/* Description returns the resource description */
func (r *IndexesResource) Description() string {
	return "Status and information about vector indexes"
}

/* MimeType returns the MIME type */
func (r *IndexesResource) MimeType() string {
	return "application/json"
}

/* GetContent returns the indexes content */
func (r *IndexesResource) GetContent(ctx context.Context) (interface{}, error) {
	query := `
		SELECT 
			i.schemaname,
			i.tablename,
			i.indexname,
			i.indexdef,
			pg_size_pretty(pg_relation_size(c.oid)) AS index_size,
			idx_scan AS index_scans,
			idx_tup_read AS tuples_read,
			idx_tup_fetch AS tuples_fetched,
			CASE 
				WHEN i.indexdef LIKE '%hnsw%' THEN 'HNSW'
				WHEN i.indexdef LIKE '%ivf%' THEN 'IVF'
				WHEN i.indexdef LIKE '%btree%' THEN 'BTREE'
				WHEN i.indexdef LIKE '%gin%' THEN 'GIN'
				WHEN i.indexdef LIKE '%gist%' THEN 'GIST'
				ELSE 'UNKNOWN'
			END AS index_type
		FROM pg_indexes i
		LEFT JOIN pg_class c ON c.relname = i.indexname
		LEFT JOIN pg_namespace n ON n.oid = c.relnamespace AND n.nspname = i.schemaname
		LEFT JOIN pg_stat_user_indexes s ON s.indexrelid = c.oid
		WHERE i.schemaname NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
		ORDER BY i.schemaname, i.tablename, i.indexname
	`
	indexes, err := r.executeQuery(ctx, query, nil)
	if err != nil {
		return nil, err
	}

	result := map[string]interface{}{
		"indexes": indexes,
		"count":   len(indexes),
	}

	return result, nil
}

