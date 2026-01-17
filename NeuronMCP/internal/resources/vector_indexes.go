/*-------------------------------------------------------------------------
 *
 * vector_indexes.go
 *    Vector indexes resource for NeuronMCP
 *
 * Provides information about NeuronDB vector indexes including HNSW and IVF
 * indexes with their parameters and status.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/resources/vector_indexes.go
 *
 *-------------------------------------------------------------------------
 */

package resources

import (
	"context"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/database"
)

/* VectorIndexesResource provides vector index information */
type VectorIndexesResource struct {
	*BaseResource
}

/* NewVectorIndexesResource creates a new vector indexes resource */
func NewVectorIndexesResource(db *database.Database) *VectorIndexesResource {
	return &VectorIndexesResource{BaseResource: NewBaseResource(db)}
}

/* URI returns the resource URI */
func (r *VectorIndexesResource) URI() string {
	return "neurondb://vector_indexes"
}

/* Name returns the resource name */
func (r *VectorIndexesResource) Name() string {
	return "Vector Indexes"
}

/* Description returns the resource description */
func (r *VectorIndexesResource) Description() string {
	return "NeuronDB vector indexes (HNSW, IVF) with parameters and status"
}

/* MimeType returns the MIME type */
func (r *VectorIndexesResource) MimeType() string {
	return "application/json"
}

/* GetContent returns the vector indexes content */
func (r *VectorIndexesResource) GetContent(ctx context.Context) (interface{}, error) {
	/* Query for vector indexes - check pg_indexes for HNSW and IVF indexes */
	query := `
		SELECT 
			schemaname,
			tablename,
			indexname,
			indexdef,
			CASE 
				WHEN indexdef LIKE '%hnsw%' THEN 'HNSW'
				WHEN indexdef LIKE '%ivf%' THEN 'IVF'
				ELSE 'UNKNOWN'
			END AS index_type
		FROM pg_indexes
		WHERE (indexdef LIKE '%hnsw%' OR indexdef LIKE '%ivf%' OR indexdef LIKE '%vector%')
			AND schemaname NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
		ORDER BY schemaname, tablename, indexname
	`
	indexes, err := r.executeQuery(ctx, query, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to query vector indexes: %w", err)
	}

	/* Try to get more detailed information from neurondb schema if available */
	detailedIndexes := make([]interface{}, 0, len(indexes))
	for _, idx := range indexes {
		indexData := map[string]interface{}{
			"schema":     idx["schemaname"],
			"table":      idx["tablename"],
			"name":       idx["indexname"],
			"type":       idx["index_type"],
			"definition": idx["indexdef"],
		}

		/* Try to get index status from neurondb schema */
		statusQuery := `
			SELECT 
				indexname,
				indexstatus,
				indexsize
			FROM neurondb.v_index_status
			WHERE indexname = $1
		`
		if status, err := r.executeQueryOne(ctx, statusQuery, []interface{}{idx["indexname"]}); err == nil {
			indexData["status"] = status["indexstatus"]
			indexData["size"] = status["indexsize"]
		}

		detailedIndexes = append(detailedIndexes, indexData)
	}

	result := map[string]interface{}{
		"indexes": detailedIndexes,
		"count":   len(detailedIndexes),
	}

	return result, nil
}



