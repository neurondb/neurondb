/*-------------------------------------------------------------------------
 *
 * workers.go
 *    Database operations
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/resources/workers.go
 *
 *-------------------------------------------------------------------------
 */

package resources

import (
	"context"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/database"
)

/* WorkersResource provides worker status information */
type WorkersResource struct {
	*BaseResource
}

/* NewWorkersResource creates a new workers resource */
func NewWorkersResource(db *database.Database) *WorkersResource {
	return &WorkersResource{BaseResource: NewBaseResource(db)}
}

/* URI returns the resource URI */
func (r *WorkersResource) URI() string {
	return "neurondb://workers"
}

/* Name returns the resource name */
func (r *WorkersResource) Name() string {
	return "Background Workers Status"
}

/* Description returns the resource description */
func (r *WorkersResource) Description() string {
	return "Status of background workers"
}

/* MimeType returns the MIME type */
func (r *WorkersResource) MimeType() string {
	return "application/json"
}

/* GetContent returns the workers content */
func (r *WorkersResource) GetContent(ctx context.Context) (interface{}, error) {
	/* Check if table exists first */
	checkQuery := `
		SELECT EXISTS (
			SELECT FROM information_schema.tables 
			WHERE table_schema = 'neurondb' 
			AND table_name = 'neurondb_workers'
		) AS table_exists
	`
	exists, err := r.executeQueryOne(ctx, checkQuery, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to check if workers table exists: %w", err)
	}
	
	if tableExists, ok := exists["table_exists"].(bool); !ok || !tableExists {
		return []map[string]interface{}{}, nil
	}
	
	query := `SELECT * FROM neurondb.neurondb_workers`
	return r.executeQuery(ctx, query, nil)
}

