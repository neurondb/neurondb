/*-------------------------------------------------------------------------
 *
 * extensions.go
 *    Extensions resource for NeuronMCP
 *
 * Provides information about installed PostgreSQL extensions including
 * name, version, and schema.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/resources/extensions.go
 *
 *-------------------------------------------------------------------------
 */

package resources

import (
	"context"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/database"
)

/* ExtensionsResource provides extension information */
type ExtensionsResource struct {
	*BaseResource
}

/* NewExtensionsResource creates a new extensions resource */
func NewExtensionsResource(db *database.Database) *ExtensionsResource {
	return &ExtensionsResource{BaseResource: NewBaseResource(db)}
}

/* URI returns the resource URI */
func (r *ExtensionsResource) URI() string {
	return "neurondb://extensions"
}

/* Name returns the resource name */
func (r *ExtensionsResource) Name() string {
	return "PostgreSQL Extensions"
}

/* Description returns the resource description */
func (r *ExtensionsResource) Description() string {
	return "Installed PostgreSQL extensions with version and schema information"
}

/* MimeType returns the MIME type */
func (r *ExtensionsResource) MimeType() string {
	return "application/json"
}

/* GetContent returns the extensions content */
func (r *ExtensionsResource) GetContent(ctx context.Context) (interface{}, error) {
	query := `
		SELECT 
			e.extname AS name,
			e.extversion AS version,
			n.nspname AS schema,
			e.extrelocatable AS relocatable,
			obj_description(e.oid, 'pg_extension') AS comment
		FROM pg_extension e
		LEFT JOIN pg_namespace n ON n.oid = e.extnamespace
		ORDER BY e.extname
	`
	extensions, err := r.executeQuery(ctx, query, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to query extensions: %w", err)
	}

	result := map[string]interface{}{
		"extensions": extensions,
		"count":      len(extensions),
	}

	return result, nil
}



