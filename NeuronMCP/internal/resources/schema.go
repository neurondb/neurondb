package resources

import (
	"context"

	"github.com/neurondb/NeuronMCP/internal/database"
)

// SchemaResource provides database schema information
type SchemaResource struct {
	*BaseResource
}

// NewSchemaResource creates a new schema resource
func NewSchemaResource(db *database.Database) *SchemaResource {
	return &SchemaResource{BaseResource: NewBaseResource(db)}
}

// URI returns the resource URI
func (r *SchemaResource) URI() string {
	return "neurondb://schema"
}

// Name returns the resource name
func (r *SchemaResource) Name() string {
	return "Database Schema"
}

// Description returns the resource description
func (r *SchemaResource) Description() string {
	return "NeurondB database schema information"
}

// MimeType returns the MIME type
func (r *SchemaResource) MimeType() string {
	return "application/json"
}

// GetContent returns the schema content
func (r *SchemaResource) GetContent(ctx context.Context) (interface{}, error) {
	query := `
		SELECT 
			table_schema,
			table_name,
			column_name,
			data_type,
			udt_name
		FROM information_schema.columns
		WHERE table_schema = 'neurondb' OR table_schema = 'public'
		ORDER BY table_schema, table_name, ordinal_position
	`
	return r.executeQuery(ctx, query, nil)
}


