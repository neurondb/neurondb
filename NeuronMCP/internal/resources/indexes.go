package resources

import (
	"context"

	"github.com/neurondb/NeuronMCP/internal/database"
)

// IndexesResource provides vector indexes information
type IndexesResource struct {
	*BaseResource
}

// NewIndexesResource creates a new indexes resource
func NewIndexesResource(db *database.Database) *IndexesResource {
	return &IndexesResource{BaseResource: NewBaseResource(db)}
}

// URI returns the resource URI
func (r *IndexesResource) URI() string {
	return "neurondb://indexes"
}

// Name returns the resource name
func (r *IndexesResource) Name() string {
	return "Vector Indexes"
}

// Description returns the resource description
func (r *IndexesResource) Description() string {
	return "Status and information about vector indexes"
}

// MimeType returns the MIME type
func (r *IndexesResource) MimeType() string {
	return "application/json"
}

// GetContent returns the indexes content
func (r *IndexesResource) GetContent(ctx context.Context) (interface{}, error) {
	query := `
		SELECT 
			schemaname,
			tablename,
			indexname,
			indexdef
		FROM pg_indexes
		WHERE indexdef LIKE '%hnsw%' OR indexdef LIKE '%ivf%'
		ORDER BY schemaname, tablename, indexname
	`
	return r.executeQuery(ctx, query, nil)
}

