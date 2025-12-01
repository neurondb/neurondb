package resources

import (
	"context"

	"github.com/neurondb/NeuronMCP/internal/database"
)

// WorkersResource provides worker status information
type WorkersResource struct {
	*BaseResource
}

// NewWorkersResource creates a new workers resource
func NewWorkersResource(db *database.Database) *WorkersResource {
	return &WorkersResource{BaseResource: NewBaseResource(db)}
}

// URI returns the resource URI
func (r *WorkersResource) URI() string {
	return "neurondb://workers"
}

// Name returns the resource name
func (r *WorkersResource) Name() string {
	return "Background Workers Status"
}

// Description returns the resource description
func (r *WorkersResource) Description() string {
	return "Status of background workers"
}

// MimeType returns the MIME type
func (r *WorkersResource) MimeType() string {
	return "application/json"
}

// GetContent returns the workers content
func (r *WorkersResource) GetContent(ctx context.Context) (interface{}, error) {
	query := `SELECT * FROM neurondb.neurondb_workers`
	return r.executeQuery(ctx, query, nil)
}

