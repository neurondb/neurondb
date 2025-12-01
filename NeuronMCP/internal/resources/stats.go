package resources

import (
	"context"

	"github.com/neurondb/NeuronMCP/internal/database"
)

// VectorStatsResource provides vector statistics
type VectorStatsResource struct {
	*BaseResource
}

// NewVectorStatsResource creates a new vector stats resource
func NewVectorStatsResource(db *database.Database) *VectorStatsResource {
	return &VectorStatsResource{BaseResource: NewBaseResource(db)}
}

// URI returns the resource URI
func (r *VectorStatsResource) URI() string {
	return "neurondb://vector_stats"
}

// Name returns the resource name
func (r *VectorStatsResource) Name() string {
	return "Vector Statistics"
}

// Description returns the resource description
func (r *VectorStatsResource) Description() string {
	return "Aggregate vector statistics"
}

// MimeType returns the MIME type
func (r *VectorStatsResource) MimeType() string {
	return "application/json"
}

// GetContent returns the vector stats content
func (r *VectorStatsResource) GetContent(ctx context.Context) (interface{}, error) {
	query := `SELECT * FROM neurondb.vector_stats`
	return r.executeQuery(ctx, query, nil)
}

// IndexHealthResource provides index health information
type IndexHealthResource struct {
	*BaseResource
}

// NewIndexHealthResource creates a new index health resource
func NewIndexHealthResource(db *database.Database) *IndexHealthResource {
	return &IndexHealthResource{BaseResource: NewBaseResource(db)}
}

// URI returns the resource URI
func (r *IndexHealthResource) URI() string {
	return "neurondb://index_health"
}

// Name returns the resource name
func (r *IndexHealthResource) Name() string {
	return "Index Health"
}

// Description returns the resource description
func (r *IndexHealthResource) Description() string {
	return "Index health dashboard"
}

// MimeType returns the MIME type
func (r *IndexHealthResource) MimeType() string {
	return "application/json"
}

// GetContent returns the index health content
func (r *IndexHealthResource) GetContent(ctx context.Context) (interface{}, error) {
	query := `SELECT * FROM neurondb.index_health`
	return r.executeQuery(ctx, query, nil)
}

