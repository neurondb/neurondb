package resources

import (
	"context"

	"github.com/neurondb/NeuronMCP/internal/database"
)

// ModelsResource provides ML models catalog
type ModelsResource struct {
	*BaseResource
}

// NewModelsResource creates a new models resource
func NewModelsResource(db *database.Database) *ModelsResource {
	return &ModelsResource{BaseResource: NewBaseResource(db)}
}

// URI returns the resource URI
func (r *ModelsResource) URI() string {
	return "neurondb://models"
}

// Name returns the resource name
func (r *ModelsResource) Name() string {
	return "ML Models"
}

// Description returns the resource description
func (r *ModelsResource) Description() string {
	return "Catalog of trained ML models"
}

// MimeType returns the MIME type
func (r *ModelsResource) MimeType() string {
	return "application/json"
}

// GetContent returns the models content
func (r *ModelsResource) GetContent(ctx context.Context) (interface{}, error) {
	query := `
		SELECT 
			model_id,
			algorithm,
			training_table,
			created_at,
			updated_at
		FROM neurondb.ml_models
		ORDER BY model_id DESC
	`
	return r.executeQuery(ctx, query, nil)
}

