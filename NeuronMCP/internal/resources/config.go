package resources

import (
	"context"

	"github.com/neurondb/NeuronMCP/internal/database"
)

// ConfigResource provides configuration information
type ConfigResource struct {
	*BaseResource
}

// NewConfigResource creates a new config resource
func NewConfigResource(db *database.Database) *ConfigResource {
	return &ConfigResource{BaseResource: NewBaseResource(db)}
}

// URI returns the resource URI
func (r *ConfigResource) URI() string {
	return "neurondb://config"
}

// Name returns the resource name
func (r *ConfigResource) Name() string {
	return "Neurondb Configuration"
}

// Description returns the resource description
func (r *ConfigResource) Description() string {
	return "Current Neurondb configuration settings"
}

// MimeType returns the MIME type
func (r *ConfigResource) MimeType() string {
	return "application/json"
}

// GetContent returns the config content
func (r *ConfigResource) GetContent(ctx context.Context) (interface{}, error) {
	query := `
		SELECT 
			name,
			setting,
			unit,
			category
		FROM pg_settings
		WHERE name LIKE 'neurondb%'
		ORDER BY name
	`
	return r.executeQuery(ctx, query, nil)
}

