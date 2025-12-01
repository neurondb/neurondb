package resources

import (
	"context"
	"encoding/json"

	"github.com/jackc/pgx/v5"
	"github.com/neurondb/NeuronMCP/internal/database"
)

// Resource is the interface that all resources must implement
type Resource interface {
	URI() string
	Name() string
	Description() string
	MimeType() string
	GetContent(ctx context.Context) (interface{}, error)
}

// BaseResource provides common functionality for resources
type BaseResource struct {
	db *database.Database
}

// NewBaseResource creates a new base resource
func NewBaseResource(db *database.Database) *BaseResource {
	return &BaseResource{db: db}
}

// executeQuery executes a query and returns results
func (r *BaseResource) executeQuery(ctx context.Context, query string, params []interface{}) ([]map[string]interface{}, error) {
	rows, err := r.db.Query(ctx, query, params...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	return scanRowsToMaps(rows)
}

// scanRowsToMaps scans all rows into maps
func scanRowsToMaps(rows pgx.Rows) ([]map[string]interface{}, error) {
	var results []map[string]interface{}

	for rows.Next() {
		row, err := scanRowToMap(rows)
		if err != nil {
			return nil, err
		}
		results = append(results, row)
	}

	return results, rows.Err()
}

// scanRowToMap scans a single row into a map
func scanRowToMap(rows pgx.Rows) (map[string]interface{}, error) {
	fieldDescriptions := rows.FieldDescriptions()
	values := make([]interface{}, len(fieldDescriptions))
	valuePointers := make([]interface{}, len(fieldDescriptions))

	for i := range values {
		valuePointers[i] = &values[i]
	}

	if err := rows.Scan(valuePointers...); err != nil {
		return nil, err
	}

	result := make(map[string]interface{})
	for i, desc := range fieldDescriptions {
		result[string(desc.Name)] = values[i]
	}

	return result, nil
}

// Manager manages all resources
type Manager struct {
	resources map[string]Resource
	db        *database.Database
}

// NewManager creates a new resource manager
func NewManager(db *database.Database) *Manager {
	m := &Manager{
		resources: make(map[string]Resource),
		db:        db,
	}

	// Register built-in resources
	m.Register(NewSchemaResource(db))
	m.Register(NewModelsResource(db))
	m.Register(NewIndexesResource(db))
	m.Register(NewConfigResource(db))
	m.Register(NewWorkersResource(db))
	m.Register(NewVectorStatsResource(db))
	m.Register(NewIndexHealthResource(db))

	return m
}

// Register registers a resource
func (m *Manager) Register(resource Resource) {
	m.resources[resource.URI()] = resource
}

// HandleResource handles a resource request
func (m *Manager) HandleResource(ctx context.Context, uri string) (*ReadResourceResponse, error) {
	resource, exists := m.resources[uri]
	if !exists {
		return nil, &ResourceNotFoundError{URI: uri}
	}

	content, err := resource.GetContent(ctx)
	if err != nil {
		return nil, err
	}

	contentJSON, err := json.MarshalIndent(content, "", "  ")
	if err != nil {
		return nil, err
	}

	return &ReadResourceResponse{
		Contents: []ResourceContent{
			{
				URI:      uri,
				MimeType: resource.MimeType(),
				Text:     string(contentJSON),
			},
		},
	}, nil
}

// ListResources returns all available resources
func (m *Manager) ListResources() []ResourceDefinition {
	definitions := make([]ResourceDefinition, 0, len(m.resources))
	for _, resource := range m.resources {
		definitions = append(definitions, ResourceDefinition{
			URI:         resource.URI(),
			Name:        resource.Name(),
			Description: resource.Description(),
			MimeType:    resource.MimeType(),
		})
	}
	return definitions
}

// ResourceDefinition represents a resource definition
type ResourceDefinition struct {
	URI         string `json:"uri"`
	Name        string `json:"name"`
	Description string `json:"description"`
	MimeType    string `json:"mimeType"`
}

// ReadResourceResponse represents a resource read response
type ReadResourceResponse struct {
	Contents []ResourceContent `json:"contents"`
}

// ResourceContent represents resource content
type ResourceContent struct {
	URI      string `json:"uri"`
	MimeType string `json:"mimeType"`
	Text     string `json:"text"`
}

// ResourceNotFoundError is returned when a resource is not found
type ResourceNotFoundError struct {
	URI string
}

func (e *ResourceNotFoundError) Error() string {
	return "resource not found: " + e.URI
}

