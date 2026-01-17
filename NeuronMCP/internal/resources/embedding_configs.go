/*-------------------------------------------------------------------------
 *
 * embedding_configs.go
 *    Embedding configurations resource for NeuronMCP
 *
 * Provides information about embedding model configurations including
 * model name, provider, dimensions, and status.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/resources/embedding_configs.go
 *
 *-------------------------------------------------------------------------
 */

package resources

import (
	"context"

	"github.com/neurondb/NeuronMCP/internal/database"
)

/* EmbeddingConfigsResource provides embedding configuration information */
type EmbeddingConfigsResource struct {
	*BaseResource
}

/* NewEmbeddingConfigsResource creates a new embedding configs resource */
func NewEmbeddingConfigsResource(db *database.Database) *EmbeddingConfigsResource {
	return &EmbeddingConfigsResource{BaseResource: NewBaseResource(db)}
}

/* URI returns the resource URI */
func (r *EmbeddingConfigsResource) URI() string {
	return "neurondb://embedding_configs"
}

/* Name returns the resource name */
func (r *EmbeddingConfigsResource) Name() string {
	return "Embedding Configurations"
}

/* Description returns the resource description */
func (r *EmbeddingConfigsResource) Description() string {
	return "Embedding model configurations with provider, dimensions, and status"
}

/* MimeType returns the MIME type */
func (r *EmbeddingConfigsResource) MimeType() string {
	return "application/json"
}

/* GetContent returns the embedding configs content */
func (r *EmbeddingConfigsResource) GetContent(ctx context.Context) (interface{}, error) {
	/* Try to query from neurondb schema first */
	query := `
		SELECT 
			model_name,
			provider,
			dimensions,
			status,
			api_key_configured,
			created_at,
			updated_at
		FROM neurondb.v_llm_models_ready
		WHERE model_type = 'embedding'
		ORDER BY model_name
	`
	configs, err := r.executeQuery(ctx, query, nil)
	if err != nil {
		/* If neurondb schema doesn't exist or view doesn't exist, return empty result */
		/* This is expected in some setups */
		return map[string]interface{}{
			"configs": []interface{}{},
			"count":   0,
			"note":    "Embedding configurations not available (neurondb schema may not be initialized)",
		}, nil
	}

	/* Format the results */
	formattedConfigs := make([]interface{}, 0, len(configs))
	for _, config := range configs {
		formattedConfigs = append(formattedConfigs, map[string]interface{}{
			"model_name":         config["model_name"],
			"provider":           config["provider"],
			"dimensions":         config["dimensions"],
			"status":            config["status"],
			"api_key_configured": config["api_key_configured"],
			"created_at":         config["created_at"],
			"updated_at":         config["updated_at"],
		})
	}

	result := map[string]interface{}{
		"configs": formattedConfigs,
		"count":   len(formattedConfigs),
	}

	return result, nil
}

