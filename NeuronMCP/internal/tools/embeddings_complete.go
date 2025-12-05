package tools

import (
	"context"
	"encoding/base64"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

// EmbedImageTool generates image embeddings
type EmbedImageTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewEmbedImageTool creates a new image embedding tool
func NewEmbedImageTool(db *database.Database, logger *logging.Logger) *EmbedImageTool {
	return &EmbedImageTool{
		BaseTool: NewBaseTool(
			"embed_image",
			"Generate image embedding from image bytes",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"image_data": map[string]interface{}{
						"type":        "string",
						"description": "Base64-encoded image data",
					},
					"model": map[string]interface{}{
						"type":        "string",
						"default":     "clip",
						"description": "Model name (default: clip)",
					},
				},
				"required": []interface{}{"image_data"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes the image embedding generation
func (t *EmbedImageTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for embed_image tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	imageData, _ := params["image_data"].(string)
	model := "clip"
	if m, ok := params["model"].(string); ok && m != "" {
		model = m
	}

	if imageData == "" {
		return Error("image_data is required and cannot be empty", "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "image_data",
			"params":    params,
		}), nil
	}

	// Decode base64 image data
	imageBytes, err := base64.StdEncoding.DecodeString(imageData)
	if err != nil {
		return Error(fmt.Sprintf("Invalid base64 image data: %v", err), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "image_data",
			"error":     err.Error(),
		}), nil
	}

	query := "SELECT embed_image($1::bytea, $2::text)::text AS embedding"
	queryParams := []interface{}{imageBytes, model}

	result, err := t.executor.ExecuteQueryOne(ctx, query, queryParams)
	if err != nil {
		t.logger.Error("Image embedding generation failed", err, map[string]interface{}{
			"model": model,
		})
		return Error(fmt.Sprintf("Image embedding generation failed: model='%s', error=%v", model, err), "EXECUTION_ERROR", map[string]interface{}{
			"model": model,
			"error": err.Error(),
		}), nil
	}

	return Success(result, map[string]interface{}{
		"model": model,
		"type":  "image",
	}), nil
}

// EmbedMultimodalTool generates multimodal embeddings
type EmbedMultimodalTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewEmbedMultimodalTool creates a new multimodal embedding tool
func NewEmbedMultimodalTool(db *database.Database, logger *logging.Logger) *EmbedMultimodalTool {
	return &EmbedMultimodalTool{
		BaseTool: NewBaseTool(
			"embed_multimodal",
			"Generate multimodal embedding from text and image",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"text": map[string]interface{}{
						"type":        "string",
						"description": "Text input",
					},
					"image_data": map[string]interface{}{
						"type":        "string",
						"description": "Base64-encoded image data",
					},
					"model": map[string]interface{}{
						"type":        "string",
						"default":     "clip",
						"description": "Model name (default: clip)",
					},
				},
				"required": []interface{}{"text", "image_data"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes the multimodal embedding generation
func (t *EmbedMultimodalTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for embed_multimodal tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	text, _ := params["text"].(string)
	imageData, _ := params["image_data"].(string)
	model := "clip"
	if m, ok := params["model"].(string); ok && m != "" {
		model = m
	}

	if text == "" {
		return Error("text is required and cannot be empty", "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "text",
			"params":    params,
		}), nil
	}

	if imageData == "" {
		return Error("image_data is required and cannot be empty", "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "image_data",
			"params":    params,
		}), nil
	}

	// Decode base64 image data
	imageBytes, err := base64.StdEncoding.DecodeString(imageData)
	if err != nil {
		return Error(fmt.Sprintf("Invalid base64 image data: %v", err), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "image_data",
			"error":     err.Error(),
		}), nil
	}

	query := "SELECT embed_multimodal($1::text, $2::bytea, $3::text)::text AS embedding"
	queryParams := []interface{}{text, imageBytes, model}

	result, err := t.executor.ExecuteQueryOne(ctx, query, queryParams)
	if err != nil {
		t.logger.Error("Multimodal embedding generation failed", err, map[string]interface{}{
			"model": model,
		})
		return Error(fmt.Sprintf("Multimodal embedding generation failed: model='%s', error=%v", model, err), "EXECUTION_ERROR", map[string]interface{}{
			"model": model,
			"error": err.Error(),
		}), nil
	}

	return Success(result, map[string]interface{}{
		"model": model,
		"type":  "multimodal",
	}), nil
}

// EmbedCachedTool generates cached embeddings
type EmbedCachedTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewEmbedCachedTool creates a new cached embedding tool
func NewEmbedCachedTool(db *database.Database, logger *logging.Logger) *EmbedCachedTool {
	return &EmbedCachedTool{
		BaseTool: NewBaseTool(
			"embed_cached",
			"Generate cached text embedding (uses cache if available)",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"text": map[string]interface{}{
						"type":        "string",
						"description": "Text to embed",
					},
					"model": map[string]interface{}{
						"type":        "string",
						"default":     "all-MiniLM-L6-v2",
						"description": "Model name (default: all-MiniLM-L6-v2)",
					},
				},
				"required": []interface{}{"text"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes the cached embedding generation
func (t *EmbedCachedTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for embed_cached tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	text, _ := params["text"].(string)
	model := "all-MiniLM-L6-v2"
	if m, ok := params["model"].(string); ok && m != "" {
		model = m
	}

	if text == "" {
		return Error("text is required and cannot be empty", "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "text",
			"params":    params,
		}), nil
	}

	query := "SELECT embed_cached($1::text, $2::text)::text AS embedding"
	queryParams := []interface{}{text, model}

	result, err := t.executor.ExecuteQueryOne(ctx, query, queryParams)
	if err != nil {
		t.logger.Error("Cached embedding generation failed", err, map[string]interface{}{
			"model": model,
		})
		return Error(fmt.Sprintf("Cached embedding generation failed: model='%s', error=%v", model, err), "EXECUTION_ERROR", map[string]interface{}{
			"model": model,
			"error": err.Error(),
		}), nil
	}

	return Success(result, map[string]interface{}{
		"model": model,
		"type":  "cached",
	}), nil
}

// ConfigureEmbeddingModelTool configures embedding model
type ConfigureEmbeddingModelTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewConfigureEmbeddingModelTool creates a new embedding model configuration tool
func NewConfigureEmbeddingModelTool(db *database.Database, logger *logging.Logger) *ConfigureEmbeddingModelTool {
	return &ConfigureEmbeddingModelTool{
		BaseTool: NewBaseTool(
			"configure_embedding_model",
			"Configure embedding model settings",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"model_name": map[string]interface{}{
						"type":        "string",
						"description": "Model name",
					},
					"config_json": map[string]interface{}{
						"type":        "string",
						"description": "JSON configuration string",
					},
				},
				"required": []interface{}{"model_name", "config_json"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes the embedding model configuration
func (t *ConfigureEmbeddingModelTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for configure_embedding_model tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	modelName, _ := params["model_name"].(string)
	configJSON, _ := params["config_json"].(string)

	if modelName == "" {
		return Error("model_name is required and cannot be empty", "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "model_name",
			"params":    params,
		}), nil
	}

	if configJSON == "" {
		return Error("config_json is required and cannot be empty", "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "config_json",
			"params":    params,
		}), nil
	}

	query := "SELECT configure_embedding_model($1::text, $2::text) AS success"
	queryParams := []interface{}{modelName, configJSON}

	result, err := t.executor.ExecuteQueryOne(ctx, query, queryParams)
	if err != nil {
		t.logger.Error("Embedding model configuration failed", err, map[string]interface{}{
			"model_name": modelName,
		})
		return Error(fmt.Sprintf("Embedding model configuration failed: model_name='%s', error=%v", modelName, err), "EXECUTION_ERROR", map[string]interface{}{
			"model_name": modelName,
			"error":      err.Error(),
		}), nil
	}

	return Success(result, map[string]interface{}{
		"model_name": modelName,
	}), nil
}

// GetEmbeddingModelConfigTool gets embedding model configuration
type GetEmbeddingModelConfigTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewGetEmbeddingModelConfigTool creates a new get embedding model config tool
func NewGetEmbeddingModelConfigTool(db *database.Database, logger *logging.Logger) *GetEmbeddingModelConfigTool {
	return &GetEmbeddingModelConfigTool{
		BaseTool: NewBaseTool(
			"get_embedding_model_config",
			"Get embedding model configuration",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"model_name": map[string]interface{}{
						"type":        "string",
						"description": "Model name",
					},
				},
				"required": []interface{}{"model_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes the get embedding model config
func (t *GetEmbeddingModelConfigTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for get_embedding_model_config tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	modelName, _ := params["model_name"].(string)

	if modelName == "" {
		return Error("model_name is required and cannot be empty", "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "model_name",
			"params":    params,
		}), nil
	}

	query := "SELECT get_embedding_model_config($1::text) AS config"
	queryParams := []interface{}{modelName}

	result, err := t.executor.ExecuteQueryOne(ctx, query, queryParams)
	if err != nil {
		t.logger.Error("Get embedding model config failed", err, map[string]interface{}{
			"model_name": modelName,
		})
		return Error(fmt.Sprintf("Get embedding model config failed: model_name='%s', error=%v", modelName, err), "EXECUTION_ERROR", map[string]interface{}{
			"model_name": modelName,
			"error":      err.Error(),
		}), nil
	}

	return Success(result, map[string]interface{}{
		"model_name": modelName,
	}), nil
}

// ListEmbeddingModelConfigsTool lists all embedding model configurations
type ListEmbeddingModelConfigsTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewListEmbeddingModelConfigsTool creates a new list embedding model configs tool
func NewListEmbeddingModelConfigsTool(db *database.Database, logger *logging.Logger) *ListEmbeddingModelConfigsTool {
	return &ListEmbeddingModelConfigsTool{
		BaseTool: NewBaseTool(
			"list_embedding_model_configs",
			"List all embedding model configurations",
			map[string]interface{}{
				"type":       "object",
				"properties": map[string]interface{}{},
				"required":   []interface{}{},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes the list embedding model configs
func (t *ListEmbeddingModelConfigsTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	query := "SELECT * FROM list_embedding_model_configs()"
	results, err := t.executor.ExecuteQuery(ctx, query, nil)
	if err != nil {
		t.logger.Error("List embedding model configs failed", err, nil)
		return Error(fmt.Sprintf("List embedding model configs failed: error=%v", err), "EXECUTION_ERROR", map[string]interface{}{
			"error": err.Error(),
		}), nil
	}

	return Success(map[string]interface{}{
		"configs": results,
		"count":   len(results),
	}, map[string]interface{}{
		"count": len(results),
	}), nil
}

// DeleteEmbeddingModelConfigTool deletes embedding model configuration
type DeleteEmbeddingModelConfigTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewDeleteEmbeddingModelConfigTool creates a new delete embedding model config tool
func NewDeleteEmbeddingModelConfigTool(db *database.Database, logger *logging.Logger) *DeleteEmbeddingModelConfigTool {
	return &DeleteEmbeddingModelConfigTool{
		BaseTool: NewBaseTool(
			"delete_embedding_model_config",
			"Delete embedding model configuration",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"model_name": map[string]interface{}{
						"type":        "string",
						"description": "Model name to delete",
					},
				},
				"required": []interface{}{"model_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes the delete embedding model config
func (t *DeleteEmbeddingModelConfigTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for delete_embedding_model_config tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	modelName, _ := params["model_name"].(string)

	if modelName == "" {
		return Error("model_name is required and cannot be empty", "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "model_name",
			"params":    params,
		}), nil
	}

	query := "SELECT delete_embedding_model_config($1::text) AS success"
	queryParams := []interface{}{modelName}

	result, err := t.executor.ExecuteQueryOne(ctx, query, queryParams)
	if err != nil {
		t.logger.Error("Delete embedding model config failed", err, map[string]interface{}{
			"model_name": modelName,
		})
		return Error(fmt.Sprintf("Delete embedding model config failed: model_name='%s', error=%v", modelName, err), "EXECUTION_ERROR", map[string]interface{}{
			"model_name": modelName,
			"error":      err.Error(),
		}), nil
	}

	return Success(result, map[string]interface{}{
		"model_name": modelName,
	}), nil
}

