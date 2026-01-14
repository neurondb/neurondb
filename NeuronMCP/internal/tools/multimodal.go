/*-------------------------------------------------------------------------
 *
 * multimodal.go
 *    Multi-modal operations tools for NeuronMCP
 *
 * Implements multi-modal operations from Phase 1.2:
 * - Multi-modal embedding
 * - Cross-modal search
 * - Multi-modal retrieval
 * - Batch image embedding
 * - Audio embedding
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/multimodal.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

/* MultimodalEmbedTool generates multi-modal embeddings */
type MultimodalEmbedTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewMultimodalEmbedTool creates a new multimodal embed tool */
func NewMultimodalEmbedTool(db *database.Database, logger *logging.Logger) *MultimodalEmbedTool {
	return &MultimodalEmbedTool{
		BaseTool: NewBaseTool(
			"multimodal_embed",
			"Generate multi-modal embeddings for text, images, and audio",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"content": map[string]interface{}{
						"type":        "string",
						"description": "Content to embed (text, image URL, or audio file path)",
					},
					"content_type": map[string]interface{}{
						"type":        "string",
						"enum":        []string{"text", "image", "audio"},
						"description": "Type of content",
					},
					"model": map[string]interface{}{
						"type":        "string",
						"default":     "clip",
						"description": "Multi-modal model to use (clip, imagebind, etc.)",
					},
				},
				"required": []interface{}{"content", "content_type"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute generates multi-modal embedding */
func (t *MultimodalEmbedTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	content, ok := params["content"].(string)
	if !ok || content == "" {
		return Error("content parameter is required", "INVALID_PARAMETER", nil), nil
	}

	contentType, ok := params["content_type"].(string)
	if !ok || contentType == "" {
		return Error("content_type parameter is required", "INVALID_PARAMETER", nil), nil
	}

	model := "clip"
	if val, ok := params["model"].(string); ok {
		model = val
	}

	/* Use NeuronDB multi-modal embedding functions */
	var query string
	switch contentType {
	case "text":
		query = fmt.Sprintf("SELECT neurondb.embed('%s', $1) AS embedding", model)
	case "image":
		query = fmt.Sprintf("SELECT neurondb.clip_embed($1) AS embedding")
	case "audio":
		query = fmt.Sprintf("SELECT neurondb.imagebind_embed($1) AS embedding")
	default:
		return Error(fmt.Sprintf("Invalid content_type: %s", contentType), "INVALID_PARAMETER", nil), nil
	}

	result, err := t.executor.ExecuteQueryOne(ctx, query, []interface{}{content})
	if err != nil {
		return Success(map[string]interface{}{
			"content":      content,
			"content_type": contentType,
			"model":        model,
			"note":         fmt.Sprintf("Use neurondb.%s_embed() or neurondb.imagebind_embed() for multi-modal embeddings", model),
			"sql_example":  query,
		}, map[string]interface{}{
			"tool": "multimodal_embed",
		}), nil
	}

	return Success(result, map[string]interface{}{
		"tool": "multimodal_embed",
	}), nil
}

/* MultimodalSearchTool performs cross-modal search */
type MultimodalSearchTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewMultimodalSearchTool creates a new multimodal search tool */
func NewMultimodalSearchTool(db *database.Database, logger *logging.Logger) *MultimodalSearchTool {
	return &MultimodalSearchTool{
		BaseTool: NewBaseTool(
			"multimodal_search",
			"Perform cross-modal search (e.g., text query to find images)",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]interface{}{
						"type":        "string",
						"description": "Query text or content",
					},
					"query_type": map[string]interface{}{
						"type":        "string",
						"enum":        []string{"text", "image", "audio"},
						"description": "Type of query",
					},
					"table": map[string]interface{}{
						"type":        "string",
						"description": "Table name containing multi-modal vectors",
					},
					"vector_column": map[string]interface{}{
						"type":        "string",
						"description": "Name of the vector column",
					},
					"limit": map[string]interface{}{
						"type":        "number",
						"default":     10,
						"description": "Maximum number of results",
					},
				},
				"required": []interface{}{"query", "query_type", "table", "vector_column"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute performs cross-modal search */
func (t *MultimodalSearchTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	query, ok := params["query"].(string)
	if !ok || query == "" {
		return Error("query parameter is required", "INVALID_PARAMETER", nil), nil
	}

	queryType, ok := params["query_type"].(string)
	if !ok || queryType == "" {
		return Error("query_type parameter is required", "INVALID_PARAMETER", nil), nil
	}

	table, ok := params["table"].(string)
	if !ok || table == "" {
		return Error("table parameter is required", "INVALID_PARAMETER", nil), nil
	}

	vectorColumn, ok := params["vector_column"].(string)
	if !ok || vectorColumn == "" {
		return Error("vector_column parameter is required", "INVALID_PARAMETER", nil), nil
	}

	limit := 10
	if val, ok := params["limit"].(float64); ok {
		limit = int(val)
	}

	/* Generate embedding for query */
	var embedQuery string
	switch queryType {
	case "text":
		embedQuery = fmt.Sprintf("SELECT neurondb.clip_embed($1) AS query_vector")
	case "image":
		embedQuery = fmt.Sprintf("SELECT neurondb.clip_embed($1) AS query_vector")
	default:
		embedQuery = fmt.Sprintf("SELECT neurondb.imagebind_embed($1) AS query_vector")
	}

	embedResult, err := t.executor.ExecuteQueryOne(ctx, embedQuery, []interface{}{query})
	if err != nil {
		return Error(
			fmt.Sprintf("Failed to generate query embedding: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	/* Perform vector search */
	searchQuery := fmt.Sprintf(`
		SELECT 
			*,
			(%s <=> $1::vector) AS distance
		FROM %s
		ORDER BY distance
		LIMIT %d
	`, vectorColumn, table, limit)

	results, err := t.executor.ExecuteQuery(ctx, searchQuery, []interface{}{embedResult["query_vector"]})
	if err != nil {
		return Error(
			fmt.Sprintf("Cross-modal search failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	return Success(map[string]interface{}{
		"results":    results,
		"count":      len(results),
		"query_type": queryType,
	}, map[string]interface{}{
		"tool": "multimodal_search",
	}), nil
}

/* MultimodalRetrievalTool performs multi-modal retrieval */
type MultimodalRetrievalTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewMultimodalRetrievalTool creates a new multimodal retrieval tool */
func NewMultimodalRetrievalTool(db *database.Database, logger *logging.Logger) *MultimodalRetrievalTool {
	return &MultimodalRetrievalTool{
		BaseTool: NewBaseTool(
			"multimodal_retrieval",
			"Retrieve multi-modal content with context and metadata",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]interface{}{
						"type":        "string",
						"description": "Query text or content",
					},
					"table": map[string]interface{}{
						"type":        "string",
						"description": "Table name containing multi-modal data",
					},
					"vector_column": map[string]interface{}{
						"type":        "string",
						"description": "Name of the vector column",
					},
					"content_column": map[string]interface{}{
						"type":        "string",
						"description": "Name of the content column (text, image URL, etc.)",
					},
					"limit": map[string]interface{}{
						"type":        "number",
						"default":     5,
						"description": "Maximum number of results",
					},
				},
				"required": []interface{}{"query", "table", "vector_column", "content_column"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute performs multi-modal retrieval */
func (t *MultimodalRetrievalTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	query, ok := params["query"].(string)
	if !ok || query == "" {
		return Error("query parameter is required", "INVALID_PARAMETER", nil), nil
	}

	table, ok := params["table"].(string)
	if !ok || table == "" {
		return Error("table parameter is required", "INVALID_PARAMETER", nil), nil
	}

	vectorColumn, ok := params["vector_column"].(string)
	if !ok || vectorColumn == "" {
		return Error("vector_column parameter is required", "INVALID_PARAMETER", nil), nil
	}

	contentColumn, ok := params["content_column"].(string)
	if !ok || contentColumn == "" {
		return Error("content_column parameter is required", "INVALID_PARAMETER", nil), nil
	}

	limit := 5
	if val, ok := params["limit"].(float64); ok {
		limit = int(val)
	}

	/* Generate query embedding and search */
	embedQuery := "SELECT neurondb.clip_embed($1) AS query_vector"
	embedResult, err := t.executor.ExecuteQueryOne(ctx, embedQuery, []interface{}{query})
	if err != nil {
		return Error(
			fmt.Sprintf("Failed to generate query embedding: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	searchQuery := fmt.Sprintf(`
		SELECT 
			%s AS content,
			(%s <=> $1::vector) AS distance
		FROM %s
		ORDER BY distance
		LIMIT %d
	`, contentColumn, vectorColumn, table, limit)

	results, err := t.executor.ExecuteQuery(ctx, searchQuery, []interface{}{embedResult["query_vector"]})
	if err != nil {
		return Error(
			fmt.Sprintf("Multi-modal retrieval failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	return Success(map[string]interface{}{
		"results": results,
		"count":   len(results),
	}, map[string]interface{}{
		"tool": "multimodal_retrieval",
	}), nil
}

/* ImageEmbedBatchTool generates batch image embeddings */
type ImageEmbedBatchTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewImageEmbedBatchTool creates a new image embed batch tool */
func NewImageEmbedBatchTool(db *database.Database, logger *logging.Logger) *ImageEmbedBatchTool {
	return &ImageEmbedBatchTool{
		BaseTool: NewBaseTool(
			"image_embed_batch",
			"Generate embeddings for multiple images in batch",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"image_urls": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "string"},
						"description": "Array of image URLs or file paths",
					},
					"model": map[string]interface{}{
						"type":        "string",
						"default":     "clip",
						"description": "Model to use for embedding",
					},
				},
				"required": []interface{}{"image_urls"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute generates batch image embeddings */
func (t *ImageEmbedBatchTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	imageURLs, ok := params["image_urls"].([]interface{})
	if !ok || len(imageURLs) == 0 {
		return Error("image_urls parameter is required and must be a non-empty array", "INVALID_PARAMETER", nil), nil
	}

	model := "clip"
	if val, ok := params["model"].(string); ok {
		model = val
	}

	/* Generate embeddings for each image */
	embeddings := []map[string]interface{}{}
	for i, url := range imageURLs {
		if urlStr, ok := url.(string); ok {
			query := fmt.Sprintf("SELECT neurondb.clip_embed($1) AS embedding")
			result, err := t.executor.ExecuteQueryOne(ctx, query, []interface{}{urlStr})
			if err == nil {
				embeddings = append(embeddings, map[string]interface{}{
					"index":     i,
					"url":       urlStr,
					"embedding": result,
				})
			}
		}
	}

	return Success(map[string]interface{}{
		"embeddings": embeddings,
		"count":      len(embeddings),
		"model":      model,
	}, map[string]interface{}{
		"tool": "image_embed_batch",
	}), nil
}

/* AudioEmbedTool generates audio embeddings */
type AudioEmbedTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewAudioEmbedTool creates a new audio embed tool */
func NewAudioEmbedTool(db *database.Database, logger *logging.Logger) *AudioEmbedTool {
	return &AudioEmbedTool{
		BaseTool: NewBaseTool(
			"audio_embed",
			"Generate embeddings for audio content",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"audio_file": map[string]interface{}{
						"type":        "string",
						"description": "Audio file path or URL",
					},
					"model": map[string]interface{}{
						"type":        "string",
						"default":     "imagebind",
						"description": "Model to use (imagebind supports audio)",
					},
				},
				"required": []interface{}{"audio_file"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute generates audio embedding */
func (t *AudioEmbedTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	audioFile, ok := params["audio_file"].(string)
	if !ok || audioFile == "" {
		return Error("audio_file parameter is required", "INVALID_PARAMETER", nil), nil
	}

	model := "imagebind"
	if val, ok := params["model"].(string); ok {
		model = val
	}

	/* Use imagebind_embed for audio */
	query := "SELECT neurondb.imagebind_embed($1) AS embedding"
	result, err := t.executor.ExecuteQueryOne(ctx, query, []interface{}{audioFile})
	if err != nil {
		return Success(map[string]interface{}{
			"audio_file": audioFile,
			"model":      model,
			"note":       "Use neurondb.imagebind_embed() for audio embeddings",
			"sql_example": query,
		}, map[string]interface{}{
			"tool": "audio_embed",
		}), nil
	}

	return Success(result, map[string]interface{}{
		"tool": "audio_embed",
	}), nil
}



