package tools

import (
	"context"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

// VectorSearchTool performs vector similarity search
type VectorSearchTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewVectorSearchTool creates a new vector search tool
func NewVectorSearchTool(db *database.Database, logger *logging.Logger) *VectorSearchTool {
	return &VectorSearchTool{
		BaseTool: NewBaseTool(
			"vector_search",
			"Perform vector similarity search using L2, cosine, inner product, L1, Hamming, Chebyshev, or Minkowski distance",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table": map[string]interface{}{
						"type":        "string",
						"description": "Table name containing vectors",
					},
					"vector_column": map[string]interface{}{
						"type":        "string",
						"description": "Name of the vector column",
					},
					"query_vector": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "number"},
						"description": "Query vector for similarity search",
					},
					"limit": map[string]interface{}{
						"type":        "number",
						"default":     10,
						"minimum":     1,
						"maximum":     1000,
						"description": "Maximum number of results",
					},
					"distance_metric": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"l2", "cosine", "inner_product", "l1", "hamming", "chebyshev", "minkowski"},
						"default":     "l2",
						"description": "Distance metric to use",
					},
					"additional_columns": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "string"},
						"description": "Additional columns to return in results",
					},
				},
				"required": []interface{}{"table", "vector_column", "query_vector"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes the vector search
func (t *VectorSearchTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error("Invalid parameters", "VALIDATION_ERROR", map[string]interface{}{"errors": errors}), nil
	}

	table, _ := params["table"].(string)
	vectorColumn, _ := params["vector_column"].(string)
	queryVector, _ := params["query_vector"].([]interface{})
	limit := 10
	if l, ok := params["limit"].(float64); ok {
		limit = int(l)
	}
	distanceMetric := "l2"
	if dm, ok := params["distance_metric"].(string); ok {
		distanceMetric = dm
	}
	additionalColumns := []interface{}{}
	if ac, ok := params["additional_columns"].([]interface{}); ok {
		additionalColumns = ac
	}

	results, err := t.executor.ExecuteVectorSearch(ctx, table, vectorColumn, queryVector, distanceMetric, limit, additionalColumns)
	if err != nil {
		t.logger.Error("Vector search failed", err, params)
		return Error(fmt.Sprintf("Vector search failed: %v", err), "SEARCH_ERROR", nil), nil
	}

	return Success(results, map[string]interface{}{
		"count":          len(results),
		"distance_metric": distanceMetric,
	}), nil
}

// VectorSearchL2Tool performs L2 distance vector search
type VectorSearchL2Tool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewVectorSearchL2Tool creates a new L2 vector search tool
func NewVectorSearchL2Tool(db *database.Database, logger *logging.Logger) *VectorSearchL2Tool {
	return &VectorSearchL2Tool{
		BaseTool: NewBaseTool(
			"vector_search_l2",
			"Perform vector similarity search using L2 (Euclidean) distance",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table":         map[string]interface{}{"type": "string"},
					"vector_column": map[string]interface{}{"type": "string"},
					"query_vector":  map[string]interface{}{"type": "array", "items": map[string]interface{}{"type": "number"}},
					"limit":         map[string]interface{}{"type": "number", "default": 10, "minimum": 1, "maximum": 1000},
				},
				"required": []interface{}{"table", "vector_column", "query_vector"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes the L2 vector search
func (t *VectorSearchL2Tool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error("Invalid parameters", "VALIDATION_ERROR", map[string]interface{}{"errors": errors}), nil
	}

	table, _ := params["table"].(string)
	vectorColumn, _ := params["vector_column"].(string)
	queryVector, _ := params["query_vector"].([]interface{})
	limit := 10
	if l, ok := params["limit"].(float64); ok {
		limit = int(l)
	}

	results, err := t.executor.ExecuteVectorSearch(ctx, table, vectorColumn, queryVector, "l2", limit, nil)
	if err != nil {
		t.logger.Error("L2 vector search failed", err, params)
		return Error(fmt.Sprintf("L2 vector search failed: %v", err), "SEARCH_ERROR", nil), nil
	}

	return Success(results, map[string]interface{}{
		"count":          len(results),
		"distance_metric": "l2",
	}), nil
}

// VectorSearchCosineTool performs cosine distance vector search
type VectorSearchCosineTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewVectorSearchCosineTool creates a new cosine vector search tool
func NewVectorSearchCosineTool(db *database.Database, logger *logging.Logger) *VectorSearchCosineTool {
	return &VectorSearchCosineTool{
		BaseTool: NewBaseTool(
			"vector_search_cosine",
			"Perform vector similarity search using cosine distance",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table":         map[string]interface{}{"type": "string"},
					"vector_column": map[string]interface{}{"type": "string"},
					"query_vector":  map[string]interface{}{"type": "array", "items": map[string]interface{}{"type": "number"}},
					"limit":         map[string]interface{}{"type": "number", "default": 10, "minimum": 1, "maximum": 1000},
				},
				"required": []interface{}{"table", "vector_column", "query_vector"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes the cosine vector search
func (t *VectorSearchCosineTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error("Invalid parameters", "VALIDATION_ERROR", map[string]interface{}{"errors": errors}), nil
	}

	table, _ := params["table"].(string)
	vectorColumn, _ := params["vector_column"].(string)
	queryVector, _ := params["query_vector"].([]interface{})
	limit := 10
	if l, ok := params["limit"].(float64); ok {
		limit = int(l)
	}

	results, err := t.executor.ExecuteVectorSearch(ctx, table, vectorColumn, queryVector, "cosine", limit, nil)
	if err != nil {
		t.logger.Error("Cosine vector search failed", err, params)
		return Error(fmt.Sprintf("Cosine vector search failed: %v", err), "SEARCH_ERROR", nil), nil
	}

	return Success(results, map[string]interface{}{
		"count":          len(results),
		"distance_metric": "cosine",
	}), nil
}

// VectorSearchInnerProductTool performs inner product distance vector search
type VectorSearchInnerProductTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewVectorSearchInnerProductTool creates a new inner product vector search tool
func NewVectorSearchInnerProductTool(db *database.Database, logger *logging.Logger) *VectorSearchInnerProductTool {
	return &VectorSearchInnerProductTool{
		BaseTool: NewBaseTool(
			"vector_search_inner_product",
			"Perform vector similarity search using inner product distance",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table":         map[string]interface{}{"type": "string"},
					"vector_column": map[string]interface{}{"type": "string"},
					"query_vector":  map[string]interface{}{"type": "array", "items": map[string]interface{}{"type": "number"}},
					"limit":         map[string]interface{}{"type": "number", "default": 10, "minimum": 1, "maximum": 1000},
				},
				"required": []interface{}{"table", "vector_column", "query_vector"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes the inner product vector search
func (t *VectorSearchInnerProductTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error("Invalid parameters", "VALIDATION_ERROR", map[string]interface{}{"errors": errors}), nil
	}

	table, _ := params["table"].(string)
	vectorColumn, _ := params["vector_column"].(string)
	queryVector, _ := params["query_vector"].([]interface{})
	limit := 10
	if l, ok := params["limit"].(float64); ok {
		limit = int(l)
	}

	results, err := t.executor.ExecuteVectorSearch(ctx, table, vectorColumn, queryVector, "inner_product", limit, nil)
	if err != nil {
		t.logger.Error("Inner product vector search failed", err, params)
		return Error(fmt.Sprintf("Inner product vector search failed: %v", err), "SEARCH_ERROR", nil), nil
	}

	return Success(results, map[string]interface{}{
		"count":          len(results),
		"distance_metric": "inner_product",
	}), nil
}

// GenerateEmbeddingTool generates text embeddings
type GenerateEmbeddingTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewGenerateEmbeddingTool creates a new embedding generation tool
func NewGenerateEmbeddingTool(db *database.Database, logger *logging.Logger) *GenerateEmbeddingTool {
	return &GenerateEmbeddingTool{
		BaseTool: NewBaseTool(
			"generate_embedding",
			"Generate text embedding using configured model",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"text": map[string]interface{}{
						"type":        "string",
						"description": "Text to embed",
					},
					"model": map[string]interface{}{
						"type":        "string",
						"description": "Model name (optional, uses default if not specified)",
					},
				},
				"required": []interface{}{"text"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes the embedding generation
func (t *GenerateEmbeddingTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error("Invalid parameters", "VALIDATION_ERROR", map[string]interface{}{"errors": errors}), nil
	}

	text, _ := params["text"].(string)
	model, _ := params["model"].(string)

	query := "SELECT embed_text($1) AS embedding"
	queryParams := []interface{}{text}
	if model != "" {
		query = "SELECT embed_text($1, $2) AS embedding"
		queryParams = append(queryParams, model)
	}

	result, err := t.executor.ExecuteQueryOne(ctx, query, queryParams)
	if err != nil {
		t.logger.Error("Embedding generation failed", err, params)
		return Error(fmt.Sprintf("Embedding generation failed: %v", err), "EMBEDDING_ERROR", nil), nil
	}

	modelName := model
	if modelName == "" {
		modelName = "default"
	}

	return Success(result, map[string]interface{}{"model": modelName}), nil
}

// BatchEmbeddingTool generates embeddings for multiple texts
type BatchEmbeddingTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewBatchEmbeddingTool creates a new batch embedding tool
func NewBatchEmbeddingTool(db *database.Database, logger *logging.Logger) *BatchEmbeddingTool {
	return &BatchEmbeddingTool{
		BaseTool: NewBaseTool(
			"batch_embedding",
			"Generate embeddings for multiple texts efficiently",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"texts": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "string"},
						"description": "Array of texts to embed",
						"minItems":    1,
						"maxItems":    1000,
					},
					"model": map[string]interface{}{
						"type":        "string",
						"description": "Model name (optional)",
					},
				},
				"required": []interface{}{"texts"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes the batch embedding
func (t *BatchEmbeddingTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error("Invalid parameters", "VALIDATION_ERROR", map[string]interface{}{"errors": errors}), nil
	}

	texts, _ := params["texts"].([]interface{})
	model, _ := params["model"].(string)

	query := "SELECT embed_text_batch($1) AS embeddings"
	queryParams := []interface{}{texts}
	if model != "" {
		query = "SELECT embed_text_batch($1, $2) AS embeddings"
		queryParams = append(queryParams, model)
	}

	result, err := t.executor.ExecuteQueryOne(ctx, query, queryParams)
	if err != nil {
		t.logger.Error("Batch embedding failed", err, params)
		return Error(fmt.Sprintf("Batch embedding failed: %v", err), "EMBEDDING_ERROR", nil), nil
	}

	modelName := model
	if modelName == "" {
		modelName = "default"
	}

	return Success(result, map[string]interface{}{
		"count": len(texts),
		"model": modelName,
	}), nil
}

