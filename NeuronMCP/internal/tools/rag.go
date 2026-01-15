/*-------------------------------------------------------------------------
 *
 * rag.go
 *    Tool implementation for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/rag.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"fmt"
	"strings"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/validation"
)

/* ProcessDocumentTool processes a document for RAG */
type ProcessDocumentTool struct {
	*BaseTool
	executor     *QueryExecutor
	logger       *logging.Logger
	configHelper *database.ConfigHelper
}

/* NewProcessDocumentTool creates a new process document tool */
func NewProcessDocumentTool(db *database.Database, logger *logging.Logger) *ProcessDocumentTool {
	return &ProcessDocumentTool{
		BaseTool: NewBaseTool(
			"postgresql_process_document",
			"Process a document: chunk text and generate embeddings",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"text": map[string]interface{}{
						"type":        "string",
						"description": "Document text to process",
					},
					"chunk_size": map[string]interface{}{
						"type":        "number",
						"default":     500,
						"minimum":     1,
						"description": "Chunk size in characters",
					},
					"overlap": map[string]interface{}{
						"type":        "number",
						"default":     50,
						"minimum":     0,
						"description": "Overlap between chunks",
					},
					"generate_embeddings": map[string]interface{}{
						"type":        "boolean",
						"default":     true,
						"description": "Whether to generate embeddings for chunks",
					},
				},
				"required": []interface{}{"text"},
			},
		),
		executor:     NewQueryExecutor(db),
		logger:       logger,
		configHelper: database.NewConfigHelper(db),
	}
}

/* Execute executes the document processing */
func (t *ProcessDocumentTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for neurondb_process_document tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	/* Accept both "text" and "document" parameters */
	text, _ := params["text"].(string)
	if text == "" {
		text, _ = params["document"].(string)
	}
	chunkSize := 500
	if c, ok := params["chunk_size"].(float64); ok {
		chunkSize = int(c)
	}
	overlap := 50
	/* Accept both "overlap" and "chunk_overlap" parameters */
	if o, ok := params["overlap"].(float64); ok {
		overlap = int(o)
	} else if o, ok := params["chunk_overlap"].(float64); ok {
		overlap = int(o)
	}
	generateEmbeddings := true
	if g, ok := params["generate_embeddings"].(bool); ok {
		generateEmbeddings = g
	}

	/* Validate text is required */
	if err := validation.ValidateRequired(text, "text"); err != nil {
		return Error(fmt.Sprintf("Invalid text parameter: %v", err), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "text",
			"error":     err.Error(),
			"params":    params,
		}), nil
	}

	/* Validate text length (max 10MB) */
	textLen := len(text)
	if err := validation.ValidateMaxLength(text, "text", 10*1024*1024); err != nil {
		return Error(fmt.Sprintf("Invalid text parameter: %v", err), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "text",
			"error":     err.Error(),
			"params":    params,
		}), nil
	}

	/* Validate chunk_size */
	if err := validation.ValidatePositive(chunkSize, "chunk_size"); err != nil {
		return Error(fmt.Sprintf("Invalid chunk_size parameter: %v", err), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":   "chunk_size",
			"text_length": textLen,
			"error":       err.Error(),
			"params":      params,
		}), nil
	}

	/* Validate overlap */
	if err := validation.ValidateNonNegative(overlap, "overlap"); err != nil {
		return Error(fmt.Sprintf("Invalid overlap parameter: %v", err), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":   "overlap",
			"text_length": textLen,
			"chunk_size":  chunkSize,
			"error":       err.Error(),
			"params":      params,
		}), nil
	}

	/* Validate overlap < chunkSize */
	if overlap >= chunkSize {
		return Error(fmt.Sprintf("overlap must be less than chunk_size: overlap=%d, chunk_size=%d", overlap, chunkSize), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":   "overlap",
			"text_length": textLen,
			"chunk_size":  chunkSize,
			"overlap":     overlap,
			"params":      params,
		}), nil
	}

  /* Use NeuronDB's unified chunking function: neurondb.chunk(document_text, chunk_size, chunk_overlap, method) */
	query := `SELECT json_agg(json_build_object('chunk_id', chunk_id, 'chunk_text', chunk_text, 'start_pos', start_pos, 'end_pos', end_pos)) AS chunks FROM neurondb.chunk($1, $2, $3, 'fixed')`
	result, err := t.executor.ExecuteQueryOne(ctx, query, []interface{}{text, chunkSize, overlap})
	if err != nil {
		t.logger.Error("Document processing failed", err, params)
		return Error(fmt.Sprintf("Document processing execution failed: text_length=%d, chunk_size=%d, overlap=%d, generate_embeddings=%v, error=%v", textLen, chunkSize, overlap, generateEmbeddings, err), "RAG_ERROR", map[string]interface{}{
			"text_length":        textLen,
			"chunk_size":          chunkSize,
			"overlap":             overlap,
			"generate_embeddings": generateEmbeddings,
			"error":               err.Error(),
		}), nil
	}

  /* If embeddings requested, generate them for each chunk */
	if generateEmbeddings {
		chunksInterface, ok := result["chunks"]
		if !ok {
			t.logger.Warn("No chunks in result, skipping embedding generation", params)
		} else if chunksArray, ok := chunksInterface.([]interface{}); ok {
			/* Get default model if available */
			modelName := "default"
			if defaultModel, err := t.configHelper.GetDefaultModel(ctx, "embedding"); err == nil && defaultModel != "" {
				modelName = defaultModel
			}

			/* Generate embeddings for each chunk */
			chunksWithEmbeddings := make([]map[string]interface{}, 0, len(chunksArray))
			for i, chunkInterface := range chunksArray {
				chunkMap, ok := chunkInterface.(map[string]interface{})
				if !ok {
					t.logger.Warn(fmt.Sprintf("Invalid chunk format at index %d, skipping", i), params)
					continue
				}

				chunkText, ok := chunkMap["chunk_text"].(string)
				if !ok || chunkText == "" {
					t.logger.Warn(fmt.Sprintf("Missing or invalid chunk_text at index %d, skipping", i), params)
					continue
				}

				/* Generate embedding for this chunk */
				embedQuery := `SELECT embed_text($1::text, $2::text)::text AS embedding`
				embedResult, err := t.executor.ExecuteQueryOne(ctx, embedQuery, []interface{}{chunkText, modelName})
				if err != nil {
					t.logger.Warn(fmt.Sprintf("Failed to generate embedding for chunk %d", i), map[string]interface{}{
						"chunk_index": i,
						"chunk_text_length": len(chunkText),
						"model": modelName,
						"error": err.Error(),
					})
					/* Continue without embedding for this chunk */
					chunkMap["embedding"] = nil
				} else {
					/* Extract embedding string */
					if embStr, ok := embedResult["embedding"].(string); ok {
						chunkMap["embedding"] = embStr
					} else if embArr, ok := embedResult["embedding"].([]interface{}); ok {
						/* Convert array to vector string format */
						parts := make([]string, 0, len(embArr))
						for _, v := range embArr {
							if f, ok := v.(float64); ok {
								parts = append(parts, fmt.Sprintf("%g", f))
							} else if f, ok := v.(float32); ok {
								parts = append(parts, fmt.Sprintf("%g", f))
							} else {
								parts = append(parts, fmt.Sprintf("%v", v))
							}
						}
						chunkMap["embedding"] = "[" + strings.Join(parts, ",") + "]"
					} else {
						t.logger.Warn(fmt.Sprintf("Invalid embedding format for chunk %d", i), map[string]interface{}{
							"chunk_index": i,
							"embedding_type": fmt.Sprintf("%T", embedResult["embedding"]),
						})
						chunkMap["embedding"] = nil
					}
				}

				chunksWithEmbeddings = append(chunksWithEmbeddings, chunkMap)
			}

			/* Update result with chunks that have embeddings */
			result["chunks"] = chunksWithEmbeddings
		}
	}

	return Success(result, map[string]interface{}{
		"chunk_size": chunkSize,
		"overlap":    overlap,
		"embeddings_generated": generateEmbeddings,
	}), nil
}

/* RetrieveContextTool retrieves context for RAG */
type RetrieveContextTool struct {
	*BaseTool
	executor     *QueryExecutor
	logger       *logging.Logger
	configHelper *database.ConfigHelper
}

/* NewRetrieveContextTool creates a new retrieve context tool */
func NewRetrieveContextTool(db *database.Database, logger *logging.Logger) *RetrieveContextTool {
	return &RetrieveContextTool{
		BaseTool: NewBaseTool(
			"postgresql_retrieve_context",
			"Retrieve relevant context using vector search",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]interface{}{
						"type":        "string",
						"description": "Query text",
					},
					"table": map[string]interface{}{
						"type":        "string",
						"description": "Table name containing documents",
					},
					"vector_column": map[string]interface{}{
						"type":        "string",
						"description": "Vector column name",
					},
					"limit": map[string]interface{}{
						"type":        "number",
						"default":     5,
						"minimum":     1,
						"maximum":     100,
						"description": "Number of results to return",
					},
				},
				"required": []interface{}{"query", "table", "vector_column"},
			},
		),
		executor:     NewQueryExecutor(db),
		logger:       logger,
		configHelper: database.NewConfigHelper(db),
	}
}

/* Execute executes the context retrieval */
func (t *RetrieveContextTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for neurondb_retrieve_context tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	queryText, _ := params["query"].(string)
	table, _ := params["table"].(string)
	vectorColumn, _ := params["vector_column"].(string)
	limit := 5
	if l, ok := params["limit"].(float64); ok {
		limit = int(l)
	}

	if queryText == "" {
		return Error("query parameter is required and cannot be empty for neurondb_retrieve_context tool", "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "query",
			"params":    params,
		}), nil
	}

	if table == "" {
		return Error(fmt.Sprintf("table parameter is required and cannot be empty for neurondb_retrieve_context tool: query_length=%d", len(queryText)), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":   "table",
			"query_length": len(queryText),
			"params":      params,
		}), nil
	}

	if vectorColumn == "" {
		return Error(fmt.Sprintf("vector_column parameter is required and cannot be empty for neurondb_retrieve_context tool: query_length=%d, table='%s'", len(queryText), table), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":   "vector_column",
			"query_length": len(queryText),
			"table":       table,
			"params":      params,
		}), nil
	}

	if limit <= 0 {
		return Error(fmt.Sprintf("limit must be greater than 0 for neurondb_retrieve_context tool: query_length=%d, table='%s', vector_column='%s', received limit=%d", len(queryText), table, vectorColumn, limit), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":     "limit",
			"query_length":  len(queryText),
			"table":         table,
			"vector_column": vectorColumn,
			"limit":         limit,
			"params":        params,
		}), nil
	}

	if limit > 100 {
		return Error(fmt.Sprintf("limit exceeds maximum value of 100 for neurondb_retrieve_context tool: query_length=%d, table='%s', vector_column='%s', received limit=%d", len(queryText), table, vectorColumn, limit), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":     "limit",
			"query_length":  len(queryText),
			"table":         table,
			"vector_column": vectorColumn,
			"limit":         limit,
			"max_limit":     100,
			"params":        params,
		}), nil
	}

	/* Validate identifiers to prevent SQL injection */
	if err := validation.ValidateSQLIdentifier(table, "table"); err != nil {
		return Error(fmt.Sprintf("Invalid table name for neurondb_retrieve_context tool: %v", err), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":   "table",
			"query_length": len(queryText),
			"error":       err.Error(),
			"params":      params,
		}), nil
	}
	if err := validation.ValidateSQLIdentifier(vectorColumn, "vector_column"); err != nil {
		return Error(fmt.Sprintf("Invalid vector column name for neurondb_retrieve_context tool: %v", err), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":   "vector_column",
			"query_length": len(queryText),
			"table":       table,
			"error":       err.Error(),
			"params":      params,
		}), nil
	}

  /* Generate embedding for query */
	/* Get default model if available */
	modelName := "default"
	if defaultModel, err := t.configHelper.GetDefaultModel(ctx, "embedding"); err == nil && defaultModel != "" {
		modelName = defaultModel
	}

	embedQuery := `SELECT embed_text($1::text, $2::text)::text AS embedding`
	embedResult, err := t.executor.ExecuteQueryOne(ctx, embedQuery, []interface{}{queryText, modelName})
	if err != nil {
		t.logger.Error("Embedding generation failed", err, params)
		return Error(fmt.Sprintf("Embedding generation failed for retrieve_context: query_length=%d, table='%s', vector_column='%s', limit=%d, model='%s', error=%v", len(queryText), table, vectorColumn, limit, modelName, err), "RAG_ERROR", map[string]interface{}{
			"query_length":  len(queryText),
			"table":         table,
			"vector_column": vectorColumn,
			"limit":         limit,
			"model":         modelName,
			"error":         err.Error(),
		}), nil
	}

	/* Extract embedding vector from result */
	var embeddingStr string
	if embStr, ok := embedResult["embedding"].(string); ok {
		embeddingStr = embStr
	} else if embArr, ok := embedResult["embedding"].([]interface{}); ok {
		/* Convert array to vector string format */
		parts := make([]string, 0, len(embArr))
		for _, v := range embArr {
			if f, ok := v.(float64); ok {
				parts = append(parts, fmt.Sprintf("%g", f))
			} else if f, ok := v.(float32); ok {
				parts = append(parts, fmt.Sprintf("%g", f))
			} else {
				parts = append(parts, fmt.Sprintf("%v", v))
			}
		}
		embeddingStr = "[" + strings.Join(parts, ",") + "]"
	} else {
		return Error("Invalid embedding format from embed_text function", "RAG_ERROR", map[string]interface{}{
			"query_length":  len(queryText),
			"table":         table,
			"vector_column": vectorColumn,
			"embedding_type": fmt.Sprintf("%T", embedResult["embedding"]),
		}), nil
	}

	/* Perform vector search using the embedding */
	/* Use EscapeIdentifier to prevent SQL injection (table and vectorColumn are validated above) */
	escapedTable := database.EscapeIdentifier(table)
	escapedVectorCol := database.EscapeIdentifier(vectorColumn)
	retrieveQuery := fmt.Sprintf(`
		SELECT 
			*,
			1 - (%s <=> $1::vector) AS similarity
		FROM %s
		ORDER BY %s <=> $1::vector
		LIMIT $2
	`, escapedVectorCol, escapedTable, escapedVectorCol)
	
	result, err := t.executor.ExecuteQuery(ctx, retrieveQuery, []interface{}{embeddingStr, limit})
	if err != nil {
		t.logger.Error("Context retrieval failed", err, params)
		return Error(fmt.Sprintf("Context retrieval execution failed: query_length=%d, table='%s', vector_column='%s', limit=%d, error=%v", len(queryText), table, vectorColumn, limit, err), "RAG_ERROR", map[string]interface{}{
			"query_length":  len(queryText),
			"table":         table,
			"vector_column": vectorColumn,
			"limit":         limit,
			"error":         err.Error(),
		}), nil
	}

	return Success(result, map[string]interface{}{
		"limit": limit,
		"count": len(result),
	}), nil
}

/* GenerateResponseTool generates a response using RAG */
type GenerateResponseTool struct {
	*BaseTool
	executor     *QueryExecutor
	logger       *logging.Logger
	configHelper *database.ConfigHelper
}

/* NewGenerateResponseTool creates a new generate response tool */
func NewGenerateResponseTool(db *database.Database, logger *logging.Logger) *GenerateResponseTool {
	return &GenerateResponseTool{
		BaseTool: NewBaseTool(
			"postgresql_generate_response",
			"Generate a response using RAG pipeline",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]interface{}{
						"type":        "string",
						"description": "User query",
					},
					"context": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "string"},
						"description": "Retrieved context chunks",
					},
				},
				"required": []interface{}{"query", "context"},
			},
		),
		executor:     NewQueryExecutor(db),
		logger:       logger,
		configHelper: database.NewConfigHelper(db),
	}
}

/* Execute executes the response generation */
func (t *GenerateResponseTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for neurondb_generate_response tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	query, _ := params["query"].(string)
	context, _ := params["context"].([]interface{})

	if query == "" {
		return Error("query parameter is required and cannot be empty for neurondb_generate_response tool", "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "query",
			"params":    params,
		}), nil
	}

	if context == nil || len(context) == 0 {
		return Error(fmt.Sprintf("context parameter is required and cannot be empty array for neurondb_generate_response tool: query_length=%d", len(query)), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":    "context",
			"query_length": len(query),
			"context_count": 0,
			"params":       params,
		}), nil
	}

	contextCount := len(context)
	contextStr := ""
	for i, c := range context {
		if i > 0 {
			contextStr += "\n\n"
		}
		if s, ok := c.(string); ok {
			if s == "" {
				return Error(fmt.Sprintf("context element at index %d is empty string for neurondb_generate_response tool: query_length=%d, context_count=%d", i, len(query), contextCount), "VALIDATION_ERROR", map[string]interface{}{
					"parameter":     "context",
					"query_length":  len(query),
					"context_count": contextCount,
					"empty_index":   i,
					"params":        params,
				}), nil
			}
			contextStr += s
		} else {
			return Error(fmt.Sprintf("context element at index %d has invalid type for neurondb_generate_response tool: query_length=%d, context_count=%d, expected string, got %T", i, len(query), contextCount, c), "VALIDATION_ERROR", map[string]interface{}{
				"parameter":     "context",
				"query_length":  len(query),
				"context_count": contextCount,
				"invalid_index": i,
				"received_type": fmt.Sprintf("%T", c),
				"params":        params,
			}), nil
		}
	}

  /* Use NeuronDB's LLM function for response generation */
  /* neurondb.llm(task, model, input_text, input_array, params, max_length) */
	modelName := "default"
	if m, ok := params["model"].(string); ok && m != "" {
		modelName = m
	}

  /* Build prompt with context */
	prompt := fmt.Sprintf("Context:\n%s\n\nQuestion: %s\n\nAnswer:", contextStr, query)
	
	llmParams := fmt.Sprintf(`{"temperature": 0.7, "max_tokens": 500}`)
	generateQuery := `SELECT neurondb.llm('generation', $1, $2, NULL, $3::jsonb, 512) AS response`
	result, err := t.executor.ExecuteQueryOne(ctx, generateQuery, []interface{}{modelName, prompt, llmParams})
	if err != nil {
		t.logger.Error("Response generation failed", err, params)
		return Error(fmt.Sprintf("Response generation execution failed: query_length=%d, context_count=%d, context_total_length=%d, error=%v", len(query), contextCount, len(contextStr), err), "RAG_ERROR", map[string]interface{}{
			"query_length":        len(query),
			"context_count":       contextCount,
			"context_total_length": len(contextStr),
			"error":               err.Error(),
		}), nil
	}

	return Success(result, nil), nil
}

/* ChunkDocumentTool chunks a document into smaller pieces */
type ChunkDocumentTool struct {
	*BaseTool
	executor     *QueryExecutor
	logger       *logging.Logger
	configHelper *database.ConfigHelper
}

/* NewChunkDocumentTool creates a new chunk document tool */
func NewChunkDocumentTool(db *database.Database, logger *logging.Logger) *ChunkDocumentTool {
	return &ChunkDocumentTool{
		BaseTool: NewBaseTool(
			"postgresql_chunk_document",
			"Chunk a document into smaller pieces with optional overlap",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"text": map[string]interface{}{
						"type":        "string",
						"description": "Text to chunk",
					},
					"chunk_size": map[string]interface{}{
						"type":        "number",
						"default":     500,
						"minimum":     1,
						"description": "Chunk size in characters",
					},
					"overlap": map[string]interface{}{
						"type":        "number",
						"default":     50,
						"minimum":     0,
						"description": "Overlap between chunks",
					},
				},
				"required": []interface{}{"text"},
			},
		),
		executor:     NewQueryExecutor(db),
		logger:       logger,
		configHelper: database.NewConfigHelper(db),
	}
}

/* Execute executes the document chunking */
func (t *ChunkDocumentTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for neurondb_chunk_document tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	/* Accept both "text" and "document" parameters */
	text, _ := params["text"].(string)
	if text == "" {
		text, _ = params["document"].(string)
	}
	chunkSize := 500
	if c, ok := params["chunk_size"].(float64); ok {
		chunkSize = int(c)
	}
	overlap := 50
	/* Accept both "overlap" and "chunk_overlap" parameters */
	if o, ok := params["overlap"].(float64); ok {
		overlap = int(o)
	} else if o, ok := params["chunk_overlap"].(float64); ok {
		overlap = int(o)
	}

	if text == "" {
		return Error("text or document parameter is required and cannot be empty for neurondb_chunk_document tool", "VALIDATION_ERROR", map[string]interface{}{
			"parameter":   "text",
			"text_length": 0,
			"params":      params,
		}), nil
	}

	textLen := len(text)
	if chunkSize <= 0 {
		return Error(fmt.Sprintf("chunk_size must be greater than 0 for neurondb_chunk_document tool: text_length=%d, received chunk_size=%d", textLen, chunkSize), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":   "chunk_size",
			"text_length": textLen,
			"chunk_size":  chunkSize,
			"params":      params,
		}), nil
	}

	if overlap < 0 {
		return Error(fmt.Sprintf("overlap cannot be negative for neurondb_chunk_document tool: text_length=%d, chunk_size=%d, received overlap=%d", textLen, chunkSize, overlap), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":   "overlap",
			"text_length": textLen,
			"chunk_size":  chunkSize,
			"overlap":     overlap,
			"params":      params,
		}), nil
	}

	if overlap >= chunkSize {
		return Error(fmt.Sprintf("overlap must be less than chunk_size for neurondb_chunk_document tool: text_length=%d, chunk_size=%d, overlap=%d", textLen, chunkSize, overlap), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":   "overlap",
			"text_length": textLen,
			"chunk_size":  chunkSize,
			"overlap":     overlap,
			"params":      params,
		}), nil
	}

  /* Use NeuronDB's unified chunking function: neurondb.chunk(document_text, chunk_size, chunk_overlap, method) */
	query := `SELECT json_agg(json_build_object('chunk_id', chunk_id, 'chunk_text', chunk_text, 'start_pos', start_pos, 'end_pos', end_pos)) AS chunks FROM neurondb.chunk($1, $2, $3, 'fixed')`
	result, err := t.executor.ExecuteQueryOne(ctx, query, []interface{}{text, chunkSize, overlap})
	if err != nil {
		t.logger.Error("Document chunking failed", err, params)
		return Error(fmt.Sprintf("Document chunking execution failed: text_length=%d, chunk_size=%d, overlap=%d, error=%v", textLen, chunkSize, overlap, err), "RAG_ERROR", map[string]interface{}{
			"text_length": textLen,
			"chunk_size":  chunkSize,
			"overlap":     overlap,
			"error":       err.Error(),
		}), nil
	}

	return Success(result, map[string]interface{}{
		"chunk_size": chunkSize,
		"overlap":    overlap,
	}), nil
}

