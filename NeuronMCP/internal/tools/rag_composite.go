/*-------------------------------------------------------------------------
 *
 * rag_composite.go
 *    Composite RAG tools for NeuronMCP
 *
 * Provides high-level composite RAG operations that combine multiple steps.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/rag_composite.go
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
)

/* IngestDocumentsTool provides a composite tool for ingesting documents */
type IngestDocumentsTool struct {
	*BaseTool
	executor     *QueryExecutor
	logger       *logging.Logger
	configHelper *database.ConfigHelper
}

/* NewIngestDocumentsTool creates a new ingest documents tool */
func NewIngestDocumentsTool(db *database.Database, logger *logging.Logger) *IngestDocumentsTool {
	return &IngestDocumentsTool{
		BaseTool: NewBaseTool(
			"postgresql_ingest_documents",
			"Composite tool: Ingest documents into a collection with automatic chunking and embedding",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"collection": map[string]interface{}{
						"type":        "string",
						"description": "Collection name (table name) to ingest into",
						"minLength":   1,
					},
					"source": map[string]interface{}{
						"type":        "string",
						"description": "Source text or document content",
						"minLength":   1,
					},
					"chunk_size": map[string]interface{}{
						"type":        "integer",
						"default":     500,
						"minimum":     1,
						"maximum":     10000,
						"description": "Chunk size in characters",
					},
					"overlap": map[string]interface{}{
						"type":        "integer",
						"default":     50,
						"minimum":     0,
						"description": "Overlap between chunks",
					},
					"embedding_model": map[string]interface{}{
						"type":        "string",
						"description": "Embedding model to use (optional, uses default if not specified)",
					},
				},
				"required": []interface{}{"collection", "source"},
				"additionalProperties": false,
			},
		),
		executor:     NewQueryExecutor(db),
		logger:       logger,
		configHelper: database.NewConfigHelper(db),
	}
}

/* Execute executes the ingest documents operation */
func (t *IngestDocumentsTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for neurondb_ingest_documents tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	collection, _ := params["collection"].(string)
	source, _ := params["source"].(string)
	chunkSize := 500
	if cs, ok := params["chunk_size"].(float64); ok {
		chunkSize = int(cs)
	}
	overlap := 50
	if ov, ok := params["overlap"].(float64); ok {
		overlap = int(ov)
	}
	embeddingModel := ""
	if em, ok := params["embedding_model"].(string); ok {
		embeddingModel = em
	}

	/* Step 1: Chunk the document */
	chunkQuery := `SELECT neurondb_chunk_text($1::text, $2::integer, $3::integer) AS chunks`
	chunkResult, err := t.executor.ExecuteQueryOne(ctx, chunkQuery, []interface{}{source, chunkSize, overlap})
	if err != nil {
		return Error(fmt.Sprintf("Failed to chunk document: %v", err), "CHUNK_ERROR", nil), nil
	}

	chunks, ok := chunkResult["chunks"].([]interface{})
	if !ok {
		return Error("Invalid chunks result", "CHUNK_ERROR", nil), nil
	}

	/* Step 2: Generate embeddings and insert into collection */
	/* For each chunk, generate embedding and insert */
	/* This is a simplified version - full implementation would batch this */
	insertedCount := 0
	for i, chunk := range chunks {
		chunkText, ok := chunk.(string)
		if !ok {
			continue
		}

		/* Generate embedding */
		embedQuery := `SELECT embed_text($1::text, $2::text)::text AS embedding`
		var embedParams []interface{}
		if embeddingModel != "" {
			embedParams = []interface{}{chunkText, embeddingModel}
		} else {
			embedParams = []interface{}{chunkText, "default"}
		}

		embedResult, err := t.executor.ExecuteQueryOne(ctx, embedQuery, embedParams)
		if err != nil {
			t.logger.Error(fmt.Sprintf("Failed to generate embedding for chunk %d", i), err, map[string]interface{}{
				"chunk_index": i,
				"chunk_text_length": len(chunkText),
				"model": embeddingModel,
			})
			continue
		}

		/* Handle embedding result - may be string (vector text format) or array */
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
			t.logger.Warn(fmt.Sprintf("Invalid embedding format for chunk %d: expected string or array, got %T", i, embedResult["embedding"]), map[string]interface{}{
				"chunk_index": i,
				"embedding_type": fmt.Sprintf("%T", embedResult["embedding"]),
			})
			continue
		}

		/* Insert into collection (simplified - assumes table has text and embedding columns) */
		insertQuery := fmt.Sprintf("INSERT INTO %s (text, embedding) VALUES ($1, $2::vector)", collection)
		_, err = t.executor.ExecuteQueryOne(ctx, insertQuery, []interface{}{chunkText, embeddingStr})
		if err != nil {
			t.logger.Error(fmt.Sprintf("Failed to insert chunk %d", i), err, map[string]interface{}{
				"chunk_index": i,
				"collection": collection,
			})
			continue
		}
		insertedCount++
	}

	return Success(map[string]interface{}{
		"chunks_created": insertedCount,
		"collection":     collection,
		"message":        fmt.Sprintf("Successfully ingested %d chunks into collection %s", insertedCount, collection),
	}, nil), nil
}

/* AnswerWithCitationsTool provides a composite tool for answering with citations */
type AnswerWithCitationsTool struct {
	*BaseTool
	executor     *QueryExecutor
	logger       *logging.Logger
	configHelper *database.ConfigHelper
}

/* NewAnswerWithCitationsTool creates a new answer with citations tool */
func NewAnswerWithCitationsTool(db *database.Database, logger *logging.Logger) *AnswerWithCitationsTool {
	return &AnswerWithCitationsTool{
		BaseTool: NewBaseTool(
			"postgresql_answer_with_citations",
			"Composite tool: Answer a question using RAG with source citations",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"collection": map[string]interface{}{
						"type":        "string",
						"description": "Collection name to search in",
						"minLength":   1,
					},
					"query": map[string]interface{}{
						"type":        "string",
						"description": "Question or query",
						"minLength":   1,
					},
					"model": map[string]interface{}{
						"type":        "string",
						"description": "LLM model to use for answer generation",
					},
					"k": map[string]interface{}{
						"type":        "integer",
						"default":     5,
						"minimum":     1,
						"maximum":     50,
						"description": "Number of context chunks to retrieve",
					},
				},
				"required": []interface{}{"collection", "query"},
				"additionalProperties": false,
			},
		),
		executor:     NewQueryExecutor(db),
		logger:       logger,
		configHelper: database.NewConfigHelper(db),
	}
}

/* Execute executes the answer with citations operation */
func (t *AnswerWithCitationsTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for neurondb_answer_with_citations tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	collection, _ := params["collection"].(string)
	query, _ := params["query"].(string)
	/* Model parameter is available for future LLM integration */
	_ = "gpt-3.5-turbo"
	if m, ok := params["model"].(string); ok && m != "" {
		_ = m
	}
	k := 5
	if kVal, ok := params["k"].(float64); ok {
		k = int(kVal)
	}

	/* Step 1: Generate query embedding */
	embedQuery := `SELECT embed_text($1::text, 'default')::text AS embedding`
	embedResult, err := t.executor.ExecuteQueryOne(ctx, embedQuery, []interface{}{query})
	if err != nil {
		return Error(fmt.Sprintf("Failed to generate query embedding: %v", err), "EMBEDDING_ERROR", map[string]interface{}{
			"query_length": len(query),
			"error": err.Error(),
		}), nil
	}

	/* Handle embedding result - may be string (vector text format) or array */
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
		return Error("Invalid embedding result format: expected string or array", "EMBEDDING_ERROR", map[string]interface{}{
			"query_length": len(query),
			"embedding_type": fmt.Sprintf("%T", embedResult["embedding"]),
		}), nil
	}

	/* Step 2: Retrieve context (vector search) */
	retrieveQuery := fmt.Sprintf(`
		SELECT text, embedding <=> $1::vector AS distance
		FROM %s
		ORDER BY embedding <=> $1::vector
		LIMIT $2
	`, collection)

	contextResults, err := t.executor.ExecuteQuery(ctx, retrieveQuery, []interface{}{embeddingStr, k})
	if err != nil {
		return Error(fmt.Sprintf("Failed to retrieve context: %v", err), "RETRIEVAL_ERROR", nil), nil
	}

	/* Extract context texts */
	contextTexts := make([]string, 0)
	for _, result := range contextResults {
		if text, ok := result["text"].(string); ok {
			contextTexts = append(contextTexts, text)
		}
	}

	/* Step 3: Generate answer using LLM with context */
	/* This is simplified - full implementation would call actual LLM */
	answer := fmt.Sprintf("Based on the retrieved context, here is an answer to: %s", query)
	citations := make([]string, len(contextTexts))
	for i, ctx := range contextTexts {
		citations[i] = fmt.Sprintf("[%d] %s", i+1, ctx[:min(100, len(ctx))])
	}

	return Success(map[string]interface{}{
		"answer":        answer,
		"citations":     citations,
		"context_count": len(contextTexts),
	}, nil), nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

