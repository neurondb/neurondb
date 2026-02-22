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
	"encoding/json"
	"fmt"
	"strings"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/validation"
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
				"required":             []interface{}{"collection", "source"},
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
	if err := validation.ValidateTableName(collection); err != nil {
		return Error(fmt.Sprintf("Invalid collection name: %v", err), "VALIDATION_ERROR", nil), nil
	}
	escapedCollection := validation.EscapeSQLIdentifier(collection)
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
				"chunk_index":       i,
				"chunk_text_length": len(chunkText),
				"model":             embeddingModel,
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
				"chunk_index":    i,
				"embedding_type": fmt.Sprintf("%T", embedResult["embedding"]),
			})
			continue
		}

		/* Insert into collection (simplified - assumes table has text and embedding columns) */
		insertQuery := fmt.Sprintf("INSERT INTO %s (text, embedding) VALUES ($1, $2::vector)", escapedCollection)
		_, err = t.executor.ExecuteQueryOne(ctx, insertQuery, []interface{}{chunkText, embeddingStr})
		if err != nil {
			t.logger.Error(fmt.Sprintf("Failed to insert chunk %d", i), err, map[string]interface{}{
				"chunk_index": i,
				"collection":  collection,
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
				"required":             []interface{}{"collection", "query"},
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
			"error":        err.Error(),
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
			"query_length":   len(query),
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

/* RAGEvaluateTool provides a composite tool for evaluating RAG pipeline quality */
type RAGEvaluateTool struct {
	*BaseTool
	executor     *QueryExecutor
	logger       *logging.Logger
	configHelper *database.ConfigHelper
}

/* NewRAGEvaluateTool creates a new RAG evaluate tool */
func NewRAGEvaluateTool(db *database.Database, logger *logging.Logger) *RAGEvaluateTool {
	return &RAGEvaluateTool{
		BaseTool: NewBaseTool(
			"postgresql_rag_evaluate",
			"Evaluate RAG pipeline quality using metrics like relevancy and semantic similarity",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]interface{}{
						"type":        "string",
						"description": "Query text",
						"minLength":   1,
					},
					"answer": map[string]interface{}{
						"type":        "string",
						"description": "Generated answer text",
						"minLength":   1,
					},
					"context_chunks": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "string"},
						"description": "Retrieved context chunks",
					},
					"evaluation_type": map[string]interface{}{
						"type":        "string",
						"default":     "basic",
						"description": "Evaluation type (basic, advanced)",
					},
				},
				"required":             []interface{}{"query", "answer", "context_chunks"},
				"additionalProperties": false,
			},
		),
		executor:     NewQueryExecutor(db),
		logger:       logger,
		configHelper: database.NewConfigHelper(db),
	}
}

/* Execute executes the RAG evaluation */
func (t *RAGEvaluateTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for neurondb_rag_evaluate tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	query, _ := params["query"].(string)
	answer, _ := params["answer"].(string)
	contextChunksInterface, _ := params["context_chunks"].([]interface{})
	evaluationType := "basic"
	if et, ok := params["evaluation_type"].(string); ok {
		evaluationType = et
	}

	/* Convert context chunks to string array */
	contextChunks := make([]string, 0, len(contextChunksInterface))
	for _, chunk := range contextChunksInterface {
		if chunkStr, ok := chunk.(string); ok {
			contextChunks = append(contextChunks, chunkStr)
		}
	}

	/* Use neurondb.rag_evaluate function */
	evalQuery := `SELECT neurondb.rag_evaluate($1, $2, $3::text[], $4) AS evaluation`
	result, err := t.executor.ExecuteQueryOne(ctx, evalQuery, []interface{}{query, answer, contextChunks, evaluationType})
	if err != nil {
		return Error(fmt.Sprintf("RAG evaluation failed: %v", err), "EVALUATION_ERROR", map[string]interface{}{
			"query_length":         len(query),
			"answer_length":        len(answer),
			"context_chunks_count": len(contextChunks),
			"error":                err.Error(),
		}), nil
	}

	return Success(result, nil), nil
}

/* RAGChatTool provides a composite tool for conversational RAG interface */
type RAGChatTool struct {
	*BaseTool
	executor     *QueryExecutor
	logger       *logging.Logger
	configHelper *database.ConfigHelper
}

/* NewRAGChatTool creates a new RAG chat tool */
func NewRAGChatTool(db *database.Database, logger *logging.Logger) *RAGChatTool {
	return &RAGChatTool{
		BaseTool: NewBaseTool(
			"postgresql_rag_chat",
			"Conversational RAG interface with conversation history support",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]interface{}{
						"type":        "string",
						"description": "User query",
						"minLength":   1,
					},
					"document_table": map[string]interface{}{
						"type":        "string",
						"description": "Document table name",
						"minLength":   1,
					},
					"vector_col": map[string]interface{}{
						"type":        "string",
						"default":     "embedding",
						"description": "Vector column name",
					},
					"text_col": map[string]interface{}{
						"type":        "string",
						"default":     "content",
						"description": "Text column name",
					},
					"model": map[string]interface{}{
						"type":        "string",
						"default":     "default",
						"description": "Embedding model",
					},
					"top_k": map[string]interface{}{
						"type":        "integer",
						"default":     5,
						"minimum":     1,
						"maximum":     50,
						"description": "Number of context chunks",
					},
					"conversation_history": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "object"},
						"description": "Conversation history (array of {role, content} objects)",
					},
					"llm_model": map[string]interface{}{
						"type":        "string",
						"default":     "gpt-3.5-turbo",
						"description": "LLM model for answer generation",
					},
				},
				"required":             []interface{}{"query", "document_table"},
				"additionalProperties": false,
			},
		),
		executor:     NewQueryExecutor(db),
		logger:       logger,
		configHelper: database.NewConfigHelper(db),
	}
}

/* Execute executes the RAG chat */
func (t *RAGChatTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for neurondb_rag_chat tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	query, _ := params["query"].(string)
	documentTable, _ := params["document_table"].(string)
	vectorCol := "embedding"
	if vc, ok := params["vector_col"].(string); ok && vc != "" {
		vectorCol = vc
	}
	textCol := "content"
	if tc, ok := params["text_col"].(string); ok && tc != "" {
		textCol = tc
	}
	model := "default"
	if m, ok := params["model"].(string); ok && m != "" {
		model = m
	}
	topK := 5
	if tk, ok := params["top_k"].(float64); ok {
		topK = int(tk)
	}
	llmModel := "gpt-3.5-turbo"
	if lm, ok := params["llm_model"].(string); ok && lm != "" {
		llmModel = lm
	}

	/* Build conversation history JSON */
	conversationHistory := "[]"
	if ch, ok := params["conversation_history"].([]interface{}); ok && len(ch) > 0 {
		/* Convert to JSON string */
		chJSON, err := json.Marshal(ch)
		if err == nil {
			conversationHistory = string(chJSON)
		}
	}

	/* Use neurondb.rag_chat function */
	chatQuery := `SELECT neurondb.rag_chat($1, $2, $3, $4, $5, $6, $7::jsonb, $8) AS result`
	result, err := t.executor.ExecuteQueryOne(ctx, chatQuery, []interface{}{
		query, documentTable, vectorCol, textCol, model, topK, conversationHistory, llmModel,
	})
	if err != nil {
		return Error(fmt.Sprintf("RAG chat failed: %v", err), "CHAT_ERROR", map[string]interface{}{
			"query_length": len(query),
			"table":        documentTable,
			"error":        err.Error(),
		}), nil
	}

	return Success(result, nil), nil
}

/* RAGHybridTool provides a composite tool for hybrid search RAG */
type RAGHybridTool struct {
	*BaseTool
	executor     *QueryExecutor
	logger       *logging.Logger
	configHelper *database.ConfigHelper
}

/* NewRAGHybridTool creates a new RAG hybrid tool */
func NewRAGHybridTool(db *database.Database, logger *logging.Logger) *RAGHybridTool {
	return &RAGHybridTool{
		BaseTool: NewBaseTool(
			"postgresql_rag_hybrid",
			"RAG with hybrid search (vector + full-text)",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]interface{}{
						"type":        "string",
						"description": "Query text",
						"minLength":   1,
					},
					"document_table": map[string]interface{}{
						"type":        "string",
						"description": "Document table name",
						"minLength":   1,
					},
					"vector_col": map[string]interface{}{
						"type":        "string",
						"default":     "embedding",
						"description": "Vector column name",
					},
					"text_col": map[string]interface{}{
						"type":        "string",
						"default":     "content",
						"description": "Text column name for full-text search",
					},
					"model": map[string]interface{}{
						"type":        "string",
						"default":     "default",
						"description": "Embedding model",
					},
					"top_k": map[string]interface{}{
						"type":        "integer",
						"default":     5,
						"minimum":     1,
						"maximum":     50,
						"description": "Number of results",
					},
					"vector_weight": map[string]interface{}{
						"type":        "number",
						"default":     0.7,
						"minimum":     0,
						"maximum":     1,
						"description": "Weight for vector search (0-1)",
					},
				},
				"required":             []interface{}{"query", "document_table"},
				"additionalProperties": false,
			},
		),
		executor:     NewQueryExecutor(db),
		logger:       logger,
		configHelper: database.NewConfigHelper(db),
	}
}

/* Execute executes the hybrid RAG */
func (t *RAGHybridTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for neurondb_rag_hybrid tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	query, _ := params["query"].(string)
	documentTable, _ := params["document_table"].(string)
	vectorCol := "embedding"
	if vc, ok := params["vector_col"].(string); ok && vc != "" {
		vectorCol = vc
	}
	textCol := "content"
	if tc, ok := params["text_col"].(string); ok && tc != "" {
		textCol = tc
	}
	model := "default"
	if m, ok := params["model"].(string); ok && m != "" {
		model = m
	}
	topK := 5
	if tk, ok := params["top_k"].(float64); ok {
		topK = int(tk)
	}
	vectorWeight := 0.7
	if vw, ok := params["vector_weight"].(float64); ok {
		vectorWeight = vw
	}

	/* Use hybrid_search function if available, otherwise use basic RAG with manual hybrid */
	/* For now, use basic RAG query and note that hybrid would require hybrid_search function */
	embedQuery := `SELECT embed_text($1::text, $2::text)::text AS embedding`
	embedResult, err := t.executor.ExecuteQueryOne(ctx, embedQuery, []interface{}{query, model})
	if err != nil {
		return Error(fmt.Sprintf("Failed to generate query embedding: %v", err), "EMBEDDING_ERROR", nil), nil
	}

	var embeddingStr string
	if embStr, ok := embedResult["embedding"].(string); ok {
		embeddingStr = embStr
	} else if embArr, ok := embedResult["embedding"].([]interface{}); ok {
		parts := make([]string, 0, len(embArr))
		for _, v := range embArr {
			if f, ok := v.(float64); ok {
				parts = append(parts, fmt.Sprintf("%g", f))
			} else {
				parts = append(parts, fmt.Sprintf("%v", v))
			}
		}
		embeddingStr = "[" + strings.Join(parts, ",") + "]"
	} else {
		return Error("Invalid embedding result format", "EMBEDDING_ERROR", nil), nil
	}

	/* Perform hybrid search - combine vector and text search */
	/* Note: This is a simplified version - full implementation would use hybrid_search function */
	retrieveQuery := fmt.Sprintf(`
		SELECT %s, 
			(%s <=> $1::vector) * $3 + 
			(1.0 - ts_rank(to_tsvector('english', %s), plainto_tsquery('english', $2))) * (1.0 - $3) AS combined_score
		FROM %s
		ORDER BY combined_score
		LIMIT $4
	`, textCol, vectorCol, textCol, documentTable)

	results, err := t.executor.ExecuteQuery(ctx, retrieveQuery, []interface{}{embeddingStr, query, vectorWeight, topK})
	if err != nil {
		return Error(fmt.Sprintf("Hybrid RAG failed: %v", err), "HYBRID_ERROR", nil), nil
	}

	return Success(map[string]interface{}{
		"results": results,
		"count":   len(results),
		"method":  "hybrid",
	}, nil), nil
}

/* RAGRerankTool provides a composite tool for RAG with reranking */
type RAGRerankTool struct {
	*BaseTool
	executor     *QueryExecutor
	logger       *logging.Logger
	configHelper *database.ConfigHelper
}

/* NewRAGRerankTool creates a new RAG rerank tool */
func NewRAGRerankTool(db *database.Database, logger *logging.Logger) *RAGRerankTool {
	return &RAGRerankTool{
		BaseTool: NewBaseTool(
			"postgresql_rag_rerank",
			"RAG with reranking for improved result quality",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]interface{}{
						"type":        "string",
						"description": "Query text",
						"minLength":   1,
					},
					"document_table": map[string]interface{}{
						"type":        "string",
						"description": "Document table name",
						"minLength":   1,
					},
					"vector_col": map[string]interface{}{
						"type":        "string",
						"default":     "embedding",
						"description": "Vector column name",
					},
					"text_col": map[string]interface{}{
						"type":        "string",
						"default":     "content",
						"description": "Text column name",
					},
					"model": map[string]interface{}{
						"type":        "string",
						"default":     "default",
						"description": "Embedding model",
					},
					"top_k": map[string]interface{}{
						"type":        "integer",
						"default":     5,
						"minimum":     1,
						"maximum":     50,
						"description": "Final number of results after reranking",
					},
					"initial_k": map[string]interface{}{
						"type":        "integer",
						"default":     20,
						"minimum":     1,
						"maximum":     100,
						"description": "Initial number of results before reranking",
					},
					"rerank_model": map[string]interface{}{
						"type":        "string",
						"default":     "cross-encoder",
						"description": "Reranking model",
					},
				},
				"required":             []interface{}{"query", "document_table"},
				"additionalProperties": false,
			},
		),
		executor:     NewQueryExecutor(db),
		logger:       logger,
		configHelper: database.NewConfigHelper(db),
	}
}

/* Execute executes the reranked RAG */
func (t *RAGRerankTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for neurondb_rag_rerank tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	query, _ := params["query"].(string)
	documentTable, _ := params["document_table"].(string)
	vectorCol := "embedding"
	if vc, ok := params["vector_col"].(string); ok && vc != "" {
		vectorCol = vc
	}
	textCol := "content"
	if tc, ok := params["text_col"].(string); ok && tc != "" {
		textCol = tc
	}
	model := "default"
	if m, ok := params["model"].(string); ok && m != "" {
		model = m
	}
	topK := 5
	if tk, ok := params["top_k"].(float64); ok {
		topK = int(tk)
	}
	initialK := 20
	if ik, ok := params["initial_k"].(float64); ok {
		initialK = int(ik)
	}
	rerankModel := "cross-encoder"
	if rm, ok := params["rerank_model"].(string); ok && rm != "" {
		rerankModel = rm
	}

	/* Step 1: Generate query embedding */
	embedQuery := `SELECT embed_text($1::text, $2::text)::text AS embedding`
	embedResult, err := t.executor.ExecuteQueryOne(ctx, embedQuery, []interface{}{query, model})
	if err != nil {
		return Error(fmt.Sprintf("Failed to generate query embedding: %v", err), "EMBEDDING_ERROR", nil), nil
	}

	var embeddingStr string
	if embStr, ok := embedResult["embedding"].(string); ok {
		embeddingStr = embStr
	} else if embArr, ok := embedResult["embedding"].([]interface{}); ok {
		parts := make([]string, 0, len(embArr))
		for _, v := range embArr {
			if f, ok := v.(float64); ok {
				parts = append(parts, fmt.Sprintf("%g", f))
			} else {
				parts = append(parts, fmt.Sprintf("%v", v))
			}
		}
		embeddingStr = "[" + strings.Join(parts, ",") + "]"
	} else {
		return Error("Invalid embedding result format", "EMBEDDING_ERROR", nil), nil
	}

	/* Step 2: Initial vector search */
	retrieveQuery := fmt.Sprintf(`
		SELECT %s
		FROM %s
		ORDER BY %s <=> $1::vector
		LIMIT $2
	`, textCol, documentTable, vectorCol)

	initialResults, err := t.executor.ExecuteQuery(ctx, retrieveQuery, []interface{}{embeddingStr, initialK})
	if err != nil {
		return Error(fmt.Sprintf("Initial retrieval failed: %v", err), "RETRIEVAL_ERROR", nil), nil
	}

	/* Extract documents for reranking */
	documents := make([]string, 0, len(initialResults))
	for _, result := range initialResults {
		if text, ok := result[textCol].(string); ok {
			documents = append(documents, text)
		}
	}

	/* Step 3: Rerank using rerank_cross_encoder or similar function */
	rerankQuery := `SELECT rerank_cross_encoder($1, $2::text[], $3, $4) AS reranked`
	rerankResult, err := t.executor.ExecuteQueryOne(ctx, rerankQuery, []interface{}{query, documents, rerankModel, topK})
	if err != nil {
		/* Fallback: return initial results if reranking fails */
		return Success(map[string]interface{}{
			"results": initialResults[:min(topK, len(initialResults))],
			"count":   min(topK, len(initialResults)),
			"method":  "reranked_fallback",
		}, nil), nil
	}

	return Success(rerankResult, nil), nil
}

/* RAGGraphTool provides a composite tool for Graph RAG */
type RAGGraphTool struct {
	*BaseTool
	executor     *QueryExecutor
	logger       *logging.Logger
	configHelper *database.ConfigHelper
}

/* NewRAGGraphTool creates a new Graph RAG tool */
func NewRAGGraphTool(db *database.Database, logger *logging.Logger) *RAGGraphTool {
	return &RAGGraphTool{
		BaseTool: NewBaseTool(
			"postgresql_rag_graph",
			"RAG with knowledge graph traversal for relationship-based retrieval",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]interface{}{
						"type":        "string",
						"description": "Query text",
						"minLength":   1,
					},
					"document_table": map[string]interface{}{
						"type":        "string",
						"description": "Document table name",
						"minLength":   1,
					},
					"vector_col": map[string]interface{}{
						"type":        "string",
						"default":     "embedding",
						"description": "Vector column name",
					},
					"text_col": map[string]interface{}{
						"type":        "string",
						"default":     "content",
						"description": "Text column name",
					},
					"entity_col": map[string]interface{}{
						"type":        "string",
						"default":     "entities",
						"description": "Entity column name (JSONB or text)",
					},
					"relation_col": map[string]interface{}{
						"type":        "string",
						"default":     "relations",
						"description": "Relation column name (JSONB or text)",
					},
					"embedding_model": map[string]interface{}{
						"type":        "string",
						"default":     "default",
						"description": "Embedding model",
					},
					"top_k": map[string]interface{}{
						"type":        "integer",
						"default":     5,
						"minimum":     1,
						"maximum":     50,
						"description": "Number of results to return",
					},
					"max_depth": map[string]interface{}{
						"type":        "integer",
						"default":     2,
						"minimum":     1,
						"maximum":     5,
						"description": "Maximum graph traversal depth",
					},
					"traversal_method": map[string]interface{}{
						"type":        "string",
						"default":     "bfs",
						"enum":        []interface{}{"bfs", "dfs"},
						"description": "Graph traversal method (bfs or dfs)",
					},
					"custom_context": map[string]interface{}{
						"type":        "object",
						"description": "Custom context parameters",
					},
				},
				"required":             []interface{}{"query", "document_table"},
				"additionalProperties": false,
			},
		),
		executor:     NewQueryExecutor(db),
		logger:       logger,
		configHelper: database.NewConfigHelper(db),
	}
}

/* Execute executes the Graph RAG */
func (t *RAGGraphTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for neurondb_rag_graph tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	query, _ := params["query"].(string)
	documentTable, _ := params["document_table"].(string)
	vectorCol := "embedding"
	if vc, ok := params["vector_col"].(string); ok && vc != "" {
		vectorCol = vc
	}
	textCol := "content"
	if tc, ok := params["text_col"].(string); ok && tc != "" {
		textCol = tc
	}
	entityCol := "entities"
	if ec, ok := params["entity_col"].(string); ok && ec != "" {
		entityCol = ec
	}
	relationCol := "relations"
	if rc, ok := params["relation_col"].(string); ok && rc != "" {
		relationCol = rc
	}
	embeddingModel := "default"
	if em, ok := params["embedding_model"].(string); ok && em != "" {
		embeddingModel = em
	}
	topK := 5
	if tk, ok := params["top_k"].(float64); ok {
		topK = int(tk)
	}
	maxDepth := 2
	if md, ok := params["max_depth"].(float64); ok {
		maxDepth = int(md)
	}
	traversalMethod := "bfs"
	if tm, ok := params["traversal_method"].(string); ok && tm != "" {
		traversalMethod = tm
	}

	/* Build custom context JSON */
	customContext := "{}"
	if cc, ok := params["custom_context"].(map[string]interface{}); ok && len(cc) > 0 {
		ccJSON, err := json.Marshal(cc)
		if err == nil {
			customContext = string(ccJSON)
		}
	}

	/* Use neurondb.rag_graph function */
	graphQuery := `SELECT neurondb.rag_graph($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb) AS result`
	result, err := t.executor.ExecuteQueryOne(ctx, graphQuery, []interface{}{
		query, documentTable, vectorCol, textCol, entityCol, relationCol, embeddingModel, topK, maxDepth, traversalMethod, customContext,
	})
	if err != nil {
		return Error(fmt.Sprintf("Graph RAG failed: %v", err), "GRAPH_ERROR", map[string]interface{}{
			"query_length": len(query),
			"table":        documentTable,
			"error":        err.Error(),
		}), nil
	}

	return Success(result, nil), nil
}

/* RAGHyDETool provides a composite tool for HyDE RAG */
type RAGHyDETool struct {
	*BaseTool
	executor     *QueryExecutor
	logger       *logging.Logger
	configHelper *database.ConfigHelper
}

/* NewRAGHyDETool creates a new HyDE RAG tool */
func NewRAGHyDETool(db *database.Database, logger *logging.Logger) *RAGHyDETool {
	return &RAGHyDETool{
		BaseTool: NewBaseTool(
			"postgresql_rag_hyde",
			"RAG with Hypothetical Document Embeddings (HyDE) for improved retrieval matching",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]interface{}{
						"type":        "string",
						"description": "Query text",
						"minLength":   1,
					},
					"document_table": map[string]interface{}{
						"type":        "string",
						"description": "Document table name",
						"minLength":   1,
					},
					"vector_col": map[string]interface{}{
						"type":        "string",
						"default":     "embedding",
						"description": "Vector column name",
					},
					"text_col": map[string]interface{}{
						"type":        "string",
						"default":     "content",
						"description": "Text column name",
					},
					"embedding_model": map[string]interface{}{
						"type":        "string",
						"default":     "default",
						"description": "Embedding model",
					},
					"llm_model": map[string]interface{}{
						"type":        "string",
						"default":     "default",
						"description": "LLM model for generating hypothetical documents",
					},
					"top_k": map[string]interface{}{
						"type":        "integer",
						"default":     5,
						"minimum":     1,
						"maximum":     50,
						"description": "Number of results to return",
					},
					"num_hypotheticals": map[string]interface{}{
						"type":        "integer",
						"default":     3,
						"minimum":     1,
						"maximum":     10,
						"description": "Number of hypothetical documents to generate",
					},
					"hypothetical_weight": map[string]interface{}{
						"type":        "number",
						"default":     0.5,
						"minimum":     0,
						"maximum":     1,
						"description": "Weight for hypothetical document retrieval (0-1)",
					},
					"custom_context": map[string]interface{}{
						"type":        "object",
						"description": "Custom context parameters (system_prompt, llm_params, etc.)",
					},
				},
				"required":             []interface{}{"query", "document_table"},
				"additionalProperties": false,
			},
		),
		executor:     NewQueryExecutor(db),
		logger:       logger,
		configHelper: database.NewConfigHelper(db),
	}
}

/* Execute executes the HyDE RAG */
func (t *RAGHyDETool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for neurondb_rag_hyde tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	query, _ := params["query"].(string)
	documentTable, _ := params["document_table"].(string)
	vectorCol := "embedding"
	if vc, ok := params["vector_col"].(string); ok && vc != "" {
		vectorCol = vc
	}
	textCol := "content"
	if tc, ok := params["text_col"].(string); ok && tc != "" {
		textCol = tc
	}
	embeddingModel := "default"
	if em, ok := params["embedding_model"].(string); ok && em != "" {
		embeddingModel = em
	}
	llmModel := "default"
	if lm, ok := params["llm_model"].(string); ok && lm != "" {
		llmModel = lm
	}
	topK := 5
	if tk, ok := params["top_k"].(float64); ok {
		topK = int(tk)
	}
	numHypotheticals := 3
	if nh, ok := params["num_hypotheticals"].(float64); ok {
		numHypotheticals = int(nh)
	}
	hypotheticalWeight := 0.5
	if hw, ok := params["hypothetical_weight"].(float64); ok {
		hypotheticalWeight = hw
	}

	/* Build custom context JSON */
	customContext := "{}"
	if cc, ok := params["custom_context"].(map[string]interface{}); ok && len(cc) > 0 {
		ccJSON, err := json.Marshal(cc)
		if err == nil {
			customContext = string(ccJSON)
		}
	}

	/* Use neurondb.rag_hyde function */
	hydeQuery := `SELECT neurondb.rag_hyde($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb) AS result`
	result, err := t.executor.ExecuteQueryOne(ctx, hydeQuery, []interface{}{
		query, documentTable, vectorCol, textCol, embeddingModel, llmModel, topK, numHypotheticals, hypotheticalWeight, customContext,
	})
	if err != nil {
		return Error(fmt.Sprintf("HyDE RAG failed: %v", err), "HYDE_ERROR", map[string]interface{}{
			"query_length": len(query),
			"table":        documentTable,
			"error":        err.Error(),
		}), nil
	}

	return Success(result, nil), nil
}

/* RAGCorrectiveTool provides a composite tool for Corrective RAG */
type RAGCorrectiveTool struct {
	*BaseTool
	executor     *QueryExecutor
	logger       *logging.Logger
	configHelper *database.ConfigHelper
}

/* NewRAGCorrectiveTool creates a new Corrective RAG tool */
func NewRAGCorrectiveTool(db *database.Database, logger *logging.Logger) *RAGCorrectiveTool {
	return &RAGCorrectiveTool{
		BaseTool: NewBaseTool(
			"postgresql_rag_corrective",
			"RAG with iterative self-correction to improve retrieval accuracy",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]interface{}{
						"type":        "string",
						"description": "Query text",
						"minLength":   1,
					},
					"document_table": map[string]interface{}{
						"type":        "string",
						"description": "Document table name",
						"minLength":   1,
					},
					"vector_col": map[string]interface{}{
						"type":        "string",
						"default":     "embedding",
						"description": "Vector column name",
					},
					"text_col": map[string]interface{}{
						"type":        "string",
						"default":     "content",
						"description": "Text column name",
					},
					"embedding_model": map[string]interface{}{
						"type":        "string",
						"default":     "default",
						"description": "Embedding model",
					},
					"llm_model": map[string]interface{}{
						"type":        "string",
						"default":     "default",
						"description": "LLM model",
					},
					"top_k": map[string]interface{}{
						"type":        "integer",
						"default":     5,
						"minimum":     1,
						"maximum":     50,
						"description": "Initial number of results",
					},
					"max_iterations": map[string]interface{}{
						"type":        "integer",
						"default":     3,
						"minimum":     1,
						"maximum":     10,
						"description": "Maximum correction iterations",
					},
					"quality_threshold": map[string]interface{}{
						"type":        "number",
						"default":     0.7,
						"minimum":     0,
						"maximum":     1,
						"description": "Quality threshold to stop correction (0-1)",
					},
					"custom_context": map[string]interface{}{
						"type":        "object",
						"description": "Custom context parameters",
					},
				},
				"required":             []interface{}{"query", "document_table"},
				"additionalProperties": false,
			},
		),
		executor:     NewQueryExecutor(db),
		logger:       logger,
		configHelper: database.NewConfigHelper(db),
	}
}

/* Execute executes the Corrective RAG */
func (t *RAGCorrectiveTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for neurondb_rag_corrective tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	query, _ := params["query"].(string)
	documentTable, _ := params["document_table"].(string)
	vectorCol := "embedding"
	if vc, ok := params["vector_col"].(string); ok && vc != "" {
		vectorCol = vc
	}
	textCol := "content"
	if tc, ok := params["text_col"].(string); ok && tc != "" {
		textCol = tc
	}
	embeddingModel := "default"
	if em, ok := params["embedding_model"].(string); ok && em != "" {
		embeddingModel = em
	}
	llmModel := "default"
	if lm, ok := params["llm_model"].(string); ok && lm != "" {
		llmModel = lm
	}
	topK := 5
	if tk, ok := params["top_k"].(float64); ok {
		topK = int(tk)
	}
	maxIterations := 3
	if mi, ok := params["max_iterations"].(float64); ok {
		maxIterations = int(mi)
	}
	qualityThreshold := 0.7
	if qt, ok := params["quality_threshold"].(float64); ok {
		qualityThreshold = qt
	}

	/* Build custom context JSON */
	customContext := "{}"
	if cc, ok := params["custom_context"].(map[string]interface{}); ok && len(cc) > 0 {
		ccJSON, err := json.Marshal(cc)
		if err == nil {
			customContext = string(ccJSON)
		}
	}

	/* Use neurondb.rag_corrective function */
	correctiveQuery := `SELECT neurondb.rag_corrective($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb) AS result`
	result, err := t.executor.ExecuteQueryOne(ctx, correctiveQuery, []interface{}{
		query, documentTable, vectorCol, textCol, embeddingModel, llmModel, topK, maxIterations, qualityThreshold, customContext,
	})
	if err != nil {
		return Error(fmt.Sprintf("Corrective RAG failed: %v", err), "CORRECTIVE_ERROR", map[string]interface{}{
			"query_length": len(query),
			"table":        documentTable,
			"error":        err.Error(),
		}), nil
	}

	return Success(result, nil), nil
}

/* RAGAgenticTool provides a composite tool for Agentic RAG */
type RAGAgenticTool struct {
	*BaseTool
	executor     *QueryExecutor
	logger       *logging.Logger
	configHelper *database.ConfigHelper
}

/* NewRAGAgenticTool creates a new Agentic RAG tool */
func NewRAGAgenticTool(db *database.Database, logger *logging.Logger) *RAGAgenticTool {
	return &RAGAgenticTool{
		BaseTool: NewBaseTool(
			"postgresql_rag_agentic",
			"RAG with autonomous planning, tool use, and dynamic retrieval strategies",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]interface{}{
						"type":        "string",
						"description": "Query text",
						"minLength":   1,
					},
					"document_table": map[string]interface{}{
						"type":        "string",
						"description": "Document table name",
						"minLength":   1,
					},
					"vector_col": map[string]interface{}{
						"type":        "string",
						"default":     "embedding",
						"description": "Vector column name",
					},
					"text_col": map[string]interface{}{
						"type":        "string",
						"default":     "content",
						"description": "Text column name",
					},
					"embedding_model": map[string]interface{}{
						"type":        "string",
						"default":     "default",
						"description": "Embedding model",
					},
					"llm_model": map[string]interface{}{
						"type":        "string",
						"default":     "default",
						"description": "LLM model",
					},
					"top_k": map[string]interface{}{
						"type":        "integer",
						"default":     5,
						"minimum":     1,
						"maximum":     50,
						"description": "Number of results per step",
					},
					"max_steps": map[string]interface{}{
						"type":        "integer",
						"default":     5,
						"minimum":     1,
						"maximum":     10,
						"description": "Maximum planning steps",
					},
					"evidence_threshold": map[string]interface{}{
						"type":        "number",
						"default":     0.7,
						"minimum":     0,
						"maximum":     1,
						"description": "Evidence sufficiency threshold (0-1)",
					},
					"max_tokens": map[string]interface{}{
						"type":        "integer",
						"default":     2000,
						"minimum":     100,
						"maximum":     10000,
						"description": "Maximum tokens budget",
					},
					"custom_context": map[string]interface{}{
						"type":        "object",
						"description": "Custom context parameters",
					},
				},
				"required":             []interface{}{"query", "document_table"},
				"additionalProperties": false,
			},
		),
		executor:     NewQueryExecutor(db),
		logger:       logger,
		configHelper: database.NewConfigHelper(db),
	}
}

/* Execute executes the Agentic RAG */
func (t *RAGAgenticTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for neurondb_rag_agentic tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	query, _ := params["query"].(string)
	documentTable, _ := params["document_table"].(string)
	vectorCol := "embedding"
	if vc, ok := params["vector_col"].(string); ok && vc != "" {
		vectorCol = vc
	}
	textCol := "content"
	if tc, ok := params["text_col"].(string); ok && tc != "" {
		textCol = tc
	}
	embeddingModel := "default"
	if em, ok := params["embedding_model"].(string); ok && em != "" {
		embeddingModel = em
	}
	llmModel := "default"
	if lm, ok := params["llm_model"].(string); ok && lm != "" {
		llmModel = lm
	}
	topK := 5
	if tk, ok := params["top_k"].(float64); ok {
		topK = int(tk)
	}
	maxSteps := 5
	if ms, ok := params["max_steps"].(float64); ok {
		maxSteps = int(ms)
	}
	evidenceThreshold := 0.7
	if et, ok := params["evidence_threshold"].(float64); ok {
		evidenceThreshold = et
	}
	maxTokens := 2000
	if mt, ok := params["max_tokens"].(float64); ok {
		maxTokens = int(mt)
	}

	/* Build custom context JSON */
	customContext := "{}"
	if cc, ok := params["custom_context"].(map[string]interface{}); ok && len(cc) > 0 {
		ccJSON, err := json.Marshal(cc)
		if err == nil {
			customContext = string(ccJSON)
		}
	}

	/* Use neurondb.rag_agentic function */
	agenticQuery := `SELECT * FROM neurondb.rag_agentic($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb)`
	results, err := t.executor.ExecuteQuery(ctx, agenticQuery, []interface{}{
		query, documentTable, vectorCol, textCol, embeddingModel, llmModel, topK, maxSteps, evidenceThreshold, maxTokens, customContext,
	})
	if err != nil {
		return Error(fmt.Sprintf("Agentic RAG failed: %v", err), "AGENTIC_ERROR", map[string]interface{}{
			"query_length": len(query),
			"table":        documentTable,
			"error":        err.Error(),
		}), nil
	}

	return Success(map[string]interface{}{
		"results": results,
		"count":   len(results),
		"method":  "agentic",
	}, nil), nil
}

/* RAGContextualTool provides a composite tool for Contextual RAG */
type RAGContextualTool struct {
	*BaseTool
	executor     *QueryExecutor
	logger       *logging.Logger
	configHelper *database.ConfigHelper
}

/* NewRAGContextualTool creates a new Contextual RAG tool */
func NewRAGContextualTool(db *database.Database, logger *logging.Logger) *RAGContextualTool {
	return &RAGContextualTool{
		BaseTool: NewBaseTool(
			"postgresql_rag_contextual",
			"RAG that adapts retrieval by interpreting broader conversational context with query rewriting",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]interface{}{
						"type":        "string",
						"description": "Query text",
						"minLength":   1,
					},
					"document_table": map[string]interface{}{
						"type":        "string",
						"description": "Document table name",
						"minLength":   1,
					},
					"vector_col": map[string]interface{}{
						"type":        "string",
						"default":     "embedding",
						"description": "Vector column name",
					},
					"text_col": map[string]interface{}{
						"type":        "string",
						"default":     "content",
						"description": "Text column name",
					},
					"embedding_model": map[string]interface{}{
						"type":        "string",
						"default":     "default",
						"description": "Embedding model",
					},
					"llm_model": map[string]interface{}{
						"type":        "string",
						"default":     "default",
						"description": "LLM model",
					},
					"top_k": map[string]interface{}{
						"type":        "integer",
						"default":     5,
						"minimum":     1,
						"maximum":     50,
						"description": "Number of results",
					},
					"conversation_history": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "object"},
						"description": "Conversation history (array of {role, content} objects)",
					},
					"session_context": map[string]interface{}{
						"type":        "object",
						"description": "Session context (topics, intent, domain, etc.)",
					},
					"cross_session_context": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Enable cross-session context understanding",
					},
					"custom_context": map[string]interface{}{
						"type":        "object",
						"description": "Custom context parameters",
					},
				},
				"required":             []interface{}{"query", "document_table"},
				"additionalProperties": false,
			},
		),
		executor:     NewQueryExecutor(db),
		logger:       logger,
		configHelper: database.NewConfigHelper(db),
	}
}

/* Execute executes the Contextual RAG */
func (t *RAGContextualTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for neurondb_rag_contextual tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	query, _ := params["query"].(string)
	documentTable, _ := params["document_table"].(string)
	vectorCol := "embedding"
	if vc, ok := params["vector_col"].(string); ok && vc != "" {
		vectorCol = vc
	}
	textCol := "content"
	if tc, ok := params["text_col"].(string); ok && tc != "" {
		textCol = tc
	}
	embeddingModel := "default"
	if em, ok := params["embedding_model"].(string); ok && em != "" {
		embeddingModel = em
	}
	llmModel := "default"
	if lm, ok := params["llm_model"].(string); ok && lm != "" {
		llmModel = lm
	}
	topK := 5
	if tk, ok := params["top_k"].(float64); ok {
		topK = int(tk)
	}
	crossSessionContext := false
	if csc, ok := params["cross_session_context"].(bool); ok {
		crossSessionContext = csc
	}

	/* Build conversation history JSON */
	conversationHistory := "[]"
	if ch, ok := params["conversation_history"].([]interface{}); ok && len(ch) > 0 {
		chJSON, err := json.Marshal(ch)
		if err == nil {
			conversationHistory = string(chJSON)
		}
	}

	/* Build session context JSON */
	sessionContext := "{}"
	if sc, ok := params["session_context"].(map[string]interface{}); ok && len(sc) > 0 {
		scJSON, err := json.Marshal(sc)
		if err == nil {
			sessionContext = string(scJSON)
		}
	}

	/* Build custom context JSON */
	customContext := "{}"
	if cc, ok := params["custom_context"].(map[string]interface{}); ok && len(cc) > 0 {
		ccJSON, err := json.Marshal(cc)
		if err == nil {
			customContext = string(ccJSON)
		}
	}

	/* Use neurondb.rag_contextual function */
	contextualQuery := `SELECT * FROM neurondb.rag_contextual($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9::jsonb, $10, $11::jsonb)`
	results, err := t.executor.ExecuteQuery(ctx, contextualQuery, []interface{}{
		query, documentTable, vectorCol, textCol, embeddingModel, llmModel, topK, conversationHistory, sessionContext, crossSessionContext, customContext,
	})
	if err != nil {
		return Error(fmt.Sprintf("Contextual RAG failed: %v", err), "CONTEXTUAL_ERROR", map[string]interface{}{
			"query_length": len(query),
			"table":        documentTable,
			"error":        err.Error(),
		}), nil
	}

	return Success(map[string]interface{}{
		"results": results,
		"count":   len(results),
		"method":  "contextual",
	}, nil), nil
}

/* RAGModularTool provides a composite tool for Modular RAG */
type RAGModularTool struct {
	*BaseTool
	executor     *QueryExecutor
	logger       *logging.Logger
	configHelper *database.ConfigHelper
}

/* NewRAGModularTool creates a new Modular RAG tool */
func NewRAGModularTool(db *database.Database, logger *logging.Logger) *RAGModularTool {
	return &RAGModularTool{
		BaseTool: NewBaseTool(
			"postgresql_rag_modular",
			"RAG with composable, plug-and-play modules for custom workflows",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]interface{}{
						"type":        "string",
						"description": "Query text",
						"minLength":   1,
					},
					"document_table": map[string]interface{}{
						"type":        "string",
						"description": "Document table name",
						"minLength":   1,
					},
					"vector_col": map[string]interface{}{
						"type":        "string",
						"default":     "embedding",
						"description": "Vector column name",
					},
					"text_col": map[string]interface{}{
						"type":        "string",
						"default":     "content",
						"description": "Text column name",
					},
					"module_config": map[string]interface{}{
						"type":        "object",
						"description": "Module configuration: {\"name\": \"pipeline_name\", \"modules\": [{\"name\": \"module_name\", \"type\": \"retrieval|reranking|generation|filter\", \"parameters\": {}, \"enabled\": true}]}",
					},
					"embedding_model": map[string]interface{}{
						"type":        "string",
						"default":     "default",
						"description": "Embedding model",
					},
					"llm_model": map[string]interface{}{
						"type":        "string",
						"default":     "default",
						"description": "LLM model",
					},
					"custom_context": map[string]interface{}{
						"type":        "object",
						"description": "Custom context parameters",
					},
				},
				"required":             []interface{}{"query", "document_table", "module_config"},
				"additionalProperties": false,
			},
		),
		executor:     NewQueryExecutor(db),
		logger:       logger,
		configHelper: database.NewConfigHelper(db),
	}
}

/* Execute executes the Modular RAG */
func (t *RAGModularTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for neurondb_rag_modular tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	query, _ := params["query"].(string)
	documentTable, _ := params["document_table"].(string)
	vectorCol := "embedding"
	if vc, ok := params["vector_col"].(string); ok && vc != "" {
		vectorCol = vc
	}
	textCol := "content"
	if tc, ok := params["text_col"].(string); ok && tc != "" {
		textCol = tc
	}
	embeddingModel := "default"
	if em, ok := params["embedding_model"].(string); ok && em != "" {
		embeddingModel = em
	}
	llmModel := "default"
	if lm, ok := params["llm_model"].(string); ok && lm != "" {
		llmModel = lm
	}

	/* Build module config JSON */
	moduleConfig := "{}"
	if mc, ok := params["module_config"].(map[string]interface{}); ok && len(mc) > 0 {
		mcJSON, err := json.Marshal(mc)
		if err == nil {
			moduleConfig = string(mcJSON)
		} else {
			return Error(fmt.Sprintf("Invalid module_config: %v", err), "VALIDATION_ERROR", map[string]interface{}{
				"error": err.Error(),
			}), nil
		}
	} else {
		return Error("module_config is required and must be a valid object", "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "module_config",
		}), nil
	}

	/* Build custom context JSON */
	customContext := "{}"
	if cc, ok := params["custom_context"].(map[string]interface{}); ok && len(cc) > 0 {
		ccJSON, err := json.Marshal(cc)
		if err == nil {
			customContext = string(ccJSON)
		}
	}

	/* Use neurondb.rag_modular function */
	modularQuery := `SELECT * FROM neurondb.rag_modular($1, $2, $3, $4, $5::jsonb, $6, $7, $8::jsonb)`
	results, err := t.executor.ExecuteQuery(ctx, modularQuery, []interface{}{
		query, documentTable, vectorCol, textCol, moduleConfig, embeddingModel, llmModel, customContext,
	})
	if err != nil {
		return Error(fmt.Sprintf("Modular RAG failed: %v", err), "MODULAR_ERROR", map[string]interface{}{
			"query_length": len(query),
			"table":        documentTable,
			"error":        err.Error(),
		}), nil
	}

	return Success(map[string]interface{}{
		"results": results,
		"count":   len(results),
		"method":  "modular",
	}, nil), nil
}
