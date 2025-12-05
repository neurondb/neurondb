package tools

import (
	"context"
	"fmt"
	"strings"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

// RerankCrossEncoderTool performs cross-encoder reranking
type RerankCrossEncoderTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewRerankCrossEncoderTool creates a new cross-encoder reranking tool
func NewRerankCrossEncoderTool(db *database.Database, logger *logging.Logger) *RerankCrossEncoderTool {
	return &RerankCrossEncoderTool{
		BaseTool: NewBaseTool(
			"rerank_cross_encoder",
			"Rerank documents using cross-encoder model",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]interface{}{
						"type":        "string",
						"description": "Query text",
					},
					"documents": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "string"},
						"description": "Array of document texts to rerank",
					},
					"model": map[string]interface{}{
						"type":        "string",
						"default":     "ms-marco-MiniLM-L-6-v2",
						"description": "Cross-encoder model name",
					},
					"top_k": map[string]interface{}{
						"type":        "number",
						"default":     10,
						"minimum":     1,
						"maximum":     1000,
						"description": "Number of top results to return",
					},
				},
				"required": []interface{}{"query", "documents"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes cross-encoder reranking
func (t *RerankCrossEncoderTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for rerank_cross_encoder tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	query, _ := params["query"].(string)
	documents, _ := params["documents"].([]interface{})
	model := "ms-marco-MiniLM-L-6-v2"
	if m, ok := params["model"].(string); ok && m != "" {
		model = m
	}
	topK := 10
	if k, ok := params["top_k"].(float64); ok {
		topK = int(k)
	}

	if query == "" || len(documents) == 0 {
		return Error("query and documents are required", "VALIDATION_ERROR", nil), nil
	}

	// Format documents array
	var docStrs []string
	for _, doc := range documents {
		if docStr, ok := doc.(string); ok {
			docStrs = append(docStrs, fmt.Sprintf("'%s'", strings.ReplaceAll(docStr, "'", "''")))
		}
	}
	docsStr := "ARRAY[" + strings.Join(docStrs, ",") + "]::text[]"

	sqlQuery := fmt.Sprintf("SELECT * FROM rerank_cross_encoder($1::text, %s, $2::text, $3::int)", docsStr)
	queryParams := []interface{}{query, model, topK}

	results, err := t.executor.ExecuteQuery(ctx, sqlQuery, queryParams)
	if err != nil {
		t.logger.Error("Cross-encoder reranking failed", err, params)
		return Error(fmt.Sprintf("Cross-encoder reranking failed: error=%v", err), "EXECUTION_ERROR", map[string]interface{}{
			"error": err.Error(),
		}), nil
	}

	return Success(map[string]interface{}{
		"results": results,
		"count":   len(results),
	}, map[string]interface{}{
		"count": len(results),
	}), nil
}

// RerankLLMTool performs LLM-based reranking
type RerankLLMTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewRerankLLMTool creates a new LLM reranking tool
func NewRerankLLMTool(db *database.Database, logger *logging.Logger) *RerankLLMTool {
	return &RerankLLMTool{
		BaseTool: NewBaseTool(
			"rerank_llm",
			"Rerank documents using LLM",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]interface{}{
						"type":        "string",
						"description": "Query text",
					},
					"documents": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "string"},
						"description": "Array of document texts to rerank",
					},
					"model": map[string]interface{}{
						"type":        "string",
						"default":     "gpt-3.5-turbo",
						"description": "LLM model name",
					},
					"top_k": map[string]interface{}{
						"type":        "number",
						"default":     10,
						"minimum":     1,
						"maximum":     1000,
						"description": "Number of top results to return",
					},
				},
				"required": []interface{}{"query", "documents"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes LLM reranking
func (t *RerankLLMTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for rerank_llm tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	query, _ := params["query"].(string)
	documents, _ := params["documents"].([]interface{})
	model := "gpt-3.5-turbo"
	if m, ok := params["model"].(string); ok && m != "" {
		model = m
	}
	topK := 10
	if k, ok := params["top_k"].(float64); ok {
		topK = int(k)
	}

	if query == "" || len(documents) == 0 {
		return Error("query and documents are required", "VALIDATION_ERROR", nil), nil
	}

	// Format documents array
	var docStrs []string
	for _, doc := range documents {
		if docStr, ok := doc.(string); ok {
			docStrs = append(docStrs, fmt.Sprintf("'%s'", strings.ReplaceAll(docStr, "'", "''")))
		}
	}
	docsStr := "ARRAY[" + strings.Join(docStrs, ",") + "]::text[]"

	sqlQuery := fmt.Sprintf("SELECT * FROM rerank_llm($1::text, %s, $2::text, $3::int)", docsStr)
	queryParams := []interface{}{query, model, topK}

	results, err := t.executor.ExecuteQuery(ctx, sqlQuery, queryParams)
	if err != nil {
		t.logger.Error("LLM reranking failed", err, params)
		return Error(fmt.Sprintf("LLM reranking failed: error=%v", err), "EXECUTION_ERROR", map[string]interface{}{
			"error": err.Error(),
		}), nil
	}

	return Success(map[string]interface{}{
		"results": results,
		"count":   len(results),
	}, map[string]interface{}{
		"count": len(results),
	}), nil
}

// RerankCohereTool performs Cohere reranking
type RerankCohereTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewRerankCohereTool creates a new Cohere reranking tool
func NewRerankCohereTool(db *database.Database, logger *logging.Logger) *RerankCohereTool {
	return &RerankCohereTool{
		BaseTool: NewBaseTool(
			"rerank_cohere",
			"Rerank documents using Cohere API",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]interface{}{
						"type":        "string",
						"description": "Query text",
					},
					"documents": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "string"},
						"description": "Array of document texts to rerank",
					},
					"top_k": map[string]interface{}{
						"type":        "number",
						"default":     10,
						"minimum":     1,
						"maximum":     1000,
						"description": "Number of top results to return",
					},
				},
				"required": []interface{}{"query", "documents"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes Cohere reranking
func (t *RerankCohereTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for rerank_cohere tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	query, _ := params["query"].(string)
	documents, _ := params["documents"].([]interface{})
	topK := 10
	if k, ok := params["top_k"].(float64); ok {
		topK = int(k)
	}

	if query == "" || len(documents) == 0 {
		return Error("query and documents are required", "VALIDATION_ERROR", nil), nil
	}

	// Format documents array
	var docStrs []string
	for _, doc := range documents {
		if docStr, ok := doc.(string); ok {
			docStrs = append(docStrs, fmt.Sprintf("'%s'", strings.ReplaceAll(docStr, "'", "''")))
		}
	}
	docsStr := "ARRAY[" + strings.Join(docStrs, ",") + "]::text[]"

	sqlQuery := fmt.Sprintf("SELECT * FROM rerank_cohere($1::text, %s, $2::int)", docsStr)
	queryParams := []interface{}{query, topK}

	results, err := t.executor.ExecuteQuery(ctx, sqlQuery, queryParams)
	if err != nil {
		t.logger.Error("Cohere reranking failed", err, params)
		return Error(fmt.Sprintf("Cohere reranking failed: error=%v", err), "EXECUTION_ERROR", map[string]interface{}{
			"error": err.Error(),
		}), nil
	}

	return Success(map[string]interface{}{
		"results": results,
		"count":   len(results),
	}, map[string]interface{}{
		"count": len(results),
	}), nil
}

// RerankColBERTTool performs ColBERT reranking
type RerankColBERTTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewRerankColBERTTool creates a new ColBERT reranking tool
func NewRerankColBERTTool(db *database.Database, logger *logging.Logger) *RerankColBERTTool {
	return &RerankColBERTTool{
		BaseTool: NewBaseTool(
			"rerank_colbert",
			"Rerank documents using ColBERT model",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]interface{}{
						"type":        "string",
						"description": "Query text",
					},
					"documents": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "string"},
						"description": "Array of document texts to rerank",
					},
					"model": map[string]interface{}{
						"type":        "string",
						"default":     "colbert-v2",
						"description": "ColBERT model name",
					},
				},
				"required": []interface{}{"query", "documents"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes ColBERT reranking
func (t *RerankColBERTTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for rerank_colbert tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	query, _ := params["query"].(string)
	documents, _ := params["documents"].([]interface{})
	model := "colbert-v2"
	if m, ok := params["model"].(string); ok && m != "" {
		model = m
	}

	if query == "" || len(documents) == 0 {
		return Error("query and documents are required", "VALIDATION_ERROR", nil), nil
	}

	// Format documents array
	var docStrs []string
	for _, doc := range documents {
		if docStr, ok := doc.(string); ok {
			docStrs = append(docStrs, fmt.Sprintf("'%s'", strings.ReplaceAll(docStr, "'", "''")))
		}
	}
	docsStr := "ARRAY[" + strings.Join(docStrs, ",") + "]::text[]"

	sqlQuery := fmt.Sprintf("SELECT * FROM rerank_colbert($1::text, %s, $2::text)", docsStr)
	queryParams := []interface{}{query, model}

	results, err := t.executor.ExecuteQuery(ctx, sqlQuery, queryParams)
	if err != nil {
		t.logger.Error("ColBERT reranking failed", err, params)
		return Error(fmt.Sprintf("ColBERT reranking failed: error=%v", err), "EXECUTION_ERROR", map[string]interface{}{
			"error": err.Error(),
		}), nil
	}

	return Success(map[string]interface{}{
		"results": results,
		"count":   len(results),
	}, map[string]interface{}{
		"count": len(results),
	}), nil
}

// RerankLTRTool performs learning-to-rank reranking
type RerankLTRTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewRerankLTRTool creates a new LTR reranking tool
func NewRerankLTRTool(db *database.Database, logger *logging.Logger) *RerankLTRTool {
	return &RerankLTRTool{
		BaseTool: NewBaseTool(
			"rerank_ltr",
			"Rerank documents using learning-to-rank",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]interface{}{
						"type":        "string",
						"description": "Query text",
					},
					"documents": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "string"},
						"description": "Array of document texts to rerank",
					},
					"feature_table": map[string]interface{}{
						"type":        "string",
						"description": "Feature table name",
					},
					"model_table": map[string]interface{}{
						"type":        "string",
						"description": "LTR model table name",
					},
				},
				"required": []interface{}{"query", "documents", "feature_table", "model_table"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes LTR reranking
func (t *RerankLTRTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for rerank_ltr tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	query, _ := params["query"].(string)
	documents, _ := params["documents"].([]interface{})
	featureTable, _ := params["feature_table"].(string)
	modelTable, _ := params["model_table"].(string)

	if query == "" || len(documents) == 0 || featureTable == "" || modelTable == "" {
		return Error("query, documents, feature_table, and model_table are required", "VALIDATION_ERROR", nil), nil
	}

	// Format documents array
	var docStrs []string
	for _, doc := range documents {
		if docStr, ok := doc.(string); ok {
			docStrs = append(docStrs, fmt.Sprintf("'%s'", strings.ReplaceAll(docStr, "'", "''")))
		}
	}
	docsStr := "ARRAY[" + strings.Join(docStrs, ",") + "]::text[]"

	sqlQuery := fmt.Sprintf("SELECT * FROM rerank_ltr($1::text, %s, $2::text, $3::text)", docsStr)
	queryParams := []interface{}{query, featureTable, modelTable}

	results, err := t.executor.ExecuteQuery(ctx, sqlQuery, queryParams)
	if err != nil {
		t.logger.Error("LTR reranking failed", err, params)
		return Error(fmt.Sprintf("LTR reranking failed: error=%v", err), "EXECUTION_ERROR", map[string]interface{}{
			"error": err.Error(),
		}), nil
	}

	return Success(map[string]interface{}{
		"results": results,
		"count":   len(results),
	}, map[string]interface{}{
		"count": len(results),
	}), nil
}

// RerankEnsembleTool performs ensemble reranking
type RerankEnsembleTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewRerankEnsembleTool creates a new ensemble reranking tool
func NewRerankEnsembleTool(db *database.Database, logger *logging.Logger) *RerankEnsembleTool {
	return &RerankEnsembleTool{
		BaseTool: NewBaseTool(
			"rerank_ensemble",
			"Rerank documents using ensemble of multiple rerankers",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]interface{}{
						"type":        "string",
						"description": "Query text",
					},
					"documents": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "string"},
						"description": "Array of document texts to rerank",
					},
					"rerankers": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "string"},
						"description": "Array of reranker names",
					},
					"weights": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "number"},
						"description": "Array of weights for each reranker",
					},
				},
				"required": []interface{}{"query", "documents", "rerankers", "weights"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes ensemble reranking
func (t *RerankEnsembleTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for rerank_ensemble tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	query, _ := params["query"].(string)
	documents, _ := params["documents"].([]interface{})
	rerankers, _ := params["rerankers"].([]interface{})
	weights, _ := params["weights"].([]interface{})

	if query == "" || len(documents) == 0 || len(rerankers) == 0 || len(weights) == 0 {
		return Error("query, documents, rerankers, and weights are required", "VALIDATION_ERROR", nil), nil
	}

	if len(rerankers) != len(weights) {
		return Error("rerankers and weights arrays must have the same length", "VALIDATION_ERROR", nil), nil
	}

	// Format arrays
	var docStrs []string
	for _, doc := range documents {
		if docStr, ok := doc.(string); ok {
			docStrs = append(docStrs, fmt.Sprintf("'%s'", strings.ReplaceAll(docStr, "'", "''")))
		}
	}
	docsStr := "ARRAY[" + strings.Join(docStrs, ",") + "]::text[]"

	var rerankerStrs []string
	for _, reranker := range rerankers {
		if rStr, ok := reranker.(string); ok {
			rerankerStrs = append(rerankerStrs, fmt.Sprintf("'%s'", strings.ReplaceAll(rStr, "'", "''")))
		}
	}
	rerankersStr := "ARRAY[" + strings.Join(rerankerStrs, ",") + "]::text[]"

	var weightStrs []string
	for _, weight := range weights {
		if w, ok := weight.(float64); ok {
			weightStrs = append(weightStrs, fmt.Sprintf("%g", w))
		}
	}
	weightsStr := "ARRAY[" + strings.Join(weightStrs, ",") + "]::float8[]"

	sqlQuery := fmt.Sprintf("SELECT * FROM rerank_ensemble($1::text, %s, %s, %s)", docsStr, rerankersStr, weightsStr)
	queryParams := []interface{}{query}

	results, err := t.executor.ExecuteQuery(ctx, sqlQuery, queryParams)
	if err != nil {
		t.logger.Error("Ensemble reranking failed", err, params)
		return Error(fmt.Sprintf("Ensemble reranking failed: error=%v", err), "EXECUTION_ERROR", map[string]interface{}{
			"error": err.Error(),
		}), nil
	}

	return Success(map[string]interface{}{
		"results": results,
		"count":   len(results),
	}, map[string]interface{}{
		"count": len(results),
	}), nil
}

