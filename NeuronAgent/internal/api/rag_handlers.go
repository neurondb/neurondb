/*-------------------------------------------------------------------------
 *
 * rag_handlers.go
 *    RAG API handlers for NeuronAgent
 *
 * Provides REST API endpoints for RAG operations including query, ingest,
 * evaluate, and pipeline management.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/api/rag_handlers.go
 *
 *-------------------------------------------------------------------------
 */

package api

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/google/uuid"
	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronAgent/internal/agent"
	"github.com/neurondb/NeuronAgent/internal/auth"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/pkg/neurondb"
)

type RAGHandlers struct {
	queries       *db.Queries
	advancedRAG   *agent.AdvancedRAG
	ragClient     *neurondb.RAGClient
	hybridClient  *neurondb.HybridSearchClient
	rerankClient  *neurondb.RerankingClient
	embedClient   *neurondb.EmbeddingClient
}

func NewRAGHandlers(
	queries *db.Queries,
	advancedRAG *agent.AdvancedRAG,
	ragClient *neurondb.RAGClient,
	hybridClient *neurondb.HybridSearchClient,
	rerankClient *neurondb.RerankingClient,
	embedClient *neurondb.EmbeddingClient,
) *RAGHandlers {
	return &RAGHandlers{
		queries:      queries,
		advancedRAG:  advancedRAG,
		ragClient:    ragClient,
		hybridClient: hybridClient,
		rerankClient: rerankClient,
		embedClient:  embedClient,
	}
}

/* RAG Query Request/Response */

type RAGQueryRequest struct {
	Query           string                 `json:"query"`
	TableName       string                 `json:"table_name"`
	VectorCol       string                 `json:"vector_col"`
	TextCol         string                 `json:"text_col"`
	Model           string                 `json:"model,omitempty"`
	TopK            int                    `json:"top_k,omitempty"`
	Rerank          bool                   `json:"rerank,omitempty"`
	RerankModel     string                 `json:"rerank_model,omitempty"`
	Hybrid          bool                   `json:"hybrid,omitempty"`
	VectorWeight    float64                `json:"vector_weight,omitempty"`
	Temporal        bool                   `json:"temporal,omitempty"`
	RecencyWeight   float64                `json:"recency_weight,omitempty"`
	Faceted         bool                   `json:"faceted,omitempty"`
	Categories      []string               `json:"categories,omitempty"`
	CustomContext   map[string]interface{} `json:"custom_context,omitempty"`
	/* RAG Architecture Selection */
	Architecture    string                 `json:"architecture,omitempty"` /* naive, hyde, graph, corrective, hybrid, agentic, contextual, modular */
	/* HyDE parameters */
	NumHypotheticals     int               `json:"num_hypotheticals,omitempty"`
	HypotheticalWeight   float64           `json:"hypothetical_weight,omitempty"`
	/* Graph RAG parameters */
	EntityCol            string            `json:"entity_col,omitempty"`
	RelationCol          string            `json:"relation_col,omitempty"`
	MaxDepth             int               `json:"max_depth,omitempty"`
	TraversalMethod      string            `json:"traversal_method,omitempty"`
	/* Corrective RAG parameters */
	MaxIterations        int               `json:"max_iterations,omitempty"`
	QualityThreshold     float64           `json:"quality_threshold,omitempty"`
	/* Agentic RAG parameters */
	MaxSteps             int               `json:"max_steps,omitempty"`
	EvidenceThreshold    float64           `json:"evidence_threshold,omitempty"`
	MaxTokens            int               `json:"max_tokens,omitempty"`
	/* Contextual RAG parameters */
	ConversationHistory  []map[string]interface{} `json:"conversation_history,omitempty"`
	SessionContext       map[string]interface{}   `json:"session_context,omitempty"`
	CrossSessionContext  bool                     `json:"cross_session_context,omitempty"`
	/* Modular RAG parameters */
	ModuleConfig         map[string]interface{}   `json:"module_config,omitempty"`
}

type RAGQueryResponse struct {
	Answer    string                   `json:"answer"`
	Documents []string                 `json:"documents"`
	Count     int                      `json:"count"`
	Method    string                   `json:"method"`
	Metadata  map[string]interface{}   `json:"metadata,omitempty"`
}

/* RAG Ingest Request/Response */

type RAGIngestRequest struct {
	DocumentText   string                 `json:"document_text"`
	TableName      string                 `json:"table_name"`
	TextCol        string                 `json:"text_col,omitempty"`
	VectorCol      string                 `json:"vector_col,omitempty"`
	EmbeddingModel string                 `json:"embedding_model,omitempty"`
	ChunkSize      int                    `json:"chunk_size,omitempty"`
	ChunkOverlap   int                    `json:"chunk_overlap,omitempty"`
	Metadata       map[string]interface{} `json:"metadata,omitempty"`
}

type RAGIngestResponse struct {
	ChunksCreated int      `json:"chunks_created"`
	ChunkIDs      []int64  `json:"chunk_ids,omitempty"`
	Message       string   `json:"message"`
}

/* RAG Evaluate Request/Response */

type RAGEvaluateRequest struct {
	Query         string   `json:"query"`
	Answer        string   `json:"answer"`
	ContextChunks []string `json:"context_chunks"`
	EvaluationType string  `json:"evaluation_type,omitempty"`
}

type RAGEvaluateResponse struct {
	Faithfulness       float64                `json:"faithfulness"`
	Relevancy          float64                `json:"relevancy"`
	ContextPrecision   float64                `json:"context_precision"`
	ContextRecall      float64                `json:"context_recall"`
	SemanticSimilarity float64                `json:"semantic_similarity"`
	OverallScore       float64                `json:"overall_score"`
	Metadata           map[string]interface{} `json:"metadata,omitempty"`
}

/* RAG Pipeline Request/Response */

type RAGPipelineRequest struct {
	PipelineName    string                 `json:"pipeline_name"`
	EmbeddingModel  string                 `json:"embedding_model"`
	ChunkSize       int                    `json:"chunk_size,omitempty"`
	ChunkOverlap    int                    `json:"chunk_overlap,omitempty"`
	RerankEnabled   bool                   `json:"rerank_enabled,omitempty"`
	RerankTopK      int                    `json:"rerank_top_k,omitempty"`
	HybridEnabled   bool                   `json:"hybrid_enabled,omitempty"`
	VectorWeight    float64                `json:"vector_weight,omitempty"`
	EvaluationEnabled bool                 `json:"evaluation_enabled,omitempty"`
	LLMModel        string                 `json:"llm_model,omitempty"`
	Config          map[string]interface{} `json:"config,omitempty"`
}

type RAGPipelineResponse struct {
	PipelineID     int                    `json:"pipeline_id"`
	PipelineName   string                 `json:"pipeline_name"`
	EmbeddingModel string                 `json:"embedding_model"`
	Config         map[string]interface{} `json:"config"`
	CreatedAt      string                 `json:"created_at"`
}

/* RAG Query Handler */

func (h *RAGHandlers) RAGQuery(w http.ResponseWriter, r *http.Request) {
	requestID := GetRequestID(r.Context())
	endpoint := r.URL.Path
	method := r.Method

	/* Check authorization */
	apiKey, ok := GetAPIKeyFromContext(r.Context())
	if !ok {
		respondError(w, WrapError(ErrUnauthorized, requestID))
		return
	}
	if err := auth.RequireAnyRole(apiKey, auth.RoleAdmin, auth.RoleUser); err != nil {
		respondError(w, NewErrorWithContext(http.StatusForbidden, "insufficient permissions", err, requestID, endpoint, method, "rag", "", nil))
		return
	}

	/* Parse request */
	var req RAGQueryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "RAG query failed: request parsing error", err, requestID, endpoint, method, "rag", "", nil))
		return
	}

	/* Validate request */
	if req.Query == "" {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "RAG query failed: query is required", nil, requestID, endpoint, method, "rag", "", nil))
		return
	}
	if req.TableName == "" {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "RAG query failed: table_name is required", nil, requestID, endpoint, method, "rag", "", nil))
		return
	}

	/* Set defaults */
	if req.VectorCol == "" {
		req.VectorCol = "embedding"
	}
	if req.TextCol == "" {
		req.TextCol = "content"
	}
	if req.Model == "" {
		req.Model = "default"
	}
	if req.TopK == 0 {
		req.TopK = 5
	}

	ctx := r.Context()
	
	/* Determine architecture - use explicit architecture parameter if provided, otherwise infer from flags */
	architecture := req.Architecture
	if architecture == "" {
		/* Infer from flags for backward compatibility */
		if req.Hybrid {
			architecture = "hybrid"
		} else if req.Rerank {
			architecture = "reranked"
		} else if req.Temporal {
			architecture = "temporal"
		} else if req.Faceted {
			architecture = "faceted"
		} else {
			architecture = "naive"
		}
	}

	var result *agent.RAGResult
	var agenticResult *agent.AgenticRAGResult
	var contextualResult *agent.ContextualRAGResult
	var modularResult *agent.ModularRAGResult
	var err error

	/* Execute RAG based on architecture */
	switch architecture {
	case "hyde":
		/* HyDE RAG */
		numHypotheticals := req.NumHypotheticals
		if numHypotheticals == 0 {
			numHypotheticals = 3
		}
		hypotheticalWeight := req.HypotheticalWeight
		if hypotheticalWeight == 0 {
			hypotheticalWeight = 0.5
		}
		result, err = h.advancedRAG.HyDERAG(ctx, req.Query, req.TableName, req.VectorCol, req.TextCol, req.TopK, numHypotheticals, hypotheticalWeight)

	case "graph":
		/* Graph RAG */
		entityCol := req.EntityCol
		if entityCol == "" {
			entityCol = "entities"
		}
		relationCol := req.RelationCol
		if relationCol == "" {
			relationCol = "relations"
		}
		maxDepth := req.MaxDepth
		if maxDepth == 0 {
			maxDepth = 2
		}
		traversalMethod := req.TraversalMethod
		if traversalMethod == "" {
			traversalMethod = "bfs"
		}
		result, err = h.advancedRAG.GraphRAG(ctx, req.Query, req.TableName, req.VectorCol, req.TextCol, entityCol, relationCol, req.TopK, maxDepth, traversalMethod)

	case "corrective":
		/* Corrective RAG */
		maxIterations := req.MaxIterations
		if maxIterations == 0 {
			maxIterations = 3
		}
		qualityThreshold := req.QualityThreshold
		if qualityThreshold == 0 {
			qualityThreshold = 0.7
		}
		result, err = h.advancedRAG.CorrectiveRAG(ctx, req.Query, req.TableName, req.VectorCol, req.TextCol, req.TopK, maxIterations, qualityThreshold)

	case "hybrid":
		/* Hybrid RAG */
		vectorWeight := req.VectorWeight
		if vectorWeight == 0 {
			vectorWeight = 0.7
		}
		result, err = h.advancedRAG.HybridRAG(ctx, req.Query, req.TableName, req.VectorCol, req.TextCol, req.TopK, vectorWeight)

	case "agentic":
		/* Agentic RAG */
		maxSteps := req.MaxSteps
		if maxSteps == 0 {
			maxSteps = 5
		}
		evidenceThreshold := req.EvidenceThreshold
		if evidenceThreshold == 0 {
			evidenceThreshold = 0.7
		}
		maxTokens := req.MaxTokens
		if maxTokens == 0 {
			maxTokens = 2000
		}
		agenticResult, err = h.advancedRAG.AgenticRAG(ctx, req.Query, req.TableName, req.VectorCol, req.TextCol, req.TopK, maxSteps, evidenceThreshold, maxTokens)
		if err == nil && agenticResult != nil {
			/* Convert AgenticRAGResult to RAGResult for response */
			result = &agent.RAGResult{
				Query:     agenticResult.Query,
				Answer:    agenticResult.Answer,
				Documents: agenticResult.Documents,
				Count:     agenticResult.Count,
				Method:    agenticResult.Method,
			}
		}

	case "contextual":
		/* Contextual RAG */
		conversationHistory := req.ConversationHistory
		if conversationHistory == nil {
			conversationHistory = []map[string]interface{}{}
		}
		sessionContext := req.SessionContext
		if sessionContext == nil {
			sessionContext = make(map[string]interface{})
		}
		contextualResult, err = h.advancedRAG.ContextualRAG(ctx, req.Query, req.TableName, req.VectorCol, req.TextCol, req.TopK, conversationHistory, sessionContext, req.CrossSessionContext)
		if err == nil && contextualResult != nil {
			/* Convert ContextualRAGResult to RAGResult for response */
			result = &agent.RAGResult{
				Query:     contextualResult.Query,
				Answer:    contextualResult.Answer,
				Documents: contextualResult.Documents,
				Count:     contextualResult.Count,
				Method:    contextualResult.Method,
			}
		}

	case "modular":
		/* Modular RAG */
		if req.ModuleConfig == nil || len(req.ModuleConfig) == 0 {
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "RAG query failed: module_config is required for modular RAG", nil, requestID, endpoint, method, "rag", "", nil))
			return
		}
		moduleConfigJSON, err2 := json.Marshal(req.ModuleConfig)
		if err2 != nil {
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "RAG query failed: invalid module_config", err2, requestID, endpoint, method, "rag", "", nil))
			return
		}
		modularResult, err = h.advancedRAG.ModularRAG(ctx, req.Query, req.TableName, req.VectorCol, req.TextCol, string(moduleConfigJSON))
		if err == nil && modularResult != nil {
			/* Convert ModularRAGResult to RAGResult for response */
			result = &agent.RAGResult{
				Query:     modularResult.Query,
				Answer:    "", /* Modular RAG may not generate answer directly */
				Documents: modularResult.Documents,
				Count:     modularResult.Count,
				Method:    modularResult.Method,
			}
		}

	case "reranked":
		/* Reranked RAG (backward compatibility) */
		initialLimit := req.TopK * 3
		if initialLimit > 50 {
			initialLimit = 50
		}
		rerankModel := req.RerankModel
		if rerankModel == "" {
			rerankModel = "cross-encoder"
		}
		result, err = h.advancedRAG.RerankedRAG(ctx, req.Query, req.TableName, req.VectorCol, initialLimit, req.TopK, rerankModel)

	case "temporal":
		/* Temporal RAG (backward compatibility) */
		recencyWeight := req.RecencyWeight
		if recencyWeight == 0 {
			recencyWeight = 0.3
		}
		result, err = h.advancedRAG.TemporalRAG(ctx, req.Query, req.TableName, req.VectorCol, "created_at", req.TopK, recencyWeight)

	case "faceted":
		/* Faceted RAG (backward compatibility) */
		result, err = h.advancedRAG.FacetedRAG(ctx, req.Query, req.TableName, req.VectorCol, "category", req.Categories, req.TopK)

	default:
		/* Naive RAG - use RAGClient if available, otherwise fallback */
		if h.ragClient != nil {
			/* Use RAGClient for basic RAG */
			answer, err2 := h.ragClient.Query(ctx, req.Query, req.TableName, req.VectorCol, req.TextCol, req.TopK)
			if err2 != nil {
				respondError(w, NewErrorWithContext(http.StatusInternalServerError, "RAG query failed", err2, requestID, endpoint, method, "rag", "", nil))
				return
			}
			result = &agent.RAGResult{
				Query:     req.Query,
				Answer:    answer,
				Documents: []string{},
				Count:     0,
				Method:    "naive",
			}
		} else {
			respondError(w, NewErrorWithContext(http.StatusInternalServerError, "RAG query failed: RAG client not configured", nil, requestID, endpoint, method, "rag", "", nil))
			return
		}
	}

	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "RAG query failed", err, requestID, endpoint, method, "rag", "", nil))
		return
	}

	if result == nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "RAG query failed: no result returned", nil, requestID, endpoint, method, "rag", "", nil))
		return
	}

	/* Build response with architecture-specific metadata */
	metadata := make(map[string]interface{})
	if agenticResult != nil {
		metadata["execution_trace"] = agenticResult.ExecutionTrace
		metadata["reasoning_path"] = agenticResult.ReasoningPath
		metadata["steps_executed"] = agenticResult.StepsExecuted
	}
	if contextualResult != nil {
		metadata["rewritten_query"] = contextualResult.RewrittenQuery
		metadata["context_adaptation"] = contextualResult.ContextAdaptation
	}
	if modularResult != nil {
		metadata["pipeline"] = modularResult.Pipeline
		metadata["module_metadata"] = modularResult.Metadata
	}

	response := RAGQueryResponse{
		Answer:    result.Answer,
		Documents: result.Documents,
		Count:     result.Count,
		Method:    result.Method,
		Metadata:  metadata,
	}

	respondJSON(w, http.StatusOK, response)
}

/* RAG Ingest Handler */

func (h *RAGHandlers) RAGIngest(w http.ResponseWriter, r *http.Request) {
	requestID := GetRequestID(r.Context())
	endpoint := r.URL.Path
	method := r.Method

	/* Check authorization */
	apiKey, ok := GetAPIKeyFromContext(r.Context())
	if !ok {
		respondError(w, WrapError(ErrUnauthorized, requestID))
		return
	}
	if err := auth.RequireAnyRole(apiKey, auth.RoleAdmin, auth.RoleUser); err != nil {
		respondError(w, NewErrorWithContext(http.StatusForbidden, "insufficient permissions", err, requestID, endpoint, method, "rag", "", nil))
		return
	}

	/* Parse request */
	var req RAGIngestRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "RAG ingest failed: request parsing error", err, requestID, endpoint, method, "rag", "", nil))
		return
	}

	/* Validate request */
	if req.DocumentText == "" {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "RAG ingest failed: document_text is required", nil, requestID, endpoint, method, "rag", "", nil))
		return
	}
	if req.TableName == "" {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "RAG ingest failed: table_name is required", nil, requestID, endpoint, method, "rag", "", nil))
		return
	}

	/* Set defaults */
	if req.TextCol == "" {
		req.TextCol = "content"
	}
	if req.VectorCol == "" {
		req.VectorCol = "embedding"
	}
	if req.EmbeddingModel == "" {
		req.EmbeddingModel = "default"
	}
	if req.ChunkSize == 0 {
		req.ChunkSize = 512
	}
	if req.ChunkOverlap == 0 {
		req.ChunkOverlap = 128
	}

	ctx := r.Context()

	/* Use RAGClient to ingest document */
	if h.ragClient == nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "RAG ingest failed: RAG client not configured", nil, requestID, endpoint, method, "rag", "", nil))
		return
	}

	chunkIDs, err := h.ragClient.IngestDocument(ctx, req.DocumentText, req.TableName, req.TextCol, req.VectorCol, req.EmbeddingModel, req.ChunkSize, req.ChunkOverlap, req.Metadata)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "RAG ingest failed", err, requestID, endpoint, method, "rag", "", nil))
		return
	}

	response := RAGIngestResponse{
		ChunksCreated: len(chunkIDs),
		ChunkIDs:      chunkIDs,
		Message:       fmt.Sprintf("Successfully ingested document into %d chunks", len(chunkIDs)),
	}

	respondJSON(w, http.StatusOK, response)
}

/* RAG Evaluate Handler */

func (h *RAGHandlers) RAGEvaluate(w http.ResponseWriter, r *http.Request) {
	requestID := GetRequestID(r.Context())
	endpoint := r.URL.Path
	method := r.Method

	/* Check authorization */
	apiKey, ok := GetAPIKeyFromContext(r.Context())
	if !ok {
		respondError(w, WrapError(ErrUnauthorized, requestID))
		return
	}
	if err := auth.RequireAnyRole(apiKey, auth.RoleAdmin, auth.RoleUser); err != nil {
		respondError(w, NewErrorWithContext(http.StatusForbidden, "insufficient permissions", err, requestID, endpoint, method, "rag", "", nil))
		return
	}

	/* Parse request */
	var req RAGEvaluateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "RAG evaluate failed: request parsing error", err, requestID, endpoint, method, "rag", "", nil))
		return
	}

	/* Validate request */
	if req.Query == "" {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "RAG evaluate failed: query is required", nil, requestID, endpoint, method, "rag", "", nil))
		return
	}
	if req.Answer == "" {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "RAG evaluate failed: answer is required", nil, requestID, endpoint, method, "rag", "", nil))
		return
	}

	ctx := r.Context()

	/* Evaluate RAG */
	evaluation, err := h.advancedRAG.EvaluateRAG(ctx, req.Query, req.Answer, req.ContextChunks)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "RAG evaluate failed", err, requestID, endpoint, method, "rag", "", nil))
		return
	}

	response := RAGEvaluateResponse{
		Faithfulness:       evaluation.Faithfulness,
		Relevancy:          evaluation.Relevancy,
		ContextPrecision:   evaluation.ContextPrecision,
		ContextRecall:      evaluation.ContextRecall,
		SemanticSimilarity: evaluation.SemanticSimilarity,
		OverallScore:       evaluation.OverallScore,
		Metadata: map[string]interface{}{
			"evaluation_type": req.EvaluationType,
		},
	}

	respondJSON(w, http.StatusOK, response)
}

/* RAG Pipeline List Handler */

func (h *RAGHandlers) ListRAGPipelines(w http.ResponseWriter, r *http.Request) {
	requestID := GetRequestID(r.Context())
	endpoint := r.URL.Path
	method := r.Method

	/* Check authorization */
	apiKey, ok := GetAPIKeyFromContext(r.Context())
	if !ok {
		respondError(w, WrapError(ErrUnauthorized, requestID))
		return
	}
	if err := auth.RequireAnyRole(apiKey, auth.RoleAdmin, auth.RoleUser); err != nil {
		respondError(w, NewErrorWithContext(http.StatusForbidden, "insufficient permissions", err, requestID, endpoint, method, "rag", "", nil))
		return
	}

	/* Query pipelines from database */
	/* Note: This would require a query method in the database layer */
	/* For now, return empty list - this would need to be implemented with actual DB query */
	_ = r.Context() /* Reserved for future DB query implementation */
	pipelines := []RAGPipelineResponse{}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"pipelines": pipelines,
		"count":     len(pipelines),
	})
}

/* RAG Pipeline Create Handler */

func (h *RAGHandlers) CreateRAGPipeline(w http.ResponseWriter, r *http.Request) {
	requestID := GetRequestID(r.Context())
	endpoint := r.URL.Path
	method := r.Method

	/* Check authorization */
	apiKey, ok := GetAPIKeyFromContext(r.Context())
	if !ok {
		respondError(w, WrapError(ErrUnauthorized, requestID))
		return
	}
	if err := auth.RequireAnyRole(apiKey, auth.RoleAdmin, auth.RoleUser); err != nil {
		respondError(w, NewErrorWithContext(http.StatusForbidden, "insufficient permissions", err, requestID, endpoint, method, "rag", "", nil))
		return
	}

	/* Parse request */
	var req RAGPipelineRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "RAG pipeline creation failed: request parsing error", err, requestID, endpoint, method, "rag", "", nil))
		return
	}

	/* Validate request */
	if req.PipelineName == "" {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "RAG pipeline creation failed: pipeline_name is required", nil, requestID, endpoint, method, "rag", "", nil))
		return
	}
	if req.EmbeddingModel == "" {
		req.EmbeddingModel = "default"
	}

	/* Create pipeline in database */
	/* Note: This would require using SQL to insert into neurondb.rag_pipelines */
	/* For now, return a mock response - this would need actual DB implementation */
	_ = r.Context() /* Reserved for future DB insert implementation */
	pipelineID := 1 // This would come from DB insert

	response := RAGPipelineResponse{
		PipelineID:     pipelineID,
		PipelineName:   req.PipelineName,
		EmbeddingModel: req.EmbeddingModel,
		Config:         req.Config,
		CreatedAt:      "2024-01-01T00:00:00Z", // This would come from DB
	}

	respondJSON(w, http.StatusCreated, response)
}

/* RAG Pipeline Get Handler */

func (h *RAGHandlers) GetRAGPipeline(w http.ResponseWriter, r *http.Request) {
	requestID := GetRequestID(r.Context())
	endpoint := r.URL.Path
	method := r.Method

	/* Check authorization */
	apiKey, ok := GetAPIKeyFromContext(r.Context())
	if !ok {
		respondError(w, WrapError(ErrUnauthorized, requestID))
		return
	}
	if err := auth.RequireAnyRole(apiKey, auth.RoleAdmin, auth.RoleUser); err != nil {
		respondError(w, NewErrorWithContext(http.StatusForbidden, "insufficient permissions", err, requestID, endpoint, method, "rag", "", nil))
		return
	}

	vars := mux.Vars(r)
	pipelineIDStr := vars["id"]
	pipelineID, err := uuid.Parse(pipelineIDStr)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "RAG pipeline get failed: invalid pipeline ID", err, requestID, endpoint, method, "rag", "", nil))
		return
	}

	/* Query pipeline from database */
	/* Note: This would require a query method in the database layer */
	/* For now, return 404 - this would need actual DB implementation */
	_ = r.Context() /* Reserved for future DB query implementation */
	_ = pipelineID  /* Reserved for future DB query implementation */
	respondError(w, NewErrorWithContext(http.StatusNotFound, "RAG pipeline not found", nil, requestID, endpoint, method, "rag", "", nil))
}
