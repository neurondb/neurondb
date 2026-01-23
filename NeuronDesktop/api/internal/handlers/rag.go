package handlers

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
)

/* RAGHandlers handles RAG-related endpoints */
type RAGHandlers struct {
	queries *db.Queries
	neurondbHandlers *NeuronDBHandlers
}

/* NewRAGHandlers creates new RAG handlers */
func NewRAGHandlers(queries *db.Queries, neurondbHandlers *NeuronDBHandlers) *RAGHandlers {
	return &RAGHandlers{
		queries:          queries,
		neurondbHandlers: neurondbHandlers,
	}
}

/* RAGQueryRequest represents a RAG query request */
type RAGQueryRequest struct {
	Query                string                 `json:"query"`
	TableName            string                 `json:"table_name"`
	VectorCol            string                 `json:"vector_col,omitempty"`
	TextCol              string                 `json:"text_col,omitempty"`
	Model                string                 `json:"model,omitempty"`
	TopK                 int                    `json:"top_k,omitempty"`
	Rerank               bool                   `json:"rerank,omitempty"`
	RerankModel          string                 `json:"rerank_model,omitempty"`
	Hybrid               bool                   `json:"hybrid,omitempty"`
	VectorWeight         float64                `json:"vector_weight,omitempty"`
	Temporal             bool                   `json:"temporal,omitempty"`
	RecencyWeight        float64                `json:"recency_weight,omitempty"`
	Faceted              bool                   `json:"faceted,omitempty"`
	Categories           []string               `json:"categories,omitempty"`
	CustomContext        map[string]interface{} `json:"custom_context,omitempty"`
	/* RAG Architecture Selection */
	Architecture         string                 `json:"architecture,omitempty"` /* naive, hyde, graph, corrective, hybrid, agentic, contextual, modular */
	/* HyDE parameters */
	NumHypotheticals     int                    `json:"num_hypotheticals,omitempty"`
	HypotheticalWeight   float64                `json:"hypothetical_weight,omitempty"`
	/* Graph RAG parameters */
	EntityCol            string                 `json:"entity_col,omitempty"`
	RelationCol          string                 `json:"relation_col,omitempty"`
	MaxDepth             int                    `json:"max_depth,omitempty"`
	TraversalMethod      string                 `json:"traversal_method,omitempty"`
	/* Corrective RAG parameters */
	MaxIterations        int                    `json:"max_iterations,omitempty"`
	QualityThreshold     float64                `json:"quality_threshold,omitempty"`
	/* Agentic RAG parameters */
	MaxSteps             int                    `json:"max_steps,omitempty"`
	EvidenceThreshold    float64                `json:"evidence_threshold,omitempty"`
	MaxTokens            int                    `json:"max_tokens,omitempty"`
	/* Contextual RAG parameters */
	ConversationHistory  []map[string]interface{} `json:"conversation_history,omitempty"`
	SessionContext       map[string]interface{}  `json:"session_context,omitempty"`
	CrossSessionContext  bool                    `json:"cross_session_context,omitempty"`
	/* Modular RAG parameters */
	ModuleConfig         map[string]interface{}  `json:"module_config,omitempty"`
}

/* RAGQueryResponse represents a RAG query response */
type RAGQueryResponse struct {
	Answer    string                   `json:"answer"`
	Documents []string                 `json:"documents"`
	Count     int                      `json:"count"`
	Method    string                   `json:"method"`
	Metadata  map[string]interface{}   `json:"metadata,omitempty"`
}

/* RAGIngestRequest represents a RAG ingest request */
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

/* RAGIngestResponse represents a RAG ingest response */
type RAGIngestResponse struct {
	ChunksCreated int      `json:"chunks_created"`
	ChunkIDs      []int64  `json:"chunk_ids,omitempty"`
	Message       string   `json:"message"`
}

/* RAGEvaluateRequest represents a RAG evaluate request */
type RAGEvaluateRequest struct {
	Query          string   `json:"query"`
	Answer         string   `json:"answer"`
	ContextChunks  []string `json:"context_chunks"`
	EvaluationType string   `json:"evaluation_type,omitempty"`
}

/* RAGEvaluateResponse represents a RAG evaluate response */
type RAGEvaluateResponse struct {
	Faithfulness       float64                `json:"faithfulness"`
	Relevancy          float64                `json:"relevancy"`
	ContextPrecision   float64                `json:"context_precision"`
	ContextRecall      float64                `json:"context_recall"`
	SemanticSimilarity float64                `json:"semantic_similarity"`
	OverallScore       float64                `json:"overall_score"`
	Metadata           map[string]interface{} `json:"metadata,omitempty"`
}

/* RAGPipeline represents a RAG pipeline */
type RAGPipeline struct {
	PipelineID     int                    `json:"pipeline_id"`
	PipelineName   string                 `json:"pipeline_name"`
	EmbeddingModel string                 `json:"embedding_model"`
	Config         map[string]interface{} `json:"config,omitempty"`
	CreatedAt      string                 `json:"created_at"`
}

/* RAGQuery handles RAG query requests */
func (h *RAGHandlers) RAGQuery(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profileId"]

	var req RAGQueryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, http.StatusBadRequest, "Invalid request body", err)
		return
	}

	/* Validate request */
	if req.Query == "" {
		respondError(w, http.StatusBadRequest, "query is required", nil)
		return
	}
	if req.TableName == "" {
		respondError(w, http.StatusBadRequest, "table_name is required", nil)
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
	client, err := h.neurondbHandlers.getClient(ctx, profileID)
	if err != nil {
		respondError(w, http.StatusInternalServerError, "Failed to get NeuronDB client", err)
		return
	}

	/* Build SQL query based on architecture */
	var querySQL string
	var queryParams []interface{}
	method := "basic"

	/* Determine which RAG architecture to use */
	architecture := req.Architecture
	if architecture == "" {
		/* Default to basic if not specified */
		architecture = "naive"
	}

	customContextJSON, _ := json.Marshal(req.CustomContext)
	if customContextJSON == nil {
		customContextJSON = []byte("{}")
	}

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
		querySQL = `SELECT chunk_text, relevance_score, answer, hypothetical_docs FROM neurondb.rag_hyde($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb)`
		queryParams = []interface{}{req.Query, req.TableName, req.VectorCol, req.TextCol, req.Model, req.Model, req.TopK, numHypotheticals, hypotheticalWeight, string(customContextJSON)}
		method = "hyde"

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
		querySQL = `SELECT chunk_text, relevance_score, answer, graph_path FROM neurondb.rag_graph($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb)`
		queryParams = []interface{}{req.Query, req.TableName, req.VectorCol, req.TextCol, entityCol, relationCol, req.Model, req.TopK, maxDepth, traversalMethod, string(customContextJSON)}
		method = "graph"

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
		querySQL = `SELECT chunk_text, relevance_score, answer, iterations, quality_score FROM neurondb.rag_corrective($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb)`
		queryParams = []interface{}{req.Query, req.TableName, req.VectorCol, req.TextCol, req.Model, req.Model, req.TopK, maxIterations, qualityThreshold, string(customContextJSON)}
		method = "corrective"

	case "hybrid":
		/* Hybrid RAG */
		vectorWeight := req.VectorWeight
		if vectorWeight == 0 {
			vectorWeight = 0.7
		}
		/* Use hybrid_search function */
		embedQuery := `SELECT embed_text($1::text, $2::text)::text AS embedding`
		var embeddingStr string
		err := client.ExecuteQueryRow(ctx, embedQuery, req.Query, req.Model).Scan(&embeddingStr)
		if err != nil {
			respondError(w, http.StatusInternalServerError, "Failed to generate embedding", err)
			return
		}
		querySQL = `SELECT * FROM hybrid_search($1, $2::vector, $3, $4::jsonb, $5, $6, 'plain')`
		queryParams = []interface{}{req.TableName, embeddingStr, req.Query, "{}", vectorWeight, req.TopK}
		method = "hybrid"

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
		querySQL = `SELECT chunk_text, relevance_score, answer, execution_trace, reasoning_path FROM neurondb.rag_agentic($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb)`
		queryParams = []interface{}{req.Query, req.TableName, req.VectorCol, req.TextCol, req.Model, req.Model, req.TopK, maxSteps, evidenceThreshold, maxTokens, string(customContextJSON)}
		method = "agentic"

	case "contextual":
		/* Contextual RAG */
		conversationHistoryJSON, _ := json.Marshal(req.ConversationHistory)
		if conversationHistoryJSON == nil {
			conversationHistoryJSON = []byte("[]")
		}
		sessionContextJSON, _ := json.Marshal(req.SessionContext)
		if sessionContextJSON == nil {
			sessionContextJSON = []byte("{}")
		}
		querySQL = `SELECT chunk_text, relevance_score, answer, rewritten_query, context_adaptation FROM neurondb.rag_contextual($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9::jsonb, $10, $11::jsonb)`
		queryParams = []interface{}{req.Query, req.TableName, req.VectorCol, req.TextCol, req.Model, req.Model, req.TopK, string(conversationHistoryJSON), string(sessionContextJSON), req.CrossSessionContext, string(customContextJSON)}
		method = "contextual"

	case "modular":
		/* Modular RAG */
		if req.ModuleConfig == nil || len(req.ModuleConfig) == 0 {
			respondError(w, http.StatusBadRequest, "module_config is required for modular RAG", nil)
			return
		}
		moduleConfigJSON, _ := json.Marshal(req.ModuleConfig)
		querySQL = `SELECT chunk_text, relevance_score, answer, pipeline_name, module_trace FROM neurondb.rag_modular($1, $2, $3, $4, $5::jsonb, $6, $7, $8::jsonb)`
		queryParams = []interface{}{req.Query, req.TableName, req.VectorCol, req.TextCol, string(moduleConfigJSON), req.Model, req.Model, string(customContextJSON)}
		method = "modular"

	default:
		/* Naive RAG - basic retrieval */
		if req.CustomContext != nil && len(req.CustomContext) > 0 {
			/* Use rag_query_with_context */
			querySQL = `SELECT chunk_text, relevance_score, answer FROM neurondb.rag_query_with_context($1, $2, $3, $4, $5, $6, $7::jsonb)`
			queryParams = []interface{}{req.Query, req.TableName, req.VectorCol, req.TextCol, req.Model, req.TopK, string(customContextJSON)}
			method = "with_context"
		} else {
			/* Use basic rag_query */
			querySQL = `SELECT chunk_text, relevance_score FROM neurondb.rag_query($1, $2, $3, $4, $5, $6)`
			queryParams = []interface{}{req.Query, req.TableName, req.VectorCol, req.TextCol, req.Model, req.TopK}
			method = "naive"
		}
	}

	/* Execute query */
	rows, err := client.ExecuteQuery(ctx, querySQL, queryParams...)
	if err != nil {
		respondError(w, http.StatusInternalServerError, "RAG query failed", err)
		return
	}
	defer rows.Close()

	var documents []string
	var answer string

	/* Scan results based on architecture */
	for rows.Next() {
		var chunkText string
		var relevanceScore float64
		
		switch method {
		case "hyde":
			var answerVal sql.NullString
			var hypotheticalDocs sql.NullString
			if err := rows.Scan(&chunkText, &relevanceScore, &answerVal, &hypotheticalDocs); err != nil {
				continue
			}
			if answerVal.Valid && answer == "" {
				answer = answerVal.String
			}
			
		case "graph":
			var answerVal sql.NullString
			var graphPath sql.NullString
			if err := rows.Scan(&chunkText, &relevanceScore, &answerVal, &graphPath); err != nil {
				continue
			}
			if answerVal.Valid && answer == "" {
				answer = answerVal.String
			}
			
		case "corrective":
			var answerVal sql.NullString
			var iterations sql.NullInt64
			var qualityScore sql.NullFloat64
			if err := rows.Scan(&chunkText, &relevanceScore, &answerVal, &iterations, &qualityScore); err != nil {
				continue
			}
			if answerVal.Valid && answer == "" {
				answer = answerVal.String
			}
			
		case "hybrid":
			/* Hybrid search returns different structure */
			if err := rows.Scan(&chunkText, &relevanceScore); err != nil {
				continue
			}
			
		case "agentic":
			var answerVal sql.NullString
			var executionTrace sql.NullString
			var reasoningPath sql.NullString
			if err := rows.Scan(&chunkText, &relevanceScore, &answerVal, &executionTrace, &reasoningPath); err != nil {
				continue
			}
			if answerVal.Valid && answer == "" {
				answer = answerVal.String
			}
			
		case "contextual":
			var answerVal sql.NullString
			var rewrittenQuery sql.NullString
			var contextAdaptation sql.NullString
			if err := rows.Scan(&chunkText, &relevanceScore, &answerVal, &rewrittenQuery, &contextAdaptation); err != nil {
				continue
			}
			if answerVal.Valid && answer == "" {
				answer = answerVal.String
			}
			
		case "modular":
			var answerVal sql.NullString
			var pipelineName sql.NullString
			var moduleTrace sql.NullString
			if err := rows.Scan(&chunkText, &relevanceScore, &answerVal, &pipelineName, &moduleTrace); err != nil {
				continue
			}
			if answerVal.Valid && answer == "" {
				answer = answerVal.String
			}
			
		case "with_context":
			var answerVal sql.NullString
			if err := rows.Scan(&chunkText, &relevanceScore, &answerVal); err != nil {
				continue
			}
			if answerVal.Valid && answer == "" {
				answer = answerVal.String
			}
			
		default:
			/* Naive RAG - basic retrieval */
			if err := rows.Scan(&chunkText, &relevanceScore); err != nil {
				continue
			}
		}
		documents = append(documents, chunkText)
	}

	/* If answer not generated yet, generate using LLM if available */
	if answer == "" && len(documents) > 0 {
		contextText := ""
		for i, doc := range documents {
			if i > 0 {
				contextText += "\n\n"
			}
			contextText += doc
		}
		prompt := fmt.Sprintf("Context:\n%s\n\nQuestion: %s\n\nAnswer:", contextText, req.Query)
		
		/* Use LLM completion if available */
		llmQuery := `SELECT ndb_llm_complete($1, '{"temperature": 0.7, "max_tokens": 500}') AS answer`
		var answerVal sql.NullString
		err := client.ExecuteQueryRow(ctx, llmQuery, prompt).Scan(&answerVal)
		if err == nil && answerVal.Valid {
			answer = answerVal.String
		} else {
			answer = "Answer generation not available"
		}
	}
	
	response := RAGQueryResponse{
		Answer:    answer,
		Documents: documents,
		Count:     len(documents),
		Method:    method,
		Metadata:  make(map[string]interface{}),
	}

	respondJSON(w, http.StatusOK, response)
}

/* RAGIngest handles RAG document ingestion requests */
func (h *RAGHandlers) RAGIngest(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profileId"]

	var req RAGIngestRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, http.StatusBadRequest, "Invalid request body", err)
		return
	}

	/* Validate request */
	if req.DocumentText == "" {
		respondError(w, http.StatusBadRequest, "document_text is required", nil)
		return
	}
	if req.TableName == "" {
		respondError(w, http.StatusBadRequest, "table_name is required", nil)
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
	client, err := h.neurondbHandlers.getClient(ctx, profileID)
	if err != nil {
		respondError(w, http.StatusInternalServerError, "Failed to get NeuronDB client", err)
		return
	}

	/* Use rag_ingest_document function */
	metadataJSON, _ := json.Marshal(req.Metadata)
	querySQL := `SELECT * FROM neurondb.rag_ingest_document($1, $2, $3, $4, $5, $6, $7, $8::jsonb)`
	
	rows, err := client.ExecuteQuery(ctx, querySQL, 
		req.DocumentText, req.TableName, req.TextCol, req.VectorCol, 
		req.EmbeddingModel, req.ChunkSize, req.ChunkOverlap, string(metadataJSON))
	if err != nil {
		respondError(w, http.StatusInternalServerError, "RAG ingest failed", err)
		return
	}
	defer rows.Close()

	var chunkIDs []int64
	for rows.Next() {
		var chunkID int64
		var chunkText string
		var embedding string
		if err := rows.Scan(&chunkID, &chunkText, &embedding); err != nil {
			continue
		}
		chunkIDs = append(chunkIDs, chunkID)
	}

	response := RAGIngestResponse{
		ChunksCreated: len(chunkIDs),
		ChunkIDs:      chunkIDs,
		Message:       fmt.Sprintf("Successfully ingested document into %d chunks", len(chunkIDs)),
	}

	respondJSON(w, http.StatusOK, response)
}

/* RAGEvaluate handles RAG evaluation requests */
func (h *RAGHandlers) RAGEvaluate(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profileId"]

	var req RAGEvaluateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, http.StatusBadRequest, "Invalid request body", err)
		return
	}

	/* Validate request */
	if req.Query == "" {
		respondError(w, http.StatusBadRequest, "query is required", nil)
		return
	}
	if req.Answer == "" {
		respondError(w, http.StatusBadRequest, "answer is required", nil)
		return
	}
	if len(req.ContextChunks) == 0 {
		respondError(w, http.StatusBadRequest, "context_chunks is required", nil)
		return
	}

	if req.EvaluationType == "" {
		req.EvaluationType = "basic"
	}

	ctx := r.Context()
	client, err := h.neurondbHandlers.getClient(ctx, profileID)
	if err != nil {
		respondError(w, http.StatusInternalServerError, "Failed to get NeuronDB client", err)
		return
	}

	/* Use rag_evaluate function */
	querySQL := `SELECT neurondb.rag_evaluate($1, $2, $3::text[], $4) AS evaluation`
	
	var evaluationJSON string
	err = client.ExecuteQueryRow(ctx, querySQL, req.Query, req.Answer, req.ContextChunks, req.EvaluationType).Scan(&evaluationJSON)
	if err != nil {
		respondError(w, http.StatusInternalServerError, "RAG evaluation failed", err)
		return
	}

	var evaluation map[string]interface{}
	if err := json.Unmarshal([]byte(evaluationJSON), &evaluation); err != nil {
		respondError(w, http.StatusInternalServerError, "Failed to parse evaluation results", err)
		return
	}

	/* Extract values from evaluation JSON */
	relevancy := getFloat64(evaluation, "relevancy", 0.0)
	semanticSimilarity := getFloat64(evaluation, "semantic_similarity", 0.0)
	
	/* Get similarity stats if available */
	var avgSimilarity float64 = 0.0
	if stats, ok := evaluation["similarity_stats"].(map[string]interface{}); ok {
		avgSimilarity = getFloat64(stats, "avg", 0.0)
	}

	response := RAGEvaluateResponse{
		Faithfulness:       relevancy, /* Using relevancy as proxy for faithfulness */
		Relevancy:          relevancy,
		ContextPrecision:   avgSimilarity, /* Using average similarity as proxy */
		ContextRecall:      relevancy, /* Using relevancy as proxy */
		SemanticSimilarity: semanticSimilarity,
		OverallScore:       (relevancy + semanticSimilarity + avgSimilarity) / 3.0, /* Average of metrics */
		Metadata:           evaluation,
	}

	respondJSON(w, http.StatusOK, response)
}

/* ListRAGPipelines handles listing RAG pipelines */
func (h *RAGHandlers) ListRAGPipelines(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profileId"]

	ctx := r.Context()
	client, err := h.neurondbHandlers.getClient(ctx, profileID)
	if err != nil {
		respondError(w, http.StatusInternalServerError, "Failed to get NeuronDB client", err)
		return
	}

	/* Query pipelines from neurondb.rag_pipelines table */
	querySQL := `SELECT pipeline_id, pipeline_name, embedding_model, configuration, created_at 
		FROM neurondb.rag_pipelines 
		ORDER BY created_at DESC`

	rows, err := client.ExecuteQuery(ctx, querySQL)
	if err != nil {
		respondError(w, http.StatusInternalServerError, "Failed to query pipelines", err)
		return
	}
	defer rows.Close()

	var pipelines []RAGPipeline
	for rows.Next() {
		var pipeline RAGPipeline
		var configJSON string
		if err := rows.Scan(&pipeline.PipelineID, &pipeline.PipelineName, &pipeline.EmbeddingModel, &configJSON, &pipeline.CreatedAt); err != nil {
			continue
		}
		if configJSON != "" {
			json.Unmarshal([]byte(configJSON), &pipeline.Config)
		}
		pipelines = append(pipelines, pipeline)
	}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"pipelines": pipelines,
		"count":     len(pipelines),
	})
}

/* CreateRAGPipeline handles creating a new RAG pipeline */
func (h *RAGHandlers) CreateRAGPipeline(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profileId"]

	var req struct {
		PipelineName   string                 `json:"pipeline_name"`
		EmbeddingModel string                 `json:"embedding_model"`
		ChunkSize      int                    `json:"chunk_size,omitempty"`
		ChunkOverlap   int                    `json:"chunk_overlap,omitempty"`
		Config         map[string]interface{} `json:"config,omitempty"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, http.StatusBadRequest, "Invalid request body", err)
		return
	}

	if req.PipelineName == "" {
		respondError(w, http.StatusBadRequest, "pipeline_name is required", nil)
		return
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
	client, err := h.neurondbHandlers.getClient(ctx, profileID)
	if err != nil {
		respondError(w, http.StatusInternalServerError, "Failed to get NeuronDB client", err)
		return
	}

	/* Use create_rag_pipeline function */
	configJSON, _ := json.Marshal(req.Config)
	querySQL := `SELECT neurondb.create_rag_pipeline($1, $2, $3, $4, $5::jsonb) AS pipeline_id`
	
	var pipelineID int
	err = client.ExecuteQueryRow(ctx, querySQL, req.PipelineName, req.EmbeddingModel, req.ChunkSize, req.ChunkOverlap, string(configJSON)).Scan(&pipelineID)
	if err != nil {
		respondError(w, http.StatusInternalServerError, "Failed to create pipeline", err)
		return
	}

	/* Get created pipeline */
	getSQL := `SELECT pipeline_id, pipeline_name, embedding_model, configuration, created_at 
		FROM neurondb.rag_pipelines 
		WHERE pipeline_id = $1`
	
	var pipeline RAGPipeline
	var configJSONStr string
	err = client.ExecuteQueryRow(ctx, getSQL, pipelineID).Scan(&pipeline.PipelineID, &pipeline.PipelineName, &pipeline.EmbeddingModel, &configJSONStr, &pipeline.CreatedAt)
	if err != nil {
		respondError(w, http.StatusInternalServerError, "Failed to retrieve created pipeline", err)
		return
	}

	if configJSONStr != "" {
		json.Unmarshal([]byte(configJSONStr), &pipeline.Config)
	}

	respondJSON(w, http.StatusCreated, pipeline)
}

/* GetRAGPipeline handles getting a specific RAG pipeline */
func (h *RAGHandlers) GetRAGPipeline(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profileId"]
	pipelineID := vars["id"]

	ctx := r.Context()
	client, err := h.neurondbHandlers.getClient(ctx, profileID)
	if err != nil {
		respondError(w, http.StatusInternalServerError, "Failed to get NeuronDB client", err)
		return
	}

	querySQL := `SELECT pipeline_id, pipeline_name, embedding_model, configuration, created_at 
		FROM neurondb.rag_pipelines 
		WHERE pipeline_id = $1`

	var pipeline RAGPipeline
	var configJSON string
	err = client.ExecuteQueryRow(ctx, querySQL, pipelineID).Scan(&pipeline.PipelineID, &pipeline.PipelineName, &pipeline.EmbeddingModel, &configJSON, &pipeline.CreatedAt)
	if err != nil {
		respondError(w, http.StatusNotFound, "Pipeline not found", err)
		return
	}

	if configJSON != "" {
		json.Unmarshal([]byte(configJSON), &pipeline.Config)
	}

	respondJSON(w, http.StatusOK, pipeline)
}

/* Helper functions */

func getFloat64(m map[string]interface{}, key string, defaultValue float64) float64 {
	if val, ok := m[key]; ok {
		if f, ok := val.(float64); ok {
			return f
		}
	}
	return defaultValue
}

func respondJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

func respondError(w http.ResponseWriter, status int, message string, err error) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	errorMsg := message
	if err != nil {
		errorMsg = fmt.Sprintf("%s: %v", message, err)
	}
	json.NewEncoder(w).Encode(map[string]interface{}{
		"error": errorMsg,
	})
}
