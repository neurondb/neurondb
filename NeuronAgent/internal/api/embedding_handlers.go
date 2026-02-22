/*-------------------------------------------------------------------------
 *
 * embedding_handlers.go
 *    Embedding API handlers for NeuronAgent
 *
 * Provides REST API endpoints for embedding operations including generation,
 * batch generation, and model listing.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/api/embedding_handlers.go
 *
 *-------------------------------------------------------------------------
 */

package api

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/neurondb/NeuronAgent/internal/auth"
	"github.com/neurondb/NeuronAgent/internal/validation"
	"github.com/neurondb/NeuronAgent/pkg/neurondb"
)

type EmbeddingHandlers struct {
	embedClient *neurondb.EmbeddingClient
}

func NewEmbeddingHandlers(embedClient *neurondb.EmbeddingClient) *EmbeddingHandlers {
	return &EmbeddingHandlers{
		embedClient: embedClient,
	}
}

/* Embedding Generate Request/Response */

type EmbeddingGenerateRequest struct {
	Text  string `json:"text"`
	Model string `json:"model,omitempty"`
}

type EmbeddingGenerateResponse struct {
	Embedding []float32              `json:"embedding"`
	Model     string                 `json:"model"`
	Dimension int                    `json:"dimension"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

/* Embedding Batch Request/Response */

type EmbeddingBatchRequest struct {
	Texts []string `json:"texts"`
	Model string   `json:"model,omitempty"`
}

type EmbeddingBatchResponse struct {
	Embeddings [][]float32            `json:"embeddings"`
	Model      string                 `json:"model"`
	Dimension  int                    `json:"dimension"`
	Count      int                    `json:"count"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
}

/* Embedding Models Response */

type EmbeddingModelsResponse struct {
	Models   []string               `json:"models"`
	Default  string                 `json:"default"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

/* Embedding Generate Handler */

func (h *EmbeddingHandlers) GenerateEmbedding(w http.ResponseWriter, r *http.Request) {
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
		respondError(w, NewErrorWithContext(http.StatusForbidden, "insufficient permissions", err, requestID, endpoint, method, "embedding", "", nil))
		return
	}

	/* Parse request */
	var req EmbeddingGenerateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "embedding generation failed: request parsing error", err, requestID, endpoint, method, "embedding", "", nil))
		return
	}

	/* Validate request */
	if req.Text == "" {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "embedding generation failed: text is required", nil, requestID, endpoint, method, "embedding", "", nil))
		return
	}

	/* Set default model */
	if req.Model == "" {
		req.Model = "default"
	}

	ctx := r.Context()

	/* Generate embedding */
	if h.embedClient == nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "embedding generation failed: embedding client not configured", nil, requestID, endpoint, method, "embedding", "", nil))
		return
	}

	embedding, err := h.embedClient.Embed(ctx, req.Text, req.Model)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "embedding generation failed", err, requestID, endpoint, method, "embedding", "", nil))
		return
	}

	response := EmbeddingGenerateResponse{
		Embedding: embedding,
		Model:     req.Model,
		Dimension: len(embedding),
		Metadata:  make(map[string]interface{}),
	}

	respondJSON(w, http.StatusOK, response)
}

/* Embedding Batch Handler */

func (h *EmbeddingHandlers) BatchGenerateEmbeddings(w http.ResponseWriter, r *http.Request) {
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
		respondError(w, NewErrorWithContext(http.StatusForbidden, "insufficient permissions", err, requestID, endpoint, method, "embedding", "", nil))
		return
	}

	/* Parse request with body size limit (1MB) and Content-Type check */
	const maxEmbeddingBodySize = 1024 * 1024
	var req EmbeddingBatchRequest
	if err := validation.DecodeJSONBody(r, maxEmbeddingBodySize, &req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "batch embedding generation failed: request parsing error", err, requestID, endpoint, method, "embedding", "", nil))
		return
	}

	/* Validate request and enforce batch size limit to prevent resource exhaustion */
	const maxBatchSize = 100
	if len(req.Texts) == 0 {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "batch embedding generation failed: texts array is required and cannot be empty", nil, requestID, endpoint, method, "embedding", "", nil))
		return
	}
	if len(req.Texts) > maxBatchSize {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, fmt.Sprintf("batch embedding generation failed: texts count %d exceeds maximum %d", len(req.Texts), maxBatchSize), nil, requestID, endpoint, method, "embedding", "", nil))
		return
	}

	/* Set default model */
	if req.Model == "" {
		req.Model = "default"
	}

	ctx := r.Context()

	/* Generate embeddings in batch */
	if h.embedClient == nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "batch embedding generation failed: embedding client not configured", nil, requestID, endpoint, method, "embedding", "", nil))
		return
	}

	embeddings := make([][]float32, 0, len(req.Texts))
	dimension := 0

	for _, text := range req.Texts {
		if text == "" {
			continue
		}
		embedding, err := h.embedClient.Embed(ctx, text, req.Model)
		if err != nil {
			/* Log error but continue with other texts */
			continue
		}
		if dimension == 0 {
			dimension = len(embedding)
		}
		embeddings = append(embeddings, embedding)
	}

	response := EmbeddingBatchResponse{
		Embeddings: embeddings,
		Model:      req.Model,
		Dimension:  dimension,
		Count:      len(embeddings),
		Metadata:   make(map[string]interface{}),
	}

	respondJSON(w, http.StatusOK, response)
}

/* Embedding Models Handler */

func (h *EmbeddingHandlers) ListEmbeddingModels(w http.ResponseWriter, r *http.Request) {
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
		respondError(w, NewErrorWithContext(http.StatusForbidden, "insufficient permissions", err, requestID, endpoint, method, "embedding", "", nil))
		return
	}

	/* Return list of available models */
	/* Note: This would typically query from a configuration or database */
	models := []string{
		"default",
		"all-MiniLM-L6-v2",
		"text-embedding-ada-002",
		"sentence-transformers/all-mpnet-base-v2",
	}

	response := EmbeddingModelsResponse{
		Models:   models,
		Default:  "default",
		Metadata: make(map[string]interface{}),
	}

	respondJSON(w, http.StatusOK, response)
}
