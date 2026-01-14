/*-------------------------------------------------------------------------
 *
 * memory_handlers.go
 *    API handlers for memory management
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/api/memory_handlers.go
 *
 *-------------------------------------------------------------------------
 */

package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/google/uuid"
	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronAgent/internal/validation"
	"github.com/neurondb/NeuronAgent/pkg/neurondb"
)

/* ListMemoryChunks lists memory chunks for an agent */
func (h *Handlers) ListMemoryChunks(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	if err := validation.ValidateUUIDRequired(vars["id"], "agent_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID", err, requestID, r.URL.Path, r.Method, "memory", "", nil))
		return
	}

	agentID, err := uuid.Parse(vars["id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent id format", err, requestID, r.URL.Path, r.Method, "memory", "", nil))
		return
	}

	limit := 50
	offset := 0
	if l := r.URL.Query().Get("limit"); l != "" {
		fmt.Sscanf(l, "%d", &limit)
		if err := validation.ValidateLimit(limit); err != nil {
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid limit parameter", err, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
			return
		}
	}
	if o := r.URL.Query().Get("offset"); o != "" {
		fmt.Sscanf(o, "%d", &offset)
		if err := validation.ValidateOffset(offset); err != nil {
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid offset parameter", err, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
			return
		}
	}

	chunks, err := h.queries.ListMemoryChunks(r.Context(), agentID, limit, offset)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to list memory chunks", err, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
		return
	}

	respondJSON(w, http.StatusOK, chunks)
}

/* GetMemoryChunk gets a memory chunk by ID */
func (h *Handlers) GetMemoryChunk(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	var id int64
	if _, err := fmt.Sscanf(vars["chunk_id"], "%d", &id); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid chunk id", err, requestID, r.URL.Path, r.Method, "memory", "", nil))
		return
	}

	chunk, err := h.queries.GetMemoryChunk(r.Context(), id)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusNotFound, "memory chunk not found", err, requestID, r.URL.Path, r.Method, "memory", fmt.Sprintf("%d", id), nil))
		return
	}

	respondJSON(w, http.StatusOK, chunk)
}

/* DeleteMemoryChunk deletes a memory chunk */
func (h *Handlers) DeleteMemoryChunk(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	var id int64
	if _, err := fmt.Sscanf(vars["chunk_id"], "%d", &id); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid chunk id", err, requestID, r.URL.Path, r.Method, "memory", "", nil))
		return
	}

	if err := h.queries.DeleteMemoryChunk(r.Context(), id); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusNotFound, "memory chunk not found", err, requestID, r.URL.Path, r.Method, "memory", fmt.Sprintf("%d", id), nil))
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

/* SearchMemory searches memory chunks by query text */
func (h *Handlers) SearchMemory(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	if err := validation.ValidateUUIDRequired(vars["id"], "agent_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID", err, requestID, r.URL.Path, r.Method, "memory", "", nil))
		return
	}

	agentID, err := uuid.Parse(vars["id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent id format", err, requestID, r.URL.Path, r.Method, "memory", "", nil))
		return
	}

	/* Validate request body size (max 1MB) */
	const maxBodySize = 1024 * 1024
	bodyBytes, err := validation.ReadAndValidateBody(r, maxBodySize)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body validation failed", err, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var req struct {
		Query string `json:"query"`
		TopK  int    `json:"top_k"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid request body", err, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
		return
	}

	if err := validation.ValidateRequired(req.Query, "query"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "query validation failed", err, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
		return
	}
	if err := validation.ValidateMaxLength(req.Query, "query", 10000); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "query too long", err, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
		return
	}

	if req.TopK <= 0 {
		req.TopK = 5
	}
	if req.TopK > 100 {
		req.TopK = 100
	}

	/* Generate embedding for query */
	embedClient := neurondb.NewClient(h.queries.GetDB()).Embedding
	embeddingModel := "all-MiniLM-L6-v2"
	embedding, err := embedClient.Embed(r.Context(), req.Query, embeddingModel)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to generate embedding", err, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
		return
	}

	/* Search memory */
	chunks, err := h.queries.SearchMemory(r.Context(), agentID, embedding, req.TopK)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to search memory", err, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
		return
	}

	respondJSON(w, http.StatusOK, chunks)
}
