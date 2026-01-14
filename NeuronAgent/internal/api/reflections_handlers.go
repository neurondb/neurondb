/*-------------------------------------------------------------------------
 *
 * reflections_handlers.go
 *    API handlers for reflections
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/api/reflections_handlers.go
 *
 *-------------------------------------------------------------------------
 */

package api

import (
	"fmt"
	"net/http"

	"github.com/google/uuid"
	"github.com/gorilla/mux"
)

/* ListReflections lists reflections with optional filters */
func (h *Handlers) ListReflections(w http.ResponseWriter, r *http.Request) {
	var agentID, sessionID *uuid.UUID

	if agentIDStr := r.URL.Query().Get("agent_id"); agentIDStr != "" {
		id, err := uuid.Parse(agentIDStr)
		if err == nil {
			agentID = &id
		}
	}

	if sessionIDStr := r.URL.Query().Get("session_id"); sessionIDStr != "" {
		id, err := uuid.Parse(sessionIDStr)
		if err == nil {
			sessionID = &id
		}
	}

	limit := 50
	offset := 0
	if l := r.URL.Query().Get("limit"); l != "" {
		if _, err := fmt.Sscanf(l, "%d", &limit); err != nil {
			requestID := GetRequestID(r.Context())
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid limit parameter", err, requestID, r.URL.Path, r.Method, "reflection", "", nil))
			return
		}
	}
	if o := r.URL.Query().Get("offset"); o != "" {
		if _, err := fmt.Sscanf(o, "%d", &offset); err != nil {
			requestID := GetRequestID(r.Context())
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid offset parameter", err, requestID, r.URL.Path, r.Method, "reflection", "", nil))
			return
		}
	}

	/* Validate pagination */
	if err := ValidatePaginationParams(limit, offset); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "pagination validation failed", err, requestID, r.URL.Path, r.Method, "reflection", "", nil))
		return
	}

	reflections, err := h.queries.ListReflections(r.Context(), agentID, sessionID, limit, offset)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to list reflections", err, requestID, r.URL.Path, r.Method, "reflection", "", nil))
		return
	}

	respondJSON(w, http.StatusOK, reflections)
}

/* GetReflection gets a reflection by ID */
func (h *Handlers) GetReflection(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	var id int64
	if _, err := fmt.Sscanf(vars["id"], "%d", &id); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid reflection id", err, requestID, r.URL.Path, r.Method, "reflection", "", nil))
		return
	}

	reflection, err := h.queries.GetReflection(r.Context(), id)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusNotFound, "reflection not found", err, requestID, r.URL.Path, r.Method, "reflection", fmt.Sprintf("%d", id), nil))
		return
	}

	respondJSON(w, http.StatusOK, reflection)
}


