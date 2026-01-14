/*-------------------------------------------------------------------------
 *
 * plans_handlers.go
 *    API handlers for plans
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/api/plans_handlers.go
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
	"github.com/neurondb/NeuronAgent/internal/db"
)

/* ListPlans lists plans with optional filters */
func (h *Handlers) ListPlans(w http.ResponseWriter, r *http.Request) {
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
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid limit parameter", err, requestID, r.URL.Path, r.Method, "plan", "", nil))
			return
		}
	}
	if o := r.URL.Query().Get("offset"); o != "" {
		if _, err := fmt.Sscanf(o, "%d", &offset); err != nil {
			requestID := GetRequestID(r.Context())
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid offset parameter", err, requestID, r.URL.Path, r.Method, "plan", "", nil))
			return
		}
	}

	/* Validate pagination */
	if err := ValidatePaginationParams(limit, offset); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "pagination validation failed", err, requestID, r.URL.Path, r.Method, "plan", "", nil))
		return
	}

	plans, err := h.queries.ListPlans(r.Context(), agentID, sessionID, limit, offset)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to list plans", err, requestID, r.URL.Path, r.Method, "plan", "", nil))
		return
	}

	respondJSON(w, http.StatusOK, plans)
}

/* GetPlan gets a plan by ID */
func (h *Handlers) GetPlan(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id, err := uuid.Parse(vars["id"])
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid plan id", err, requestID, r.URL.Path, r.Method, "plan", "", nil))
		return
	}

	plan, err := h.queries.GetPlan(r.Context(), id)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusNotFound, "plan not found", err, requestID, r.URL.Path, r.Method, "plan", id.String(), nil))
		return
	}

	respondJSON(w, http.StatusOK, plan)
}

/* UpdatePlanStatus updates a plan's status */
func (h *Handlers) UpdatePlanStatus(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id, err := uuid.Parse(vars["id"])
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid plan id", err, requestID, r.URL.Path, r.Method, "plan", "", nil))
		return
	}

	var req struct {
		Status string                 `json:"status"`
		Result map[string]interface{} `json:"result"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid request body", err, requestID, r.URL.Path, r.Method, "plan", id.String(), nil))
		return
	}

	validStatuses := map[string]bool{
		"created":   true,
		"executing": true,
		"completed": true,
		"failed":    true,
		"cancelled": true,
	}
	if !validStatuses[req.Status] {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid status", nil, requestID, r.URL.Path, r.Method, "plan", id.String(), nil))
		return
	}

	var result db.JSONBMap
	if req.Result != nil {
		result = db.FromMap(req.Result)
	}

	plan, err := h.queries.UpdatePlanStatus(r.Context(), id, req.Status, result)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to update plan status", err, requestID, r.URL.Path, r.Method, "plan", id.String(), nil))
		return
	}

	respondJSON(w, http.StatusOK, plan)
}


