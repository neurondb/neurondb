/*-------------------------------------------------------------------------
 *
 * observability_handlers.go
 *    API handlers for observability features
 *
 * Provides REST API endpoints for decision trees, performance profiling,
 * and observability dashboards.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/api/observability_handlers.go
 *
 *-------------------------------------------------------------------------
 */

package api

import (
	"net/http"

	"github.com/google/uuid"
	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/observability"
)

/* ObservabilityHandlers provides observability API handlers */
type ObservabilityHandlers struct {
	queries              *db.Queries
	decisionTreeViz      *observability.DecisionTreeVisualizer
	profiler             *observability.Profiler
}

/* NewObservabilityHandlers creates new observability handlers */
func NewObservabilityHandlers(queries *db.Queries) *ObservabilityHandlers {
	return &ObservabilityHandlers{
		queries:         queries,
		decisionTreeViz: observability.NewDecisionTreeVisualizer(queries),
		profiler:        observability.NewProfiler(queries),
	}
}

/* GetDecisionTree gets decision tree for an execution */
func (h *ObservabilityHandlers) GetDecisionTree(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	executionID := vars["id"]

	execUUID, err := uuid.Parse(executionID)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid execution ID", err, requestID, r.URL.Path, r.Method, "", "", nil))
		return
	}

	tree, err := h.decisionTreeViz.BuildDecisionTree(r.Context(), execUUID)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to build decision tree", err, requestID, r.URL.Path, r.Method, "", "", nil))
		return
	}

	respondJSON(w, http.StatusOK, tree)
}

/* GetToolCallChain gets tool call chain for an execution */
func (h *ObservabilityHandlers) GetToolCallChain(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	executionID := vars["id"]

	execUUID, err := uuid.Parse(executionID)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid execution ID", err, requestID, r.URL.Path, r.Method, "", "", nil))
		return
	}

	chain, err := h.decisionTreeViz.GetToolCallChain(r.Context(), execUUID)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to get tool call chain", err, requestID, r.URL.Path, r.Method, "", "", nil))
		return
	}

	respondJSON(w, http.StatusOK, chain)
}

/* GetPerformanceProfile gets performance profile for an execution */
func (h *ObservabilityHandlers) GetPerformanceProfile(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	executionID := vars["id"]

	execUUID, err := uuid.Parse(executionID)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid execution ID", err, requestID, r.URL.Path, r.Method, "", "", nil))
		return
	}

	profile, err := h.profiler.ProfileExecution(r.Context(), execUUID)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to get performance profile", err, requestID, r.URL.Path, r.Method, "", "", nil))
		return
	}

	respondJSON(w, http.StatusOK, profile)
}


