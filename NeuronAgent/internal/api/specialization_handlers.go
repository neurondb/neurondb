/*-------------------------------------------------------------------------
 *
 * specialization_handlers.go
 *    Agent Specialization API handlers for NeuronAgent
 *
 * Provides REST API endpoints for agent specialization management.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/api/specialization_handlers.go
 *
 *-------------------------------------------------------------------------
 */

package api

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"

	"github.com/google/uuid"
	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/validation"
)

/* SpecializationHandlers handles agent specialization API requests */
type SpecializationHandlers struct {
	queries *db.Queries
}

/* NewSpecializationHandlers creates new specialization handlers */
func NewSpecializationHandlers(queries *db.Queries) *SpecializationHandlers {
	return &SpecializationHandlers{
		queries: queries,
	}
}

/* CreateSpecializationRequest represents a request to create a specialization */
type CreateSpecializationRequest struct {
	SpecializationType string                 `json:"specialization_type"`
	Capabilities       []string               `json:"capabilities,omitempty"`
	Config             map[string]interface{} `json:"config,omitempty"`
}

/* UpdateSpecializationRequest represents a request to update a specialization */
type UpdateSpecializationRequest struct {
	SpecializationType string                 `json:"specialization_type,omitempty"`
	Capabilities       []string               `json:"capabilities,omitempty"`
	Config             map[string]interface{} `json:"config,omitempty"`
}

/* SpecializationResponse represents a specialization in API responses */
type SpecializationResponse struct {
	ID                 string                 `json:"id"`
	AgentID            string                 `json:"agent_id"`
	SpecializationType string                 `json:"specialization_type"`
	Capabilities       []string               `json:"capabilities"`
	Config             map[string]interface{} `json:"config"`
	CreatedAt          string                 `json:"created_at"`
	UpdatedAt          string                 `json:"updated_at"`
}

/* CreateSpecialization creates a new agent specialization */
func (h *SpecializationHandlers) CreateSpecialization(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	/* Validate agent ID */
	if err := validation.ValidateUUIDRequired(vars["agent_id"], "agent_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID", err, requestID, r.URL.Path, r.Method, "specialization", "", nil))
		return
	}

	agentID, err := uuid.Parse(vars["agent_id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID format", err, requestID, r.URL.Path, r.Method, "specialization", "", nil))
		return
	}

	/* Verify agent exists */
	_, err = h.queries.GetAgentByID(r.Context(), agentID)
	if err != nil {
		status := http.StatusInternalServerError
		if err.Error() == "agent not found: sql: no rows in result set" {
			status = http.StatusNotFound
		}
		respondError(w, NewErrorWithContext(status, "agent not found", err, requestID, r.URL.Path, r.Method, "specialization", "", nil))
		return
	}

	/* Validate request body size */
	const maxBodySize = 1024 * 1024
	bodyBytes, err := validation.ReadAndValidateBody(r, maxBodySize)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body validation failed", err, requestID, r.URL.Path, r.Method, "specialization", "", nil))
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var req CreateSpecializationRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body parsing error", err, requestID, r.URL.Path, r.Method, "specialization", "", nil))
		return
	}

	/* Validate required fields */
	if req.SpecializationType == "" {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "specialization_type is required", nil, requestID, r.URL.Path, r.Method, "specialization", "", nil))
		return
	}

	/* Validate specialization type */
	validTypes := map[string]bool{
		"planning":  true,
		"research":  true,
		"coding":    true,
		"execution": true,
		"analysis":  true,
		"general":   true,
	}
	if !validTypes[req.SpecializationType] {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid specialization_type", nil, requestID, r.URL.Path, r.Method, "specialization", "", map[string]interface{}{
			"valid_types": []string{"planning", "research", "coding", "execution", "analysis", "general"},
		}))
		return
	}

	specialization := &db.AgentSpecialization{
		AgentID:            agentID,
		SpecializationType: req.SpecializationType,
		Capabilities:       req.Capabilities,
		Config:             db.FromMap(req.Config),
	}

	if err := h.queries.CreateAgentSpecialization(r.Context(), specialization); err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "specialization creation failed", err, requestID, r.URL.Path, r.Method, "specialization", "", nil))
		return
	}

	respondJSON(w, http.StatusCreated, toSpecializationResponse(specialization))
}

/* GetSpecialization gets an agent specialization by agent ID */
func (h *SpecializationHandlers) GetSpecialization(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	/* Validate agent ID */
	if err := validation.ValidateUUIDRequired(vars["agent_id"], "agent_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID", err, requestID, r.URL.Path, r.Method, "specialization", "", nil))
		return
	}

	agentID, err := uuid.Parse(vars["agent_id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID format", err, requestID, r.URL.Path, r.Method, "specialization", "", nil))
		return
	}

	specialization, err := h.queries.GetAgentSpecializationByAgentID(r.Context(), agentID)
	if err != nil {
		status := http.StatusInternalServerError
		if err.Error() == "agent specialization not found: sql: no rows in result set" {
			status = http.StatusNotFound
		}
		respondError(w, NewErrorWithContext(status, "failed to get specialization", err, requestID, r.URL.Path, r.Method, "specialization", "", nil))
		return
	}

	respondJSON(w, http.StatusOK, toSpecializationResponse(specialization))
}

/* UpdateSpecialization updates an agent specialization */
func (h *SpecializationHandlers) UpdateSpecialization(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	/* Validate agent ID */
	if err := validation.ValidateUUIDRequired(vars["agent_id"], "agent_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID", err, requestID, r.URL.Path, r.Method, "specialization", "", nil))
		return
	}

	agentID, err := uuid.Parse(vars["agent_id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID format", err, requestID, r.URL.Path, r.Method, "specialization", "", nil))
		return
	}

	/* Get existing specialization */
	specialization, err := h.queries.GetAgentSpecializationByAgentID(r.Context(), agentID)
	if err != nil {
		status := http.StatusInternalServerError
		if err.Error() == "agent specialization not found: sql: no rows in result set" {
			status = http.StatusNotFound
		}
		respondError(w, NewErrorWithContext(status, "failed to get specialization", err, requestID, r.URL.Path, r.Method, "specialization", "", nil))
		return
	}

	/* Validate request body size */
	const maxBodySize = 1024 * 1024
	bodyBytes, err := validation.ReadAndValidateBody(r, maxBodySize)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body validation failed", err, requestID, r.URL.Path, r.Method, "specialization", "", nil))
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var req UpdateSpecializationRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body parsing error", err, requestID, r.URL.Path, r.Method, "specialization", "", nil))
		return
	}

	/* Validate specialization type if provided */
	if req.SpecializationType != "" {
		validTypes := map[string]bool{
			"planning":  true,
			"research":  true,
			"coding":    true,
			"execution": true,
			"analysis":  true,
			"general":   true,
		}
		if !validTypes[req.SpecializationType] {
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid specialization_type", nil, requestID, r.URL.Path, r.Method, "specialization", "", map[string]interface{}{
				"valid_types": []string{"planning", "research", "coding", "execution", "analysis", "general"},
			}))
			return
		}
		specialization.SpecializationType = req.SpecializationType
	}

	/* Update fields */
	if req.Capabilities != nil {
		specialization.Capabilities = req.Capabilities
	}
	if req.Config != nil {
		specialization.Config = db.FromMap(req.Config)
	}

	if err := h.queries.UpdateAgentSpecialization(r.Context(), specialization); err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "specialization update failed", err, requestID, r.URL.Path, r.Method, "specialization", "", nil))
		return
	}

	/* Get updated specialization */
	updatedSpecialization, err := h.queries.GetAgentSpecializationByAgentID(r.Context(), agentID)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to get updated specialization", err, requestID, r.URL.Path, r.Method, "specialization", "", nil))
		return
	}

	respondJSON(w, http.StatusOK, toSpecializationResponse(updatedSpecialization))
}

/* DeleteSpecialization deletes an agent specialization */
func (h *SpecializationHandlers) DeleteSpecialization(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	/* Validate agent ID */
	if err := validation.ValidateUUIDRequired(vars["agent_id"], "agent_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID", err, requestID, r.URL.Path, r.Method, "specialization", "", nil))
		return
	}

	agentID, err := uuid.Parse(vars["agent_id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID format", err, requestID, r.URL.Path, r.Method, "specialization", "", nil))
		return
	}

	if err := h.queries.DeleteAgentSpecializationByAgentID(r.Context(), agentID); err != nil {
		status := http.StatusInternalServerError
		if err.Error() == "agent specialization not found" {
			status = http.StatusNotFound
		}
		respondError(w, NewErrorWithContext(status, "failed to delete specialization", err, requestID, r.URL.Path, r.Method, "specialization", "", nil))
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

/* ListSpecializations lists all agent specializations */
func (h *SpecializationHandlers) ListSpecializations(w http.ResponseWriter, r *http.Request) {
	requestID := GetRequestID(r.Context())

	/* Parse query parameters */
	specializationType := r.URL.Query().Get("specialization_type")
	var specializationTypePtr *string
	if specializationType != "" {
		specializationTypePtr = &specializationType
	}

	specializations, err := h.queries.ListAgentSpecializations(r.Context(), specializationTypePtr)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to list specializations", err, requestID, r.URL.Path, r.Method, "specialization", "", nil))
		return
	}

	responses := make([]SpecializationResponse, len(specializations))
	for i, spec := range specializations {
		responses[i] = toSpecializationResponse(&spec)
	}

	respondJSON(w, http.StatusOK, responses)
}

/* Helper function to convert DB model to API response */

func toSpecializationResponse(specialization *db.AgentSpecialization) SpecializationResponse {
	return SpecializationResponse{
		ID:                 specialization.ID.String(),
		AgentID:            specialization.AgentID.String(),
		SpecializationType: specialization.SpecializationType,
		Capabilities:       specialization.Capabilities,
		Config:             specialization.Config.ToMap(),
		CreatedAt:          specialization.CreatedAt.Format("2006-01-02T15:04:05Z07:00"),
		UpdatedAt:          specialization.UpdatedAt.Format("2006-01-02T15:04:05Z07:00"),
	}
}







