/*-------------------------------------------------------------------------
 *
 * versions_handlers.go
 *    API handlers for agent versioning
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/api/versions_handlers.go
 *
 *-------------------------------------------------------------------------
 */

package api

import (
	"encoding/json"
	"net/http"
	"strconv"

	"github.com/google/uuid"
	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronAgent/internal/db"
)

/* ListAgentVersions lists all versions for an agent */
func (h *Handlers) ListAgentVersions(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	agentID, err := uuid.Parse(vars["id"])
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent id", err, requestID, r.URL.Path, r.Method, "agent_version", "", nil))
		return
	}

	versions, err := h.queries.ListAgentVersions(r.Context(), agentID)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to list agent versions", err, requestID, r.URL.Path, r.Method, "agent_version", agentID.String(), nil))
		return
	}

	respondJSON(w, http.StatusOK, versions)
}

/* CreateAgentVersion creates a new agent version */
func (h *Handlers) CreateAgentVersion(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	agentID, err := uuid.Parse(vars["id"])
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent id", err, requestID, r.URL.Path, r.Method, "agent_version", "", nil))
		return
	}

	/* Verify agent exists */
	_, err = h.queries.GetAgentByID(r.Context(), agentID)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusNotFound, "agent not found", err, requestID, r.URL.Path, r.Method, "agent_version", agentID.String(), nil))
		return
	}

	var req struct {
		VersionNumber int                    `json:"version_number"`
		Name          *string                `json:"name"`
		Description   *string                `json:"description"`
		SystemPrompt  string                 `json:"system_prompt"`
		ModelName     string                 `json:"model_name"`
		EnabledTools  []string               `json:"enabled_tools"`
		Config        map[string]interface{} `json:"config"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid request body", err, requestID, r.URL.Path, r.Method, "agent_version", agentID.String(), nil))
		return
	}

	if req.VersionNumber <= 0 {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "version_number must be greater than 0", nil, requestID, r.URL.Path, r.Method, "agent_version", agentID.String(), nil))
		return
	}

	if req.SystemPrompt == "" {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "system_prompt is required", nil, requestID, r.URL.Path, r.Method, "agent_version", agentID.String(), nil))
		return
	}

	if req.ModelName == "" {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "model_name is required", nil, requestID, r.URL.Path, r.Method, "agent_version", agentID.String(), nil))
		return
	}

	if req.Config == nil {
		req.Config = make(map[string]interface{})
	}

	version := &db.AgentVersion{
		AgentID:       agentID,
		VersionNumber: req.VersionNumber,
		Name:          req.Name,
		Description:   req.Description,
		SystemPrompt:  req.SystemPrompt,
		ModelName:     req.ModelName,
		EnabledTools:  req.EnabledTools,
		Config:        db.FromMap(req.Config),
		IsActive:      false,
	}

	if err := h.queries.CreateAgentVersion(r.Context(), version); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to create agent version", err, requestID, r.URL.Path, r.Method, "agent_version", agentID.String(), nil))
		return
	}

	respondJSON(w, http.StatusCreated, version)
}

/* GetAgentVersion gets a specific agent version */
func (h *Handlers) GetAgentVersion(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	agentID, err := uuid.Parse(vars["id"])
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent id", err, requestID, r.URL.Path, r.Method, "agent_version", "", nil))
		return
	}

	versionNumber, err := strconv.Atoi(vars["version"])
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid version number", err, requestID, r.URL.Path, r.Method, "agent_version", agentID.String(), nil))
		return
	}

	version, err := h.queries.GetAgentVersion(r.Context(), agentID, versionNumber)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusNotFound, "agent version not found", err, requestID, r.URL.Path, r.Method, "agent_version", agentID.String(), nil))
		return
	}

	respondJSON(w, http.StatusOK, version)
}

/* ActivateAgentVersion activates a specific agent version */
func (h *Handlers) ActivateAgentVersion(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	agentID, err := uuid.Parse(vars["id"])
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent id", err, requestID, r.URL.Path, r.Method, "agent_version", "", nil))
		return
	}

	versionNumber, err := strconv.Atoi(vars["version"])
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid version number", err, requestID, r.URL.Path, r.Method, "agent_version", agentID.String(), nil))
		return
	}

	/* Verify version exists */
	_, err = h.queries.GetAgentVersion(r.Context(), agentID, versionNumber)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusNotFound, "agent version not found", err, requestID, r.URL.Path, r.Method, "agent_version", agentID.String(), nil))
		return
	}

	if err := h.queries.ActivateAgentVersion(r.Context(), agentID, versionNumber); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to activate agent version", err, requestID, r.URL.Path, r.Method, "agent_version", agentID.String(), nil))
		return
	}

	/* Get updated version */
	version, err := h.queries.GetAgentVersion(r.Context(), agentID, versionNumber)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to get activated version", err, requestID, r.URL.Path, r.Method, "agent_version", agentID.String(), nil))
		return
	}

	respondJSON(w, http.StatusOK, version)
}


