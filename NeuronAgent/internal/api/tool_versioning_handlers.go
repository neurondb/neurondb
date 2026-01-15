/*-------------------------------------------------------------------------
 *
 * tool_versioning_handlers.go
 *    API handlers for tool versioning
 *
 * Provides REST API endpoints for tool version management.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/api/tool_versioning_handlers.go
 *
 *-------------------------------------------------------------------------
 */

package api

import (
	"encoding/json"
	"net/http"

	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/tools"
)

/* ToolVersioningHandlers provides tool versioning API handlers */
type ToolVersioningHandlers struct {
	queries           *db.Queries
	versionManager    *tools.ToolVersionManager
}

/* NewToolVersioningHandlers creates new tool versioning handlers */
func NewToolVersioningHandlers(queries *db.Queries) *ToolVersioningHandlers {
	return &ToolVersioningHandlers{
		queries:        queries,
		versionManager: tools.NewToolVersionManager(queries),
	}
}

/* CreateToolVersion creates a new tool version */
func (h *ToolVersioningHandlers) CreateToolVersion(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	toolName := vars["name"]

	var version tools.ToolVersion
	if err := json.NewDecoder(r.Body).Decode(&version); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid request body", err, requestID, r.URL.Path, r.Method, "", "", nil))
		return
	}

	version.ToolName = toolName

	id, err := h.versionManager.CreateVersion(r.Context(), &version)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to create tool version", err, requestID, r.URL.Path, r.Method, "", "", nil))
		return
	}

	respondJSON(w, http.StatusCreated, map[string]interface{}{"id": id.String()})
}

/* ListToolVersions lists all versions of a tool */
func (h *ToolVersioningHandlers) ListToolVersions(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	toolName := vars["name"]

	versions, err := h.versionManager.ListVersions(r.Context(), toolName)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to list tool versions", err, requestID, r.URL.Path, r.Method, "", "", nil))
		return
	}

	respondJSON(w, http.StatusOK, versions)
}

/* GetToolVersion gets a specific tool version */
func (h *ToolVersioningHandlers) GetToolVersion(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	toolName := vars["name"]
	version := vars["version"]

	toolVersion, err := h.versionManager.GetVersion(r.Context(), toolName, version)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusNotFound, "tool version not found", err, requestID, r.URL.Path, r.Method, "", "", nil))
		return
	}

	respondJSON(w, http.StatusOK, toolVersion)
}

/* DeprecateToolVersion deprecates a tool version */
func (h *ToolVersioningHandlers) DeprecateToolVersion(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	toolName := vars["name"]
	version := vars["version"]

	if err := h.versionManager.DeprecateVersion(r.Context(), toolName, version); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to deprecate tool version", err, requestID, r.URL.Path, r.Method, "", "", nil))
		return
	}

	respondJSON(w, http.StatusOK, map[string]string{"status": "deprecated"})
}

