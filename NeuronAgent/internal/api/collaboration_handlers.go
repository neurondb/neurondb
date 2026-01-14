/*-------------------------------------------------------------------------
 *
 * collaboration_handlers.go
 *    API handlers for collaboration workspace endpoints
 *
 * Provides HTTP handlers for workspace creation, participant management,
 * and real-time collaboration features.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/api/collaboration_handlers.go
 *
 *-------------------------------------------------------------------------
 */

package api

import (
	"encoding/json"
	"net/http"

	"github.com/google/uuid"
	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronAgent/internal/collaboration"
	"github.com/neurondb/NeuronAgent/internal/db"
)

type CollaborationHandlers struct {
	queries   *db.Queries
	workspace *collaboration.WorkspaceManager
}

func NewCollaborationHandlers(queries *db.Queries, workspace *collaboration.WorkspaceManager) *CollaborationHandlers {
	return &CollaborationHandlers{
		queries:   queries,
		workspace: workspace,
	}
}

type CreateWorkspaceRequest struct {
	Name        string  `json:"name"`
	Description *string `json:"description,omitempty"`
}

type AddParticipantRequest struct {
	UserID  *uuid.UUID `json:"user_id,omitempty"`
	AgentID *uuid.UUID `json:"agent_id,omitempty"`
	Role    string     `json:"role"`
}

/* CreateWorkspace creates a new collaboration workspace */
func (h *CollaborationHandlers) CreateWorkspace(w http.ResponseWriter, r *http.Request) {
	requestID := GetRequestID(r.Context())

	var req CreateWorkspaceRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, WrapError(NewError(http.StatusBadRequest, "workspace creation failed: request body parsing error", err), requestID))
		return
	}

	if req.Name == "" {
		respondError(w, WrapError(NewError(http.StatusBadRequest, "workspace name required", nil), requestID))
		return
	}

	/* Get owner from context */
	var ownerID *uuid.UUID
	/* Owner ID would come from authenticated user context */

	workspaceID, err := h.workspace.CreateWorkspace(r.Context(), req.Name, ownerID)
	if err != nil {
		respondError(w, WrapError(NewError(http.StatusInternalServerError, "workspace creation failed", err), requestID))
		return
	}

	respondJSON(w, http.StatusCreated, map[string]interface{}{
		"workspace_id": workspaceID.String(),
		"name":         req.Name,
		"status":       "created",
	})
}

/* GetWorkspace retrieves workspace state */
func (h *CollaborationHandlers) GetWorkspace(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	workspaceID, err := uuid.Parse(vars["id"])
	if err != nil {
		respondError(w, WrapError(NewError(http.StatusBadRequest, "invalid workspace ID", err), requestID))
		return
	}

	workspace, participants, err := h.workspace.GetWorkspaceState(r.Context(), workspaceID)
	if err != nil {
		respondError(w, WrapError(NewError(http.StatusInternalServerError, "workspace retrieval failed", err), requestID))
		return
	}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"workspace":    workspace,
		"participants": participants,
	})
}

/* AddParticipant adds a participant to a workspace */
func (h *CollaborationHandlers) AddParticipant(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	workspaceID, err := uuid.Parse(vars["id"])
	if err != nil {
		respondError(w, WrapError(NewError(http.StatusBadRequest, "invalid workspace ID", err), requestID))
		return
	}

	var req AddParticipantRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, WrapError(NewError(http.StatusBadRequest, "participant addition failed: request body parsing error", err), requestID))
		return
	}

	if req.UserID == nil && req.AgentID == nil {
		respondError(w, WrapError(NewError(http.StatusBadRequest, "user_id or agent_id required", nil), requestID))
		return
	}

	err = h.workspace.AddParticipant(r.Context(), workspaceID, req.UserID, req.AgentID, req.Role)
	if err != nil {
		respondError(w, WrapError(NewError(http.StatusInternalServerError, "participant addition failed", err), requestID))
		return
	}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"workspace_id": workspaceID.String(),
		"status":       "participant_added",
	})
}
