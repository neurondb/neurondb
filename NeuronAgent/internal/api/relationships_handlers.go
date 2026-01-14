/*-------------------------------------------------------------------------
 *
 * relationships_handlers.go
 *    API handlers for agent relationships
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/api/relationships_handlers.go
 *
 *-------------------------------------------------------------------------
 */

package api

import (
	"encoding/json"
	"net/http"

	"github.com/google/uuid"
	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronAgent/internal/agent"
)

/* ListAgentRelationships lists all relationships for an agent */
func (h *Handlers) ListAgentRelationships(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	agentID, err := uuid.Parse(vars["id"])
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent id", err, requestID, r.URL.Path, r.Method, "agent_relationship", "", nil))
		return
	}

	collab := agent.NewCollaborationManager(h.queries, h.runtime)
	relationships, err := collab.GetAgentRelationships(r.Context(), agentID)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to get agent relationships", err, requestID, r.URL.Path, r.Method, "agent_relationship", agentID.String(), nil))
		return
	}

	respondJSON(w, http.StatusOK, relationships)
}

/* CreateAgentRelationship creates a relationship between agents */
func (h *Handlers) CreateAgentRelationship(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	agentID, err := uuid.Parse(vars["id"])
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent id", err, requestID, r.URL.Path, r.Method, "agent_relationship", "", nil))
		return
	}

	var req struct {
		ToAgentID        uuid.UUID              `json:"to_agent_id"`
		RelationshipType string                 `json:"relationship_type"`
		Metadata         map[string]interface{} `json:"metadata"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid request body", err, requestID, r.URL.Path, r.Method, "agent_relationship", agentID.String(), nil))
		return
	}

	if req.ToAgentID == uuid.Nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "to_agent_id is required", nil, requestID, r.URL.Path, r.Method, "agent_relationship", agentID.String(), nil))
		return
	}

	if req.RelationshipType == "" {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "relationship_type is required", nil, requestID, r.URL.Path, r.Method, "agent_relationship", agentID.String(), nil))
		return
	}

	validTypes := map[string]bool{
		"delegates_to":      true,
		"collaborates_with": true,
		"supervises":        true,
		"reports_to":        true,
	}
	if !validTypes[req.RelationshipType] {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid relationship_type", nil, requestID, r.URL.Path, r.Method, "agent_relationship", agentID.String(), nil))
		return
	}

	/* Verify both agents exist */
	_, err = h.queries.GetAgentByID(r.Context(), agentID)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusNotFound, "from agent not found", err, requestID, r.URL.Path, r.Method, "agent_relationship", agentID.String(), nil))
		return
	}

	_, err = h.queries.GetAgentByID(r.Context(), req.ToAgentID)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusNotFound, "to agent not found", err, requestID, r.URL.Path, r.Method, "agent_relationship", agentID.String(), nil))
		return
	}

	collab := agent.NewCollaborationManager(h.queries, h.runtime)
	if err := collab.CreateRelationship(r.Context(), agentID, req.ToAgentID, req.RelationshipType, req.Metadata); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to create relationship", err, requestID, r.URL.Path, r.Method, "agent_relationship", agentID.String(), nil))
		return
	}

	w.WriteHeader(http.StatusCreated)
}

/* DeleteAgentRelationship deletes a relationship */
func (h *Handlers) DeleteAgentRelationship(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	agentID, err := uuid.Parse(vars["id"])
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent id", err, requestID, r.URL.Path, r.Method, "agent_relationship", "", nil))
		return
	}

	relationshipID, err := uuid.Parse(vars["relationship_id"])
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid relationship id", err, requestID, r.URL.Path, r.Method, "agent_relationship", agentID.String(), nil))
		return
	}

	query := `DELETE FROM neurondb_agent.agent_relationships WHERE id = $1 AND (from_agent_id = $2 OR to_agent_id = $2)`
	result, err := h.queries.GetDB().ExecContext(r.Context(), query, relationshipID, agentID)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to delete relationship", err, requestID, r.URL.Path, r.Method, "agent_relationship", agentID.String(), nil))
		return
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to check delete result", err, requestID, r.URL.Path, r.Method, "agent_relationship", agentID.String(), nil))
		return
	}

	if rowsAffected == 0 {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusNotFound, "relationship not found", nil, requestID, r.URL.Path, r.Method, "agent_relationship", agentID.String(), nil))
		return
	}

	w.WriteHeader(http.StatusNoContent)
}


