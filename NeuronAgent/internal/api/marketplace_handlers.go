/*-------------------------------------------------------------------------
 *
 * marketplace_handlers.go
 *    API handlers for marketplace features
 *
 * Provides REST API endpoints for tool, agent, and workflow marketplace.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/api/marketplace_handlers.go
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
	"github.com/neurondb/NeuronAgent/internal/marketplace"
)

/* MarketplaceHandlers provides marketplace API handlers */
type MarketplaceHandlers struct {
	queries          *db.Queries
	toolMarketplace  *marketplace.ToolMarketplace
	agentMarketplace *marketplace.AgentMarketplace
}

/* NewMarketplaceHandlers creates new marketplace handlers */
func NewMarketplaceHandlers(queries *db.Queries) *MarketplaceHandlers {
	return &MarketplaceHandlers{
		queries:          queries,
		toolMarketplace:  marketplace.NewToolMarketplace(queries),
		agentMarketplace: marketplace.NewAgentMarketplace(queries),
	}
}

/* ListMarketplaceTools lists tools in the marketplace */
func (h *MarketplaceHandlers) ListMarketplaceTools(w http.ResponseWriter, r *http.Request) {
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	offset, _ := strconv.Atoi(r.URL.Query().Get("offset"))

	if limit <= 0 {
		limit = 20
	}

	tools, err := h.toolMarketplace.ListTools(r.Context(), limit, offset)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to list marketplace tools", err, requestID, r.URL.Path, r.Method, "", "", nil))
		return
	}

	respondJSON(w, http.StatusOK, tools)
}

/* PublishTool publishes a tool to the marketplace */
func (h *MarketplaceHandlers) PublishTool(w http.ResponseWriter, r *http.Request) {
	var tool marketplace.MarketplaceTool
	if err := json.NewDecoder(r.Body).Decode(&tool); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid request body", err, requestID, r.URL.Path, r.Method, "", "", nil))
		return
	}

	id, err := h.toolMarketplace.PublishTool(r.Context(), &tool)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to publish tool", err, requestID, r.URL.Path, r.Method, "", "", nil))
		return
	}

	respondJSON(w, http.StatusCreated, map[string]interface{}{"id": id.String()})
}

/* RateTool rates a tool in the marketplace */
func (h *MarketplaceHandlers) RateTool(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	toolID := vars["id"]

	var req struct {
		UserID string  `json:"user_id"`
		Rating float64 `json:"rating"`
		Review string  `json:"review"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid request body", err, requestID, r.URL.Path, r.Method, "", "", nil))
		return
	}

	toolUUID, err := uuid.Parse(toolID)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid tool ID", err, requestID, r.URL.Path, r.Method, "", "", nil))
		return
	}

	if err := h.toolMarketplace.RateTool(r.Context(), toolUUID, req.UserID, req.Rating, req.Review); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to rate tool", err, requestID, r.URL.Path, r.Method, "", "", nil))
		return
	}

	respondJSON(w, http.StatusOK, map[string]string{"status": "success"})
}

/* ListMarketplaceAgents lists agents in the marketplace */
func (h *MarketplaceHandlers) ListMarketplaceAgents(w http.ResponseWriter, r *http.Request) {
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	offset, _ := strconv.Atoi(r.URL.Query().Get("offset"))

	if limit <= 0 {
		limit = 20
	}

	agents, err := h.agentMarketplace.ListAgents(r.Context(), limit, offset)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to list marketplace agents", err, requestID, r.URL.Path, r.Method, "", "", nil))
		return
	}

	respondJSON(w, http.StatusOK, agents)
}

/* PublishAgent publishes an agent to the marketplace */
func (h *MarketplaceHandlers) PublishAgent(w http.ResponseWriter, r *http.Request) {
	var agent marketplace.MarketplaceAgent
	if err := json.NewDecoder(r.Body).Decode(&agent); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid request body", err, requestID, r.URL.Path, r.Method, "", "", nil))
		return
	}

	id, err := h.agentMarketplace.PublishAgent(r.Context(), &agent)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to publish agent", err, requestID, r.URL.Path, r.Method, "", "", nil))
		return
	}

	respondJSON(w, http.StatusCreated, map[string]interface{}{"id": id.String()})
}

