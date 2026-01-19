package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronDesktop/api/internal/agent"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
)

/* AgentMemoryHandlers handles agent memory-related endpoints */
type AgentMemoryHandlers struct {
	queries *db.Queries
	agentHandlers *AgentHandlers
}

/* NewAgentMemoryHandlers creates new agent memory handlers */
func NewAgentMemoryHandlers(queries *db.Queries, agentHandlers *AgentHandlers) *AgentMemoryHandlers {
	return &AgentMemoryHandlers{
		queries: queries,
		agentHandlers: agentHandlers,
	}
}

/* SubmitMemoryFeedback submits feedback on a memory retrieval */
func (h *AgentMemoryHandlers) SubmitMemoryFeedback(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]
	memoryID := vars["memory_id"]

	var req agent.MemoryFeedbackRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid request body"), nil)
		return
	}

	client, err := h.agentHandlers.getClient(r.Context(), profileID)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	response, err := client.SubmitMemoryFeedback(r.Context(), memoryID, req)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	WriteSuccess(w, response, http.StatusOK)
}

/* ConsolidateMemory consolidates similar memories */
func (h *AgentMemoryHandlers) ConsolidateMemory(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]
	agentID := vars["agent_id"]

	var req agent.ConsolidateMemoryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid request body"), nil)
		return
	}

	client, err := h.agentHandlers.getClient(r.Context(), profileID)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	response, err := client.ConsolidateMemory(r.Context(), agentID, req)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	WriteSuccess(w, response, http.StatusOK)
}

/* GetMemoryQuality gets memory quality metrics */
func (h *AgentMemoryHandlers) GetMemoryQuality(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]
	agentID := vars["agent_id"]

	memoryID := r.URL.Query().Get("memory_id")
	tier := r.URL.Query().Get("tier")

	client, err := h.agentHandlers.getClient(r.Context(), profileID)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	quality, err := client.GetMemoryQuality(r.Context(), agentID, memoryID, tier)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	WriteSuccess(w, quality, http.StatusOK)
}
