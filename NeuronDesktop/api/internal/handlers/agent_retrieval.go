package handlers

import (
	"net/http"
	"strconv"

	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
)

/* AgentRetrievalHandlers handles agent retrieval-related endpoints */
type AgentRetrievalHandlers struct {
	queries *db.Queries
	agentHandlers *AgentHandlers
}

/* NewAgentRetrievalHandlers creates new agent retrieval handlers */
func NewAgentRetrievalHandlers(queries *db.Queries, agentHandlers *AgentHandlers) *AgentRetrievalHandlers {
	return &AgentRetrievalHandlers{
		queries: queries,
		agentHandlers: agentHandlers,
	}
}

/* GetRetrievalStats gets retrieval statistics for an agent */
func (h *AgentRetrievalHandlers) GetRetrievalStats(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]
	agentID := vars["agent_id"]

	/* Parse days query parameter (default: 30) */
	days := 30
	if daysStr := r.URL.Query().Get("days"); daysStr != "" {
		if parsedDays, err := strconv.Atoi(daysStr); err == nil && parsedDays > 0 && parsedDays <= 365 {
			days = parsedDays
		}
	}

	client, err := h.agentHandlers.getClient(r.Context(), profileID)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	stats, err := client.GetRetrievalStats(r.Context(), agentID, days)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	WriteSuccess(w, stats, http.StatusOK)
}
