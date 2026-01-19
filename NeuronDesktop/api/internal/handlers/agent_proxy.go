package handlers

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
)

/* AgentProxyHandlers provides generic proxy handlers for NeuronAgent endpoints */
type AgentProxyHandlers struct {
	queries       *db.Queries
	agentHandlers *AgentHandlers
}

/* NewAgentProxyHandlers creates new agent proxy handlers */
func NewAgentProxyHandlers(queries *db.Queries, agentHandlers *AgentHandlers) *AgentProxyHandlers {
	return &AgentProxyHandlers{
		queries:       queries,
		agentHandlers: agentHandlers,
	}
}

/* ProxyRequest proxies a request to NeuronAgent */
/* This is a generic handler that extracts the path from the request URL */
func (h *AgentProxyHandlers) ProxyRequest(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]

	/* Extract the path after /profiles/{profile_id}/agent/ */
	/* The route pattern should match what comes after /agent/ */
	requestPath := r.URL.Path
	prefix := fmt.Sprintf("/api/v1/profiles/%s/agent/", profileID)
	if !strings.HasPrefix(requestPath, prefix) {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid path"), nil)
		return
	}
	agentPath := strings.TrimPrefix(requestPath, prefix)

	/* Read request body if present */
	var body interface{}
	if r.Body != nil && r.ContentLength > 0 {
		bodyBytes, err := io.ReadAll(r.Body)
		if err != nil {
			WriteError(w, r, http.StatusBadRequest, fmt.Errorf("failed to read request body"), nil)
			return
		}
		if len(bodyBytes) > 0 {
			if err := json.Unmarshal(bodyBytes, &body); err != nil {
				body = string(bodyBytes)
			}
		}
	}

	client, err := h.agentHandlers.getClient(r.Context(), profileID)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	/* Construct full path for NeuronAgent API */
	fullPath := "/api/v1/" + agentPath
	if r.URL.RawQuery != "" {
		fullPath += "?" + r.URL.RawQuery
	}

	result, err := client.ProxyRequest(r.Context(), r.Method, fullPath, body)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	WriteSuccess(w, result, http.StatusOK)
}
