package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronDesktop/api/internal/agent"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
	"github.com/neurondb/NeuronDesktop/api/internal/logging"
)

/* AgentHandlers handles NeuronAgent proxy endpoints */
type AgentHandlers struct {
	queries    *db.Queries
	clients    map[string]*agent.Client
	mu         sync.RWMutex
	corsConfig *CORSConfig
	logger     *logging.Logger
}

/* GetQueries returns the queries instance (for use in websocket handler) */
func (h *AgentHandlers) GetQueries() *db.Queries {
	return h.queries
}

/* NewAgentHandlers creates new agent handlers */
func NewAgentHandlers(queries *db.Queries) *AgentHandlers {
	return &AgentHandlers{
		queries: queries,
		clients: make(map[string]*agent.Client),
		corsConfig: &CORSConfig{
			AllowedOrigins: []string{"*"}, /* Default to allow all */
		},
		logger: nil, /* Will be set via SetLogger if needed */
	}
}

/* SetCORSConfig sets the CORS configuration for WebSocket */
func (h *AgentHandlers) SetCORSConfig(allowedOrigins []string) {
	h.corsConfig = &CORSConfig{
		AllowedOrigins: allowedOrigins,
	}
}

/* SetLogger sets the logger for Agent handlers */
func (h *AgentHandlers) SetLogger(logger *logging.Logger) {
	h.logger = logger
}

/* InvalidateClient removes and closes a client for a profile */
func (h *AgentHandlers) InvalidateClient(profileID string) {
	h.mu.Lock()
	defer h.mu.Unlock()

	if _, ok := h.clients[profileID]; ok {
		/* Agent clients don't have a Close method, just remove from cache */
		delete(h.clients, profileID)
	}
}

/* CloseClient closes a specific client */
func (h *AgentHandlers) CloseClient(profileID string) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	if _, ok := h.clients[profileID]; ok {
		delete(h.clients, profileID)
	}
	return nil
}

/* CloseAll closes all cached clients */
func (h *AgentHandlers) CloseAll() {
	h.mu.Lock()
	defer h.mu.Unlock()

	for profileID := range h.clients {
		delete(h.clients, profileID)
	}
}

/* getClient gets or creates an agent client for a profile */
func (h *AgentHandlers) getClient(ctx context.Context, profileID string) (*agent.Client, error) {
	/* First check with read lock */
	h.mu.RLock()
	client, ok := h.clients[profileID]
	h.mu.RUnlock()

	if ok {
		return client, nil
	}

	/* Need to create client - acquire write lock */
	h.mu.Lock()
	defer h.mu.Unlock()

	/* Double-check after acquiring write lock (another goroutine might have created it) */
	if _, ok := h.clients[profileID]; ok {
		return client, nil
	}

	profile, err := h.queries.GetProfile(ctx, profileID)
	if err != nil {
		return nil, fmt.Errorf("failed to get profile: %w", err)
	}

	if profile.AgentEndpoint == "" {
		return nil, fmt.Errorf("agent endpoint not configured")
	}

	client = agent.NewClient(profile.AgentEndpoint, profile.AgentAPIKey)

	h.clients[profileID] = client

	return client, nil
}

/* ListAgents lists agents */
func (h *AgentHandlers) ListAgents(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]

	client, err := h.getClient(r.Context(), profileID)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	agents, err := client.ListAgents(r.Context())
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	WriteSuccess(w, agents, http.StatusOK)
}

/* CreateSession creates a session */
func (h *AgentHandlers) CreateSession(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]

	var req agent.CreateSessionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid request body"), nil)
		return
	}

	client, err := h.getClient(r.Context(), profileID)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	session, err := client.CreateSession(r.Context(), req)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	WriteSuccess(w, session, http.StatusCreated)
}

/* SendMessage sends a message */
func (h *AgentHandlers) SendMessage(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]
	sessionID := vars["session_id"]

	var req agent.SendMessageRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	client, err := h.getClient(r.Context(), profileID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	message, err := client.SendMessage(r.Context(), sessionID, req)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(message)
}

/* GetAgent gets a single agent */
func (h *AgentHandlers) GetAgent(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]
	agentID := vars["agent_id"]

	client, err := h.getClient(r.Context(), profileID)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	agent, err := client.GetAgent(r.Context(), agentID)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	WriteSuccess(w, agent, http.StatusOK)
}

/* CreateAgent creates a new agent */
func (h *AgentHandlers) CreateAgent(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]

	var req agent.CreateAgentRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid request body"), nil)
		return
	}

	if req.Name == "" {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("name is required"), nil)
		return
	}

	client, err := h.getClient(r.Context(), profileID)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	agent, err := client.CreateAgent(r.Context(), req)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	WriteSuccess(w, agent, http.StatusCreated)
}

/* UpdateAgent updates an existing agent */
func (h *AgentHandlers) UpdateAgent(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]
	agentID := vars["agent_id"]

	var req agent.UpdateAgentRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid request body"), nil)
		return
	}

	client, err := h.getClient(r.Context(), profileID)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	updatedAgent, err := client.UpdateAgent(r.Context(), agentID, req)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	WriteSuccess(w, updatedAgent, http.StatusOK)
}

/* DeleteAgent deletes an agent */
func (h *AgentHandlers) DeleteAgent(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]
	agentID := vars["agent_id"]

	client, err := h.getClient(r.Context(), profileID)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	if err := client.DeleteAgent(r.Context(), agentID); err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	WriteSuccess(w, map[string]interface{}{"success": true}, http.StatusOK)
}

/* TestAgentConfig tests an Agent configuration without saving it */
func (h *AgentHandlers) TestAgentConfig(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Endpoint string `json:"endpoint"`
		APIKey   string `json:"api_key"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid request body"), nil)
		return
	}

	if req.Endpoint == "" {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("endpoint is required"), nil)
		return
	}

	testClient := agent.NewClient(req.Endpoint, req.APIKey)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	_, err := testClient.ListAgents(ctx)
	if err != nil {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("failed to connect to agent: %w", err), nil)
		return
	}

	WriteSuccess(w, map[string]interface{}{
		"success": true,
		"message": "Agent configuration test passed",
	}, http.StatusOK)
}

/* ListModels returns available LLM models */
func (h *AgentHandlers) ListModels(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]

	/* Try to get models from NeuronAgent API */
	client, err := h.getClient(r.Context(), profileID)
	if err == nil {
		agentModels, err := client.ListModels(r.Context())
		if err == nil && len(agentModels) > 0 {
			/* Convert agent models to response format */
			models := make([]map[string]interface{}, 0, len(agentModels))
			for _, m := range agentModels {
				modelMap := map[string]interface{}{
					"name": m.Name,
				}
				if m.Provider != "" {
					modelMap["provider"] = m.Provider
				}
				if m.Description != "" {
					modelMap["description"] = m.Description
				}
				if m.DisplayName != "" {
					modelMap["display_name"] = m.DisplayName
				} else {
					modelMap["display_name"] = m.Name
				}
				models = append(models, modelMap)
			}
			WriteSuccess(w, map[string]interface{}{"models": models}, http.StatusOK)
			return
		}
	}

	/* Fallback to hardcoded models if API call fails or returns empty */
	models := []map[string]interface{}{
		{
			"name":         "gpt-4",
			"display_name": "GPT-4",
			"provider":     "OpenAI",
			"description":  "Most capable GPT-4 model",
		},
		{
			"name":         "gpt-4-turbo",
			"display_name": "GPT-4 Turbo",
			"provider":     "OpenAI",
			"description":  "Faster and cheaper GPT-4 variant",
		},
		{
			"name":         "gpt-3.5-turbo",
			"display_name": "GPT-3.5 Turbo",
			"provider":     "OpenAI",
			"description":  "Fast and cost-effective model",
		},
		{
			"name":         "claude-3-opus",
			"display_name": "Claude 3 Opus",
			"provider":     "Anthropic",
			"description":  "Most capable Claude model",
		},
		{
			"name":         "claude-3-sonnet",
			"display_name": "Claude 3 Sonnet",
			"provider":     "Anthropic",
			"description":  "Balanced performance and speed",
		},
		{
			"name":         "claude-3-haiku",
			"display_name": "Claude 3 Haiku",
			"provider":     "Anthropic",
			"description":  "Fastest Claude model",
		},
		{
			"name":         "gemini-pro",
			"display_name": "Gemini Pro",
			"provider":     "Google",
			"description":  "Google's advanced model",
		},
		{
			"name":         "llama-2-70b",
			"display_name": "Llama 2 70B",
			"provider":     "Meta",
			"description":  "Open-source large model",
		},
		{
			"name":         "custom",
			"display_name": "Custom Model",
			"provider":     "Custom",
			"description":  "Enter a custom model name",
		},
	}

	WriteSuccess(w, map[string]interface{}{"models": models}, http.StatusOK)
}

/* ListSessions lists sessions for an agent */
func (h *AgentHandlers) ListSessions(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]
	agentID := vars["agent_id"]

	client, err := h.getClient(r.Context(), profileID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	sessions, err := client.ListSessions(r.Context(), agentID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(sessions)
}

/* GetSession gets a session */
func (h *AgentHandlers) GetSession(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]
	sessionID := vars["session_id"]

	client, err := h.getClient(r.Context(), profileID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	session, err := client.GetSession(r.Context(), sessionID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(session)
}

/* GetMessages gets messages from a session */
func (h *AgentHandlers) GetMessages(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]
	sessionID := vars["session_id"]

	client, err := h.getClient(r.Context(), profileID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	messages, err := client.GetMessages(r.Context(), sessionID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(messages)
}

/* ExportAgent exports an agent as JSON */
func (h *AgentHandlers) ExportAgent(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]
	agentID := vars["agent_id"]

	client, err := h.getClient(r.Context(), profileID)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	agent, err := client.GetAgent(r.Context(), agentID)
	if err != nil {
		WriteError(w, r, http.StatusNotFound, fmt.Errorf("agent not found"), nil)
		return
	}

	exportData := map[string]interface{}{
		"version":     "1.0",
		"type":        "agent",
		"exported_at": time.Now().Format(time.RFC3339),
		"agent": map[string]interface{}{
			"name":          agent.Name,
			"description":   agent.Description,
			"system_prompt": agent.SystemPrompt,
			"model_name":    agent.ModelName,
			"enabled_tools": agent.EnabledTools,
			"config":        agent.Config,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=agent-%s-%s.json", agent.Name, agentID[:8]))
	json.NewEncoder(w).Encode(exportData)
}

/* ImportAgent imports an agent from JSON */
func (h *AgentHandlers) ImportAgent(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]

	var importData struct {
		Version string                 `json:"version"`
		Type    string                 `json:"type"`
		Agent   map[string]interface{} `json:"agent"`
	}

	if err := json.NewDecoder(r.Body).Decode(&importData); err != nil {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid JSON: %w", err), nil)
		return
	}

	if importData.Type != "agent" {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid import type: expected 'agent'"), nil)
		return
	}

	agentData := importData.Agent
	name, _ := agentData["name"].(string)
	if name == "" {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("agent name is required"), nil)
		return
	}

	client, err := h.getClient(r.Context(), profileID)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	existingAgents, err := client.ListAgents(r.Context())
	if err == nil {
		for _, a := range existingAgents {
			if a.Name == name {
				WriteError(w, r, http.StatusConflict, fmt.Errorf("agent with name '%s' already exists", name), nil)
				return
			}
		}
	}

	createReq := agent.CreateAgentRequest{
		Name:         name,
		Description:  getStringFromMap(agentData, "description"),
		SystemPrompt: getStringFromMap(agentData, "system_prompt"),
		ModelName:    getStringFromMap(agentData, "model_name"),
		EnabledTools: getStringSliceFromMap(agentData, "enabled_tools"),
		Config:       getMapFromMap(agentData, "config"),
	}

	if createReq.Name == "" {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("name is required"), nil)
		return
	}

	createdAgent, err := client.CreateAgent(r.Context(), createReq)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	WriteSuccess(w, createdAgent, http.StatusCreated)
}

/* Helper functions for agent import */
func getStringFromMap(m map[string]interface{}, key string) string {
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}

func getStringSliceFromMap(m map[string]interface{}, key string) []string {
	if v, ok := m[key].([]interface{}); ok {
		result := make([]string, 0, len(v))
		for _, item := range v {
			if s, ok := item.(string); ok {
				result = append(result, s)
			}
		}
		return result
	}
	return nil
}

func getMapFromMap(m map[string]interface{}, key string) map[string]interface{} {
	if v, ok := m[key].(map[string]interface{}); ok {
		return v
	}
	return nil
}
