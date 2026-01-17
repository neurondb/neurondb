package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"

	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
	"github.com/neurondb/NeuronDesktop/api/internal/logging"
	"github.com/neurondb/NeuronDesktop/api/internal/mcp"
	"github.com/neurondb/NeuronDesktop/api/internal/utils"
)

/* MCPHandlers handles MCP-related endpoints */
type MCPHandlers struct {
	mcpManager *MCPManager
	corsConfig *CORSConfig
	logger     *logging.Logger
}

/* CORSConfig holds CORS configuration for WebSocket */
type CORSConfig struct {
	AllowedOrigins []string
}

/* NewMCPHandlers creates new MCP handlers */
func NewMCPHandlers(mcpManager *MCPManager) *MCPHandlers {
	return &MCPHandlers{
		mcpManager: mcpManager,
		corsConfig: &CORSConfig{
			AllowedOrigins: []string{"*"}, /* Default to allow all */
		},
		logger: nil, /* Will be set via SetLogger if needed */
	}
}

/* SetLogger sets the logger for MCP handlers */
func (h *MCPHandlers) SetLogger(logger *logging.Logger) {
	h.logger = logger
}

/* SetCORSConfig sets the CORS configuration for WebSocket */
func (h *MCPHandlers) SetCORSConfig(allowedOrigins []string) {
	h.corsConfig = &CORSConfig{
		AllowedOrigins: allowedOrigins,
	}
}

/* ListTools lists tools from the MCP server */
func (h *MCPHandlers) ListTools(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]

	client, err := h.mcpManager.GetClient(r.Context(), profileID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	tools, err := client.ListTools()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(tools)
}

/* CallTool calls a tool on the MCP server */
func (h *MCPHandlers) CallTool(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]

	var req struct {
		Name      string                 `json:"name"`
		Arguments map[string]interface{} `json:"arguments"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid request body"), nil)
		return
	}

	/* Validate tool call */
	if errors := utils.ValidateToolCall(req.Name, req.Arguments); len(errors) > 0 {
		WriteValidationErrors(w, r, errors)
		return
	}

	client, err := h.mcpManager.GetClient(r.Context(), profileID)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	result, err := client.CallTool(req.Name, req.Arguments)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	WriteSuccess(w, result, http.StatusOK)
}

/* ListConnections lists active MCP connections */
func (h *MCPHandlers) ListConnections(w http.ResponseWriter, r *http.Request) {
	connections := h.mcpManager.ListConnections()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(connections)
}

/* TestMCPConfig tests an MCP configuration without saving it */
func (h *MCPHandlers) TestMCPConfig(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Command string            `json:"command"`
		Args    []string          `json:"args,omitempty"`
		Env     map[string]string `json:"env,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid request body"), nil)
		return
	}

	/* Create temporary MCP config */
	mcpConfig := mcp.MCPConfig{
		Command: req.Command,
		Args:    req.Args,
		Env:     req.Env,
	}

	if mcpConfig.Command == "" {
		mcpConfig.Command = "/home/pge/pge/neurondb/NeuronMCP/bin/neurondb-mcp"
	}
	if mcpConfig.Args == nil {
		mcpConfig.Args = []string{}
	}
	if mcpConfig.Env == nil {
		mcpConfig.Env = make(map[string]string)
	}

	/* Try to create and initialize client */
	testClient, err := mcp.NewClient(mcpConfig)
	if err != nil {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("failed to create MCP client: %w", err), nil)
		return
	}
	defer testClient.Close()

	/* Try to list tools as a test */
	_, err = testClient.ListTools()
	if err != nil {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("failed to list tools: %w", err), nil)
		return
	}

	WriteSuccess(w, map[string]interface{}{
		"success": true,
		"message": "MCP configuration test passed",
	}, http.StatusOK)
}

/* MCPManager manages MCP client connections */
type MCPManager struct {
	clients map[string]*mcp.Client
	mu      sync.RWMutex
	queries *db.Queries
}

/* NewMCPManager creates a new MCP manager */
func NewMCPManager(queries *db.Queries) *MCPManager {
	return &MCPManager{
		clients: make(map[string]*mcp.Client),
		queries: queries,
	}
}

/* InvalidateClient removes and closes a client for a profile */
func (m *MCPManager) InvalidateClient(profileID string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if client, ok := m.clients[profileID]; ok {
		client.Close()
		delete(m.clients, profileID)
	}
}

/* CloseClient closes a specific client */
func (m *MCPManager) CloseClient(profileID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if client, ok := m.clients[profileID]; ok {
		err := client.Close()
		delete(m.clients, profileID)
		return err
	}
	return nil
}

/* CloseAll closes all cached clients */
func (m *MCPManager) CloseAll() {
	m.mu.Lock()
	defer m.mu.Unlock()

	for profileID, client := range m.clients {
		client.Close()
		delete(m.clients, profileID)
	}
}

/* GetMCPManager returns the MCP manager (for use in websocket handler) */
func (h *MCPHandlers) GetMCPManager() *MCPManager {
	return h.mcpManager
}

/* GetClient gets or creates an MCP client for a profile */
func (m *MCPManager) GetClient(ctx context.Context, profileID string) (*mcp.Client, error) {
	/* First check with read lock */
	m.mu.RLock()
	client, ok := m.clients[profileID]
	m.mu.RUnlock()

	if ok && client.IsAlive() {
		return client, nil
	}

	/* Need to create client - acquire write lock */
	m.mu.Lock()
	defer m.mu.Unlock()

	/* Double-check after acquiring write lock (another goroutine might have created it) */
	if client, ok := m.clients[profileID]; ok && client.IsAlive() {
		return client, nil
	}

	/* Close old client if it exists but is not alive */
	if client, ok := m.clients[profileID]; ok {
		client.Close()
		delete(m.clients, profileID)
	}

	/* Get profile */
	profile, err := m.queries.GetProfile(ctx, profileID)
	if err != nil {
		return nil, fmt.Errorf("failed to get profile: %w", err)
	}

	if profile == nil {
		return nil, fmt.Errorf("profile not found: %s", profileID)
	}

	/* Parse MCP config */
	defaultCmd := utils.FindNeuronMCPBinary()
	if defaultCmd == "" {
		// Fallback to PATH lookup (in Docker we install to /usr/local/bin/neurondb-mcp)
		defaultCmd = "neurondb-mcp"
	}
	mcpConfig := mcp.MCPConfig{
		Command: defaultCmd,
		Args:    []string{},
		Env:     make(map[string]string),
	}

	/* Set default database environment variables from profile's NeuronDB DSN */
	if profile.NeuronDBDSN != "" {
		/* Use the same database connection as the profile's NeuronDB */
		/* This ensures MCP can connect to the same database */
		/* MCP server expects NEURONDB_CONNECTION_STRING (not NEURONDB_DSN) */
		mcpConfig.Env["NEURONDB_CONNECTION_STRING"] = profile.NeuronDBDSN
	}

	if profile.MCPConfig != nil {
		if cmd, ok := profile.MCPConfig["command"].(string); ok {
			mcpConfig.Command = cmd
		}
		if args, ok := profile.MCPConfig["args"].([]interface{}); ok {
			for _, arg := range args {
				if s, ok := arg.(string); ok {
					mcpConfig.Args = append(mcpConfig.Args, s)
				}
			}
		}
		if env, ok := profile.MCPConfig["env"].(map[string]interface{}); ok {
			for k, v := range env {
				if s, ok := v.(string); ok {
					mcpConfig.Env[k] = s
				}
			}
		}
	}

	/* Create new client */
	newClient, err := mcp.NewClient(mcpConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create MCP client: %w", err)
	}
	client = newClient

	m.clients[profileID] = client

	return client, nil
}

/* ListConnections lists all active connections */
func (m *MCPManager) ListConnections() []map[string]interface{} {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var connections []map[string]interface{}
	for profileID, client := range m.clients {
		connections = append(connections, map[string]interface{}{
			"profile_id": profileID,
			"alive":      client.IsAlive(),
		})
	}

	return connections
}

/* ListThreads lists all chat threads for a profile */
func (h *MCPHandlers) ListThreads(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]

	threads, err := h.mcpManager.queries.ListMCPThreads(r.Context(), profileID)
	if err != nil {
		/* Check if it's a "table doesn't exist" error */
		errStr := err.Error()
		if strings.Contains(errStr, "does not exist") || strings.Contains(errStr, "relation") || strings.Contains(errStr, "mcp_chat_threads") {
			WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("database tables not found. Please run migration: psql -d neurondb -f NeuronDesktop/api/migrations/007_mcp_chat_threads.sql"), nil)
			return
		}
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	WriteSuccess(w, threads, http.StatusOK)
}

/* CreateThread creates a new chat thread */
func (h *MCPHandlers) CreateThread(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]

	var req struct {
		Title string `json:"title"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid request body"), nil)
		return
	}

	title := req.Title
	if title == "" {
		title = "New chat"
	}

	thread, err := h.mcpManager.queries.CreateMCPThread(r.Context(), profileID, title)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	WriteSuccess(w, thread, http.StatusCreated)
}

/* GetThread gets a thread with its messages */
func (h *MCPHandlers) GetThread(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	threadID := vars["thread_id"]
	/* profileID is available but not needed for this query */

	thread, err := h.mcpManager.queries.GetMCPThread(r.Context(), threadID)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	if thread == nil {
		WriteError(w, r, http.StatusNotFound, fmt.Errorf("thread not found"), nil)
		return
	}

	messages, err := h.mcpManager.queries.ListMCPMessages(r.Context(), threadID)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	response := map[string]interface{}{
		"thread":   thread,
		"messages": messages,
	}

	WriteSuccess(w, response, http.StatusOK)
}

/* UpdateThread updates a thread (e.g., title) */
func (h *MCPHandlers) UpdateThread(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	threadID := vars["thread_id"]

	var req struct {
		Title string `json:"title"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid request body"), nil)
		return
	}

	thread, err := h.mcpManager.queries.UpdateMCPThread(r.Context(), threadID, req.Title)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	WriteSuccess(w, thread, http.StatusOK)
}

/* DeleteThread deletes a thread */
func (h *MCPHandlers) DeleteThread(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	threadID := vars["thread_id"]

	if err := h.mcpManager.queries.DeleteMCPThread(r.Context(), threadID); err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	WriteSuccess(w, map[string]interface{}{"success": true}, http.StatusOK)
}

/* AddMessage adds a message to a thread */
func (h *MCPHandlers) AddMessage(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	threadID := vars["thread_id"]

	var req struct {
		Role     string                 `json:"role"`
		Content  string                 `json:"content"`
		ToolName string                 `json:"tool_name,omitempty"`
		Data     map[string]interface{} `json:"data,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid request body"), nil)
		return
	}

	message, err := h.mcpManager.queries.CreateMCPMessage(r.Context(), threadID, req.Role, req.Content, req.ToolName, req.Data)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	WriteSuccess(w, message, http.StatusCreated)
}
