package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronAgent/internal/agent"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/metrics"
)

type Handlers struct {
	queries *db.Queries
	runtime *agent.Runtime
}

func NewHandlers(queries *db.Queries, runtime *agent.Runtime) *Handlers {
	return &Handlers{
		queries: queries,
		runtime: runtime,
	}
}

// Agents

func (h *Handlers) CreateAgent(w http.ResponseWriter, r *http.Request) {
	requestID := GetRequestID(r.Context())
	endpoint := r.URL.Path
	method := r.Method
	
	var req CreateAgentRequest
	bodyBytes, _ := io.ReadAll(r.Body)
	bodySize := len(bodyBytes)
	r.Body = io.NopCloser(io.Reader(bytes.NewReader(bodyBytes)))
	
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "agent creation failed: request body parsing error", err, requestID, endpoint, method, "agent", "", map[string]interface{}{
			"body_size": bodySize,
		}))
		return
	}

	// Validate request
	if !ValidateAndRespond(w, func() error { return ValidateCreateAgentRequest(&req) }) {
		return
	}

	agent := &db.Agent{
		Name:         req.Name,
		Description:  req.Description,
		SystemPrompt: req.SystemPrompt,
		ModelName:    req.ModelName,
		MemoryTable:  req.MemoryTable,
		EnabledTools: req.EnabledTools,
		Config:       db.FromMap(req.Config),
	}

	if err := h.queries.CreateAgent(r.Context(), agent); err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "agent creation failed", err, requestID, endpoint, method, "agent", "", map[string]interface{}{
			"agent_name":      req.Name,
			"model_name":      req.ModelName,
			"enabled_tools":   req.EnabledTools,
			"system_prompt_length": len(req.SystemPrompt),
		}))
		return
	}

	respondJSON(w, http.StatusCreated, toAgentResponse(agent))
}

func (h *Handlers) GetAgent(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id, err := uuid.Parse(vars["id"])
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, WrapError(ErrBadRequest, requestID))
		return
	}

	agent, err := h.queries.GetAgentByID(r.Context(), id)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, WrapError(ErrNotFound, requestID))
		return
	}

	respondJSON(w, http.StatusOK, toAgentResponse(agent))
}

func (h *Handlers) ListAgents(w http.ResponseWriter, r *http.Request) {
	agents, err := h.queries.ListAgents(r.Context())
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, WrapError(NewError(http.StatusInternalServerError, "failed to list agents", err), requestID))
		return
	}

	responses := make([]AgentResponse, len(agents))
	for i, a := range agents {
		responses[i] = toAgentResponse(&a)
	}

	respondJSON(w, http.StatusOK, responses)
}

func (h *Handlers) UpdateAgent(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id, err := uuid.Parse(vars["id"])
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, WrapError(ErrBadRequest, requestID))
		return
	}

	var req CreateAgentRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, WrapError(ErrBadRequest, requestID))
		return
	}

	// Validate request
	if !ValidateAndRespond(w, func() error { return ValidateCreateAgentRequest(&req) }) {
		return
	}

	agent, err := h.queries.GetAgentByID(r.Context(), id)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, WrapError(ErrNotFound, requestID))
		return
	}

	// Update fields
	agent.Name = req.Name
	agent.Description = req.Description
	agent.SystemPrompt = req.SystemPrompt
	agent.ModelName = req.ModelName
	agent.MemoryTable = req.MemoryTable
	agent.EnabledTools = req.EnabledTools
	agent.Config = db.FromMap(req.Config)

	if err := h.queries.UpdateAgent(r.Context(), agent); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, WrapError(NewError(http.StatusInternalServerError, "failed to update agent", err), requestID))
		return
	}

	respondJSON(w, http.StatusOK, toAgentResponse(agent))
}

func (h *Handlers) DeleteAgent(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id, err := uuid.Parse(vars["id"])
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, WrapError(ErrBadRequest, requestID))
		return
	}

	if err := h.queries.DeleteAgent(r.Context(), id); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, WrapError(ErrNotFound, requestID))
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

// Sessions

func (h *Handlers) CreateSession(w http.ResponseWriter, r *http.Request) {
	var req CreateSessionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, WrapError(ErrBadRequest, requestID))
		return
	}

	// Validate request
	if !ValidateAndRespond(w, func() error { return ValidateCreateSessionRequest(&req) }) {
		return
	}

	metadata := db.FromMap(req.Metadata)
	if req.Metadata == nil {
		metadata = make(db.JSONBMap)
	}
	session := &db.Session{
		AgentID:       req.AgentID,
		ExternalUserID: req.ExternalUserID,
		Metadata:      metadata,
	}

	if err := h.queries.CreateSession(r.Context(), session); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, WrapError(NewError(http.StatusInternalServerError, "failed to create session", err), requestID))
		return
	}

	respondJSON(w, http.StatusCreated, toSessionResponse(session))
}

func (h *Handlers) GetSession(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id, err := uuid.Parse(vars["id"])
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, WrapError(ErrBadRequest, requestID))
		return
	}

	session, err := h.queries.GetSession(r.Context(), id)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, WrapError(ErrNotFound, requestID))
		return
	}

	respondJSON(w, http.StatusOK, toSessionResponse(session))
}

func (h *Handlers) ListSessions(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	agentID, err := uuid.Parse(vars["agent_id"])
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, WrapError(ErrBadRequest, requestID))
		return
	}

	limit := 50
	offset := 0
	// Parse query parameters for pagination
	if l := r.URL.Query().Get("limit"); l != "" {
		fmt.Sscanf(l, "%d", &limit)
	}
	if o := r.URL.Query().Get("offset"); o != "" {
		fmt.Sscanf(o, "%d", &offset)
	}

	sessions, err := h.queries.ListSessions(r.Context(), agentID, limit, offset)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, WrapError(NewError(http.StatusInternalServerError, "failed to list sessions", err), requestID))
		return
	}

	responses := make([]SessionResponse, len(sessions))
	for i, s := range sessions {
		responses[i] = toSessionResponse(&s)
	}

	respondJSON(w, http.StatusOK, responses)
}

// Messages

func (h *Handlers) SendMessage(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	vars := mux.Vars(r)
	sessionID, err := uuid.Parse(vars["session_id"])
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, WrapError(ErrBadRequest, requestID))
		return
	}

	var req SendMessageRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, WrapError(ErrBadRequest, requestID))
		return
	}

	// Validate request
	if !ValidateAndRespond(w, func() error { return ValidateSendMessageRequest(&req) }) {
		return
	}

	// Check if streaming is requested
	if req.Stream {
		StreamResponse(w, r, h.runtime, sessionID.String(), req.Content)
		return
	}

	state, err := h.runtime.Execute(r.Context(), sessionID, req.Content)
	if err != nil {
		metrics.RecordAgentExecution(state.AgentID.String(), "error", time.Since(start))
		requestID := GetRequestID(r.Context())
		respondError(w, WrapError(NewError(http.StatusInternalServerError, "failed to process message", err), requestID))
		return
	}

	// Record metrics
	duration := time.Since(start)
	metrics.RecordAgentExecution(state.AgentID.String(), "success", duration)

	response := map[string]interface{}{
		"session_id":   state.SessionID,
		"agent_id":     state.AgentID,
		"response":     state.FinalAnswer,
		"tokens_used":  state.TokensUsed,
		"tool_calls":   state.ToolCalls,
		"tool_results": state.ToolResults,
	}

	respondJSON(w, http.StatusOK, response)
}

func (h *Handlers) GetMessages(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	sessionID, err := uuid.Parse(vars["session_id"])
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, WrapError(ErrBadRequest, requestID))
		return
	}

	limit := 100
	offset := 0
	// Parse query parameters
	if l := r.URL.Query().Get("limit"); l != "" {
		_, _ = fmt.Sscanf(l, "%d", &limit)
	}
	if o := r.URL.Query().Get("offset"); o != "" {
		_, _ = fmt.Sscanf(o, "%d", &offset)
	}

	messages, err := h.queries.GetMessages(r.Context(), sessionID, limit, offset)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, WrapError(NewError(http.StatusInternalServerError, "failed to get messages", err), requestID))
		return
	}

	responses := make([]MessageResponse, len(messages))
	for i, m := range messages {
		responses[i] = toMessageResponse(&m)
	}

	respondJSON(w, http.StatusOK, responses)
}

// Helper functions

func toAgentResponse(a *db.Agent) AgentResponse {
	return AgentResponse{
		ID:           a.ID,
		Name:         a.Name,
		Description:  a.Description,
		SystemPrompt: a.SystemPrompt,
		ModelName:    a.ModelName,
		MemoryTable:  a.MemoryTable,
		EnabledTools: a.EnabledTools,
		Config:       a.Config.ToMap(),
		CreatedAt:    a.CreatedAt,
		UpdatedAt:    a.UpdatedAt,
	}
}

func toSessionResponse(s *db.Session) SessionResponse {
	return SessionResponse{
		ID:             s.ID,
		AgentID:        s.AgentID,
		ExternalUserID: s.ExternalUserID,
		Metadata:       s.Metadata.ToMap(),
		CreatedAt:      s.CreatedAt,
		LastActivityAt: s.LastActivityAt,
	}
}

func toMessageResponse(m *db.Message) MessageResponse {
	metadata := make(map[string]interface{})
	if m.Metadata != nil {
		metadata = m.Metadata
	}
	return MessageResponse{
		ID:         m.ID,
		SessionID:  m.SessionID,
		Role:       m.Role,
		Content:    m.Content,
		ToolName:   m.ToolName,
		ToolCallID: m.ToolCallID,
		TokenCount: m.TokenCount,
		Metadata:   metadata,
		CreatedAt:  m.CreatedAt,
	}
}

func respondJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

func respondError(w http.ResponseWriter, err *APIError) {
	response := ErrorResponse{
		Error: err.Message,
		Code:  err.Code,
	}
	if err.Err != nil {
		response.Message = err.Err.Error()
	}
	if err.RequestID != "" {
		w.Header().Set("X-Request-ID", err.RequestID)
	}
	respondJSON(w, err.Code, response)
}
