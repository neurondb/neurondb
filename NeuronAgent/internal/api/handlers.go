/*-------------------------------------------------------------------------
 *
 * handlers.go
 *    API handlers for NeuronAgent
 *
 * Provides HTTP handlers for agents, sessions, messages, and other API endpoints.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/api/handlers.go
 *
 *-------------------------------------------------------------------------
 */

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
	"github.com/neurondb/NeuronAgent/internal/validation"
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

/* Agents */

func (h *Handlers) CreateAgent(w http.ResponseWriter, r *http.Request) {
	requestID := GetRequestID(r.Context())
	endpoint := r.URL.Path
	method := r.Method

	/* Validate request body size (max 1MB) */
	const maxBodySize = 1024 * 1024
	bodyBytes, err := validation.ReadAndValidateBody(r, maxBodySize)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body validation failed", err, requestID, endpoint, method, "agent", "", nil))
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var req CreateAgentRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "agent creation failed: request body parsing error", err, requestID, endpoint, method, "agent", "", map[string]interface{}{
			"body_size": len(bodyBytes),
		}))
		return
	}

	/* Validate request */
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
			"agent_name":           req.Name,
			"model_name":           req.ModelName,
			"enabled_tools":        req.EnabledTools,
			"system_prompt_length": len(req.SystemPrompt),
		}))
		return
	}

	respondJSON(w, http.StatusCreated, toAgentResponse(agent))
}

func (h *Handlers) GetAgent(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	if err := validation.ValidateUUIDRequired(vars["id"], "agent_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID", err, requestID, r.URL.Path, r.Method, "agent", "", nil))
		return
	}

	id, err := uuid.Parse(vars["id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID format", err, requestID, r.URL.Path, r.Method, "agent", "", nil))
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
	var agents []db.Agent
	var err error

	search := r.URL.Query().Get("search")
	if search != "" {
		agents, err = h.queries.ListAgentsWithFilter(r.Context(), &search)
	} else {
		agents, err = h.queries.ListAgents(r.Context())
	}

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
	requestID := GetRequestID(r.Context())

	if err := validation.ValidateUUIDRequired(vars["id"], "agent_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID", err, requestID, r.URL.Path, r.Method, "agent", "", nil))
		return
	}

	id, err := uuid.Parse(vars["id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID format", err, requestID, r.URL.Path, r.Method, "agent", "", nil))
		return
	}

	/* Validate request body size (max 1MB) */
	const maxBodySize = 1024 * 1024
	bodyBytes, err := validation.ReadAndValidateBody(r, maxBodySize)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body validation failed", err, requestID, r.URL.Path, r.Method, "agent", "", nil))
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var req CreateAgentRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body parsing error", err, requestID, r.URL.Path, r.Method, "agent", "", nil))
		return
	}

	/* Validate request */
	if !ValidateAndRespond(w, func() error { return ValidateCreateAgentRequest(&req) }) {
		return
	}

	agent, err := h.queries.GetAgentByID(r.Context(), id)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, WrapError(ErrNotFound, requestID))
		return
	}

	/* Update fields */
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
	requestID := GetRequestID(r.Context())

	if err := validation.ValidateUUIDRequired(vars["id"], "agent_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID", err, requestID, r.URL.Path, r.Method, "agent", "", nil))
		return
	}

	id, err := uuid.Parse(vars["id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID format", err, requestID, r.URL.Path, r.Method, "agent", "", nil))
		return
	}

	if err := h.queries.DeleteAgent(r.Context(), id); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, WrapError(ErrNotFound, requestID))
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

/* Sessions */

func (h *Handlers) CreateSession(w http.ResponseWriter, r *http.Request) {
	requestID := GetRequestID(r.Context())

	/* Validate request body size (max 1MB) */
	const maxBodySize = 1024 * 1024
	bodyBytes, err := validation.ReadAndValidateBody(r, maxBodySize)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body validation failed", err, requestID, r.URL.Path, r.Method, "session", "", nil))
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var req CreateSessionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body parsing error", err, requestID, r.URL.Path, r.Method, "session", "", nil))
		return
	}

	if err := validation.ValidateUUIDRequired(req.AgentID.String(), "agent_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID", err, requestID, r.URL.Path, r.Method, "session", "", nil))
		return
	}

	/* Validate request */
	if !ValidateAndRespond(w, func() error { return ValidateCreateSessionRequest(&req) }) {
		return
	}

	metadata := db.FromMap(req.Metadata)
	if req.Metadata == nil {
		metadata = make(db.JSONBMap)
	}
	session := &db.Session{
		AgentID:        req.AgentID,
		ExternalUserID: req.ExternalUserID,
		Metadata:       metadata,
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
	requestID := GetRequestID(r.Context())

	if err := validation.ValidateUUIDRequired(vars["id"], "session_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid session ID", err, requestID, r.URL.Path, r.Method, "session", "", nil))
		return
	}

	id, err := uuid.Parse(vars["id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid session ID format", err, requestID, r.URL.Path, r.Method, "session", "", nil))
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
	requestID := GetRequestID(r.Context())

	if err := validation.ValidateUUIDRequired(vars["agent_id"], "agent_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID", err, requestID, r.URL.Path, r.Method, "session", "", nil))
		return
	}

	agentID, err := uuid.Parse(vars["agent_id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID format", err, requestID, r.URL.Path, r.Method, "session", "", nil))
		return
	}

	limit := 50
	offset := 0
	/* Parse query parameters for pagination */
	if l := r.URL.Query().Get("limit"); l != "" {
		if _, err := fmt.Sscanf(l, "%d", &limit); err != nil {
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid limit parameter", err, requestID, r.URL.Path, r.Method, "session", "", nil))
			return
		}
		if err := validation.ValidateIntRange(limit, 1, 1000, "limit"); err != nil {
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid limit value", err, requestID, r.URL.Path, r.Method, "session", "", nil))
			return
		}
	}
	if o := r.URL.Query().Get("offset"); o != "" {
		if _, err := fmt.Sscanf(o, "%d", &offset); err != nil {
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid offset parameter", err, requestID, r.URL.Path, r.Method, "session", "", nil))
			return
		}
		if err := validation.ValidateNonNegative(offset, "offset"); err != nil {
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid offset value", err, requestID, r.URL.Path, r.Method, "session", "", nil))
			return
		}
	}

	var sessions []db.Session
	externalUserID := r.URL.Query().Get("external_user_id")
	startDate := r.URL.Query().Get("start_date")
	endDate := r.URL.Query().Get("end_date")

	if externalUserID != "" || startDate != "" || endDate != "" {
		var externalUserIDPtr *string
		if externalUserID != "" {
			externalUserIDPtr = &externalUserID
		}
		var startDatePtr, endDatePtr *string
		if startDate != "" {
			startDatePtr = &startDate
		}
		if endDate != "" {
			endDatePtr = &endDate
		}
		sessions, err = h.queries.ListSessionsWithFilter(r.Context(), agentID, externalUserIDPtr, startDatePtr, endDatePtr, limit, offset)
	} else {
		sessions, err = h.queries.ListSessions(r.Context(), agentID, limit, offset)
	}

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

func (h *Handlers) UpdateSession(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	if err := validation.ValidateUUIDRequired(vars["id"], "session_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid session ID", err, requestID, r.URL.Path, r.Method, "session", "", nil))
		return
	}

	id, err := uuid.Parse(vars["id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid session ID format", err, requestID, r.URL.Path, r.Method, "session", "", nil))
		return
	}

	/* Get existing session */
	session, err := h.queries.GetSession(r.Context(), id)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusNotFound, "session not found", err, requestID, r.URL.Path, r.Method, "session", "", nil))
		return
	}

	/* Validate request body size (max 1MB) */
	const maxBodySize = 1024 * 1024
	bodyBytes, err := validation.ReadAndValidateBody(r, maxBodySize)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body validation failed", err, requestID, r.URL.Path, r.Method, "session", "", nil))
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var req struct {
		ExternalUserID *string                `json:"external_user_id"`
		Metadata       map[string]interface{} `json:"metadata"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body parsing error", err, requestID, r.URL.Path, r.Method, "session", "", nil))
		return
	}

	/* Update fields if provided */
	if req.ExternalUserID != nil {
		session.ExternalUserID = req.ExternalUserID
	}
	if req.Metadata != nil {
		session.Metadata = db.FromMap(req.Metadata)
	}

	if err := h.queries.UpdateSession(r.Context(), session); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, WrapError(NewError(http.StatusInternalServerError, "failed to update session", err), requestID))
		return
	}

	respondJSON(w, http.StatusOK, toSessionResponse(session))
}

func (h *Handlers) DeleteSession(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	if err := validation.ValidateUUIDRequired(vars["id"], "session_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid session ID", err, requestID, r.URL.Path, r.Method, "session", "", nil))
		return
	}

	id, err := uuid.Parse(vars["id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid session ID format", err, requestID, r.URL.Path, r.Method, "session", "", nil))
		return
	}

	if err := h.queries.DeleteSession(r.Context(), id); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, WrapError(ErrNotFound, requestID))
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

/* Messages */

func (h *Handlers) SendMessage(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	if err := validation.ValidateUUIDRequired(vars["session_id"], "session_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid session ID", err, requestID, r.URL.Path, r.Method, "message", "", nil))
		return
	}

	sessionID, err := uuid.Parse(vars["session_id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid session ID format", err, requestID, r.URL.Path, r.Method, "message", "", nil))
		return
	}

	/* Validate request body size (max 10MB for messages with potentially large content) */
	const maxBodySize = 10 * 1024 * 1024
	bodyBytes, err := validation.ReadAndValidateBody(r, maxBodySize)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body validation failed", err, requestID, r.URL.Path, r.Method, "message", "", nil))
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var req SendMessageRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body parsing error", err, requestID, r.URL.Path, r.Method, "message", "", nil))
		return
	}

	/* Validate request */
	if !ValidateAndRespond(w, func() error { return ValidateSendMessageRequest(&req) }) {
		return
	}

	/* Check if streaming is requested */
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

	/* Record metrics */
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
	/* Parse query parameters */
	if l := r.URL.Query().Get("limit"); l != "" {
		if _, err := fmt.Sscanf(l, "%d", &limit); err != nil {
			requestID := GetRequestID(r.Context())
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid limit parameter", err, requestID, r.URL.Path, r.Method, "message", "", nil))
			return
		}
	}
	if o := r.URL.Query().Get("offset"); o != "" {
		if _, err := fmt.Sscanf(o, "%d", &offset); err != nil {
			requestID := GetRequestID(r.Context())
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid offset parameter", err, requestID, r.URL.Path, r.Method, "message", "", nil))
			return
		}
	}

	/* Validate pagination */
	if err := ValidatePaginationParams(limit, offset); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "pagination validation failed", err, requestID, r.URL.Path, r.Method, "message", "", nil))
		return
	}

	/* Get query parameters */
	role := r.URL.Query().Get("role")
	contentSearch := r.URL.Query().Get("search")
	startDate := r.URL.Query().Get("start_date")
	endDate := r.URL.Query().Get("end_date")

	/* Validate date range if provided */
	if startDate != "" || endDate != "" {
		if err := ValidateDateRange(startDate, endDate); err != nil {
			requestID := GetRequestID(r.Context())
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "date range validation failed", err, requestID, r.URL.Path, r.Method, "message", "", nil))
			return
		}
	}

	/* Validate search query if provided */
	if contentSearch != "" {
		if err := ValidateSearchQuery(contentSearch, 1000); err != nil {
			requestID := GetRequestID(r.Context())
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "search query validation failed", err, requestID, r.URL.Path, r.Method, "message", "", nil))
			return
		}
	}

	var messages []db.Message

	if role != "" || contentSearch != "" || startDate != "" || endDate != "" {
		var rolePtr, contentSearchPtr *string
		if role != "" {
			rolePtr = &role
		}
		if contentSearch != "" {
			contentSearchPtr = &contentSearch
		}
		var startDatePtr, endDatePtr *string
		if startDate != "" {
			startDatePtr = &startDate
		}
		if endDate != "" {
			endDatePtr = &endDate
		}
		messages, err = h.queries.GetMessagesWithFilter(r.Context(), sessionID, rolePtr, contentSearchPtr, startDatePtr, endDatePtr, limit, offset)
	} else {
		messages, err = h.queries.GetMessages(r.Context(), sessionID, limit, offset)
	}

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

func (h *Handlers) GetMessage(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	var id int64
	if _, err := fmt.Sscanf(vars["id"], "%d", &id); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, WrapError(ErrBadRequest, requestID))
		return
	}

	message, err := h.queries.GetMessageByID(r.Context(), id)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, WrapError(ErrNotFound, requestID))
		return
	}

	respondJSON(w, http.StatusOK, toMessageResponse(message))
}

func (h *Handlers) UpdateMessage(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	var id int64
	if _, err := fmt.Sscanf(vars["id"], "%d", &id); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, WrapError(ErrBadRequest, requestID))
		return
	}

	/* Get existing message */
	message, err := h.queries.GetMessageByID(r.Context(), id)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, WrapError(ErrNotFound, requestID))
		return
	}

	var req struct {
		Content  *string                `json:"content"`
		Metadata map[string]interface{} `json:"metadata"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, WrapError(ErrBadRequest, requestID))
		return
	}

	/* Update fields if provided */
	if req.Content != nil {
		message.Content = *req.Content
	}
	if req.Metadata != nil {
		message.Metadata = req.Metadata
	}

	if err := h.queries.UpdateMessage(r.Context(), message); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, WrapError(NewError(http.StatusInternalServerError, "failed to update message", err), requestID))
		return
	}

	respondJSON(w, http.StatusOK, toMessageResponse(message))
}

func (h *Handlers) DeleteMessage(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	var id int64
	if _, err := fmt.Sscanf(vars["id"], "%d", &id); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, WrapError(ErrBadRequest, requestID))
		return
	}

	if err := h.queries.DeleteMessage(r.Context(), id); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, WrapError(ErrNotFound, requestID))
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

/* Helper functions */

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
