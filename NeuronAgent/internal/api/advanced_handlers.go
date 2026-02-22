/*-------------------------------------------------------------------------
 *
 * advanced_handlers.go
 *    Advanced API handlers for NeuronAgent
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/api/advanced_handlers.go
 *
 *-------------------------------------------------------------------------
 */

package api

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronAgent/internal/agent"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/validation"
)

/* CloneAgent clones an agent */
func (h *Handlers) CloneAgent(w http.ResponseWriter, r *http.Request) {
	requestID := GetRequestID(r.Context())
	if _, err := MustGetPrincipalFromContext(r.Context()); err != nil {
		respondError(w, NewErrorWithContext(http.StatusUnauthorized, "authorization required", err, requestID, r.URL.Path, r.Method, "agent", "", nil))
		return
	}
	vars := mux.Vars(r)
	if err := validation.ValidateUUIDRequired(vars["id"], "agent_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID", err, requestID, r.URL.Path, r.Method, "agent", "", nil))
		return
	}

	id, err := uuid.Parse(vars["id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID format", err, requestID, r.URL.Path, r.Method, "agent", "", nil))
		return
	}

	/* Get original agent */
	originalAgent, err := h.queries.GetAgentByID(r.Context(), id)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusNotFound, "agent not found", err, requestID, r.URL.Path, r.Method, "agent", id.String(), nil))
		return
	}

	/* Create cloned agent */
	clonedAgent := &db.Agent{
		Name:         originalAgent.Name + "_clone",
		Description:  originalAgent.Description,
		SystemPrompt: originalAgent.SystemPrompt,
		ModelName:    originalAgent.ModelName,
		MemoryTable:  originalAgent.MemoryTable,
		EnabledTools: originalAgent.EnabledTools,
		Config:       originalAgent.Config,
	}

	if err := h.queries.CreateAgent(r.Context(), clonedAgent); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "agent cloning failed", err, requestID, r.URL.Path, r.Method, "agent", id.String(), nil))
		return
	}

	respondJSON(w, http.StatusCreated, toAgentResponse(clonedAgent))
}

/* GeneratePlan generates a plan for an agent */
func (h *Handlers) GeneratePlan(w http.ResponseWriter, r *http.Request) {
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

	/* Get agent */
	agent, err := h.queries.GetAgentByID(r.Context(), id)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusNotFound, "agent not found", err, requestID, r.URL.Path, r.Method, "agent", id.String(), nil))
		return
	}

	/* Validate request body size (max 1MB) */
	const maxBodySize = 1024 * 1024
	bodyBytes, err := validation.ReadAndValidateBody(r, maxBodySize)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body validation failed", err, requestID, r.URL.Path, r.Method, "agent", id.String(), nil))
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var req struct {
		Task string `json:"task"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid request body", err, requestID, r.URL.Path, r.Method, "agent", id.String(), nil))
		return
	}

	if err := validation.ValidateRequired(req.Task, "task"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "task validation failed", err, requestID, r.URL.Path, r.Method, "agent", id.String(), nil))
		return
	}
	if err := validation.ValidateMaxLength(req.Task, "task", 10000); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "task too long", err, requestID, r.URL.Path, r.Method, "agent", id.String(), nil))
		return
	}

	/* Generate plan */
	planner := h.runtime.GetPlanner()
	if planner == nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "planner not available", nil, requestID, r.URL.Path, r.Method, "agent", id.String(), nil))
		return
	}
	steps, err := planner.Plan(r.Context(), req.Task, agent.EnabledTools)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "plan generation failed", err, requestID, r.URL.Path, r.Method, "agent", id.String(), nil))
		return
	}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"plan":  steps,
		"steps": len(steps),
	})
}

/* ReflectOnResponse reflects on an agent response */
func (h *Handlers) ReflectOnResponse(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id, err := uuid.Parse(vars["id"])
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, WrapError(ErrBadRequest, requestID))
		return
	}

	var req struct {
		UserMessage string             `json:"user_message"`
		Response    string             `json:"response"`
		ToolCalls   []agent.ToolCall   `json:"tool_calls"`
		ToolResults []agent.ToolResult `json:"tool_results"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid request body", err, requestID, r.URL.Path, r.Method, "agent", id.String(), nil))
		return
	}

	/* Reflect */
	reflector := h.runtime.GetReflector()
	if reflector == nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "reflector not available", nil, requestID, r.URL.Path, r.Method, "agent", id.String(), nil))
		return
	}
	reflection, err := reflector.Reflect(r.Context(), req.UserMessage, req.Response, req.ToolCalls, req.ToolResults)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "reflection failed", err, requestID, r.URL.Path, r.Method, "agent", id.String(), nil))
		return
	}

	respondJSON(w, http.StatusOK, reflection)
}

/* DelegateToAgent delegates a task to another agent */
func (h *Handlers) DelegateToAgent(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id, err := uuid.Parse(vars["id"])
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, WrapError(ErrBadRequest, requestID))
		return
	}

	var req struct {
		ToAgentID uuid.UUID `json:"to_agent_id"`
		Task      string    `json:"task"`
		SessionID uuid.UUID `json:"session_id"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid request body", err, requestID, r.URL.Path, r.Method, "agent", id.String(), nil))
		return
	}

	/* Delegate */
	collab := agent.NewCollaborationManager(h.queries, h.runtime)
	result, err := collab.DelegateTask(r.Context(), id, req.ToAgentID, req.Task, req.SessionID)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "delegation failed", err, requestID, r.URL.Path, r.Method, "agent", id.String(), nil))
		return
	}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"result": result,
	})
}

/* GetAgentMetrics gets agent performance metrics */
func (h *Handlers) GetAgentMetrics(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id, err := uuid.Parse(vars["id"])
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, WrapError(ErrBadRequest, requestID))
		return
	}

	/* Get metrics from database */
	query := `SELECT 
		COALESCE(COUNT(DISTINCT s.id), 0) AS session_count,
		COALESCE(COUNT(m.id), 0) AS message_count,
		COALESCE(AVG(m.token_count), 0) AS avg_tokens_per_message
		FROM neurondb_agent.agents a
		LEFT JOIN neurondb_agent.sessions s ON s.agent_id = a.id
		LEFT JOIN neurondb_agent.messages m ON m.session_id = s.id
		WHERE a.id = $1
		GROUP BY a.id`

	var metrics struct {
		SessionCount        int      `db:"session_count"`
		MessageCount        int      `db:"message_count"`
		AvgTokensPerMessage *float64 `db:"avg_tokens_per_message"`
	}

	err = h.queries.GetDB().GetContext(r.Context(), &metrics, query, id)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "metrics retrieval failed", err, requestID, r.URL.Path, r.Method, "agent", id.String(), nil))
		return
	}

	respondJSON(w, http.StatusOK, metrics)
}

/* GetAgentCosts gets cost analytics for an agent */
func (h *Handlers) GetAgentCosts(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id, err := uuid.Parse(vars["id"])
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, WrapError(ErrBadRequest, requestID))
		return
	}

	/* Parse date range from query params */
	startDate := time.Now().AddDate(0, 0, -30) /* Default: last 30 days */
	endDate := time.Now()

	if startStr := r.URL.Query().Get("start_date"); startStr != "" {
		if parsed, err := time.Parse(time.RFC3339, startStr); err == nil {
			startDate = parsed
		}
	}

	if endStr := r.URL.Query().Get("end_date"); endStr != "" {
		if parsed, err := time.Parse(time.RFC3339, endStr); err == nil {
			endDate = parsed
		}
	}

	/* Get cost summary */
	costTracker := agent.NewCostTracker(h.queries)
	summary, err := costTracker.GetCostSummary(r.Context(), id, startDate, endDate)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "cost summary retrieval failed", err, requestID, r.URL.Path, r.Method, "agent", id.String(), nil))
		return
	}

	respondJSON(w, http.StatusOK, summary)
}

/* CreateTool creates a custom tool */
func (h *Handlers) CreateTool(w http.ResponseWriter, r *http.Request) {
	if _, err := MustGetPrincipalFromContext(r.Context()); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusUnauthorized, "authorization required", err, requestID, r.URL.Path, r.Method, "tool", "", nil))
		return
	}
	var req struct {
		Name          string                 `json:"name"`
		Description   string                 `json:"description"`
		ArgSchema     map[string]interface{} `json:"arg_schema"`
		HandlerType   string                 `json:"handler_type"`
		HandlerConfig map[string]interface{} `json:"handler_config"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid request body", err, requestID, r.URL.Path, r.Method, "tool", "", nil))
		return
	}

	/* Validate required fields */
	if req.Name == "" {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "tool name is required", nil, requestID, r.URL.Path, r.Method, "tool", "", nil))
		return
	}
	if req.Description == "" {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "tool description is required", nil, requestID, r.URL.Path, r.Method, "tool", req.Name, nil))
		return
	}
	if req.HandlerType == "" {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "tool handler_type is required", nil, requestID, r.URL.Path, r.Method, "tool", req.Name, nil))
		return
	}
	if req.ArgSchema == nil {
		req.ArgSchema = make(map[string]interface{})
	}
	if req.HandlerConfig == nil {
		req.HandlerConfig = make(map[string]interface{})
	}

	tool := &db.Tool{
		Name:          req.Name,
		Description:   req.Description,
		ArgSchema:     db.FromMap(req.ArgSchema),
		HandlerType:   req.HandlerType,
		HandlerConfig: db.FromMap(req.HandlerConfig),
		Enabled:       true,
	}

	if err := h.queries.CreateTool(r.Context(), tool); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "tool creation failed", err, requestID, r.URL.Path, r.Method, "tool", req.Name, nil))
		return
	}

	respondJSON(w, http.StatusCreated, tool)
}

/* ListTools lists all tools */
func (h *Handlers) ListTools(w http.ResponseWriter, r *http.Request) {
	if _, err := MustGetPrincipalFromContext(r.Context()); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusUnauthorized, "authorization required", err, requestID, r.URL.Path, r.Method, "tool", "", nil))
		return
	}
	var tools []db.Tool
	var err error

	enabledStr := r.URL.Query().Get("enabled")
	search := r.URL.Query().Get("search")
	handlerType := r.URL.Query().Get("handler_type")

	if enabledStr != "" || search != "" || handlerType != "" {
		var enabled *bool
		if enabledStr != "" {
			enabledVal := enabledStr == "true"
			enabled = &enabledVal
		}
		var searchPtr, handlerTypePtr *string
		if search != "" {
			searchPtr = &search
		}
		if handlerType != "" {
			handlerTypePtr = &handlerType
		}
		tools, err = h.queries.ListToolsWithFilter(r.Context(), enabled, searchPtr, handlerTypePtr)
	} else {
		tools, err = h.queries.ListTools(r.Context())
	}

	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to list tools", err, requestID, r.URL.Path, r.Method, "tool", "", nil))
		return
	}

	respondJSON(w, http.StatusOK, tools)
}

/* GetTool gets a tool by name */
func (h *Handlers) GetTool(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	toolName := vars["name"]

	tool, err := h.queries.GetTool(r.Context(), toolName)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusNotFound, "tool not found", err, requestID, r.URL.Path, r.Method, "tool", toolName, nil))
		return
	}

	respondJSON(w, http.StatusOK, tool)
}

/* UpdateTool updates a tool */
func (h *Handlers) UpdateTool(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	toolName := vars["name"]

	/* Get existing tool */
	existingTool, err := h.queries.GetTool(r.Context(), toolName)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusNotFound, "tool not found", err, requestID, r.URL.Path, r.Method, "tool", toolName, nil))
		return
	}

	var req struct {
		Description   *string                `json:"description"`
		ArgSchema     map[string]interface{} `json:"arg_schema"`
		HandlerType   *string                `json:"handler_type"`
		HandlerConfig map[string]interface{} `json:"handler_config"`
		Enabled       *bool                  `json:"enabled"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid request body", err, requestID, r.URL.Path, r.Method, "tool", toolName, nil))
		return
	}

	/* Update fields if provided */
	if req.Description != nil {
		existingTool.Description = *req.Description
	}
	if req.ArgSchema != nil {
		existingTool.ArgSchema = db.FromMap(req.ArgSchema)
	}
	if req.HandlerType != nil {
		existingTool.HandlerType = *req.HandlerType
	}
	if req.HandlerConfig != nil {
		existingTool.HandlerConfig = db.FromMap(req.HandlerConfig)
	}
	if req.Enabled != nil {
		existingTool.Enabled = *req.Enabled
	}

	if err := h.queries.UpdateTool(r.Context(), existingTool); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "tool update failed", err, requestID, r.URL.Path, r.Method, "tool", toolName, nil))
		return
	}

	respondJSON(w, http.StatusOK, existingTool)
}

/* DeleteTool deletes a tool */
func (h *Handlers) DeleteTool(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	toolName := vars["name"]

	if err := h.queries.DeleteTool(r.Context(), toolName); err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusNotFound, "tool not found", err, requestID, r.URL.Path, r.Method, "tool", toolName, nil))
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

/* GetToolAnalytics gets analytics for a tool */
func (h *Handlers) GetToolAnalytics(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	toolName := vars["name"]

	/* Parse date range */
	startDate := time.Now().AddDate(0, 0, -30)
	endDate := time.Now()

	if startStr := r.URL.Query().Get("start_date"); startStr != "" {
		if parsed, err := time.Parse(time.RFC3339, startStr); err == nil {
			startDate = parsed
		}
	}

	if endStr := r.URL.Query().Get("end_date"); endStr != "" {
		if parsed, err := time.Parse(time.RFC3339, endStr); err == nil {
			endDate = parsed
		}
	}

	/* Get stats */
	query := `SELECT 
		COUNT(*) AS total_calls,
		SUM(CASE WHEN success THEN 1 ELSE 0 END) AS success_calls,
		AVG(execution_time_ms) AS avg_execution_time_ms,
		SUM(tokens_used) AS total_tokens,
		SUM(cost) AS total_cost
		FROM neurondb_agent.tool_usage_logs
		WHERE tool_name = $1 AND created_at BETWEEN $2 AND $3`

	var stats struct {
		TotalCalls         int      `db:"total_calls"`
		SuccessCalls       int      `db:"success_calls"`
		AvgExecutionTimeMs *float64 `db:"avg_execution_time_ms"`
		TotalTokens        int      `db:"total_tokens"`
		TotalCost          float64  `db:"total_cost"`
	}

	err := h.queries.GetDB().GetContext(r.Context(), &stats, query, toolName, startDate, endDate)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "tool analytics retrieval failed", err, requestID, r.URL.Path, r.Method, "tool", toolName, nil))
		return
	}

	respondJSON(w, http.StatusOK, stats)
}

/* SummarizeMemory summarizes memory for an agent */
func (h *Handlers) SummarizeMemory(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id, err := uuid.Parse(vars["id"])
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, WrapError(ErrBadRequest, requestID))
		return
	}

	var req struct {
		MaxChunks int `json:"max_chunks"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil && err != io.EOF {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid request body", err, requestID, r.URL.Path, r.Method, "memory", id.String(), nil))
		return
	}

	if req.MaxChunks == 0 {
		req.MaxChunks = 10
	}

	/* Summarize */
	memory := h.runtime.GetMemoryManager()
	if memory == nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "memory manager not available", nil, requestID, r.URL.Path, r.Method, "memory", id.String(), nil))
		return
	}
	err = memory.SummarizeMemory(r.Context(), id, req.MaxChunks)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "memory summarization failed", err, requestID, r.URL.Path, r.Method, "memory", id.String(), nil))
		return
	}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"status":     "summarized",
		"max_chunks": req.MaxChunks,
	})
}

/* GetAnalyticsOverview gets system-wide analytics */
func (h *Handlers) GetAnalyticsOverview(w http.ResponseWriter, r *http.Request) {
	/* Get overview stats */
	query := `SELECT 
		(SELECT COUNT(*) FROM neurondb_agent.agents) AS total_agents,
		(SELECT COUNT(*) FROM neurondb_agent.sessions) AS total_sessions,
		(SELECT COUNT(*) FROM neurondb_agent.messages) AS total_messages,
		(SELECT SUM(cost) FROM neurondb_agent.cost_logs WHERE created_at > NOW() - INTERVAL '30 days') AS total_cost_30d`

	var overview struct {
		TotalAgents   int      `db:"total_agents"`
		TotalSessions int      `db:"total_sessions"`
		TotalMessages int      `db:"total_messages"`
		TotalCost30d  *float64 `db:"total_cost_30d"`
	}

	err := h.queries.GetDB().GetContext(r.Context(), &overview, query)
	if err != nil {
		requestID := GetRequestID(r.Context())
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "analytics overview retrieval failed", err, requestID, r.URL.Path, r.Method, "analytics", "", nil))
		return
	}

	respondJSON(w, http.StatusOK, overview)
}

/* GetRetrievalStats returns statistics about retrieval decisions */
func (h *Handlers) GetRetrievalStats(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	if err := validation.ValidateUUIDRequired(vars["id"], "agent_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID", err, requestID, r.URL.Path, r.Method, "retrieval", "", nil))
		return
	}

	agentID, err := uuid.Parse(vars["id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent id format", err, requestID, r.URL.Path, r.Method, "retrieval", "", nil))
		return
	}

	/* Verify agent exists */
	_, err = h.queries.GetAgentByID(r.Context(), agentID)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusNotFound, "agent not found", err, requestID, r.URL.Path, r.Method, "retrieval", agentID.String(), nil))
		return
	}

	/* Get days parameter */
	days := 30
	if d := r.URL.Query().Get("days"); d != "" {
		fmt.Sscanf(d, "%d", &days)
		if days < 1 {
			days = 1
		}
		if days > 365 {
			days = 365
		}
	}

	/* Get retrieval learning manager */
	retrievalLearning := h.runtime.GetRetrievalLearning()
	if retrievalLearning == nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "retrieval learning manager not available", nil, requestID, r.URL.Path, r.Method, "retrieval", agentID.String(), nil))
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()

	stats, err := retrievalLearning.GetRetrievalStats(ctx, agentID, days)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to get retrieval stats", err, requestID, r.URL.Path, r.Method, "retrieval", agentID.String(), nil))
		return
	}

	duration := time.Since(start)
	stats["duration_ms"] = duration.Milliseconds()
	stats["agent_id"] = agentID.String()
	stats["days"] = days

	respondJSON(w, http.StatusOK, stats)
}

/* ConsolidateMemory consolidates similar memories for an agent */
func (h *Handlers) ConsolidateMemory(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	if err := validation.ValidateUUIDRequired(vars["id"], "agent_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID", err, requestID, r.URL.Path, r.Method, "memory", "", nil))
		return
	}

	agentID, err := uuid.Parse(vars["id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent id format", err, requestID, r.URL.Path, r.Method, "memory", "", nil))
		return
	}

	/* Verify agent exists */
	_, err = h.queries.GetAgentByID(r.Context(), agentID)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusNotFound, "agent not found", err, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
		return
	}

	/* Parse request body */
	var req struct {
		Tier                string  `json:"tier"`                 /* stm, mtm, lpm */
		SimilarityThreshold float64 `json:"similarity_threshold"` /* 0.7-1.0, default 0.9 */
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "failed to read request body", err, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
		return
	}
	r.Body = io.NopCloser(bytes.NewBuffer(body))

	if len(body) > 0 {
		if err := json.Unmarshal(body, &req); err != nil {
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid request body", err, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
			return
		}
	}

	/* Validate tier */
	if req.Tier == "" {
		req.Tier = "mtm" /* Default to MTM */
	}
	if req.Tier != "stm" && req.Tier != "mtm" && req.Tier != "lpm" {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid tier: must be 'stm', 'mtm', or 'lpm'", nil, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
		return
	}

	/* Validate similarity threshold */
	if req.SimilarityThreshold == 0 {
		req.SimilarityThreshold = 0.9 /* Default */
	}
	if req.SimilarityThreshold < 0.7 || req.SimilarityThreshold > 1.0 {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "similarity_threshold must be between 0.7 and 1.0", nil, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
		return
	}

	/* Get memory adaptation manager */
	memoryAdaptation := h.runtime.GetMemoryAdaptation()
	if memoryAdaptation == nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "memory adaptation manager not available", nil, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Minute)
	defer cancel()

	consolidated, err := memoryAdaptation.ConsolidateSimilarMemories(ctx, agentID, req.Tier, req.SimilarityThreshold)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "memory consolidation failed", err, requestID, r.URL.Path, r.Method, "memory", agentID.String(), nil))
		return
	}

	duration := time.Since(start)
	respondJSON(w, http.StatusOK, map[string]interface{}{
		"agent_id":             agentID.String(),
		"tier":                 req.Tier,
		"similarity_threshold": req.SimilarityThreshold,
		"consolidated_count":   consolidated,
		"status":               "completed",
		"duration_ms":          duration.Milliseconds(),
		"completed_at":         time.Now().UTC(),
	})
}
