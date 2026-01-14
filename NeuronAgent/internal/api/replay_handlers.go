/*-------------------------------------------------------------------------
 *
 * replay_handlers.go
 *    Execution Snapshots and Replay API handlers for NeuronAgent
 *
 * Provides REST API endpoints for execution snapshot management and replay.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/api/replay_handlers.go
 *
 *-------------------------------------------------------------------------
 */

package api

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strconv"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronAgent/internal/agent"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/replay"
	"github.com/neurondb/NeuronAgent/internal/validation"
)

/* ReplayHandlers handles execution snapshot and replay API requests */
type ReplayHandlers struct {
	queries       *db.Queries
	replayManager *replay.ReplayManager
	runtime       *agent.Runtime
}

/* NewReplayHandlers creates new replay handlers */
func NewReplayHandlers(queries *db.Queries, replayManager *replay.ReplayManager, runtime *agent.Runtime) *ReplayHandlers {
	return &ReplayHandlers{
		queries:       queries,
		replayManager: replayManager,
		runtime:       runtime,
	}
}

/* CreateSnapshotRequest represents a request to create an execution snapshot */
type CreateSnapshotRequest struct {
	UserMessage       string `json:"user_message"`
	DeterministicMode bool   `json:"deterministic_mode,omitempty"`
}

/* SnapshotResponse represents an execution snapshot in API responses */
type SnapshotResponse struct {
	ID                string                 `json:"id"`
	SessionID         string                 `json:"session_id"`
	AgentID           string                 `json:"agent_id"`
	UserMessage       string                 `json:"user_message"`
	ExecutionState    map[string]interface{} `json:"execution_state"`
	DeterministicMode bool                   `json:"deterministic_mode"`
	CreatedAt         time.Time              `json:"created_at"`
}

/* ReplayResponse represents a replay execution response */
type ReplayResponse struct {
	SessionID   string                 `json:"session_id"`
	AgentID     string                 `json:"agent_id"`
	UserMessage string                 `json:"user_message"`
	FinalAnswer string                 `json:"final_answer,omitempty"`
	ToolCalls   []map[string]interface{} `json:"tool_calls,omitempty"`
	ToolResults []map[string]interface{} `json:"tool_results,omitempty"`
	TokensUsed  int                    `json:"tokens_used"`
}

/* CreateSnapshot creates a new execution snapshot */
func (h *ReplayHandlers) CreateSnapshot(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	/* Validate session ID */
	if err := validation.ValidateUUIDRequired(vars["session_id"], "session_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid session ID", err, requestID, r.URL.Path, r.Method, "snapshot", "", nil))
		return
	}

	sessionID, err := uuid.Parse(vars["session_id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid session ID format", err, requestID, r.URL.Path, r.Method, "snapshot", "", nil))
		return
	}

	/* Validate request body size */
	const maxBodySize = 1024 * 1024
	bodyBytes, err := validation.ReadAndValidateBody(r, maxBodySize)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body validation failed", err, requestID, r.URL.Path, r.Method, "snapshot", "", nil))
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var req CreateSnapshotRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body parsing error", err, requestID, r.URL.Path, r.Method, "snapshot", "", nil))
		return
	}

	/* Validate required fields */
	if req.UserMessage == "" {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "user_message is required", nil, requestID, r.URL.Path, r.Method, "snapshot", "", nil))
		return
	}

	/* Verify session exists */
	_, err = h.queries.GetSession(r.Context(), sessionID)
	if err != nil {
		status := http.StatusInternalServerError
		if err.Error() == "session not found: sql: no rows in result set" {
			status = http.StatusNotFound
		}
		respondError(w, NewErrorWithContext(status, "failed to get session", err, requestID, r.URL.Path, r.Method, "snapshot", "", nil))
		return
	}

	/* Execute the agent to get execution state */
	state, err := h.runtime.Execute(r.Context(), sessionID, req.UserMessage)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to execute agent", err, requestID, r.URL.Path, r.Method, "snapshot", "", nil))
		return
	}

	/* Store snapshot */
	if err := h.replayManager.StoreExecutionSnapshot(r.Context(), state, req.DeterministicMode); err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to store snapshot", err, requestID, r.URL.Path, r.Method, "snapshot", "", nil))
		return
	}

	/* Get the snapshot that was just created (we need to query it) */
	/* Since we don't have the ID, we'll list recent snapshots and get the first one */
	snapshots, err := h.queries.ListExecutionSnapshotsBySession(r.Context(), sessionID, 1, 0)
	if err != nil || len(snapshots) == 0 {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to retrieve created snapshot", err, requestID, r.URL.Path, r.Method, "snapshot", "", nil))
		return
	}

	respondJSON(w, http.StatusCreated, toSnapshotResponse(&snapshots[0]))
}

/* ListSnapshotsBySession lists execution snapshots for a session */
func (h *ReplayHandlers) ListSnapshotsBySession(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	/* Validate session ID */
	if err := validation.ValidateUUIDRequired(vars["session_id"], "session_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid session ID", err, requestID, r.URL.Path, r.Method, "snapshot", "", nil))
		return
	}

	sessionID, err := uuid.Parse(vars["session_id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid session ID format", err, requestID, r.URL.Path, r.Method, "snapshot", "", nil))
		return
	}

	/* Parse query parameters */
	limitStr := r.URL.Query().Get("limit")
	limit := 100
	if limitStr != "" {
		var err error
		limit, err = strconv.Atoi(limitStr)
		if err != nil || limit < 1 || limit > 1000 {
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid limit parameter", err, requestID, r.URL.Path, r.Method, "snapshot", "", nil))
			return
		}
	}

	offsetStr := r.URL.Query().Get("offset")
	offset := 0
	if offsetStr != "" {
		var err error
		offset, err = strconv.Atoi(offsetStr)
		if err != nil || offset < 0 {
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid offset parameter", err, requestID, r.URL.Path, r.Method, "snapshot", "", nil))
			return
		}
	}

	snapshots, err := h.queries.ListExecutionSnapshotsBySession(r.Context(), sessionID, limit, offset)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to list snapshots", err, requestID, r.URL.Path, r.Method, "snapshot", "", nil))
		return
	}

	responses := make([]SnapshotResponse, len(snapshots))
	for i, snapshot := range snapshots {
		responses[i] = toSnapshotResponse(&snapshot)
	}

	respondJSON(w, http.StatusOK, responses)
}

/* ListSnapshotsByAgent lists execution snapshots for an agent */
func (h *ReplayHandlers) ListSnapshotsByAgent(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	/* Validate agent ID */
	if err := validation.ValidateUUIDRequired(vars["agent_id"], "agent_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID", err, requestID, r.URL.Path, r.Method, "snapshot", "", nil))
		return
	}

	agentID, err := uuid.Parse(vars["agent_id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent ID format", err, requestID, r.URL.Path, r.Method, "snapshot", "", nil))
		return
	}

	/* Parse query parameters */
	limitStr := r.URL.Query().Get("limit")
	limit := 100
	if limitStr != "" {
		var err error
		limit, err = strconv.Atoi(limitStr)
		if err != nil || limit < 1 || limit > 1000 {
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid limit parameter", err, requestID, r.URL.Path, r.Method, "snapshot", "", nil))
			return
		}
	}

	offsetStr := r.URL.Query().Get("offset")
	offset := 0
	if offsetStr != "" {
		var err error
		offset, err = strconv.Atoi(offsetStr)
		if err != nil || offset < 0 {
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid offset parameter", err, requestID, r.URL.Path, r.Method, "snapshot", "", nil))
			return
		}
	}

	snapshots, err := h.queries.ListExecutionSnapshotsByAgent(r.Context(), agentID, limit, offset)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to list snapshots", err, requestID, r.URL.Path, r.Method, "snapshot", "", nil))
		return
	}

	responses := make([]SnapshotResponse, len(snapshots))
	for i, snapshot := range snapshots {
		responses[i] = toSnapshotResponse(&snapshot)
	}

	respondJSON(w, http.StatusOK, responses)
}

/* GetSnapshot gets an execution snapshot by ID */
func (h *ReplayHandlers) GetSnapshot(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	/* Validate snapshot ID */
	if err := validation.ValidateUUIDRequired(vars["id"], "id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid snapshot ID", err, requestID, r.URL.Path, r.Method, "snapshot", vars["id"], nil))
		return
	}

	snapshotID, err := uuid.Parse(vars["id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid snapshot ID format", err, requestID, r.URL.Path, r.Method, "snapshot", vars["id"], nil))
		return
	}

	snapshot, err := h.queries.GetExecutionSnapshotByID(r.Context(), snapshotID)
	if err != nil {
		status := http.StatusInternalServerError
		if err.Error() == "execution snapshot not found on" || err.Error() == "execution snapshot not found" {
			status = http.StatusNotFound
		}
		respondError(w, NewErrorWithContext(status, "failed to get snapshot", err, requestID, r.URL.Path, r.Method, "snapshot", vars["id"], nil))
		return
	}

	respondJSON(w, http.StatusOK, toSnapshotResponse(snapshot))
}

/* ReplaySnapshot replays an execution from a snapshot */
func (h *ReplayHandlers) ReplaySnapshot(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	/* Validate snapshot ID */
	if err := validation.ValidateUUIDRequired(vars["id"], "id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid snapshot ID", err, requestID, r.URL.Path, r.Method, "replay", vars["id"], nil))
		return
	}

	snapshotID, err := uuid.Parse(vars["id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid snapshot ID format", err, requestID, r.URL.Path, r.Method, "replay", vars["id"], nil))
		return
	}

	/* Replay execution */
	state, err := h.replayManager.ReplayExecution(r.Context(), snapshotID)
	if err != nil {
		status := http.StatusInternalServerError
		if err.Error() == "failed to get execution snapshot" {
			status = http.StatusNotFound
		}
		respondError(w, NewErrorWithContext(status, "failed to replay execution", err, requestID, r.URL.Path, r.Method, "replay", vars["id"], nil))
		return
	}

	respondJSON(w, http.StatusOK, toReplayResponse(state))
}

/* DeleteSnapshot deletes an execution snapshot */
func (h *ReplayHandlers) DeleteSnapshot(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	/* Validate snapshot ID */
	if err := validation.ValidateUUIDRequired(vars["id"], "id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid snapshot ID", err, requestID, r.URL.Path, r.Method, "snapshot", vars["id"], nil))
		return
	}

	snapshotID, err := uuid.Parse(vars["id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid snapshot ID format", err, requestID, r.URL.Path, r.Method, "snapshot", vars["id"], nil))
		return
	}

	/* Verify snapshot exists */
	_, err = h.queries.GetExecutionSnapshotByID(r.Context(), snapshotID)
	if err != nil {
		status := http.StatusInternalServerError
		if err.Error() == "execution snapshot not found on" || err.Error() == "execution snapshot not found" {
			status = http.StatusNotFound
		}
		respondError(w, NewErrorWithContext(status, "failed to get snapshot", err, requestID, r.URL.Path, r.Method, "snapshot", vars["id"], nil))
		return
	}

	if err := h.queries.DeleteExecutionSnapshot(r.Context(), snapshotID); err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to delete snapshot", err, requestID, r.URL.Path, r.Method, "snapshot", vars["id"], nil))
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

/* Helper functions to convert DB models to API responses */

func toSnapshotResponse(snapshot *db.ExecutionSnapshot) SnapshotResponse {
	return SnapshotResponse{
		ID:                snapshot.ID.String(),
		SessionID:         snapshot.SessionID.String(),
		AgentID:           snapshot.AgentID.String(),
		UserMessage:       snapshot.UserMessage,
		ExecutionState:    snapshot.ExecutionState.ToMap(),
		DeterministicMode: snapshot.DeterministicMode,
		CreatedAt:         snapshot.CreatedAt,
	}
}

func toReplayResponse(state *agent.ExecutionState) ReplayResponse {
	toolCalls := make([]map[string]interface{}, len(state.ToolCalls))
	for i, call := range state.ToolCalls {
		toolCalls[i] = map[string]interface{}{
			"id":        call.ID,
			"name":      call.Name,
			"arguments": call.Arguments,
		}
	}

	toolResults := make([]map[string]interface{}, len(state.ToolResults))
	for i, result := range state.ToolResults {
		resultMap := map[string]interface{}{
			"tool_call_id": result.ToolCallID,
			"content":      result.Content,
		}
		if result.Error != nil {
			resultMap["error"] = result.Error.Error()
		}
		toolResults[i] = resultMap
	}

	return ReplayResponse{
		SessionID:   state.SessionID.String(),
		AgentID:     state.AgentID.String(),
		UserMessage: state.UserMessage,
		FinalAnswer: state.FinalAnswer,
		ToolCalls:   toolCalls,
		ToolResults: toolResults,
		TokensUsed:  state.TokensUsed,
	}
}

