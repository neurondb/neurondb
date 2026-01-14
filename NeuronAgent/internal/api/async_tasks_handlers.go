/*-------------------------------------------------------------------------
 *
 * async_tasks_handlers.go
 *    API handlers for asynchronous task execution
 *
 * Provides HTTP handlers for creating, querying, and managing async tasks.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/api/async_tasks_handlers.go
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
	"github.com/neurondb/NeuronAgent/internal/agent"
	"github.com/neurondb/NeuronAgent/internal/db"
)

type AsyncTasksHandlers struct {
	queries       *db.Queries
	asyncExecutor *agent.AsyncTaskExecutor
}

func NewAsyncTasksHandlers(queries *db.Queries, asyncExecutor *agent.AsyncTaskExecutor) *AsyncTasksHandlers {
	return &AsyncTasksHandlers{
		queries:       queries,
		asyncExecutor: asyncExecutor,
	}
}

type CreateAsyncTaskRequest struct {
	SessionID uuid.UUID              `json:"session_id"`
	AgentID   uuid.UUID              `json:"agent_id"`
	TaskType  string                 `json:"task_type"`
	Input     map[string]interface{} `json:"input"`
	Priority  int                    `json:"priority,omitempty"`
}

/* CreateAsyncTask creates a new asynchronous task */
func (h *AsyncTasksHandlers) CreateAsyncTask(w http.ResponseWriter, r *http.Request) {
	requestID := GetRequestID(r.Context())

	var req CreateAsyncTaskRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, WrapError(NewError(http.StatusBadRequest, "async task creation failed: request body parsing error", err), requestID))
		return
	}

	if req.TaskType == "" {
		respondError(w, WrapError(NewError(http.StatusBadRequest, "task_type is required", nil), requestID))
		return
	}

	if req.Priority == 0 {
		req.Priority = 0
	}

	task, err := h.asyncExecutor.ExecuteAsync(r.Context(), req.SessionID, req.AgentID, req.TaskType, req.Input, req.Priority)
	if err != nil {
		respondError(w, WrapError(NewError(http.StatusInternalServerError, "async task creation failed", err), requestID))
		return
	}

	respondJSON(w, http.StatusCreated, task)
}

/* GetAsyncTaskStatus retrieves the status of an async task */
func (h *AsyncTasksHandlers) GetAsyncTaskStatus(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	taskID, err := uuid.Parse(vars["id"])
	if err != nil {
		respondError(w, WrapError(NewError(http.StatusBadRequest, "invalid task ID", err), requestID))
		return
	}

	task, err := h.asyncExecutor.GetTaskStatus(r.Context(), taskID)
	if err != nil {
		respondError(w, WrapError(NewError(http.StatusNotFound, "task not found", err), requestID))
		return
	}

	respondJSON(w, http.StatusOK, task)
}

/* CancelAsyncTask cancels a running or pending task */
func (h *AsyncTasksHandlers) CancelAsyncTask(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	taskID, err := uuid.Parse(vars["id"])
	if err != nil {
		respondError(w, WrapError(NewError(http.StatusBadRequest, "invalid task ID", err), requestID))
		return
	}

	err = h.asyncExecutor.CancelTask(r.Context(), taskID)
	if err != nil {
		respondError(w, WrapError(NewError(http.StatusNotFound, "task not found or cannot be cancelled", err), requestID))
		return
	}

	respondJSON(w, http.StatusOK, map[string]interface{}{
		"task_id": taskID.String(),
		"status":  "cancelled",
	})
}

/* ListAsyncTasks lists tasks with optional filters */
func (h *AsyncTasksHandlers) ListAsyncTasks(w http.ResponseWriter, r *http.Request) {
	requestID := GetRequestID(r.Context())

	/* Parse query parameters */
	var sessionID, agentID *uuid.UUID
	var status *string
	limit := 50
	offset := 0

	if sessionIDStr := r.URL.Query().Get("session_id"); sessionIDStr != "" {
		if id, err := uuid.Parse(sessionIDStr); err == nil {
			sessionID = &id
		}
	}

	if agentIDStr := r.URL.Query().Get("agent_id"); agentIDStr != "" {
		if id, err := uuid.Parse(agentIDStr); err == nil {
			agentID = &id
		}
	}

	if statusStr := r.URL.Query().Get("status"); statusStr != "" {
		status = &statusStr
	}

	/* Parse pagination */
	if limitStr := r.URL.Query().Get("limit"); limitStr != "" {
		if l, err := strconv.Atoi(limitStr); err == nil && l > 0 {
			limit = l
		}
	}

	if offsetStr := r.URL.Query().Get("offset"); offsetStr != "" {
		if o, err := strconv.Atoi(offsetStr); err == nil && o >= 0 {
			offset = o
		}
	}

	tasks, err := h.asyncExecutor.ListTasks(r.Context(), sessionID, agentID, status, limit, offset)
	if err != nil {
		respondError(w, WrapError(NewError(http.StatusInternalServerError, "task listing failed", err), requestID))
		return
	}

	respondJSON(w, http.StatusOK, tasks)
}
