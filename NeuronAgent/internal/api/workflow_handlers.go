/*-------------------------------------------------------------------------
 *
 * workflow_handlers.go
 *    API handlers for workflow engine
 *
 * Provides REST API endpoints for workflow management, execution, and monitoring.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/api/workflow_handlers.go
 *
 *-------------------------------------------------------------------------
 */

package api

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/metrics"
	"github.com/neurondb/NeuronAgent/internal/validation"
	"github.com/neurondb/NeuronAgent/internal/workflow"
)

type WorkflowHandlers struct {
	queries *db.Queries
	engine  *workflow.Engine
}

func NewWorkflowHandlers(queries *db.Queries, engine *workflow.Engine) *WorkflowHandlers {
	return &WorkflowHandlers{
		queries: queries,
		engine:  engine,
	}
}

/* CreateWorkflow creates a new workflow */
func (h *WorkflowHandlers) CreateWorkflow(w http.ResponseWriter, r *http.Request) {
	requestID := GetRequestID(r.Context())

	/* Validate request body size (max 1MB) */
	const maxBodySize = 1024 * 1024
	bodyBytes, err := validation.ReadAndValidateBody(r, maxBodySize)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body validation failed", err, requestID, r.URL.Path, r.Method, "workflow", "", nil))
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var req struct {
		Name        string                 `json:"name"`
		DAGDefinition map[string]interface{} `json:"dag_definition"`
		Status      string                 `json:"status,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body parsing error", err, requestID, r.URL.Path, r.Method, "workflow", "", nil))
		return
	}

	if req.Name == "" {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "workflow name is required", nil, requestID, r.URL.Path, r.Method, "workflow", "", nil))
		return
	}

	if req.DAGDefinition == nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "dag_definition is required", nil, requestID, r.URL.Path, r.Method, "workflow", "", nil))
		return
	}

	if req.Status == "" {
		req.Status = "active"
	}

	validStatuses := map[string]bool{"active": true, "paused": true, "archived": true}
	if !validStatuses[req.Status] {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid status, must be one of: active, paused, archived", nil, requestID, r.URL.Path, r.Method, "workflow", "", nil))
		return
	}

	workflow := &db.Workflow{
		Name:         req.Name,
		DAGDefinition: db.FromMap(req.DAGDefinition),
		Status:       req.Status,
	}

	if err := h.queries.CreateWorkflow(r.Context(), workflow); err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "workflow creation failed", err, requestID, r.URL.Path, r.Method, "workflow", "", nil))
		return
	}

	respondJSON(w, http.StatusCreated, workflow)
}

/* GetWorkflow gets a workflow by ID */
func (h *WorkflowHandlers) GetWorkflow(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	if err := validation.ValidateUUIDRequired(vars["id"], "workflow_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid workflow ID", err, requestID, r.URL.Path, r.Method, "workflow", "", nil))
		return
	}

	id, err := uuid.Parse(vars["id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid workflow ID format", err, requestID, r.URL.Path, r.Method, "workflow", "", nil))
		return
	}

	workflow, err := h.queries.GetWorkflowByID(r.Context(), id)
	if err != nil {
		respondError(w, WrapError(ErrNotFound, requestID))
		return
	}

	respondJSON(w, http.StatusOK, workflow)
}

/* ListWorkflows lists all workflows */
func (h *WorkflowHandlers) ListWorkflows(w http.ResponseWriter, r *http.Request) {
	requestID := GetRequestID(r.Context())

	status := r.URL.Query().Get("status")
	var workflows []db.Workflow
	var err error

	if status != "" {
		workflows, err = h.queries.ListWorkflowsByStatus(r.Context(), status)
	} else {
		workflows, err = h.queries.ListWorkflows(r.Context())
	}

	if err != nil {
		respondError(w, WrapError(NewError(http.StatusInternalServerError, "failed to list workflows", err), requestID))
		return
	}

	respondJSON(w, http.StatusOK, workflows)
}

/* UpdateWorkflow updates a workflow */
func (h *WorkflowHandlers) UpdateWorkflow(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	if err := validation.ValidateUUIDRequired(vars["id"], "workflow_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid workflow ID", err, requestID, r.URL.Path, r.Method, "workflow", "", nil))
		return
	}

	id, err := uuid.Parse(vars["id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid workflow ID format", err, requestID, r.URL.Path, r.Method, "workflow", "", nil))
		return
	}

	workflow, err := h.queries.GetWorkflowByID(r.Context(), id)
	if err != nil {
		respondError(w, WrapError(ErrNotFound, requestID))
		return
	}

	/* Validate request body size (max 1MB) */
	const maxBodySize = 1024 * 1024
	bodyBytes, err := validation.ReadAndValidateBody(r, maxBodySize)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body validation failed", err, requestID, r.URL.Path, r.Method, "workflow", "", nil))
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var req struct {
		Name         *string                `json:"name,omitempty"`
		DAGDefinition *map[string]interface{} `json:"dag_definition,omitempty"`
		Status       *string                `json:"status,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body parsing error", err, requestID, r.URL.Path, r.Method, "workflow", "", nil))
		return
	}

	if req.Name != nil {
		workflow.Name = *req.Name
	}
	if req.DAGDefinition != nil {
		workflow.DAGDefinition = db.FromMap(*req.DAGDefinition)
	}
	if req.Status != nil {
		validStatuses := map[string]bool{"active": true, "paused": true, "archived": true}
		if !validStatuses[*req.Status] {
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid status, must be one of: active, paused, archived", nil, requestID, r.URL.Path, r.Method, "workflow", "", nil))
			return
		}
		workflow.Status = *req.Status
	}

	if err := h.queries.UpdateWorkflow(r.Context(), workflow); err != nil {
		respondError(w, WrapError(NewError(http.StatusInternalServerError, "failed to update workflow", err), requestID))
		return
	}

	respondJSON(w, http.StatusOK, workflow)
}

/* DeleteWorkflow deletes a workflow */
func (h *WorkflowHandlers) DeleteWorkflow(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	if err := validation.ValidateUUIDRequired(vars["id"], "workflow_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid workflow ID", err, requestID, r.URL.Path, r.Method, "workflow", "", nil))
		return
	}

	id, err := uuid.Parse(vars["id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid workflow ID format", err, requestID, r.URL.Path, r.Method, "workflow", "", nil))
		return
	}

	if err := h.queries.DeleteWorkflow(r.Context(), id); err != nil {
		respondError(w, WrapError(ErrNotFound, requestID))
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

/* CreateWorkflowStep creates a new workflow step */
func (h *WorkflowHandlers) CreateWorkflowStep(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	if err := validation.ValidateUUIDRequired(vars["workflow_id"], "workflow_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid workflow ID", err, requestID, r.URL.Path, r.Method, "workflow_step", "", nil))
		return
	}

	workflowID, err := uuid.Parse(vars["workflow_id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid workflow ID format", err, requestID, r.URL.Path, r.Method, "workflow_step", "", nil))
		return
	}

	/* Validate request body size (max 1MB) */
	const maxBodySize = 1024 * 1024
	bodyBytes, err := validation.ReadAndValidateBody(r, maxBodySize)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body validation failed", err, requestID, r.URL.Path, r.Method, "workflow_step", "", nil))
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var req struct {
		StepName          string                 `json:"step_name"`
		StepType          string                 `json:"step_type"`
		Inputs            map[string]interface{} `json:"inputs,omitempty"`
		Outputs           map[string]interface{} `json:"outputs,omitempty"`
		Dependencies      []string               `json:"dependencies,omitempty"`
		RetryConfig       map[string]interface{} `json:"retry_config,omitempty"`
		IdempotencyKey    *string                `json:"idempotency_key,omitempty"`
		CompensationStepID *string               `json:"compensation_step_id,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body parsing error", err, requestID, r.URL.Path, r.Method, "workflow_step", "", nil))
		return
	}

	if req.StepName == "" {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "step_name is required", nil, requestID, r.URL.Path, r.Method, "workflow_step", "", nil))
		return
	}

	validStepTypes := map[string]bool{"agent": true, "tool": true, "approval": true, "http": true, "sql": true, "custom": true}
	if !validStepTypes[req.StepType] {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid step_type, must be one of: agent, tool, approval, http, sql, custom", nil, requestID, r.URL.Path, r.Method, "workflow_step", "", nil))
		return
	}

	step := &db.WorkflowStep{
		WorkflowID: workflowID,
		StepName:   req.StepName,
		StepType:   req.StepType,
	}

	if req.Inputs != nil {
		step.Inputs = db.FromMap(req.Inputs)
	}
	if req.Outputs != nil {
		step.Outputs = db.FromMap(req.Outputs)
	}
	if req.Dependencies != nil {
		step.Dependencies = req.Dependencies
	}
	if req.RetryConfig != nil {
		step.RetryConfig = db.FromMap(req.RetryConfig)
	}
	if req.IdempotencyKey != nil {
		step.IdempotencyKey = req.IdempotencyKey
	}
	if req.CompensationStepID != nil {
		compensationID, err := uuid.Parse(*req.CompensationStepID)
		if err != nil {
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid compensation_step_id format", err, requestID, r.URL.Path, r.Method, "workflow_step", "", nil))
			return
		}
		step.CompensationStepID = &compensationID
	}

	if err := h.queries.CreateWorkflowStep(r.Context(), step); err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "workflow step creation failed", err, requestID, r.URL.Path, r.Method, "workflow_step", "", nil))
		return
	}

	respondJSON(w, http.StatusCreated, step)
}

/* ListWorkflowSteps lists all steps for a workflow */
func (h *WorkflowHandlers) ListWorkflowSteps(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	if err := validation.ValidateUUIDRequired(vars["workflow_id"], "workflow_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid workflow ID", err, requestID, r.URL.Path, r.Method, "workflow_step", "", nil))
		return
	}

	workflowID, err := uuid.Parse(vars["workflow_id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid workflow ID format", err, requestID, r.URL.Path, r.Method, "workflow_step", "", nil))
		return
	}

	steps, err := h.queries.ListWorkflowSteps(r.Context(), workflowID)
	if err != nil {
		respondError(w, WrapError(NewError(http.StatusInternalServerError, "failed to list workflow steps", err), requestID))
		return
	}

	respondJSON(w, http.StatusOK, steps)
}

/* ExecuteWorkflow executes a workflow */
func (h *WorkflowHandlers) ExecuteWorkflow(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	if err := validation.ValidateUUIDRequired(vars["workflow_id"], "workflow_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid workflow ID", err, requestID, r.URL.Path, r.Method, "workflow_execution", "", nil))
		return
	}

	workflowID, err := uuid.Parse(vars["workflow_id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid workflow ID format", err, requestID, r.URL.Path, r.Method, "workflow_execution", "", nil))
		return
	}

	/* Validate request body size (max 1MB) */
	const maxBodySize = 1024 * 1024
	bodyBytes, err := validation.ReadAndValidateBody(r, maxBodySize)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body validation failed", err, requestID, r.URL.Path, r.Method, "workflow_execution", "", nil))
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var req struct {
		TriggerType string                 `json:"trigger_type,omitempty"`
		TriggerData map[string]interface{} `json:"trigger_data,omitempty"`
		Inputs      map[string]interface{} `json:"inputs,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body parsing error", err, requestID, r.URL.Path, r.Method, "workflow_execution", "", nil))
		return
	}

	if req.TriggerType == "" {
		req.TriggerType = "manual"
	}

	validTriggerTypes := map[string]bool{"manual": true, "schedule": true, "webhook": true, "db_notify": true, "queue": true}
	if !validTriggerTypes[req.TriggerType] {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid trigger_type, must be one of: manual, schedule, webhook, db_notify, queue", nil, requestID, r.URL.Path, r.Method, "workflow_execution", "", nil))
		return
	}

	if req.Inputs == nil {
		req.Inputs = make(map[string]interface{})
	}
	if req.TriggerData == nil {
		req.TriggerData = make(map[string]interface{})
	}

	/* Execute workflow asynchronously */
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
		defer cancel()

		_, execErr := h.engine.ExecuteWorkflow(ctx, workflowID, req.TriggerType, req.TriggerData, req.Inputs)
		if execErr != nil {
			metrics.ErrorWithContext(ctx, "Workflow execution failed", execErr, map[string]interface{}{
				"workflow_id": workflowID.String(),
				"request_id":  requestID,
			})
		}
	}()

	respondJSON(w, http.StatusAccepted, map[string]interface{}{
		"workflow_id": workflowID.String(),
		"status":      "accepted",
		"message":     "Workflow execution started",
	})
}

/* GetWorkflowExecution gets a workflow execution by ID */
func (h *WorkflowHandlers) GetWorkflowExecution(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	if err := validation.ValidateUUIDRequired(vars["execution_id"], "execution_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid execution ID", err, requestID, r.URL.Path, r.Method, "workflow_execution", "", nil))
		return
	}

	id, err := uuid.Parse(vars["execution_id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid execution ID format", err, requestID, r.URL.Path, r.Method, "workflow_execution", "", nil))
		return
	}

	execution, err := h.queries.GetWorkflowExecutionByID(r.Context(), id)
	if err != nil {
		respondError(w, WrapError(ErrNotFound, requestID))
		return
	}

	respondJSON(w, http.StatusOK, execution)
}

/* ListWorkflowExecutions lists executions for a workflow */
func (h *WorkflowHandlers) ListWorkflowExecutions(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	if err := validation.ValidateUUIDRequired(vars["workflow_id"], "workflow_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid workflow ID", err, requestID, r.URL.Path, r.Method, "workflow_execution", "", nil))
		return
	}

	workflowID, err := uuid.Parse(vars["workflow_id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid workflow ID format", err, requestID, r.URL.Path, r.Method, "workflow_execution", "", nil))
		return
	}

	status := r.URL.Query().Get("status")
	var executions []db.WorkflowExecution
	var listErr error

	if status != "" {
		executions, listErr = h.queries.ListWorkflowExecutionsByStatus(r.Context(), workflowID, status)
	} else {
		executions, listErr = h.queries.ListWorkflowExecutions(r.Context(), workflowID)
	}

	if listErr != nil {
		respondError(w, WrapError(NewError(http.StatusInternalServerError, "failed to list workflow executions", listErr), requestID))
		return
	}

	respondJSON(w, http.StatusOK, executions)
}

/* Workflow Schedule Handlers */

/* CreateWorkflowSchedule creates or updates a workflow schedule */
func (h *WorkflowHandlers) CreateWorkflowSchedule(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	/* Validate workflow ID */
	if err := validation.ValidateUUIDRequired(vars["workflow_id"], "workflow_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid workflow ID", err, requestID, r.URL.Path, r.Method, "workflow_schedule", "", nil))
		return
	}

	workflowID, err := uuid.Parse(vars["workflow_id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid workflow ID format", err, requestID, r.URL.Path, r.Method, "workflow_schedule", "", nil))
		return
	}

	/* Verify workflow exists */
	_, err = h.queries.GetWorkflowByID(r.Context(), workflowID)
	if err != nil {
		status := http.StatusInternalServerError
		if err.Error() == "workflow not found: sql: no rows in result set" {
			status = http.StatusNotFound
		}
		respondError(w, NewErrorWithContext(status, "workflow not found", err, requestID, r.URL.Path, r.Method, "workflow_schedule", "", nil))
		return
	}

	/* Validate request body size */
	const maxBodySize = 1024 * 1024
	bodyBytes, err := validation.ReadAndValidateBody(r, maxBodySize)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body validation failed", err, requestID, r.URL.Path, r.Method, "workflow_schedule", "", nil))
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var req struct {
		CronExpression string     `json:"cron_expression"`
		Timezone       string     `json:"timezone,omitempty"`
		Enabled        bool       `json:"enabled,omitempty"`
		NextRunAt      *time.Time `json:"next_run_at,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body parsing error", err, requestID, r.URL.Path, r.Method, "workflow_schedule", "", nil))
		return
	}

	/* Validate required fields */
	if req.CronExpression == "" {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "cron_expression is required", nil, requestID, r.URL.Path, r.Method, "workflow_schedule", "", nil))
		return
	}

	if req.Timezone == "" {
		req.Timezone = "UTC"
	}

	schedule := &db.WorkflowSchedule{
		WorkflowID:     workflowID,
		CronExpression: req.CronExpression,
		Timezone:       req.Timezone,
		Enabled:        req.Enabled,
		NextRunAt:      req.NextRunAt,
	}

	if err := h.queries.CreateWorkflowSchedule(r.Context(), schedule); err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "workflow schedule creation failed", err, requestID, r.URL.Path, r.Method, "workflow_schedule", "", nil))
		return
	}

	respondJSON(w, http.StatusCreated, schedule)
}

/* GetWorkflowSchedule gets a workflow schedule by workflow ID */
func (h *WorkflowHandlers) GetWorkflowSchedule(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	/* Validate workflow ID */
	if err := validation.ValidateUUIDRequired(vars["workflow_id"], "workflow_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid workflow ID", err, requestID, r.URL.Path, r.Method, "workflow_schedule", "", nil))
		return
	}

	workflowID, err := uuid.Parse(vars["workflow_id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid workflow ID format", err, requestID, r.URL.Path, r.Method, "workflow_schedule", "", nil))
		return
	}

	schedule, err := h.queries.GetWorkflowScheduleByWorkflowID(r.Context(), workflowID)
	if err != nil {
		status := http.StatusInternalServerError
		if err.Error() == "workflow schedule not found: sql: no rows in result set" {
			status = http.StatusNotFound
		}
		respondError(w, NewErrorWithContext(status, "failed to get workflow schedule", err, requestID, r.URL.Path, r.Method, "workflow_schedule", "", nil))
		return
	}

	respondJSON(w, http.StatusOK, schedule)
}

/* UpdateWorkflowSchedule updates a workflow schedule */
func (h *WorkflowHandlers) UpdateWorkflowSchedule(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	/* Validate workflow ID */
	if err := validation.ValidateUUIDRequired(vars["workflow_id"], "workflow_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid workflow ID", err, requestID, r.URL.Path, r.Method, "workflow_schedule", "", nil))
		return
	}

	workflowID, err := uuid.Parse(vars["workflow_id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid workflow ID format", err, requestID, r.URL.Path, r.Method, "workflow_schedule", "", nil))
		return
	}

	/* Get existing schedule */
	schedule, err := h.queries.GetWorkflowScheduleByWorkflowID(r.Context(), workflowID)
	if err != nil {
		status := http.StatusInternalServerError
		if err.Error() == "workflow schedule not found: sql: no rows in result set" {
			status = http.StatusNotFound
		}
		respondError(w, NewErrorWithContext(status, "failed to get workflow schedule", err, requestID, r.URL.Path, r.Method, "workflow_schedule", "", nil))
		return
	}

	/* Validate request body size */
	const maxBodySize = 1024 * 1024
	bodyBytes, err := validation.ReadAndValidateBody(r, maxBodySize)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body validation failed", err, requestID, r.URL.Path, r.Method, "workflow_schedule", "", nil))
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var req struct {
		CronExpression string     `json:"cron_expression,omitempty"`
		Timezone       string     `json:"timezone,omitempty"`
		Enabled        *bool      `json:"enabled,omitempty"`
		NextRunAt      *time.Time `json:"next_run_at,omitempty"`
		LastRunAt      *time.Time `json:"last_run_at,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body parsing error", err, requestID, r.URL.Path, r.Method, "workflow_schedule", "", nil))
		return
	}

	/* Update fields */
	if req.CronExpression != "" {
		schedule.CronExpression = req.CronExpression
	}
	if req.Timezone != "" {
		schedule.Timezone = req.Timezone
	}
	if req.Enabled != nil {
		schedule.Enabled = *req.Enabled
	}
	if req.NextRunAt != nil {
		schedule.NextRunAt = req.NextRunAt
	}
	if req.LastRunAt != nil {
		schedule.LastRunAt = req.LastRunAt
	}

	if err := h.queries.UpdateWorkflowSchedule(r.Context(), schedule); err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "workflow schedule update failed", err, requestID, r.URL.Path, r.Method, "workflow_schedule", "", nil))
		return
	}

	/* Get updated schedule */
	updatedSchedule, err := h.queries.GetWorkflowScheduleByWorkflowID(r.Context(), workflowID)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to get updated workflow schedule", err, requestID, r.URL.Path, r.Method, "workflow_schedule", "", nil))
		return
	}

	respondJSON(w, http.StatusOK, updatedSchedule)
}

/* DeleteWorkflowSchedule deletes a workflow schedule */
func (h *WorkflowHandlers) DeleteWorkflowSchedule(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	/* Validate workflow ID */
	if err := validation.ValidateUUIDRequired(vars["workflow_id"], "workflow_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid workflow ID", err, requestID, r.URL.Path, r.Method, "workflow_schedule", "", nil))
		return
	}

	workflowID, err := uuid.Parse(vars["workflow_id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid workflow ID format", err, requestID, r.URL.Path, r.Method, "workflow_schedule", "", nil))
		return
	}

	if err := h.queries.DeleteWorkflowScheduleByWorkflowID(r.Context(), workflowID); err != nil {
		status := http.StatusInternalServerError
		if err.Error() == "workflow schedule not found" {
			status = http.StatusNotFound
		}
		respondError(w, NewErrorWithContext(status, "failed to delete workflow schedule", err, requestID, r.URL.Path, r.Method, "workflow_schedule", "", nil))
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

/* ListWorkflowSchedules lists all workflow schedules */
func (h *WorkflowHandlers) ListWorkflowSchedules(w http.ResponseWriter, r *http.Request) {
	requestID := GetRequestID(r.Context())

	schedules, err := h.queries.ListWorkflowSchedules(r.Context())
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to list workflow schedules", err, requestID, r.URL.Path, r.Method, "workflow_schedule", "", nil))
		return
	}

	respondJSON(w, http.StatusOK, schedules)
}

