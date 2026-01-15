/*-------------------------------------------------------------------------
 *
 * evaluation_handlers.go
 *    Evaluation Framework API handlers for NeuronAgent
 *
 * Provides REST API endpoints for evaluation framework operations including
 * eval task management, eval run execution, and result retrieval.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/api/evaluation_handlers.go
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
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/eval"
	"github.com/neurondb/NeuronAgent/internal/validation"
)

/* EvaluationHandlers handles evaluation framework API requests */
type EvaluationHandlers struct {
	queries   *db.Queries
	evaluator *eval.Evaluator
}

/* NewEvaluationHandlers creates new evaluation handlers */
func NewEvaluationHandlers(queries *db.Queries, evaluator *eval.Evaluator) *EvaluationHandlers {
	return &EvaluationHandlers{
		queries:   queries,
		evaluator: evaluator,
	}
}

/* Eval Task Request/Response DTOs */

type CreateEvalTaskRequest struct {
	TaskType             string                 `json:"task_type"`
	Input                string                 `json:"input"`
	ExpectedOutput       *string                `json:"expected_output,omitempty"`
	ExpectedToolSequence map[string]interface{} `json:"expected_tool_sequence,omitempty"`
	GoldenSQLSideEffects map[string]interface{} `json:"golden_sql_side_effects,omitempty"`
	Metadata             map[string]interface{} `json:"metadata,omitempty"`
}

type EvalTaskResponse struct {
	ID                   string                 `json:"id"`
	TaskType             string                 `json:"task_type"`
	Input                string                 `json:"input"`
	ExpectedOutput       *string                `json:"expected_output,omitempty"`
	ExpectedToolSequence map[string]interface{} `json:"expected_tool_sequence,omitempty"`
	GoldenSQLSideEffects map[string]interface{} `json:"golden_sql_side_effects,omitempty"`
	Metadata             map[string]interface{} `json:"metadata"`
	CreatedAt            time.Time              `json:"created_at"`
	UpdatedAt            time.Time              `json:"updated_at"`
}

/* Eval Run Request/Response DTOs */

type CreateEvalRunRequest struct {
	DatasetVersion string     `json:"dataset_version"`
	AgentID        *uuid.UUID `json:"agent_id,omitempty"`
	TotalTasks     int        `json:"total_tasks,omitempty"`
	Metadata       map[string]interface{} `json:"metadata,omitempty"`
}

type ExecuteEvalRunRequest struct {
	TaskType  *string    `json:"task_type,omitempty"`  // Filter tasks by type
	TaskLimit *int       `json:"task_limit,omitempty"` // Limit number of tasks to evaluate
}

type EvalRunResponse struct {
	ID             string                 `json:"id"`
	DatasetVersion string                 `json:"dataset_version"`
	AgentID        *string                `json:"agent_id,omitempty"`
	StartedAt      time.Time              `json:"started_at"`
	CompletedAt    *time.Time             `json:"completed_at,omitempty"`
	Score          *float64               `json:"score,omitempty"`
	TotalTasks     int                    `json:"total_tasks"`
	PassedTasks    int                    `json:"passed_tasks"`
	FailedTasks    int                    `json:"failed_tasks"`
	Metadata       map[string]interface{} `json:"metadata"`
	CreatedAt      time.Time              `json:"created_at"`
}

/* Eval Task Result Response DTOs */

type EvalTaskResultResponse struct {
	ID                   string                 `json:"id"`
	EvalRunID            string                 `json:"eval_run_id"`
	EvalTaskID           string                 `json:"eval_task_id"`
	SessionID            *string                `json:"session_id,omitempty"`
	Passed               bool                   `json:"passed"`
	ActualOutput         *string                `json:"actual_output,omitempty"`
	ActualToolSequence   map[string]interface{} `json:"actual_tool_sequence,omitempty"`
	ActualSQLSideEffects map[string]interface{} `json:"actual_sql_side_effects,omitempty"`
	Score                *float64               `json:"score,omitempty"`
	ErrorMessage         *string                `json:"error_message,omitempty"`
	Metadata             map[string]interface{} `json:"metadata"`
	CreatedAt            time.Time              `json:"created_at"`
}

/* Eval Retrieval Result Request/Response DTOs */

type CreateEvalRetrievalResultRequest struct {
	RecallAtK       *float64              `json:"recall_at_k,omitempty"`
	MRR             *float64              `json:"mrr,omitempty"`
	GroundingPassed bool                  `json:"grounding_passed"`
	RetrievedChunks []string              `json:"retrieved_chunks"`
	RelevantChunks  []string              `json:"relevant_chunks"`
}

type EvalRetrievalResultResponse struct {
	ID               string    `json:"id"`
	EvalTaskResultID string    `json:"eval_task_result_id"`
	RecallAtK        *float64  `json:"recall_at_k,omitempty"`
	MRR              *float64  `json:"mrr,omitempty"`
	GroundingPassed  bool      `json:"grounding_passed"`
	RetrievedChunks  []string  `json:"retrieved_chunks"`
	RelevantChunks   []string  `json:"relevant_chunks"`
	CreatedAt        time.Time `json:"created_at"`
}

/* CreateEvalTask creates a new eval task */
func (h *EvaluationHandlers) CreateEvalTask(w http.ResponseWriter, r *http.Request) {
	requestID := GetRequestID(r.Context())

	/* Validate request body size */
	const maxBodySize = 1024 * 1024
	bodyBytes, err := validation.ReadAndValidateBody(r, maxBodySize)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body validation failed", err, requestID, r.URL.Path, r.Method, "eval_task", "", nil))
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var req CreateEvalTaskRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body parsing error", err, requestID, r.URL.Path, r.Method, "eval_task", "", nil))
		return
	}

	/* Validate required fields */
	if req.TaskType == "" {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "task_type is required", nil, requestID, r.URL.Path, r.Method, "eval_task", "", nil))
		return
	}
	if req.Input == "" {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "input is required", nil, requestID, r.URL.Path, r.Method, "eval_task", "", nil))
		return
	}

	/* Validate task type */
	validTaskTypes := map[string]bool{
		"tool_sequence":   true,
		"sql_side_effect": true,
		"retrieval":       true,
		"end_to_end":      true,
	}
	if !validTaskTypes[req.TaskType] {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid task_type", nil, requestID, r.URL.Path, r.Method, "eval_task", "", map[string]interface{}{
			"valid_types": []string{"tool_sequence", "sql_side_effect", "retrieval", "end_to_end"},
		}))
		return
	}

	task := &db.EvalTask{
		TaskType:             req.TaskType,
		Input:                req.Input,
		ExpectedOutput:       req.ExpectedOutput,
		ExpectedToolSequence: db.FromMap(req.ExpectedToolSequence),
		GoldenSQLSideEffects: db.FromMap(req.GoldenSQLSideEffects),
		Metadata:             db.FromMap(req.Metadata),
	}

	if err := h.queries.CreateEvalTask(r.Context(), task); err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "eval task creation failed", err, requestID, r.URL.Path, r.Method, "eval_task", "", nil))
		return
	}

	respondJSON(w, http.StatusCreated, toEvalTaskResponse(task))
}

/* ListEvalTasks lists eval tasks */
func (h *EvaluationHandlers) ListEvalTasks(w http.ResponseWriter, r *http.Request) {
	requestID := GetRequestID(r.Context())

	/* Parse query parameters */
	taskType := r.URL.Query().Get("task_type")
	var taskTypePtr *string
	if taskType != "" {
		taskTypePtr = &taskType
	}

	limitStr := r.URL.Query().Get("limit")
	limit := 100
	if limitStr != "" {
		var err error
		limit, err = strconv.Atoi(limitStr)
		if err != nil || limit < 1 || limit > 1000 {
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid limit parameter", err, requestID, r.URL.Path, r.Method, "eval_task", "", nil))
			return
		}
	}

	offsetStr := r.URL.Query().Get("offset")
	offset := 0
	if offsetStr != "" {
		var err error
		offset, err = strconv.Atoi(offsetStr)
		if err != nil || offset < 0 {
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid offset parameter", err, requestID, r.URL.Path, r.Method, "eval_task", "", nil))
			return
		}
	}

	tasks, err := h.queries.ListEvalTasks(r.Context(), taskTypePtr, limit, offset)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to list eval tasks", err, requestID, r.URL.Path, r.Method, "eval_task", "", nil))
		return
	}

	responses := make([]EvalTaskResponse, len(tasks))
	for i, task := range tasks {
		responses[i] = toEvalTaskResponse(&task)
	}

	respondJSON(w, http.StatusOK, responses)
}

/* GetEvalTask gets an eval task by ID */
func (h *EvaluationHandlers) GetEvalTask(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	/* Validate task ID */
	if err := validation.ValidateUUIDRequired(vars["id"], "id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid task ID", err, requestID, r.URL.Path, r.Method, "eval_task", vars["id"], nil))
		return
	}

	taskID, err := uuid.Parse(vars["id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid task ID format", err, requestID, r.URL.Path, r.Method, "eval_task", vars["id"], nil))
		return
	}

	task, err := h.queries.GetEvalTaskByID(r.Context(), taskID)
	if err != nil {
		status := http.StatusInternalServerError
		if err.Error() == "eval task not found: sql: no rows in result set" {
			status = http.StatusNotFound
		}
		respondError(w, NewErrorWithContext(status, "failed to get eval task", err, requestID, r.URL.Path, r.Method, "eval_task", vars["id"], nil))
		return
	}

	respondJSON(w, http.StatusOK, toEvalTaskResponse(task))
}

/* CreateEvalRun creates a new eval run */
func (h *EvaluationHandlers) CreateEvalRun(w http.ResponseWriter, r *http.Request) {
	requestID := GetRequestID(r.Context())

	/* Validate request body size */
	const maxBodySize = 1024 * 1024
	bodyBytes, err := validation.ReadAndValidateBody(r, maxBodySize)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body validation failed", err, requestID, r.URL.Path, r.Method, "eval_run", "", nil))
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var req CreateEvalRunRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body parsing error", err, requestID, r.URL.Path, r.Method, "eval_run", "", nil))
		return
	}

	/* Validate required fields */
	if req.DatasetVersion == "" {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "dataset_version is required", nil, requestID, r.URL.Path, r.Method, "eval_run", "", nil))
		return
	}

	run := &db.EvalRun{
		DatasetVersion: req.DatasetVersion,
		AgentID:        req.AgentID,
		TotalTasks:     req.TotalTasks,
		Metadata:       db.FromMap(req.Metadata),
	}

	if err := h.queries.CreateEvalRun(r.Context(), run); err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "eval run creation failed", err, requestID, r.URL.Path, r.Method, "eval_run", "", nil))
		return
	}

	respondJSON(w, http.StatusCreated, toEvalRunResponse(run))
}

/* ListEvalRuns lists eval runs */
func (h *EvaluationHandlers) ListEvalRuns(w http.ResponseWriter, r *http.Request) {
	requestID := GetRequestID(r.Context())

	/* Parse query parameters */
	datasetVersion := r.URL.Query().Get("dataset_version")
	var datasetVersionPtr *string
	if datasetVersion != "" {
		datasetVersionPtr = &datasetVersion
	}

	agentIDStr := r.URL.Query().Get("agent_id")
	var agentIDPtr *uuid.UUID
	if agentIDStr != "" {
		agentID, err := uuid.Parse(agentIDStr)
		if err != nil {
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid agent_id format", err, requestID, r.URL.Path, r.Method, "eval_run", "", nil))
			return
		}
		agentIDPtr = &agentID
	}

	limitStr := r.URL.Query().Get("limit")
	limit := 100
	if limitStr != "" {
		var err error
		limit, err = strconv.Atoi(limitStr)
		if err != nil || limit < 1 || limit > 1000 {
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid limit parameter", err, requestID, r.URL.Path, r.Method, "eval_run", "", nil))
			return
		}
	}

	offsetStr := r.URL.Query().Get("offset")
	offset := 0
	if offsetStr != "" {
		var err error
		offset, err = strconv.Atoi(offsetStr)
		if err != nil || offset < 0 {
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid offset parameter", err, requestID, r.URL.Path, r.Method, "eval_run", "", nil))
			return
		}
	}

	runs, err := h.queries.ListEvalRuns(r.Context(), datasetVersionPtr, agentIDPtr, limit, offset)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to list eval runs", err, requestID, r.URL.Path, r.Method, "eval_run", "", nil))
		return
	}

	responses := make([]EvalRunResponse, len(runs))
	for i, run := range runs {
		responses[i] = toEvalRunResponse(&run)
	}

	respondJSON(w, http.StatusOK, responses)
}

/* GetEvalRun gets an eval run by ID */
func (h *EvaluationHandlers) GetEvalRun(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	/* Validate run ID */
	if err := validation.ValidateUUIDRequired(vars["id"], "id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid run ID", err, requestID, r.URL.Path, r.Method, "eval_run", vars["id"], nil))
		return
	}

	runID, err := uuid.Parse(vars["id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid run ID format", err, requestID, r.URL.Path, r.Method, "eval_run", vars["id"], nil))
		return
	}

	run, err := h.queries.GetEvalRunByID(r.Context(), runID)
	if err != nil {
		status := http.StatusInternalServerError
		if err.Error() == "eval run not found: sql: no rows in result set" {
			status = http.StatusNotFound
		}
		respondError(w, NewErrorWithContext(status, "failed to get eval run", err, requestID, r.URL.Path, r.Method, "eval_run", vars["id"], nil))
		return
	}

	respondJSON(w, http.StatusOK, toEvalRunResponse(run))
}

/* UpdateEvalRun updates an eval run (typically to mark as completed) */
func (h *EvaluationHandlers) UpdateEvalRun(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	/* Validate run ID */
	if err := validation.ValidateUUIDRequired(vars["id"], "id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid run ID", err, requestID, r.URL.Path, r.Method, "eval_run", vars["id"], nil))
		return
	}

	runID, err := uuid.Parse(vars["id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid run ID format", err, requestID, r.URL.Path, r.Method, "eval_run", vars["id"], nil))
		return
	}

	/* Get existing run */
	run, err := h.queries.GetEvalRunByID(r.Context(), runID)
	if err != nil {
		status := http.StatusInternalServerError
		if err.Error() == "eval run not found: sql: no rows in result set" {
			status = http.StatusNotFound
		}
		respondError(w, NewErrorWithContext(status, "failed to get eval run", err, requestID, r.URL.Path, r.Method, "eval_run", vars["id"], nil))
		return
	}

	/* Validate request body size */
	const maxBodySize = 1024 * 1024
	bodyBytes, err := validation.ReadAndValidateBody(r, maxBodySize)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body validation failed", err, requestID, r.URL.Path, r.Method, "eval_run", vars["id"], nil))
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var req struct {
		Score       *float64 `json:"score,omitempty"`
		PassedTasks *int     `json:"passed_tasks,omitempty"`
		FailedTasks *int     `json:"failed_tasks,omitempty"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body parsing error", err, requestID, r.URL.Path, r.Method, "eval_run", vars["id"], nil))
		return
	}

	/* Update fields */
	if req.Score != nil {
		run.Score = req.Score
	}
	if req.PassedTasks != nil {
		run.PassedTasks = *req.PassedTasks
	}
	if req.FailedTasks != nil {
		run.FailedTasks = *req.FailedTasks
	}

	if err := h.queries.UpdateEvalRun(r.Context(), run); err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "eval run update failed", err, requestID, r.URL.Path, r.Method, "eval_run", vars["id"], nil))
		return
	}

	/* Get updated run */
	updatedRun, err := h.queries.GetEvalRunByID(r.Context(), runID)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to get updated eval run", err, requestID, r.URL.Path, r.Method, "eval_run", vars["id"], nil))
		return
	}

	respondJSON(w, http.StatusOK, toEvalRunResponse(updatedRun))
}

/* ExecuteEvalRun executes an evaluation run */
func (h *EvaluationHandlers) ExecuteEvalRun(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	/* Validate run ID */
	if err := validation.ValidateUUIDRequired(vars["run_id"], "run_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid run ID", err, requestID, r.URL.Path, r.Method, "eval_run", vars["run_id"], nil))
		return
	}

	runID, err := uuid.Parse(vars["run_id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid run ID format", err, requestID, r.URL.Path, r.Method, "eval_run", vars["run_id"], nil))
		return
	}

	/* Get eval run */
	run, err := h.queries.GetEvalRunByID(r.Context(), runID)
	if err != nil {
		status := http.StatusInternalServerError
		if err.Error() == "eval run not found: sql: no rows in result set" {
			status = http.StatusNotFound
		}
		respondError(w, NewErrorWithContext(status, "failed to get eval run", err, requestID, r.URL.Path, r.Method, "eval_run", vars["run_id"], nil))
		return
	}

	if run.AgentID == nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "agent_id is required for evaluation run", nil, requestID, r.URL.Path, r.Method, "eval_run", vars["run_id"], nil))
		return
	}

	/* Parse request body (optional) */
	var req ExecuteEvalRunRequest
	bodyBytes, _ := validation.ReadAndValidateBody(r, 1024*1024)
	if len(bodyBytes) > 0 {
		r.Body = io.NopCloser(bytes.NewReader(bodyBytes))
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body parsing error", err, requestID, r.URL.Path, r.Method, "eval_run", vars["run_id"], nil))
			return
		}
	}

	/* List tasks to evaluate */
	var taskTypePtr *string
	if req.TaskType != nil {
		taskTypePtr = req.TaskType
	}

	taskLimit := 1000
	if req.TaskLimit != nil && *req.TaskLimit > 0 {
		taskLimit = *req.TaskLimit
	}

	tasks, err := h.queries.ListEvalTasks(r.Context(), taskTypePtr, taskLimit, 0)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to list eval tasks", err, requestID, r.URL.Path, r.Method, "eval_run", vars["run_id"], nil))
		return
	}

	if len(tasks) == 0 {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "no eval tasks found", nil, requestID, r.URL.Path, r.Method, "eval_run", vars["run_id"], nil))
		return
	}

	/* Execute evaluation for each task */
	passedTasks := 0
	failedTasks := 0
	totalScore := 0.0
	taskCount := 0

	for _, task := range tasks {
		result, err := h.evaluator.EvaluateTask(r.Context(), &task, *run.AgentID)
		if err != nil {
			errorMsg := err.Error()
			result = &db.EvalTaskResult{
				EvalRunID:    runID,
				EvalTaskID:   task.ID,
				Passed:       false,
				ErrorMessage: &errorMsg,
			}
		}

		/* Set eval run ID */
		result.EvalRunID = runID

		/* Save result */
		if err := h.queries.CreateEvalTaskResult(r.Context(), result); err != nil {
			/* Continue with other tasks even if one fails */
			continue
		}

		if result.Passed {
			passedTasks++
		} else {
			failedTasks++
		}

		if result.Score != nil {
			totalScore += *result.Score
			taskCount++
		}
	}

	/* Calculate overall score */
	var overallScore *float64
	if taskCount > 0 {
		score := totalScore / float64(taskCount)
		overallScore = &score
	}

	/* Update eval run */
	run.PassedTasks = passedTasks
	run.FailedTasks = failedTasks
	run.TotalTasks = len(tasks)
	run.Score = overallScore

	if err := h.queries.UpdateEvalRun(r.Context(), run); err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to update eval run", err, requestID, r.URL.Path, r.Method, "eval_run", vars["run_id"], nil))
		return
	}

	/* Get updated run */
	updatedRun, err := h.queries.GetEvalRunByID(r.Context(), runID)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to get updated eval run", err, requestID, r.URL.Path, r.Method, "eval_run", vars["run_id"], nil))
		return
	}

	respondJSON(w, http.StatusOK, toEvalRunResponse(updatedRun))
}

/* GetEvalRunResults gets task results for an eval run */
func (h *EvaluationHandlers) GetEvalRunResults(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	/* Validate run ID */
	if err := validation.ValidateUUIDRequired(vars["run_id"], "run_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid run ID", err, requestID, r.URL.Path, r.Method, "eval_run", vars["run_id"], nil))
		return
	}

	runID, err := uuid.Parse(vars["run_id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid run ID format", err, requestID, r.URL.Path, r.Method, "eval_run", vars["run_id"], nil))
		return
	}

	results, err := h.queries.GetEvalTaskResultsByRun(r.Context(), runID)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "failed to get eval run results", err, requestID, r.URL.Path, r.Method, "eval_run", vars["run_id"], nil))
		return
	}

	responses := make([]EvalTaskResultResponse, len(results))
	for i, result := range results {
		responses[i] = toEvalTaskResultResponse(&result)
	}

	respondJSON(w, http.StatusOK, responses)
}

/* CreateEvalRetrievalResult creates a retrieval result for an eval task result */
func (h *EvaluationHandlers) CreateEvalRetrievalResult(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	requestID := GetRequestID(r.Context())

	/* Validate result ID */
	if err := validation.ValidateUUIDRequired(vars["result_id"], "result_id"); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid result ID", err, requestID, r.URL.Path, r.Method, "eval_retrieval_result", vars["result_id"], nil))
		return
	}

	resultID, err := uuid.Parse(vars["result_id"])
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "invalid result ID format", err, requestID, r.URL.Path, r.Method, "eval_retrieval_result", vars["result_id"], nil))
		return
	}

	/* Validate request body size */
	const maxBodySize = 1024 * 1024
	bodyBytes, err := validation.ReadAndValidateBody(r, maxBodySize)
	if err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body validation failed", err, requestID, r.URL.Path, r.Method, "eval_retrieval_result", vars["result_id"], nil))
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	var req CreateEvalRetrievalResultRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, NewErrorWithContext(http.StatusBadRequest, "request body parsing error", err, requestID, r.URL.Path, r.Method, "eval_retrieval_result", vars["result_id"], nil))
		return
	}

	/* Convert chunks to JSONB */
	retrievedChunksMap := make(map[string]interface{})
	retrievedChunksMap["chunks"] = req.RetrievedChunks

	relevantChunksMap := make(map[string]interface{})
	relevantChunksMap["chunks"] = req.RelevantChunks

	retrievalResult := &db.EvalRetrievalResult{
		EvalTaskResultID: resultID,
		RecallAtK:        req.RecallAtK,
		MRR:              req.MRR,
		GroundingPassed:  req.GroundingPassed,
		RetrievedChunks:  db.FromMap(retrievedChunksMap),
		RelevantChunks:   db.FromMap(relevantChunksMap),
	}

	if err := h.queries.CreateEvalRetrievalResult(r.Context(), retrievalResult); err != nil {
		respondError(w, NewErrorWithContext(http.StatusInternalServerError, "eval retrieval result creation failed", err, requestID, r.URL.Path, r.Method, "eval_retrieval_result", vars["result_id"], nil))
		return
	}

	respondJSON(w, http.StatusCreated, toEvalRetrievalResultResponse(retrievalResult))
}

/* Helper functions to convert DB models to API responses */

func toEvalTaskResponse(task *db.EvalTask) EvalTaskResponse {
	return EvalTaskResponse{
		ID:                   task.ID.String(),
		TaskType:             task.TaskType,
		Input:                task.Input,
		ExpectedOutput:       task.ExpectedOutput,
		ExpectedToolSequence: task.ExpectedToolSequence.ToMap(),
		GoldenSQLSideEffects: task.GoldenSQLSideEffects.ToMap(),
		Metadata:             task.Metadata.ToMap(),
		CreatedAt:            task.CreatedAt,
		UpdatedAt:            task.UpdatedAt,
	}
}

func toEvalRunResponse(run *db.EvalRun) EvalRunResponse {
	var agentIDStr *string
	if run.AgentID != nil {
		id := run.AgentID.String()
		agentIDStr = &id
	}

	return EvalRunResponse{
		ID:             run.ID.String(),
		DatasetVersion: run.DatasetVersion,
		AgentID:        agentIDStr,
		StartedAt:      run.StartedAt,
		CompletedAt:    run.CompletedAt,
		Score:          run.Score,
		TotalTasks:     run.TotalTasks,
		PassedTasks:    run.PassedTasks,
		FailedTasks:    run.FailedTasks,
		Metadata:       run.Metadata.ToMap(),
		CreatedAt:      run.CreatedAt,
	}
}

func toEvalTaskResultResponse(result *db.EvalTaskResult) EvalTaskResultResponse {
	var sessionIDStr *string
	if result.SessionID != nil {
		id := result.SessionID.String()
		sessionIDStr = &id
	}

	return EvalTaskResultResponse{
		ID:                   result.ID.String(),
		EvalRunID:            result.EvalRunID.String(),
		EvalTaskID:           result.EvalTaskID.String(),
		SessionID:            sessionIDStr,
		Passed:               result.Passed,
		ActualOutput:         result.ActualOutput,
		ActualToolSequence:   result.ActualToolSequence.ToMap(),
		ActualSQLSideEffects: result.ActualSQLSideEffects.ToMap(),
		Score:                result.Score,
		ErrorMessage:         result.ErrorMessage,
		Metadata:             result.Metadata.ToMap(),
		CreatedAt:            result.CreatedAt,
	}
}

func toEvalRetrievalResultResponse(result *db.EvalRetrievalResult) EvalRetrievalResultResponse {
	/* Extract chunks from JSONB */
	var retrievedChunks []string
	if chunks, ok := result.RetrievedChunks["chunks"].([]interface{}); ok {
		for _, chunk := range chunks {
			if str, ok := chunk.(string); ok {
				retrievedChunks = append(retrievedChunks, str)
			}
		}
	}

	var relevantChunks []string
	if chunks, ok := result.RelevantChunks["chunks"].([]interface{}); ok {
		for _, chunk := range chunks {
			if str, ok := chunk.(string); ok {
				relevantChunks = append(relevantChunks, str)
			}
		}
	}

	return EvalRetrievalResultResponse{
		ID:               result.ID.String(),
		EvalTaskResultID: result.EvalTaskResultID.String(),
		RecallAtK:        result.RecallAtK,
		MRR:              result.MRR,
		GroundingPassed:  result.GroundingPassed,
		RetrievedChunks:  retrievedChunks,
		RelevantChunks:   relevantChunks,
		CreatedAt:        result.CreatedAt,
	}
}








