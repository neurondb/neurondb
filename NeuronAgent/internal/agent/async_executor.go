/*-------------------------------------------------------------------------
 *
 * async_executor.go
 *    Asynchronous task execution with status tracking
 *
 * Provides asynchronous task execution for long-running agent operations
 * with status tracking, result storage, and completion notifications.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/async_executor.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/tools"
)

/* AsyncTaskExecutor manages asynchronous task execution */
type AsyncTaskExecutor struct {
	queries  *db.Queries
	runtime  *Runtime
	notifier *TaskNotifier
}

/* AsyncTask represents an asynchronous task */
type AsyncTask struct {
	ID          uuid.UUID
	SessionID   uuid.UUID
	AgentID     uuid.UUID
	TaskType    string
	Status      string
	Priority    int
	Input       map[string]interface{}
	Result      map[string]interface{}
	ErrorMsg    *string
	CreatedAt   time.Time
	StartedAt   *time.Time
	CompletedAt *time.Time
	Metadata    map[string]interface{}
}

/* NewAsyncTaskExecutor creates a new async task executor */
func NewAsyncTaskExecutor(queries *db.Queries, runtime *Runtime, notifier *TaskNotifier) *AsyncTaskExecutor {
	return &AsyncTaskExecutor{
		queries:  queries,
		runtime:  runtime,
		notifier: notifier,
	}
}

/* ExecuteAsync queues an asynchronous task for execution */
func (e *AsyncTaskExecutor) ExecuteAsync(ctx context.Context, sessionID, agentID uuid.UUID, taskType string, input map[string]interface{}, priority int) (*AsyncTask, error) {
	/* Validate input */
	if taskType == "" {
		return nil, fmt.Errorf("async task execution failed: task_type_empty=true")
	}

	/* Create task record */
	taskID := uuid.New()
	now := time.Now()

	inputJSON, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("async task execution failed: input_serialization_error=true, error=%w", err)
	}

	metadata := make(map[string]interface{})
	if input["metadata"] != nil {
		if meta, ok := input["metadata"].(map[string]interface{}); ok {
			metadata = meta
		}
	}

	/* Insert task into database */
	query := `INSERT INTO neurondb_agent.async_tasks 
		(id, session_id, agent_id, task_type, status, priority, input, metadata, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
		RETURNING id, session_id, agent_id, task_type, status, priority, input, result, error_message, 
			created_at, started_at, completed_at, metadata`

	task := &AsyncTask{}
	err = e.queries.DB.QueryRowContext(ctx, query,
		taskID, sessionID, agentID, taskType, "pending", priority,
		string(inputJSON), db.FromMap(metadata), now,
	).Scan(
		&task.ID, &task.SessionID, &task.AgentID, &task.TaskType, &task.Status, &task.Priority,
		&inputJSON, &task.Result, &task.ErrorMsg,
		&task.CreatedAt, &task.StartedAt, &task.CompletedAt, &task.Metadata,
	)

	if err != nil {
		return nil, fmt.Errorf("async task execution failed: task_creation_error=true, session_id='%s', agent_id='%s', task_type='%s', error=%w",
			sessionID.String(), agentID.String(), taskType, err)
	}

	/* Parse input JSON */
	if err := json.Unmarshal(inputJSON, &task.Input); err != nil {
		task.Input = input
	}

	/* Queue task for background execution */
	go e.executeTaskInBackground(context.Background(), task)

	return task, nil
}

/* GetTaskStatus retrieves the status of an async task */
func (e *AsyncTaskExecutor) GetTaskStatus(ctx context.Context, taskID uuid.UUID) (*AsyncTask, error) {
	query := `SELECT id, session_id, agent_id, task_type, status, priority, input, result, error_message,
		created_at, started_at, completed_at, metadata
		FROM neurondb_agent.async_tasks
		WHERE id = $1`

	task := &AsyncTask{}
	var inputJSON, resultJSON []byte
	var errorMsg *string

	err := e.queries.DB.QueryRowContext(ctx, query, taskID).Scan(
		&task.ID, &task.SessionID, &task.AgentID, &task.TaskType, &task.Status, &task.Priority,
		&inputJSON, &resultJSON, &errorMsg,
		&task.CreatedAt, &task.StartedAt, &task.CompletedAt, &task.Metadata,
	)

	if err != nil {
		return nil, fmt.Errorf("task status retrieval failed: task_id='%s', error=%w", taskID.String(), err)
	}

	/* Parse JSON fields */
	if len(inputJSON) > 0 {
		json.Unmarshal(inputJSON, &task.Input)
	}
	if len(resultJSON) > 0 {
		json.Unmarshal(resultJSON, &task.Result)
	}
	task.ErrorMsg = errorMsg

	return task, nil
}

/* CancelTask cancels a running or pending task */
func (e *AsyncTaskExecutor) CancelTask(ctx context.Context, taskID uuid.UUID) error {
	query := `UPDATE neurondb_agent.async_tasks
		SET status = 'cancelled', completed_at = NOW()
		WHERE id = $1 AND status IN ('pending', 'running')
		RETURNING id`

	var cancelledID uuid.UUID
	err := e.queries.DB.QueryRowContext(ctx, query, taskID).Scan(&cancelledID)
	if err != nil {
		return fmt.Errorf("task cancellation failed: task_id='%s', error=%w", taskID.String(), err)
	}

	return nil
}

/* ListTasks lists tasks with optional filters */
func (e *AsyncTaskExecutor) ListTasks(ctx context.Context, sessionID, agentID *uuid.UUID, status *string, limit, offset int) ([]*AsyncTask, error) {
	query := `SELECT id, session_id, agent_id, task_type, status, priority, input, result, error_message,
		created_at, started_at, completed_at, metadata
		FROM neurondb_agent.async_tasks
		WHERE 1=1`
	args := []interface{}{}
	argPos := 1

	if sessionID != nil {
		query += fmt.Sprintf(" AND session_id = $%d", argPos)
		args = append(args, *sessionID)
		argPos++
	}

	if agentID != nil {
		query += fmt.Sprintf(" AND agent_id = $%d", argPos)
		args = append(args, *agentID)
		argPos++
	}

	if status != nil {
		query += fmt.Sprintf(" AND status = $%d", argPos)
		args = append(args, *status)
		argPos++
	}

	query += " ORDER BY created_at DESC"

	if limit > 0 {
		query += fmt.Sprintf(" LIMIT $%d", argPos)
		args = append(args, limit)
		argPos++
	}

	if offset > 0 {
		query += fmt.Sprintf(" OFFSET $%d", argPos)
		args = append(args, offset)
	}

	rows, err := e.queries.DB.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("task listing failed: error=%w", err)
	}
	defer rows.Close()

	var tasks []*AsyncTask
	for rows.Next() {
		task := &AsyncTask{}
		var inputJSON, resultJSON []byte
		var errorMsg *string

		err := rows.Scan(
			&task.ID, &task.SessionID, &task.AgentID, &task.TaskType, &task.Status, &task.Priority,
			&inputJSON, &resultJSON, &errorMsg,
			&task.CreatedAt, &task.StartedAt, &task.CompletedAt, &task.Metadata,
		)
		if err != nil {
			continue
		}

		/* Parse JSON fields */
		if len(inputJSON) > 0 {
			json.Unmarshal(inputJSON, &task.Input)
		}
		if len(resultJSON) > 0 {
			json.Unmarshal(resultJSON, &task.Result)
		}
		task.ErrorMsg = errorMsg

		tasks = append(tasks, task)
	}

	return tasks, nil
}

/* executeTaskInBackground executes a task in the background */
func (e *AsyncTaskExecutor) executeTaskInBackground(ctx context.Context, task *AsyncTask) {
	/* Update status to running */
	startedAt := time.Now()
	updateQuery := `UPDATE neurondb_agent.async_tasks
		SET status = 'running', started_at = $1
		WHERE id = $2 AND status = 'pending'`

	_, err := e.queries.DB.ExecContext(ctx, updateQuery, startedAt, task.ID)
	if err != nil {
		return
	}

	/* Execute task based on type */
	var result map[string]interface{}
	var taskErr error

	switch task.TaskType {
	case "agent_execution":
		result, taskErr = e.executeAgentTask(ctx, task)
	case "data_processing":
		result, taskErr = e.executeDataProcessingTask(ctx, task)
	case "code_execution":
		result, taskErr = e.executeCodeTask(ctx, task)
	default:
		taskErr = fmt.Errorf("unknown task type: %s", task.TaskType)
	}

	/* Update task with result */
	completedAt := time.Now()
	status := "completed"
	errorMsg := (*string)(nil)
	resultJSON := []byte("{}")

	if taskErr != nil {
		status = "failed"
		errStr := taskErr.Error()
		errorMsg = &errStr
	} else if result != nil {
		resultJSON, _ = json.Marshal(result)
	}

	updateResultQuery := `UPDATE neurondb_agent.async_tasks
		SET status = $1, result = $2, error_message = $3, completed_at = $4
		WHERE id = $5`

	_, err = e.queries.DB.ExecContext(ctx, updateResultQuery,
		status, string(resultJSON), errorMsg, completedAt, task.ID)
	if err != nil {
		return
	}

	/* Send completion notification */
	if e.notifier != nil {
		if status == "completed" {
			e.notifier.SendCompletionAlert(ctx, task.ID)
		} else {
			e.notifier.SendFailureAlert(ctx, task.ID, taskErr)
		}
	}
}

/* executeAgentTask executes an agent execution task */
func (e *AsyncTaskExecutor) executeAgentTask(ctx context.Context, task *AsyncTask) (map[string]interface{}, error) {
	/* Extract user message from input */
	userMessage, ok := task.Input["user_message"].(string)
	if !ok {
		return nil, fmt.Errorf("agent task requires user_message in input")
	}

	/* Execute agent */
	state, err := e.runtime.Execute(ctx, task.SessionID, userMessage)
	if err != nil {
		return nil, fmt.Errorf("agent execution failed: %w", err)
	}

	/* Return result */
	return map[string]interface{}{
		"final_answer": state.FinalAnswer,
		"tool_calls":   state.ToolCalls,
		"tokens_used":  state.TokensUsed,
	}, nil
}

/* executeDataProcessingTask executes a data processing task */
func (e *AsyncTaskExecutor) executeDataProcessingTask(ctx context.Context, task *AsyncTask) (map[string]interface{}, error) {
	/* Get input data */
	inputData, ok := task.Input["data"].(string)
	if !ok {
		return nil, fmt.Errorf("data field is required for data processing task")
	}

	/* Get processing type */
	processingType, _ := task.Input["type"].(string)
	if processingType == "" {
		processingType = "auto" /* Auto-detect */
	}

	/* Process based on type */
	var result map[string]interface{}
	var err error

	switch processingType {
	case "csv", "auto":
		if strings.HasSuffix(strings.ToLower(inputData), ".csv") || strings.Contains(inputData, ",") {
			result, err = e.processCSV(ctx, inputData, task.Input)
		}
	case "json":
		result, err = e.processJSON(ctx, inputData, task.Input)
	case "xml":
		result, err = e.processXML(ctx, inputData, task.Input)
	default:
		/* Try to auto-detect */
		if strings.TrimSpace(inputData)[0] == '{' || strings.TrimSpace(inputData)[0] == '[' {
			result, err = e.processJSON(ctx, inputData, task.Input)
		} else if strings.TrimSpace(inputData)[0] == '<' {
			result, err = e.processXML(ctx, inputData, task.Input)
		} else {
			result, err = e.processCSV(ctx, inputData, task.Input)
		}
	}

	if err != nil {
		return nil, fmt.Errorf("data processing failed: %w", err)
	}

	return result, nil
}

/* processCSV processes CSV data */
func (e *AsyncTaskExecutor) processCSV(ctx context.Context, data string, config map[string]interface{}) (map[string]interface{}, error) {
	lines := strings.Split(strings.TrimSpace(data), "\n")
	if len(lines) == 0 {
		return nil, fmt.Errorf("empty CSV data")
	}

	/* Parse header */
	header := strings.Split(lines[0], ",")
	for i, h := range header {
		header[i] = strings.TrimSpace(h)
	}

	/* Parse rows */
	rows := make([]map[string]string, 0, len(lines)-1)
	for i := 1; i < len(lines); i++ {
		values := strings.Split(lines[i], ",")
		if len(values) != len(header) {
			continue /* Skip malformed rows */
		}
		row := make(map[string]string)
		for j, val := range values {
			row[header[j]] = strings.TrimSpace(val)
		}
		rows = append(rows, row)
	}

	/* Apply transformations if specified */
	if transform, ok := config["transform"].(map[string]interface{}); ok {
		rows = e.transformCSV(rows, transform)
	}

	return map[string]interface{}{
		"status":      "completed",
		"row_count":   len(rows),
		"column_count": len(header),
		"columns":     header,
		"data":        rows,
		"summary": map[string]interface{}{
			"total_rows":    len(rows),
			"total_columns": len(header),
		},
	}, nil
}

/* processJSON processes JSON data */
func (e *AsyncTaskExecutor) processJSON(ctx context.Context, data string, config map[string]interface{}) (map[string]interface{}, error) {
	var jsonData interface{}
	if err := json.Unmarshal([]byte(data), &jsonData); err != nil {
		return nil, fmt.Errorf("failed to parse JSON: %w", err)
	}

	/* Apply transformations if specified */
	if transform, ok := config["transform"].(map[string]interface{}); ok {
		jsonData = e.transformJSON(jsonData, transform)
	}

	/* Extract summary */
	summary := e.summarizeJSON(jsonData)

	return map[string]interface{}{
		"status":  "completed",
		"data":    jsonData,
		"summary": summary,
	}, nil
}

/* processXML processes XML data */
func (e *AsyncTaskExecutor) processXML(ctx context.Context, data string, config map[string]interface{}) (map[string]interface{}, error) {
	/* Basic XML parsing - in production, use proper XML parser */
	/* For now, extract basic structure */
	lines := strings.Split(data, "\n")
	elementCount := 0
	attributeCount := 0

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "<") && strings.HasSuffix(trimmed, ">") {
			elementCount++
			/* Count attributes */
			if strings.Contains(trimmed, "=") {
				parts := strings.Fields(trimmed)
				for _, part := range parts {
					if strings.Contains(part, "=") {
						attributeCount++
					}
				}
			}
		}
	}

	return map[string]interface{}{
		"status":  "completed",
		"summary": map[string]interface{}{
			"element_count":    elementCount,
			"attribute_count":  attributeCount,
			"raw_data_length":  len(data),
		},
		"note": "Full XML parsing requires proper XML parser library",
	}, nil
}

/* transformCSV applies transformations to CSV data */
func (e *AsyncTaskExecutor) transformCSV(rows []map[string]string, transform map[string]interface{}) []map[string]string {
	/* Apply filters */
	if filter, ok := transform["filter"].(map[string]interface{}); ok {
		filtered := make([]map[string]string, 0)
		for _, row := range rows {
			match := true
			for key, value := range filter {
				if row[key] != fmt.Sprintf("%v", value) {
					match = false
					break
				}
			}
			if match {
				filtered = append(filtered, row)
			}
		}
		rows = filtered
	}

	/* Apply sorting */
	if sortBy, ok := transform["sort_by"].(string); ok && sortBy != "" {
		sortOrder := "asc"
		if order, ok := transform["sort_order"].(string); ok && (order == "desc" || order == "asc") {
			sortOrder = order
		}

		/* Sort rows by the specified column */
		sortedRows := make([]map[string]string, len(rows))
		copy(sortedRows, rows)

		/* Use sort.Slice for stable sorting */
		sort.Slice(sortedRows, func(i, j int) bool {
			valI, hasI := sortedRows[i][sortBy]
			valJ, hasJ := sortedRows[j][sortBy]

			/* Handle missing values - put them at the end */
			if !hasI && !hasJ {
				return false
			}
			if !hasI {
				return sortOrder == "desc"
			}
			if !hasJ {
				return sortOrder == "asc"
			}

			/* Try to convert to numbers first */
			numI, errI := strconv.ParseFloat(valI, 64)
			numJ, errJ := strconv.ParseFloat(valJ, 64)
			if errI == nil && errJ == nil {
				if sortOrder == "desc" {
					return numI > numJ
				}
				return numI < numJ
			}

			/* Try to parse as dates/timestamps */
			timeI, errI := time.Parse(time.RFC3339, valI)
			timeJ, errJ := time.Parse(time.RFC3339, valJ)
			if errI == nil && errJ == nil {
				if sortOrder == "desc" {
					return timeI.After(timeJ)
				}
				return timeI.Before(timeJ)
			}

			/* Fall back to string comparison */
			if sortOrder == "desc" {
				return valI > valJ
			}
			return valI < valJ
		})

		rows = sortedRows
	}

	return rows
}

/* transformJSON applies transformations to JSON data */
func (e *AsyncTaskExecutor) transformJSON(data interface{}, transform map[string]interface{}) interface{} {
	/* Apply filters, mappings, etc. */
	/* This is a simplified implementation */
	return data
}

/* summarizeJSON creates a summary of JSON data */
func (e *AsyncTaskExecutor) summarizeJSON(data interface{}) map[string]interface{} {
	summary := make(map[string]interface{})

	switch v := data.(type) {
	case map[string]interface{}:
		summary["type"] = "object"
		summary["key_count"] = len(v)
	case []interface{}:
		summary["type"] = "array"
		summary["length"] = len(v)
	case string:
		summary["type"] = "string"
		summary["length"] = len(v)
	case float64:
		summary["type"] = "number"
		summary["value"] = v
	case bool:
		summary["type"] = "boolean"
		summary["value"] = v
	default:
		summary["type"] = "unknown"
	}

	return summary
}

/* executeCodeTask executes a code execution task */
func (e *AsyncTaskExecutor) executeCodeTask(ctx context.Context, task *AsyncTask) (map[string]interface{}, error) {
	/* Get code and language */
	code, ok := task.Input["code"].(string)
	if !ok {
		return nil, fmt.Errorf("code field is required for code execution task")
	}

	language, _ := task.Input["language"].(string)
	if language == "" {
		language = "python" /* Default to Python */
	}

	/* Get timeout */
	timeout := 30 * time.Second
	if timeoutStr, ok := task.Input["timeout"].(string); ok {
		if parsed, err := time.ParseDuration(timeoutStr); err == nil {
			timeout = parsed
		}
	}

	/* Execute code in sandbox */
	sandbox := tools.NewEnhancedSandbox(tools.SandboxConfig{
		MaxMemory: 512 * 1024 * 1024, /* 512 MB */
		MaxCPU:    50.0,
		Timeout:   timeout,
		Isolation: tools.IsolationContainer, /* Use container isolation for security */
	})

	/* Prepare command based on language */
	var command string
	var args []string

	switch strings.ToLower(language) {
	case "python", "py":
		command = "python3"
		args = []string{"-c", code}
	case "javascript", "js", "node":
		command = "node"
		args = []string{"-e", code}
	case "bash", "sh", "shell":
		command = "bash"
		args = []string{"-c", code}
	default:
		return nil, fmt.Errorf("unsupported language: %s", language)
	}

	/* Execute in sandbox */
	output, err := sandbox.ExecuteCommand(ctx, command, args, "")
	if err != nil {
		return map[string]interface{}{
			"status":  "failed",
			"error":   err.Error(),
			"output":  string(output),
		}, nil /* Return error info, don't fail the task */
	}

	return map[string]interface{}{
		"status":  "completed",
		"output":  string(output),
		"language": language,
	}, nil
}
