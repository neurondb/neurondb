/*-------------------------------------------------------------------------
 *
 * engine.go
 *    Workflow DAG engine for NeuronAgent
 *
 * Provides DAG workflow execution with steps, inputs, outputs, dependencies,
 * retries, and idempotency keys.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/workflow/engine.go
 *
 *-------------------------------------------------------------------------
 */

package workflow

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/agent"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/metrics"
	"github.com/neurondb/NeuronAgent/internal/notifications"
)

type Engine struct {
	queries        *db.Queries
	runtime        *agent.Runtime
	toolRegistry   agent.ToolRegistry
	emailService   *notifications.EmailService
	webhookService *notifications.WebhookService
	baseURL        string
}

func NewEngine(queries *db.Queries) *Engine {
	return &Engine{queries: queries}
}

/* SetRuntime sets the agent runtime for agent step execution */
func (e *Engine) SetRuntime(runtime *agent.Runtime) {
	e.runtime = runtime
}

/* SetToolRegistry sets the tool registry for tool step execution */
func (e *Engine) SetToolRegistry(registry agent.ToolRegistry) {
	e.toolRegistry = registry
}

/* SetEmailService sets the email service for HITL notifications */
func (e *Engine) SetEmailService(service *notifications.EmailService) {
	e.emailService = service
}

/* SetWebhookService sets the webhook service for HITL notifications */
func (e *Engine) SetWebhookService(service *notifications.WebhookService) {
	e.webhookService = service
}

/* SetBaseURL sets the base URL for HITL approval links */
func (e *Engine) SetBaseURL(baseURL string) {
	e.baseURL = baseURL
}

/* ExecuteWorkflow executes a workflow with given inputs */
func (e *Engine) ExecuteWorkflow(ctx context.Context, workflowID uuid.UUID, triggerType string, triggerData map[string]interface{}, inputs map[string]interface{}) (*db.WorkflowExecution, error) {
	/* Create execution */
	execution := &db.WorkflowExecution{
		WorkflowID:  workflowID,
		Status:      "pending",
		TriggerType: triggerType,
		TriggerData: triggerData,
		Inputs:      inputs,
		Outputs:     make(map[string]interface{}),
	}

	/* Save execution */
	if err := e.queries.CreateWorkflowExecution(ctx, execution); err != nil {
		return nil, fmt.Errorf("failed to create workflow execution: %w", err)
	}

	/* Load workflow and steps */
	_, err := e.queries.GetWorkflowByID(ctx, workflowID)
	if err != nil {
		execution.Status = "failed"
		errorMsg := fmt.Sprintf("failed to load workflow: %v", err)
		execution.ErrorMessage = &errorMsg
		if updateErr := e.queries.UpdateWorkflowExecution(ctx, execution); updateErr != nil {
			metrics.WarnWithContext(ctx, "Failed to update workflow execution status after workflow load error", map[string]interface{}{
				"execution_id": execution.ID.String(),
				"workflow_id":  workflowID.String(),
				"error":        updateErr.Error(),
			})
		}
		return nil, fmt.Errorf("failed to load workflow: %w", err)
	}

	steps, err := e.queries.ListWorkflowSteps(ctx, workflowID)
	if err != nil {
		execution.Status = "failed"
		errorMsg := fmt.Sprintf("failed to load workflow steps: %v", err)
		execution.ErrorMessage = &errorMsg
		if updateErr := e.queries.UpdateWorkflowExecution(ctx, execution); updateErr != nil {
			metrics.WarnWithContext(ctx, "Failed to update workflow execution status after steps load error", map[string]interface{}{
				"execution_id": execution.ID.String(),
				"workflow_id":  workflowID.String(),
				"error":        updateErr.Error(),
			})
		}
		return nil, fmt.Errorf("failed to load workflow steps: %w", err)
	}

	if len(steps) == 0 {
		execution.Status = "completed"
		if err := e.queries.UpdateWorkflowExecution(ctx, execution); err != nil {
			metrics.WarnWithContext(ctx, "Failed to update workflow execution status for empty workflow", map[string]interface{}{
				"execution_id": execution.ID.String(),
				"workflow_id":  workflowID.String(),
				"error":        err.Error(),
			})
		}
		return execution, nil
	}

	/* Build dependency graph and get topological order */
	executionOrder, err := e.buildDAG(steps)
	if err != nil {
		execution.Status = "failed"
		errorMsg := fmt.Sprintf("failed to build dependency graph: %v", err)
		execution.ErrorMessage = &errorMsg
		if updateErr := e.queries.UpdateWorkflowExecution(ctx, execution); updateErr != nil {
			metrics.WarnWithContext(ctx, "Failed to update workflow execution status after DAG build error", map[string]interface{}{
				"execution_id": execution.ID.String(),
				"workflow_id":  workflowID.String(),
				"error":        updateErr.Error(),
			})
		}
		return nil, fmt.Errorf("failed to build dependency graph: %w", err)
	}

	/* Execute steps in topological order */
	execution.Status = "running"
	if err := e.queries.UpdateWorkflowExecution(ctx, execution); err != nil {
		return nil, fmt.Errorf("failed to update execution status: %w", err)
	}

	stepOutputs := make(map[uuid.UUID]map[string]interface{})
	for _, stepID := range executionOrder {
		/* Find step by ID */
		var step *db.WorkflowStep
		for i := range steps {
			if steps[i].ID == stepID {
				step = &steps[i]
				break
			}
		}
		if step == nil {
			execution.Status = "failed"
			errorMsg := fmt.Sprintf("step not found: %s", stepID.String())
			execution.ErrorMessage = &errorMsg
			if updateErr := e.queries.UpdateWorkflowExecution(ctx, execution); updateErr != nil {
				metrics.WarnWithContext(ctx, "Failed to update workflow execution status after step not found error", map[string]interface{}{
					"execution_id": execution.ID.String(),
					"workflow_id":  workflowID.String(),
					"step_id":      stepID.String(),
					"error":        updateErr.Error(),
				})
			}
			return nil, fmt.Errorf("step not found: %s", stepID.String())
		}

		/* Build inputs for this step from workflow inputs and previous step outputs */
		stepInputs := e.buildStepInputs(step, inputs, stepOutputs)

		/* Execute step */
		outputs, err := e.ExecuteStep(ctx, execution.ID, step, stepInputs)
		if err != nil {
			execution.Status = "failed"
			errorMsg := err.Error()
			execution.ErrorMessage = &errorMsg
			_ = e.queries.UpdateWorkflowExecution(ctx, execution)
			return nil, fmt.Errorf("step execution failed: step_name='%s', error=%w", step.StepName, err)
		}

		stepOutputs[step.ID] = outputs
	}

	/* Collect final outputs from all steps */
	finalOutputs := make(map[string]interface{})
	for stepID, outputs := range stepOutputs {
		/* Find step name */
		var stepName string
		for i := range steps {
			if steps[i].ID == stepID {
				stepName = steps[i].StepName
				break
			}
		}
		if stepName != "" {
			finalOutputs[stepName] = outputs
		} else {
			finalOutputs[stepID.String()] = outputs
		}
	}

	execution.Outputs = finalOutputs
	execution.Status = "completed"
	if err := e.queries.UpdateWorkflowExecution(ctx, execution); err != nil {
		return nil, fmt.Errorf("failed to update execution completion: %w", err)
	}

	return execution, nil
}

/* ExecuteStep executes a single workflow step */
func (e *Engine) ExecuteStep(ctx context.Context, executionID uuid.UUID, step *db.WorkflowStep, inputs map[string]interface{}) (map[string]interface{}, error) {
	/* Check idempotency */
	if step.IdempotencyKey != nil && *step.IdempotencyKey != "" {
		existingExecution, err := e.queries.GetWorkflowStepExecutionByIdempotencyKey(ctx, *step.IdempotencyKey)
		if err != nil {
			return nil, fmt.Errorf("failed to check idempotency: %w", err)
		}
		if existingExecution != nil && existingExecution.Status == "completed" {
			/* Return cached result */
			return existingExecution.Outputs, nil
		}
	}

	/* Create step execution */
	stepExecution := &db.WorkflowStepExecution{
		WorkflowExecutionID: executionID,
		WorkflowStepID:      step.ID,
		Status:              "running",
		Inputs:              inputs,
		Outputs:             make(map[string]interface{}),
		IdempotencyKey:      step.IdempotencyKey,
	}
	now := time.Now()
	stepExecution.StartedAt = &now

	/* Save step execution */
	if err := e.queries.CreateWorkflowStepExecution(ctx, stepExecution); err != nil {
		return nil, fmt.Errorf("failed to create step execution: %w", err)
	}

	/* Execute step based on type */
	var outputs map[string]interface{}
	var err error

	switch step.StepType {
	case "agent":
		outputs, err = e.executeAgentStep(ctx, step, inputs)
	case "tool":
		outputs, err = e.executeToolStep(ctx, step, inputs)
	case "approval":
		outputs, err = e.executeApprovalStep(ctx, executionID, stepExecution.ID, step, inputs)
	case "http":
		outputs, err = e.executeHTTPStep(ctx, step, inputs)
	case "sql":
		outputs, err = e.executeSQLStep(ctx, step, inputs)
	default:
		err = fmt.Errorf("unknown step type: %s", step.StepType)
	}

	/* Handle retries if error */
	if err != nil {
		stepExecution.Status = "failed"
		errorMsg := err.Error()
		stepExecution.ErrorMessage = &errorMsg

		retryConfig := step.RetryConfig
		if retryConfig != nil && stepExecution.RetryCount < getMaxRetries(retryConfig) {
			/* Retry with backoff */
			stepExecution.RetryCount++
			if updateErr := e.queries.UpdateWorkflowStepExecution(ctx, stepExecution); updateErr != nil {
				return nil, fmt.Errorf("failed to update step execution for retry: %w", updateErr)
			}

			/* Schedule retry - for now just return error, would need retry scheduler */
			return nil, fmt.Errorf("step failed, retry %d/%d: %w", stepExecution.RetryCount, getMaxRetries(retryConfig), err)
		}

		/* Max retries reached or no retry config */
		if updateErr := e.queries.UpdateWorkflowStepExecution(ctx, stepExecution); updateErr != nil {
			return nil, fmt.Errorf("failed to update failed step execution: %w", updateErr)
		}
		return nil, err
	}

	stepExecution.Outputs = outputs
	stepExecution.Status = "completed"
	completedAt := time.Now()
	stepExecution.CompletedAt = &completedAt

	/* Update step execution */
	if err := e.queries.UpdateWorkflowStepExecution(ctx, stepExecution); err != nil {
		return nil, fmt.Errorf("failed to update step execution: %w", err)
	}

	return outputs, nil
}

/* executeAgentStep executes an agent step */
func (e *Engine) executeAgentStep(ctx context.Context, step *db.WorkflowStep, inputs map[string]interface{}) (map[string]interface{}, error) {
	if e.runtime == nil {
		return nil, fmt.Errorf("agent runtime not configured")
	}

	/* Extract agent_id and user_message from inputs */
	agentIDStr, ok := inputs["agent_id"].(string)
	if !ok {
		/* Try UUID type */
		if agentID, ok := inputs["agent_id"].(uuid.UUID); ok {
			agentIDStr = agentID.String()
		} else {
			return nil, fmt.Errorf("agent_id is required and must be a string or UUID")
		}
	}

	agentID, err := uuid.Parse(agentIDStr)
	if err != nil {
		return nil, fmt.Errorf("invalid agent_id format: %w", err)
	}

	userMessage, ok := inputs["user_message"].(string)
	if !ok {
		return nil, fmt.Errorf("user_message is required and must be a string")
	}

	/* Get or create session for this agent */
	session, err := e.queries.GetSession(ctx, uuid.Nil) /* We'll need to create or find a session */
	if err != nil || session == nil || session.AgentID != agentID {
		/* Create a new session for this workflow execution */
		session = &db.Session{
			AgentID: agentID,
			Metadata: make(map[string]interface{}),
		}
		if err := e.queries.CreateSession(ctx, session); err != nil {
			return nil, fmt.Errorf("failed to create session: %w", err)
		}
	}

	/* Execute agent via runtime */
	state, err := e.runtime.Execute(ctx, session.ID, userMessage)
	if err != nil {
		return nil, fmt.Errorf("agent execution failed: %w", err)
	}

	/* Return outputs */
	outputs := map[string]interface{}{
		"response":     state.FinalAnswer,
		"tokens_used":  state.TokensUsed,
		"tool_calls":   state.ToolCalls,
		"tool_results": state.ToolResults,
	}

	return outputs, nil
}

/* executeToolStep executes a tool step */
func (e *Engine) executeToolStep(ctx context.Context, step *db.WorkflowStep, inputs map[string]interface{}) (map[string]interface{}, error) {
	if e.toolRegistry == nil {
		return nil, fmt.Errorf("tool registry not configured")
	}

	/* Extract tool_name and args from inputs */
	toolName, ok := inputs["tool_name"].(string)
	if !ok {
		return nil, fmt.Errorf("tool_name is required and must be a string")
	}

	args, ok := inputs["args"].(map[string]interface{})
	if !ok {
		/* Try to extract from inputs directly (args might be at top level) */
		args = make(map[string]interface{})
		for k, v := range inputs {
			if k != "tool_name" {
				args[k] = v
			}
		}
	}

	/* Get tool from registry */
	tool, err := e.toolRegistry.Get(ctx, toolName)
	if err != nil {
		return nil, fmt.Errorf("tool not found: tool_name='%s', error=%w", toolName, err)
	}

	/* Execute tool */
	result, err := e.toolRegistry.Execute(ctx, tool, args)
	if err != nil {
		return nil, fmt.Errorf("tool execution failed: tool_name='%s', error=%w", toolName, err)
	}

	/* Return outputs */
	outputs := map[string]interface{}{
		"result": result,
		"tool":   toolName,
	}

	return outputs, nil
}

/* executeApprovalStep executes an approval step */
func (e *Engine) executeApprovalStep(ctx context.Context, workflowExecutionID uuid.UUID, stepExecutionID uuid.UUID, step *db.WorkflowStep, inputs map[string]interface{}) (map[string]interface{}, error) {
	hitlManager := NewHITLManager(e.queries, e.emailService, e.webhookService, e.baseURL)
	
	/* Set services if available */
	if e.emailService != nil {
		hitlManager.SetEmailService(e.emailService)
	}
	if e.webhookService != nil {
		hitlManager.SetWebhookService(e.webhookService)
	}
	
	return hitlManager.ExecuteApprovalStep(ctx, workflowExecutionID, stepExecutionID, step, inputs)
}

/* executeHTTPStep executes an HTTP step */
func (e *Engine) executeHTTPStep(ctx context.Context, step *db.WorkflowStep, inputs map[string]interface{}) (map[string]interface{}, error) {
	/* Extract HTTP configuration from inputs */
	url, ok := inputs["url"].(string)
	if !ok {
		return nil, fmt.Errorf("url is required and must be a string")
	}

	method := "GET"
	if m, ok := inputs["method"].(string); ok && m != "" {
		method = strings.ToUpper(m)
	}

	/* Validate method */
	validMethods := map[string]bool{"GET": true, "POST": true, "PUT": true, "PATCH": true, "DELETE": true}
	if !validMethods[method] {
		return nil, fmt.Errorf("invalid HTTP method: %s", method)
	}

	/* Extract headers */
	headers := make(map[string]string)
	if h, ok := inputs["headers"].(map[string]interface{}); ok {
		for k, v := range h {
			if str, ok := v.(string); ok {
				headers[k] = str
			}
		}
	}

	/* Extract body */
	var body []byte
	if b, ok := inputs["body"].(string); ok {
		body = []byte(b)
	} else if b, ok := inputs["body"].(map[string]interface{}); ok {
		/* Convert map to JSON */
		jsonBody, err := json.Marshal(b)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal body: %w", err)
		}
		body = jsonBody
		/* Set content type if not specified */
		if _, exists := headers["Content-Type"]; !exists {
			headers["Content-Type"] = "application/json"
		}
	}

	/* Create HTTP request */
	req, err := http.NewRequestWithContext(ctx, method, url, strings.NewReader(string(body)))
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP request: %w", err)
	}

	/* Set headers */
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	/* Execute request */
	client := &http.Client{
		Timeout: 30 * time.Second,
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	/* Read response body */
	var responseBody interface{}
	contentType := resp.Header.Get("Content-Type")
	if strings.Contains(contentType, "application/json") {
		var jsonBody map[string]interface{}
		decoder := json.NewDecoder(resp.Body)
		if err := decoder.Decode(&jsonBody); err != nil {
			/* If JSON decode fails, read as string */
			var bodyBytes []byte
			buf := make([]byte, 1024)
			for {
				n, readErr := resp.Body.Read(buf)
				if n > 0 {
					bodyBytes = append(bodyBytes, buf[:n]...)
				}
				if readErr != nil {
					break
				}
			}
			responseBody = string(bodyBytes)
		} else {
			responseBody = jsonBody
		}
	} else {
		/* Read as string */
		var bodyBytes []byte
		buf := make([]byte, 1024)
		for {
			n, err := resp.Body.Read(buf)
			if n > 0 {
				bodyBytes = append(bodyBytes, buf[:n]...)
			}
			if err != nil {
				break
			}
		}
		responseBody = string(bodyBytes)
	}

	/* Convert headers to map[string]string for JSON serialization */
	headerMap := make(map[string]string)
	for k, v := range resp.Header {
		if len(v) > 0 {
			headerMap[k] = v[0]
		}
	}

	/* Return outputs */
	outputs := map[string]interface{}{
		"status_code": resp.StatusCode,
		"status":      resp.Status,
		"headers":     headerMap,
		"body":        responseBody,
	}

	return outputs, nil
}

/* executeSQLStep executes a SQL step */
func (e *Engine) executeSQLStep(ctx context.Context, step *db.WorkflowStep, inputs map[string]interface{}) (map[string]interface{}, error) {
	/* Extract SQL query from inputs */
	query, ok := inputs["query"].(string)
	if !ok {
		return nil, fmt.Errorf("query is required and must be a string")
	}

	/* Validate query - only allow SELECT, EXPLAIN, SHOW, DESCRIBE */
	queryUpper := strings.TrimSpace(strings.ToUpper(query))
	if !strings.HasPrefix(queryUpper, "SELECT") &&
		!strings.HasPrefix(queryUpper, "EXPLAIN") &&
		!strings.HasPrefix(queryUpper, "SHOW") &&
		!strings.HasPrefix(queryUpper, "DESCRIBE") {
		return nil, fmt.Errorf("only SELECT, EXPLAIN, SHOW, and DESCRIBE queries are allowed")
	}

	/* Check for dangerous keywords */
	dangerous := []string{"DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE"}
	for _, keyword := range dangerous {
		if strings.Contains(queryUpper, " "+keyword+" ") {
			return nil, fmt.Errorf("query contains forbidden keyword: %s", keyword)
		}
	}

	/* Execute query */
	rows, err := e.queries.GetDB().QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("SQL query execution failed: %w", err)
	}
	defer rows.Close()

	/* Get column names */
	columns, err := rows.Columns()
	if err != nil {
		return nil, fmt.Errorf("failed to get columns: %w", err)
	}

	/* Convert results to JSON */
	var results []map[string]interface{}
	for rows.Next() {
		values := make([]interface{}, len(columns))
		valuePtrs := make([]interface{}, len(columns))
		for i := range values {
			valuePtrs[i] = &values[i]
		}

		if err := rows.Scan(valuePtrs...); err != nil {
			return nil, fmt.Errorf("failed to scan row: %w", err)
		}

		row := make(map[string]interface{})
		for i, col := range columns {
			row[col] = values[i]
		}
		results = append(results, row)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("row iteration failed: %w", err)
	}

	/* Return outputs */
	outputs := map[string]interface{}{
		"rows":    results,
		"count":   len(results),
		"columns": columns,
	}

	return outputs, nil
}

/* CompensateStep executes compensation step for rollback */
func (e *Engine) CompensateStep(ctx context.Context, stepExecution *db.WorkflowStepExecution) error {
	if stepExecution.Status != "completed" {
		return nil /* Nothing to compensate */
	}

	step, err := e.queries.GetWorkflowStepByID(ctx, stepExecution.WorkflowStepID)
	if err != nil {
		return fmt.Errorf("failed to get workflow step: %w", err)
	}

	if step.CompensationStepID == nil {
		return nil /* No compensation step */
	}

	compensationStep, err := e.queries.GetWorkflowStepByID(ctx, *step.CompensationStepID)
	if err != nil {
		return fmt.Errorf("failed to get compensation step: %w", err)
	}

	/* Execute compensation step */
	_, err = e.ExecuteStep(ctx, stepExecution.WorkflowExecutionID, compensationStep, stepExecution.Outputs)
	if err != nil {
		return fmt.Errorf("compensation step failed: %w", err)
	}

	stepExecution.Status = "compensated"
	if err := e.queries.UpdateWorkflowStepExecution(ctx, stepExecution); err != nil {
		return fmt.Errorf("failed to update compensated step execution: %w", err)
	}

	return nil
}

/* getMaxRetries extracts max retries from retry config */
func getMaxRetries(retryConfig db.JSONBMap) int {
	if maxRetries, ok := retryConfig["max_retries"].(float64); ok {
		return int(maxRetries)
	}
	return 3 /* Default */
}

/* buildDAG builds a dependency graph and returns steps in topological order */
func (e *Engine) buildDAG(steps []db.WorkflowStep) ([]uuid.UUID, error) {
	/* Create adjacency list for dependencies */
	dependencies := make(map[uuid.UUID][]uuid.UUID)
	stepMap := make(map[uuid.UUID]*db.WorkflowStep)
	inDegree := make(map[uuid.UUID]int)

	/* Initialize maps */
	for i := range steps {
		stepMap[steps[i].ID] = &steps[i]
		dependencies[steps[i].ID] = []uuid.UUID{}
		inDegree[steps[i].ID] = 0
	}

	/* Build dependency graph */
	for i := range steps {
		if steps[i].Dependencies != nil && len(steps[i].Dependencies) > 0 {
			/* Dependencies is a PostgreSQL StringArray */
			for _, depStr := range steps[i].Dependencies {
				depID, err := uuid.Parse(depStr)
				if err == nil {
					dependencies[depID] = append(dependencies[depID], steps[i].ID)
					inDegree[steps[i].ID]++
				}
			}
		}
	}

	/* Topological sort using Kahn's algorithm */
	var queue []uuid.UUID
	var result []uuid.UUID

	/* Add all nodes with in-degree 0 to queue */
	for stepID, degree := range inDegree {
		if degree == 0 {
			queue = append(queue, stepID)
		}
	}

	/* Process queue */
	for len(queue) > 0 {
		/* Dequeue */
		current := queue[0]
		queue = queue[1:]
		result = append(result, current)

		/* Reduce in-degree of dependent nodes */
		for _, dependent := range dependencies[current] {
			inDegree[dependent]--
			if inDegree[dependent] == 0 {
				queue = append(queue, dependent)
			}
		}
	}

	/* Check for cycles */
	if len(result) != len(steps) {
		return nil, fmt.Errorf("dependency graph contains cycles")
	}

	return result, nil
}

/* buildStepInputs builds inputs for a step from workflow inputs and previous step outputs */
func (e *Engine) buildStepInputs(step *db.WorkflowStep, workflowInputs map[string]interface{}, stepOutputs map[uuid.UUID]map[string]interface{}) map[string]interface{} {
	stepInputs := make(map[string]interface{})

	/* Start with workflow inputs */
	for k, v := range workflowInputs {
		stepInputs[k] = v
	}

	/* Override with step-specific inputs if defined */
	if step.Inputs != nil {
		/* Inputs is JSONBMap which is map[string]interface{} */
		for k, v := range step.Inputs {
			stepInputs[k] = v
		}
	}

	/* Resolve dependencies - if step has dependencies, merge their outputs */
	if step.Dependencies != nil && len(step.Dependencies) > 0 {
		for _, depStr := range step.Dependencies {
			depID, err := uuid.Parse(depStr)
			if err != nil {
				continue
			}

			/* Merge outputs from dependency */
			if outputs, exists := stepOutputs[depID]; exists {
				for k, v := range outputs {
					/* Use step name as prefix if available */
					key := fmt.Sprintf("%s_%s", depID.String()[:8], k)
					stepInputs[key] = v
				}
			}
		}
	}

	return stepInputs
}
