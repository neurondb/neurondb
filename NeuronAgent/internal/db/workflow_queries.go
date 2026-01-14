/*-------------------------------------------------------------------------
 *
 * workflow_queries.go
 *    Database queries for workflow engine
 *
 * Provides database query functions for workflows, steps, executions, and schedules.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/db/workflow_queries.go
 *
 *-------------------------------------------------------------------------
 */

package db

import (
	"context"
	"database/sql"
	"fmt"
	"time"

	"github.com/google/uuid"
)

/* Workflow queries */
const (
	createWorkflowQuery = `
		INSERT INTO neurondb_agent.workflows (name, dag_definition, status)
		VALUES ($1, $2::jsonb, $3)
		RETURNING id, created_at, updated_at`

	getWorkflowByIDQuery = `SELECT * FROM neurondb_agent.workflows WHERE id = $1`

	listWorkflowsQuery = `
		SELECT * FROM neurondb_agent.workflows 
		WHERE ($1::text IS NULL OR status = $1)
		ORDER BY created_at DESC`

	updateWorkflowQuery = `
		UPDATE neurondb_agent.workflows 
		SET name = $2, dag_definition = $3::jsonb, status = $4, updated_at = NOW()
		WHERE id = $1
		RETURNING updated_at`

	deleteWorkflowQuery = `DELETE FROM neurondb_agent.workflows WHERE id = $1`

	listWorkflowsByStatusQuery = `
		SELECT * FROM neurondb_agent.workflows 
		WHERE status = $1
		ORDER BY created_at DESC`
)

/* Workflow step queries */
const (
	createWorkflowStepQuery = `
		INSERT INTO neurondb_agent.workflow_steps 
		(workflow_id, step_name, step_type, inputs, outputs, dependencies, retry_config, idempotency_key, compensation_step_id)
		VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6, $7::jsonb, $8, $9)
		RETURNING id, created_at, updated_at`

	getWorkflowStepByIDQuery = `SELECT * FROM neurondb_agent.workflow_steps WHERE id = $1`

	listWorkflowStepsQuery = `
		SELECT * FROM neurondb_agent.workflow_steps 
		WHERE workflow_id = $1 
		ORDER BY step_name`
)

/* Workflow execution queries */
const (
	createWorkflowExecutionQuery = `
		INSERT INTO neurondb_agent.workflow_executions 
		(workflow_id, status, trigger_type, trigger_data, inputs)
		VALUES ($1, $2, $3, $4::jsonb, $5::jsonb)
		RETURNING id, created_at, updated_at`

	updateWorkflowExecutionQuery = `
		UPDATE neurondb_agent.workflow_executions 
		SET status = $2, outputs = $3::jsonb, error_message = $4, updated_at = NOW()
		WHERE id = $1
		RETURNING updated_at`

	getWorkflowExecutionByIDQuery = `SELECT * FROM neurondb_agent.workflow_executions WHERE id = $1`

	listWorkflowExecutionsQuery = `
		SELECT * FROM neurondb_agent.workflow_executions 
		WHERE workflow_id = $1 
		ORDER BY created_at DESC`

	listWorkflowExecutionsByStatusQuery = `
		SELECT * FROM neurondb_agent.workflow_executions 
		WHERE workflow_id = $1 AND status = $2
		ORDER BY created_at DESC`
)

/* Workflow step execution queries */
const (
	createWorkflowStepExecutionQuery = `
		INSERT INTO neurondb_agent.workflow_step_executions 
		(workflow_execution_id, workflow_step_id, status, inputs, idempotency_key, started_at)
		VALUES ($1, $2, $3, $4::jsonb, $5, $6)
		RETURNING id, created_at, updated_at`

	updateWorkflowStepExecutionQuery = `
		UPDATE neurondb_agent.workflow_step_executions 
		SET status = $2, outputs = $3::jsonb, error_message = $4, retry_count = $5, completed_at = NOW(), updated_at = NOW()
		WHERE id = $1
		RETURNING updated_at`

	getWorkflowStepExecutionByIDQuery = `SELECT * FROM neurondb_agent.workflow_step_executions WHERE id = $1`

	getWorkflowStepExecutionByIdempotencyKeyQuery = `
		SELECT * FROM neurondb_agent.workflow_step_executions 
		WHERE idempotency_key = $1 AND status = 'completed'`
)

/* Workflow schedule queries */
const (
	createWorkflowScheduleQuery = `
		INSERT INTO neurondb_agent.workflow_schedules 
		(workflow_id, cron_expression, timezone, enabled, next_run_at)
		VALUES ($1, $2, $3, $4, $5)
		ON CONFLICT (workflow_id) 
		DO UPDATE SET cron_expression = EXCLUDED.cron_expression, 
		              timezone = EXCLUDED.timezone, 
		              enabled = EXCLUDED.enabled, 
		              next_run_at = EXCLUDED.next_run_at, 
		              updated_at = NOW()
		RETURNING id, created_at, updated_at`

	getWorkflowScheduleByWorkflowIDQuery = `SELECT * FROM neurondb_agent.workflow_schedules WHERE workflow_id = $1`

	getWorkflowScheduleByIDQuery = `SELECT * FROM neurondb_agent.workflow_schedules WHERE id = $1`

	updateWorkflowScheduleQuery = `
		UPDATE neurondb_agent.workflow_schedules 
		SET cron_expression = $2, timezone = $3, enabled = $4, next_run_at = $5, last_run_at = $6, updated_at = NOW()
		WHERE id = $1
		RETURNING updated_at`

	deleteWorkflowScheduleQuery = `DELETE FROM neurondb_agent.workflow_schedules WHERE id = $1`

	deleteWorkflowScheduleByWorkflowIDQuery = `DELETE FROM neurondb_agent.workflow_schedules WHERE workflow_id = $1`

	listWorkflowSchedulesQuery = `
		SELECT * FROM neurondb_agent.workflow_schedules 
		ORDER BY created_at DESC`

	listWorkflowSchedulesByNextRunQuery = `
		SELECT * FROM neurondb_agent.workflow_schedules 
		WHERE enabled = true AND next_run_at IS NOT NULL AND next_run_at <= $1
		ORDER BY next_run_at ASC`
)

/* Workflow methods */
func (q *Queries) CreateWorkflow(ctx context.Context, workflow *Workflow) error {
	dagDefValue, err := workflow.DAGDefinition.Value()
	if err != nil {
		return fmt.Errorf("failed to convert dag_definition: %w", err)
	}

	params := []interface{}{workflow.Name, dagDefValue, workflow.Status}
	err = q.DB.GetContext(ctx, workflow, createWorkflowQuery, params...)
	if err != nil {
		return fmt.Errorf("workflow creation failed: %w", err)
	}
	return nil
}

func (q *Queries) GetWorkflowByID(ctx context.Context, id uuid.UUID) (*Workflow, error) {
	var workflow Workflow
	err := q.DB.GetContext(ctx, &workflow, getWorkflowByIDQuery, id)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("workflow not found: %w", err)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get workflow: %w", err)
	}
	return &workflow, nil
}

func (q *Queries) ListWorkflows(ctx context.Context) ([]Workflow, error) {
	var workflows []Workflow
	err := q.DB.SelectContext(ctx, &workflows, listWorkflowsQuery, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to list workflows: %w", err)
	}
	return workflows, nil
}

func (q *Queries) ListWorkflowsByStatus(ctx context.Context, status string) ([]Workflow, error) {
	var workflows []Workflow
	err := q.DB.SelectContext(ctx, &workflows, listWorkflowsByStatusQuery, status)
	if err != nil {
		return nil, fmt.Errorf("failed to list workflows by status: %w", err)
	}
	return workflows, nil
}

func (q *Queries) UpdateWorkflow(ctx context.Context, workflow *Workflow) error {
	dagDefValue, err := workflow.DAGDefinition.Value()
	if err != nil {
		return fmt.Errorf("failed to convert dag_definition: %w", err)
	}

	params := []interface{}{workflow.ID, workflow.Name, dagDefValue, workflow.Status}
	err = q.DB.GetContext(ctx, workflow, updateWorkflowQuery, params...)
	if err != nil {
		return fmt.Errorf("workflow update failed: %w", err)
	}
	return nil
}

func (q *Queries) DeleteWorkflow(ctx context.Context, id uuid.UUID) error {
	result, err := q.DB.ExecContext(ctx, deleteWorkflowQuery, id)
	if err != nil {
		return fmt.Errorf("workflow deletion failed: %w", err)
	}
	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}
	if rowsAffected == 0 {
		return fmt.Errorf("workflow not found")
	}
	return nil
}

/* Workflow step methods */
func (q *Queries) CreateWorkflowStep(ctx context.Context, step *WorkflowStep) error {
	inputsValue, err := step.Inputs.Value()
	if err != nil {
		return fmt.Errorf("failed to convert inputs: %w", err)
	}
	outputsValue, err := step.Outputs.Value()
	if err != nil {
		return fmt.Errorf("failed to convert outputs: %w", err)
	}
	retryConfigValue, err := step.RetryConfig.Value()
	if err != nil {
		return fmt.Errorf("failed to convert retry_config: %w", err)
	}

	params := []interface{}{
		step.WorkflowID, step.StepName, step.StepType,
		inputsValue, outputsValue, step.Dependencies,
		retryConfigValue, step.IdempotencyKey, step.CompensationStepID,
	}
	err = q.DB.GetContext(ctx, step, createWorkflowStepQuery, params...)
	if err != nil {
		return fmt.Errorf("workflow step creation failed: %w", err)
	}
	return nil
}

func (q *Queries) GetWorkflowStepByID(ctx context.Context, id uuid.UUID) (*WorkflowStep, error) {
	var step WorkflowStep
	err := q.DB.GetContext(ctx, &step, getWorkflowStepByIDQuery, id)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("workflow step not found: %w", err)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get workflow step: %w", err)
	}
	return &step, nil
}

func (q *Queries) ListWorkflowSteps(ctx context.Context, workflowID uuid.UUID) ([]WorkflowStep, error) {
	var steps []WorkflowStep
	err := q.DB.SelectContext(ctx, &steps, listWorkflowStepsQuery, workflowID)
	if err != nil {
		return nil, fmt.Errorf("failed to list workflow steps: %w", err)
	}
	return steps, nil
}

/* Workflow execution methods */
func (q *Queries) CreateWorkflowExecution(ctx context.Context, execution *WorkflowExecution) error {
	triggerDataValue, err := execution.TriggerData.Value()
	if err != nil {
		return fmt.Errorf("failed to convert trigger_data: %w", err)
	}
	inputsValue, err := execution.Inputs.Value()
	if err != nil {
		return fmt.Errorf("failed to convert inputs: %w", err)
	}

	params := []interface{}{execution.WorkflowID, execution.Status, execution.TriggerType, triggerDataValue, inputsValue}
	err = q.DB.GetContext(ctx, execution, createWorkflowExecutionQuery, params...)
	if err != nil {
		return fmt.Errorf("workflow execution creation failed: %w", err)
	}
	return nil
}

func (q *Queries) UpdateWorkflowExecution(ctx context.Context, execution *WorkflowExecution) error {
	outputsValue, err := execution.Outputs.Value()
	if err != nil {
		return fmt.Errorf("failed to convert outputs: %w", err)
	}

	params := []interface{}{execution.ID, execution.Status, outputsValue, execution.ErrorMessage}
	err = q.DB.GetContext(ctx, execution, updateWorkflowExecutionQuery, params...)
	if err != nil {
		return fmt.Errorf("workflow execution update failed: %w", err)
	}
	return nil
}

func (q *Queries) GetWorkflowExecutionByID(ctx context.Context, id uuid.UUID) (*WorkflowExecution, error) {
	var execution WorkflowExecution
	err := q.DB.GetContext(ctx, &execution, getWorkflowExecutionByIDQuery, id)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("workflow execution not found: %w", err)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get workflow execution: %w", err)
	}
	return &execution, nil
}

func (q *Queries) ListWorkflowExecutions(ctx context.Context, workflowID uuid.UUID) ([]WorkflowExecution, error) {
	var executions []WorkflowExecution
	err := q.DB.SelectContext(ctx, &executions, listWorkflowExecutionsQuery, workflowID)
	if err != nil {
		return nil, fmt.Errorf("failed to list workflow executions: %w", err)
	}
	return executions, nil
}

func (q *Queries) ListWorkflowExecutionsByStatus(ctx context.Context, workflowID uuid.UUID, status string) ([]WorkflowExecution, error) {
	var executions []WorkflowExecution
	err := q.DB.SelectContext(ctx, &executions, listWorkflowExecutionsByStatusQuery, workflowID, status)
	if err != nil {
		return nil, fmt.Errorf("failed to list workflow executions by status: %w", err)
	}
	return executions, nil
}

/* Workflow step execution methods */
func (q *Queries) CreateWorkflowStepExecution(ctx context.Context, stepExecution *WorkflowStepExecution) error {
	inputsValue, err := stepExecution.Inputs.Value()
	if err != nil {
		return fmt.Errorf("failed to convert inputs: %w", err)
	}

	params := []interface{}{stepExecution.WorkflowExecutionID, stepExecution.WorkflowStepID, stepExecution.Status, inputsValue, stepExecution.IdempotencyKey, stepExecution.StartedAt}
	err = q.DB.GetContext(ctx, stepExecution, createWorkflowStepExecutionQuery, params...)
	if err != nil {
		return fmt.Errorf("workflow step execution creation failed: %w", err)
	}
	return nil
}

func (q *Queries) UpdateWorkflowStepExecution(ctx context.Context, stepExecution *WorkflowStepExecution) error {
	outputsValue, err := stepExecution.Outputs.Value()
	if err != nil {
		return fmt.Errorf("failed to convert outputs: %w", err)
	}

	params := []interface{}{stepExecution.ID, stepExecution.Status, outputsValue, stepExecution.ErrorMessage, stepExecution.RetryCount}
	err = q.DB.GetContext(ctx, stepExecution, updateWorkflowStepExecutionQuery, params...)
	if err != nil {
		return fmt.Errorf("workflow step execution update failed: %w", err)
	}
	return nil
}

func (q *Queries) GetWorkflowStepExecutionByID(ctx context.Context, id uuid.UUID) (*WorkflowStepExecution, error) {
	var stepExecution WorkflowStepExecution
	err := q.DB.GetContext(ctx, &stepExecution, getWorkflowStepExecutionByIDQuery, id)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("workflow step execution not found: %w", err)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get workflow step execution: %w", err)
	}
	return &stepExecution, nil
}

func (q *Queries) GetWorkflowStepExecutionByIdempotencyKey(ctx context.Context, key string) (*WorkflowStepExecution, error) {
	var stepExecution WorkflowStepExecution
	err := q.DB.GetContext(ctx, &stepExecution, getWorkflowStepExecutionByIdempotencyKeyQuery, key)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get workflow step execution by idempotency key: %w", err)
	}
	return &stepExecution, nil
}

/* Workflow schedule methods */
func (q *Queries) CreateWorkflowSchedule(ctx context.Context, schedule *WorkflowSchedule) error {
	params := []interface{}{schedule.WorkflowID, schedule.CronExpression, schedule.Timezone, schedule.Enabled, schedule.NextRunAt}
	err := q.DB.GetContext(ctx, schedule, createWorkflowScheduleQuery, params...)
	if err != nil {
		return fmt.Errorf("workflow schedule creation failed: %w", err)
	}
	return nil
}

func (q *Queries) GetWorkflowScheduleByWorkflowID(ctx context.Context, workflowID uuid.UUID) (*WorkflowSchedule, error) {
	var schedule WorkflowSchedule
	err := q.DB.GetContext(ctx, &schedule, getWorkflowScheduleByWorkflowIDQuery, workflowID)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("workflow schedule not found: %w", err)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get workflow schedule: %w", err)
	}
	return &schedule, nil
}

func (q *Queries) GetWorkflowScheduleByID(ctx context.Context, id uuid.UUID) (*WorkflowSchedule, error) {
	var schedule WorkflowSchedule
	err := q.DB.GetContext(ctx, &schedule, getWorkflowScheduleByIDQuery, id)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("workflow schedule not found: %w", err)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get workflow schedule: %w", err)
	}
	return &schedule, nil
}

func (q *Queries) UpdateWorkflowSchedule(ctx context.Context, schedule *WorkflowSchedule) error {
	params := []interface{}{schedule.ID, schedule.CronExpression, schedule.Timezone, schedule.Enabled, schedule.NextRunAt, schedule.LastRunAt}
	err := q.DB.GetContext(ctx, schedule, updateWorkflowScheduleQuery, params...)
	if err != nil {
		return fmt.Errorf("workflow schedule update failed: %w", err)
	}
	return nil
}

func (q *Queries) DeleteWorkflowSchedule(ctx context.Context, id uuid.UUID) error {
	result, err := q.DB.ExecContext(ctx, deleteWorkflowScheduleQuery, id)
	if err != nil {
		return fmt.Errorf("workflow schedule deletion failed: %w", err)
	}
	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}
	if rowsAffected == 0 {
		return fmt.Errorf("workflow schedule not found")
	}
	return nil
}

func (q *Queries) DeleteWorkflowScheduleByWorkflowID(ctx context.Context, workflowID uuid.UUID) error {
	result, err := q.DB.ExecContext(ctx, deleteWorkflowScheduleByWorkflowIDQuery, workflowID)
	if err != nil {
		return fmt.Errorf("workflow schedule deletion failed: %w", err)
	}
	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}
	if rowsAffected == 0 {
		return fmt.Errorf("workflow schedule not found")
	}
	return nil
}

func (q *Queries) ListWorkflowSchedules(ctx context.Context) ([]WorkflowSchedule, error) {
	var schedules []WorkflowSchedule
	err := q.DB.SelectContext(ctx, &schedules, listWorkflowSchedulesQuery)
	if err != nil {
		return nil, fmt.Errorf("failed to list workflow schedules: %w", err)
	}
	return schedules, nil
}

func (q *Queries) ListWorkflowSchedulesByNextRun(ctx context.Context, beforeTime time.Time) ([]WorkflowSchedule, error) {
	var schedules []WorkflowSchedule
	err := q.DB.SelectContext(ctx, &schedules, listWorkflowSchedulesByNextRunQuery, beforeTime)
	if err != nil {
		return nil, fmt.Errorf("failed to list workflow schedules by next run: %w", err)
	}
	return schedules, nil
}
