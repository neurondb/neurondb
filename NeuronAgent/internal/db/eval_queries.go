/*-------------------------------------------------------------------------
 *
 * eval_queries.go
 *    Database queries for evaluation framework
 *
 * Provides database query functions for eval tasks, runs, and results.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/db/eval_queries.go
 *
 *-------------------------------------------------------------------------
 */

package db

import (
	"context"
	"database/sql"
	"fmt"

	"github.com/google/uuid"
)

/* Eval task queries */
const (
	createEvalTaskQuery = `
		INSERT INTO neurondb_agent.eval_tasks 
		(task_type, input, expected_output, expected_tool_sequence, golden_sql_side_effects, metadata)
		VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6::jsonb)
		RETURNING id, created_at, updated_at`

	getEvalTaskByIDQuery = `SELECT * FROM neurondb_agent.eval_tasks WHERE id = $1`

	listEvalTasksQuery = `
		SELECT * FROM neurondb_agent.eval_tasks 
		WHERE ($1::text IS NULL OR task_type = $1)
		ORDER BY created_at DESC
		LIMIT $2 OFFSET $3`
)

/* Eval run queries */
const (
	createEvalRunQuery = `
		INSERT INTO neurondb_agent.eval_runs 
		(dataset_version, agent_id, total_tasks)
		VALUES ($1, $2, $3)
		RETURNING id, started_at, created_at`

	updateEvalRunQuery = `
		UPDATE neurondb_agent.eval_runs 
		SET completed_at = NOW(), score = $2, passed_tasks = $3, failed_tasks = $4
		WHERE id = $1
		RETURNING completed_at`

	getEvalRunByIDQuery = `SELECT * FROM neurondb_agent.eval_runs WHERE id = $1`

	listEvalRunsQuery = `
		SELECT * FROM neurondb_agent.eval_runs 
		WHERE ($1::text IS NULL OR dataset_version = $1)
		AND ($2::uuid IS NULL OR agent_id = $2)
		ORDER BY started_at DESC
		LIMIT $3 OFFSET $4`
)

/* Eval task result queries */
const (
	createEvalTaskResultQuery = `
		INSERT INTO neurondb_agent.eval_task_results 
		(eval_run_id, eval_task_id, session_id, passed, actual_output, actual_tool_sequence, actual_sql_side_effects, score, error_message, metadata)
		VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8, $9, $10::jsonb)
		RETURNING id, created_at`

	getEvalTaskResultsByRunQuery = `
		SELECT * FROM neurondb_agent.eval_task_results 
		WHERE eval_run_id = $1
		ORDER BY created_at`
)

/* Eval retrieval result queries */
const (
	createEvalRetrievalResultQuery = `
		INSERT INTO neurondb_agent.eval_retrieval_results 
		(eval_task_result_id, recall_at_k, mrr, grounding_passed, retrieved_chunks, relevant_chunks)
		VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb)
		RETURNING id, created_at`
)

/* Eval task methods */
func (q *Queries) CreateEvalTask(ctx context.Context, task *EvalTask) error {
	expectedToolSeqValue, err := task.ExpectedToolSequence.Value()
	if err != nil {
		return fmt.Errorf("failed to convert expected_tool_sequence: %w", err)
	}
	goldenSQLValue, err := task.GoldenSQLSideEffects.Value()
	if err != nil {
		return fmt.Errorf("failed to convert golden_sql_side_effects: %w", err)
	}
	metadataValue, err := task.Metadata.Value()
	if err != nil {
		return fmt.Errorf("failed to convert metadata: %w", err)
	}

	params := []interface{}{task.TaskType, task.Input, task.ExpectedOutput, expectedToolSeqValue, goldenSQLValue, metadataValue}
	err = q.DB.GetContext(ctx, task, createEvalTaskQuery, params...)
	if err != nil {
		return fmt.Errorf("eval task creation failed: %w", err)
	}
	return nil
}

func (q *Queries) GetEvalTaskByID(ctx context.Context, id uuid.UUID) (*EvalTask, error) {
	var task EvalTask
	err := q.DB.GetContext(ctx, &task, getEvalTaskByIDQuery, id)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("eval task not found: %w", err)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get eval task: %w", err)
	}
	return &task, nil
}

func (q *Queries) ListEvalTasks(ctx context.Context, taskType *string, limit, offset int) ([]EvalTask, error) {
	var tasks []EvalTask
	err := q.DB.SelectContext(ctx, &tasks, listEvalTasksQuery, taskType, limit, offset)
	if err != nil {
		return nil, fmt.Errorf("failed to list eval tasks: %w", err)
	}
	return tasks, nil
}

/* Eval run methods */
func (q *Queries) CreateEvalRun(ctx context.Context, run *EvalRun) error {
	params := []interface{}{run.DatasetVersion, run.AgentID, run.TotalTasks}
	err := q.DB.GetContext(ctx, run, createEvalRunQuery, params...)
	if err != nil {
		return fmt.Errorf("eval run creation failed: %w", err)
	}
	return nil
}

func (q *Queries) UpdateEvalRun(ctx context.Context, run *EvalRun) error {
	params := []interface{}{run.ID, run.Score, run.PassedTasks, run.FailedTasks}
	err := q.DB.GetContext(ctx, run, updateEvalRunQuery, params...)
	if err != nil {
		return fmt.Errorf("eval run update failed: %w", err)
	}
	return nil
}

func (q *Queries) GetEvalRunByID(ctx context.Context, id uuid.UUID) (*EvalRun, error) {
	var run EvalRun
	err := q.DB.GetContext(ctx, &run, getEvalRunByIDQuery, id)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("eval run not found: %w", err)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get eval run: %w", err)
	}
	return &run, nil
}

func (q *Queries) ListEvalRuns(ctx context.Context, datasetVersion *string, agentID *uuid.UUID, limit, offset int) ([]EvalRun, error) {
	var runs []EvalRun
	err := q.DB.SelectContext(ctx, &runs, listEvalRunsQuery, datasetVersion, agentID, limit, offset)
	if err != nil {
		return nil, fmt.Errorf("failed to list eval runs: %w", err)
	}
	return runs, nil
}

/* Eval task result methods */
func (q *Queries) CreateEvalTaskResult(ctx context.Context, result *EvalTaskResult) error {
	actualToolSeqValue, err := result.ActualToolSequence.Value()
	if err != nil {
		return fmt.Errorf("failed to convert actual_tool_sequence: %w", err)
	}
	actualSQLValue, err := result.ActualSQLSideEffects.Value()
	if err != nil {
		return fmt.Errorf("failed to convert actual_sql_side_effects: %w", err)
	}
	metadataValue, err := result.Metadata.Value()
	if err != nil {
		return fmt.Errorf("failed to convert metadata: %w", err)
	}

	params := []interface{}{
		result.EvalRunID, result.EvalTaskID, result.SessionID, result.Passed,
		result.ActualOutput, actualToolSeqValue, actualSQLValue,
		result.Score, result.ErrorMessage, metadataValue,
	}
	err = q.DB.GetContext(ctx, result, createEvalTaskResultQuery, params...)
	if err != nil {
		return fmt.Errorf("eval task result creation failed: %w", err)
	}
	return nil
}

func (q *Queries) GetEvalTaskResultsByRun(ctx context.Context, runID uuid.UUID) ([]EvalTaskResult, error) {
	var results []EvalTaskResult
	err := q.DB.SelectContext(ctx, &results, getEvalTaskResultsByRunQuery, runID)
	if err != nil {
		return nil, fmt.Errorf("failed to get eval task results: %w", err)
	}
	return results, nil
}

/* Eval retrieval result methods */
func (q *Queries) CreateEvalRetrievalResult(ctx context.Context, result *EvalRetrievalResult) error {
	retrievedChunksValue, err := result.RetrievedChunks.Value()
	if err != nil {
		return fmt.Errorf("failed to convert retrieved_chunks: %w", err)
	}
	relevantChunksValue, err := result.RelevantChunks.Value()
	if err != nil {
		return fmt.Errorf("failed to convert relevant_chunks: %w", err)
	}

	params := []interface{}{
		result.EvalTaskResultID, result.RecallAtK, result.MRR,
		result.GroundingPassed, retrievedChunksValue, relevantChunksValue,
	}
	err = q.DB.GetContext(ctx, result, createEvalRetrievalResultQuery, params...)
	if err != nil {
		return fmt.Errorf("eval retrieval result creation failed: %w", err)
	}
	return nil
}
