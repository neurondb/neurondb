/*-------------------------------------------------------------------------
 *
 * eval_models.go
 *    Evaluation framework models
 *
 * Defines data structures for evaluation tasks, runs, and results.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/db/eval_models.go
 *
 *-------------------------------------------------------------------------
 */

package db

import (
	"time"

	"github.com/google/uuid"
)

type EvalTask struct {
	ID                   uuid.UUID `db:"id"`
	TaskType             string    `db:"task_type"`
	Input                string    `db:"input"`
	ExpectedOutput       *string   `db:"expected_output"`
	ExpectedToolSequence JSONBMap  `db:"expected_tool_sequence"`
	GoldenSQLSideEffects JSONBMap  `db:"golden_sql_side_effects"`
	Metadata             JSONBMap  `db:"metadata"`
	CreatedAt            time.Time `db:"created_at"`
	UpdatedAt            time.Time `db:"updated_at"`
}

type EvalRun struct {
	ID             uuid.UUID  `db:"id"`
	DatasetVersion string     `db:"dataset_version"`
	AgentID        *uuid.UUID `db:"agent_id"`
	StartedAt      time.Time  `db:"started_at"`
	CompletedAt    *time.Time `db:"completed_at"`
	Score          *float64   `db:"score"`
	TotalTasks     int        `db:"total_tasks"`
	PassedTasks    int        `db:"passed_tasks"`
	FailedTasks    int        `db:"failed_tasks"`
	Metadata       JSONBMap   `db:"metadata"`
	CreatedAt      time.Time  `db:"created_at"`
}

type EvalTaskResult struct {
	ID                   uuid.UUID  `db:"id"`
	EvalRunID            uuid.UUID  `db:"eval_run_id"`
	EvalTaskID           uuid.UUID  `db:"eval_task_id"`
	SessionID            *uuid.UUID `db:"session_id"`
	Passed               bool       `db:"passed"`
	ActualOutput         *string    `db:"actual_output"`
	ActualToolSequence   JSONBMap   `db:"actual_tool_sequence"`
	ActualSQLSideEffects JSONBMap   `db:"actual_sql_side_effects"`
	Score                *float64   `db:"score"`
	ErrorMessage         *string    `db:"error_message"`
	Metadata             JSONBMap   `db:"metadata"`
	CreatedAt            time.Time  `db:"created_at"`
}

type EvalRetrievalResult struct {
	ID               uuid.UUID `db:"id"`
	EvalTaskResultID uuid.UUID `db:"eval_task_result_id"`
	RecallAtK        *float64  `db:"recall_at_k"`
	MRR              *float64  `db:"mrr"`
	GroundingPassed  bool      `db:"grounding_passed"`
	RetrievedChunks  JSONBMap  `db:"retrieved_chunks"`
	RelevantChunks   JSONBMap  `db:"relevant_chunks"`
	CreatedAt        time.Time `db:"created_at"`
}
