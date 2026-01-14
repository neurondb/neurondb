/*-------------------------------------------------------------------------
 *
 * workflow_models.go
 *    Workflow engine models
 *
 * Defines data structures for workflows, steps, executions, and schedules.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/db/workflow_models.go
 *
 *-------------------------------------------------------------------------
 */

package db

import (
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"
)

type Workflow struct {
	ID            uuid.UUID `db:"id"`
	Name          string    `db:"name"`
	DAGDefinition JSONBMap  `db:"dag_definition"`
	Status        string    `db:"status"`
	CreatedAt     time.Time `db:"created_at"`
	UpdatedAt     time.Time `db:"updated_at"`
}

type WorkflowStep struct {
	ID                 uuid.UUID      `db:"id"`
	WorkflowID         uuid.UUID      `db:"workflow_id"`
	StepName           string         `db:"step_name"`
	StepType           string         `db:"step_type"`
	Inputs             JSONBMap       `db:"inputs"`
	Outputs            JSONBMap       `db:"outputs"`
	Dependencies       pq.StringArray `db:"dependencies"`
	RetryConfig        JSONBMap       `db:"retry_config"`
	IdempotencyKey     *string        `db:"idempotency_key"`
	CompensationStepID *uuid.UUID     `db:"compensation_step_id"`
	CreatedAt          time.Time      `db:"created_at"`
	UpdatedAt          time.Time      `db:"updated_at"`
}

type WorkflowExecution struct {
	ID           uuid.UUID  `db:"id"`
	WorkflowID   uuid.UUID  `db:"workflow_id"`
	Status       string     `db:"status"`
	TriggerType  string     `db:"trigger_type"`
	TriggerData  JSONBMap   `db:"trigger_data"`
	Inputs       JSONBMap   `db:"inputs"`
	Outputs      JSONBMap   `db:"outputs"`
	ErrorMessage *string    `db:"error_message"`
	StartedAt    *time.Time `db:"started_at"`
	CompletedAt  *time.Time `db:"completed_at"`
	CreatedAt    time.Time  `db:"created_at"`
	UpdatedAt    time.Time  `db:"updated_at"`
}

type WorkflowStepExecution struct {
	ID                  uuid.UUID  `db:"id"`
	WorkflowExecutionID uuid.UUID  `db:"workflow_execution_id"`
	WorkflowStepID      uuid.UUID  `db:"workflow_step_id"`
	Status              string     `db:"status"`
	Inputs              JSONBMap   `db:"inputs"`
	Outputs             JSONBMap   `db:"outputs"`
	ErrorMessage        *string    `db:"error_message"`
	RetryCount          int        `db:"retry_count"`
	IdempotencyKey      *string    `db:"idempotency_key"`
	StartedAt           *time.Time `db:"started_at"`
	CompletedAt         *time.Time `db:"completed_at"`
	CreatedAt           time.Time  `db:"created_at"`
	UpdatedAt           time.Time  `db:"updated_at"`
}

type WorkflowSchedule struct {
	ID             uuid.UUID  `db:"id"`
	WorkflowID     uuid.UUID  `db:"workflow_id"`
	CronExpression string     `db:"cron_expression"`
	Timezone       string     `db:"timezone"`
	Enabled        bool       `db:"enabled"`
	NextRunAt      *time.Time `db:"next_run_at"`
	LastRunAt      *time.Time `db:"last_run_at"`
	CreatedAt      time.Time  `db:"created_at"`
	UpdatedAt      time.Time  `db:"updated_at"`
}
