/*-------------------------------------------------------------------------
 *
 * approval_models.go
 *    Approval request models
 *
 * Defines data structures for approval requests.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/db/approval_models.go
 *
 *-------------------------------------------------------------------------
 */

package db

import (
	"time"

	"github.com/google/uuid"
)

type ApprovalRequest struct {
	ID                  uuid.UUID  `db:"id"`
	WorkflowExecutionID *uuid.UUID `db:"workflow_execution_id"`
	StepExecutionID     *uuid.UUID `db:"step_execution_id"`
	AgentID             *uuid.UUID `db:"agent_id"`
	SessionID           *uuid.UUID `db:"session_id"`
	ApprovalType        string     `db:"approval_type"`
	Status              string     `db:"status"`
	RequestedBy         *string    `db:"requested_by"`
	ApprovedBy          *string    `db:"approved_by"`
	RequestedAt         time.Time  `db:"requested_at"`
	RespondedAt         *time.Time `db:"responded_at"`
	Reason              *string    `db:"reason"`
	Metadata            JSONBMap   `db:"metadata"`
	CreatedAt           time.Time  `db:"created_at"`
	UpdatedAt           time.Time  `db:"updated_at"`
}


