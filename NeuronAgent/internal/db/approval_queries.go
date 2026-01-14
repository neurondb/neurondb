/*-------------------------------------------------------------------------
 *
 * approval_queries.go
 *    Database queries for approval requests
 *
 * Provides database query functions for approval requests.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/db/approval_queries.go
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

/* Approval request queries */
const (
	createApprovalRequestQuery = `
		INSERT INTO neurondb_agent.approval_requests 
		(workflow_execution_id, step_execution_id, agent_id, session_id, approval_type, status, requested_by, metadata)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
		RETURNING id, requested_at, created_at, updated_at`

	getApprovalRequestQuery = `SELECT * FROM neurondb_agent.approval_requests WHERE id = $1`

	updateApprovalRequestQuery = `
		UPDATE neurondb_agent.approval_requests 
		SET status = $2, approved_by = $3, responded_at = NOW(), reason = $4, updated_at = NOW()
		WHERE id = $1
		RETURNING updated_at`

	listApprovalRequestsQuery = `
		SELECT * FROM neurondb_agent.approval_requests 
		WHERE ($1::uuid IS NULL OR workflow_execution_id = $1)
		AND ($2::uuid IS NULL OR step_execution_id = $2)
		AND ($3::text IS NULL OR status = $3)
		ORDER BY requested_at DESC
		LIMIT $4 OFFSET $5`
)

/* Approval request methods */
func (q *Queries) CreateApprovalRequest(ctx context.Context, req *ApprovalRequest) error {
	metadataValue, err := req.Metadata.Value()
	if err != nil {
		return fmt.Errorf("failed to convert metadata: %w", err)
	}

	params := []interface{}{
		req.WorkflowExecutionID, req.StepExecutionID, req.AgentID, req.SessionID,
		req.ApprovalType, req.Status, req.RequestedBy, metadataValue,
	}
	err = q.DB.GetContext(ctx, req, createApprovalRequestQuery, params...)
	if err != nil {
		return fmt.Errorf("approval request creation failed: %w", err)
	}
	return nil
}

func (q *Queries) GetApprovalRequest(ctx context.Context, id uuid.UUID) (*ApprovalRequest, error) {
	var req ApprovalRequest
	err := q.DB.GetContext(ctx, &req, getApprovalRequestQuery, id)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("approval request not found: %w", err)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get approval request: %w", err)
	}
	return &req, nil
}

func (q *Queries) UpdateApprovalRequest(ctx context.Context, req *ApprovalRequest) error {
	params := []interface{}{req.ID, req.Status, req.ApprovedBy, req.Reason}
	err := q.DB.GetContext(ctx, req, updateApprovalRequestQuery, params...)
	if err != nil {
		return fmt.Errorf("approval request update failed: %w", err)
	}
	return nil
}

func (q *Queries) ListApprovalRequests(ctx context.Context, workflowExecutionID, stepExecutionID *uuid.UUID, status *string, limit, offset int) ([]ApprovalRequest, error) {
	var requests []ApprovalRequest
	params := []interface{}{workflowExecutionID, stepExecutionID, status, limit, offset}
	err := q.DB.SelectContext(ctx, &requests, listApprovalRequestsQuery, params...)
	if err != nil {
		return nil, fmt.Errorf("failed to list approval requests: %w", err)
	}
	return requests, nil
}


