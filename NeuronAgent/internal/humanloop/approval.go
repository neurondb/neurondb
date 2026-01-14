/*-------------------------------------------------------------------------
 *
 * approval.go
 *    Human-in-the-loop approval workflows
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/humanloop/approval.go
 *
 *-------------------------------------------------------------------------
 */

package humanloop

import (
	"context"
	"database/sql"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/jmoiron/sqlx"
	"github.com/neurondb/NeuronAgent/internal/db"
)

/* ApprovalRequest represents an approval request */
type ApprovalRequest struct {
	ID                uuid.UUID   `db:"id"`
	AgentID           *uuid.UUID  `db:"agent_id"`
	SessionID         *uuid.UUID  `db:"session_id"`
	RequestType       string      `db:"request_type"`
	ActionDescription string      `db:"action_description"`
	Payload           db.JSONBMap `db:"payload"`
	Status            string      `db:"status"`
	RequestedBy       *string     `db:"requested_by"`
	ApprovedBy        *string     `db:"approved_by"`
	RejectionReason   *string     `db:"rejection_reason"`
	ExpiresAt         *time.Time  `db:"expires_at"`
	CreatedAt         string      `db:"created_at"`
	UpdatedAt         string      `db:"updated_at"`
	ResolvedAt        *string     `db:"resolved_at"`
}

/* ApprovalManager manages approval requests */
type ApprovalManager struct {
	db *sqlx.DB
}

/* NewApprovalManager creates a new approval manager */
func NewApprovalManager(db *sqlx.DB) *ApprovalManager {
	return &ApprovalManager{db: db}
}

/* CreateApprovalRequest creates a new approval request */
func (am *ApprovalManager) CreateApprovalRequest(ctx context.Context, req *ApprovalRequest) error {
	query := `INSERT INTO neurondb_agent.approval_requests 
		(id, agent_id, session_id, request_type, action_description, payload, status, requested_by, expires_at, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, $9, NOW(), NOW())`

	payloadJSON := db.FromMap(req.Payload)
	if payloadJSON == nil {
		payloadJSON = make(db.JSONBMap)
	}

	_, err := am.db.ExecContext(ctx, query,
		req.ID, req.AgentID, req.SessionID, req.RequestType, req.ActionDescription,
		payloadJSON, req.Status, req.RequestedBy, req.ExpiresAt)
	if err != nil {
		return fmt.Errorf("failed to create approval request: %w", err)
	}
	return nil
}

/* GetApprovalRequest gets an approval request by ID */
func (am *ApprovalManager) GetApprovalRequest(ctx context.Context, id uuid.UUID) (*ApprovalRequest, error) {
	var req ApprovalRequest
	query := `SELECT * FROM neurondb_agent.approval_requests WHERE id = $1`
	err := am.db.GetContext(ctx, &req, query, id)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("approval request not found")
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get approval request: %w", err)
	}
	return &req, nil
}

/* ApproveRequest approves an approval request */
func (am *ApprovalManager) ApproveRequest(ctx context.Context, id uuid.UUID, approvedBy string) error {
	query := `UPDATE neurondb_agent.approval_requests 
		SET status = 'approved', approved_by = $2, resolved_at = NOW(), updated_at = NOW()
		WHERE id = $1 AND status = 'pending'`

	result, err := am.db.ExecContext(ctx, query, id, approvedBy)
	if err != nil {
		return fmt.Errorf("failed to approve request: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to check rows affected: %w", err)
	}
	if rowsAffected == 0 {
		return fmt.Errorf("approval request not found or already resolved")
	}
	return nil
}

/* RejectRequest rejects an approval request */
func (am *ApprovalManager) RejectRequest(ctx context.Context, id uuid.UUID, rejectedBy, reason string) error {
	query := `UPDATE neurondb_agent.approval_requests 
		SET status = 'rejected', approved_by = $2, rejection_reason = $3, resolved_at = NOW(), updated_at = NOW()
		WHERE id = $1 AND status = 'pending'`

	result, err := am.db.ExecContext(ctx, query, id, rejectedBy, reason)
	if err != nil {
		return fmt.Errorf("failed to reject request: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to check rows affected: %w", err)
	}
	if rowsAffected == 0 {
		return fmt.Errorf("approval request not found or already resolved")
	}
	return nil
}

/* ListPendingApprovals lists pending approval requests */
func (am *ApprovalManager) ListPendingApprovals(ctx context.Context, agentID *uuid.UUID, limit, offset int) ([]ApprovalRequest, error) {
	var reqs []ApprovalRequest
	var query string
	var args []interface{}

	if agentID != nil {
		query = `SELECT * FROM neurondb_agent.approval_requests 
			WHERE status = 'pending' AND agent_id = $1 
			ORDER BY created_at ASC 
			LIMIT $2 OFFSET $3`
		args = []interface{}{agentID, limit, offset}
	} else {
		query = `SELECT * FROM neurondb_agent.approval_requests 
			WHERE status = 'pending' 
			ORDER BY created_at ASC 
			LIMIT $1 OFFSET $2`
		args = []interface{}{limit, offset}
	}

	err := am.db.SelectContext(ctx, &reqs, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to list pending approvals: %w", err)
	}
	return reqs, nil
}

/* CheckExpiredApprovals checks and expires old approval requests */
func (am *ApprovalManager) CheckExpiredApprovals(ctx context.Context) error {
	query := `UPDATE neurondb_agent.approval_requests 
		SET status = 'expired', updated_at = NOW()
		WHERE status = 'pending' AND expires_at IS NOT NULL AND expires_at < NOW()`

	_, err := am.db.ExecContext(ctx, query)
	return err
}
