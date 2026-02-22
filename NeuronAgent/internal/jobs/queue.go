/*-------------------------------------------------------------------------
 *
 * queue.go
 *    Job queue management for NeuronAgent
 *
 * Provides job queue operations for enqueueing, dequeuing, and
 * managing background job execution with status tracking.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/jobs/queue.go
 *
 *-------------------------------------------------------------------------
 */

package jobs

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/metrics"
)

const maxJobPayloadSize = 512 * 1024 // 512 KiB

type Queue struct {
	queries *db.Queries
}

func NewQueue(queries *db.Queries) *Queue {
	return &Queue{queries: queries}
}

/* Enqueue adds a job to the queue */
func (q *Queue) Enqueue(ctx context.Context, jobType string, agentID, sessionID *uuid.UUID, payload map[string]interface{}, priority int) (*db.Job, error) {
	if payload != nil {
		payloadBytes, err := json.Marshal(payload)
		if err != nil {
			return nil, fmt.Errorf("job enqueue failed: invalid_payload=true, error=%w", err)
		}
		if len(payloadBytes) > maxJobPayloadSize {
			return nil, fmt.Errorf("job enqueue failed: payload_too_large=true, size=%d, max=%d", len(payloadBytes), maxJobPayloadSize)
		}
	}

	job := &db.Job{
		Type:       jobType,
		Status:     "queued",
		Priority:   priority,
		Payload:    payload,
		AgentID:    agentID,
		SessionID:  sessionID,
		MaxRetries: 3,
	}

	job, err := q.queries.CreateJob(ctx, job)
	if err == nil {
		metrics.RecordJobQueued()
	}
	return job, err
}

/* ClaimJob claims the next available job using SKIP LOCKED */
func (q *Queue) ClaimJob(ctx context.Context) (*db.Job, error) {
	return q.queries.ClaimJob(ctx)
}

/* GetJob returns a job by ID (for idempotency checks) */
func (q *Queue) GetJob(ctx context.Context, id int64) (*db.Job, error) {
	return q.queries.GetJob(ctx, id)
}

/* UpdateJob updates a job's status and result */
func (q *Queue) UpdateJob(ctx context.Context, id int64, status string, result map[string]interface{}, errorMsg *string, retryCount int, completedAt *sql.NullTime) error {
	return q.queries.UpdateJob(ctx, id, status, result, errorMsg, retryCount, completedAt)
}
