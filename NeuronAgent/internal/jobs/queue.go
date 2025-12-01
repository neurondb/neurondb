package jobs

import (
	"context"
	"database/sql"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/metrics"
)

type Queue struct {
	queries *db.Queries
}

func NewQueue(queries *db.Queries) *Queue {
	return &Queue{queries: queries}
}

// Enqueue adds a job to the queue
func (q *Queue) Enqueue(ctx context.Context, jobType string, agentID, sessionID *uuid.UUID, payload map[string]interface{}, priority int) (*db.Job, error) {
	job := &db.Job{
		Type:     jobType,
		Status:   "queued",
		Priority: priority,
		Payload:  payload,
		AgentID:  agentID,
		SessionID: sessionID,
		MaxRetries: 3,
	}

	job, err := q.queries.CreateJob(ctx, job)
	if err == nil {
		metrics.RecordJobQueued()
	}
	return job, err
}

// ClaimJob claims the next available job using SKIP LOCKED
func (q *Queue) ClaimJob(ctx context.Context) (*db.Job, error) {
	return q.queries.ClaimJob(ctx)
}

// UpdateJob updates a job's status and result
func (q *Queue) UpdateJob(ctx context.Context, id int64, status string, result map[string]interface{}, errorMsg *string, retryCount int, completedAt *sql.NullTime) error {
	return q.queries.UpdateJob(ctx, id, status, result, errorMsg, retryCount, completedAt)
}

