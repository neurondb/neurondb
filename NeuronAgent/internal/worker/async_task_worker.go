/*-------------------------------------------------------------------------
 *
 * async_task_worker.go
 *    Background worker for asynchronous task execution
 *
 * Processes async tasks from the queue with priority handling and
 * automatic retry logic.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/worker/async_task_worker.go
 *
 *-------------------------------------------------------------------------
 */

package worker

import (
	"context"
	"encoding/json"
	"time"

	"github.com/neurondb/NeuronAgent/internal/agent"
	"github.com/neurondb/NeuronAgent/internal/db"
)

/* AsyncTaskWorker processes async tasks from the queue */
type AsyncTaskWorker struct {
	queries       *db.Queries
	asyncExecutor *agent.AsyncTaskExecutor
	interval      time.Duration
	workers       int
}

/* NewAsyncTaskWorker creates a new async task worker */
func NewAsyncTaskWorker(queries *db.Queries, asyncExecutor *agent.AsyncTaskExecutor, interval time.Duration, workers int) *AsyncTaskWorker {
	return &AsyncTaskWorker{
		queries:       queries,
		asyncExecutor: asyncExecutor,
		interval:      interval,
		workers:       workers,
	}
}

/* Start starts the async task worker */
func (w *AsyncTaskWorker) Start(ctx context.Context) error {
	ticker := time.NewTicker(w.interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			w.processPendingTasks(ctx)
		}
	}
}

/* processPendingTasks processes pending async tasks */
func (w *AsyncTaskWorker) processPendingTasks(ctx context.Context) {
	/* Get pending tasks ordered by priority */
	query := `SELECT id, session_id, agent_id, task_type, status, priority, input, result, error_message,
		created_at, started_at, completed_at, metadata
		FROM neurondb_agent.async_tasks
		WHERE status = 'pending'
		ORDER BY priority DESC, created_at ASC
		LIMIT $1
		FOR UPDATE SKIP LOCKED`

	rows, err := w.queries.DB.QueryContext(ctx, query, w.workers*2)
	if err != nil {
		return
	}
	defer rows.Close()

	var tasks []*agent.AsyncTask
	for rows.Next() {
		task := &agent.AsyncTask{}
		var inputJSON, resultJSON []byte
		var errorMsg *string

		err := rows.Scan(
			&task.ID, &task.SessionID, &task.AgentID, &task.TaskType, &task.Status, &task.Priority,
			&inputJSON, &resultJSON, &errorMsg,
			&task.CreatedAt, &task.StartedAt, &task.CompletedAt, &task.Metadata,
		)
		if err != nil {
			continue
		}

		/* Parse JSON fields */
		if len(inputJSON) > 0 {
			_ = json.Unmarshal(inputJSON, &task.Input)
		}
		if len(resultJSON) > 0 {
			_ = json.Unmarshal(resultJSON, &task.Result)
		}
		task.ErrorMsg = errorMsg

		tasks = append(tasks, task)
	}

	/* Process tasks in parallel (up to worker limit) */
	semaphore := make(chan struct{}, w.workers)
	for _, task := range tasks {
		semaphore <- struct{}{}
		go func(t *agent.AsyncTask) {
			defer func() { <-semaphore }()
			w.processTask(ctx, t)
		}(task)
	}
}

/* processTask processes a single async task */
func (w *AsyncTaskWorker) processTask(ctx context.Context, task *agent.AsyncTask) {
	/* Task execution is handled by AsyncTaskExecutor.executeTaskInBackground */
	/* This worker primarily monitors and manages the queue */
	_ = task
}
