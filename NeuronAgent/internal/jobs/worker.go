/*-------------------------------------------------------------------------
 *
 * worker.go
 *    Background job worker for NeuronAgent
 *
 * Provides worker pool for processing background jobs from the queue
 * with configurable concurrency and graceful shutdown support.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/jobs/worker.go
 *
 *-------------------------------------------------------------------------
 */

package jobs

import (
	"context"
	"database/sql"
	"sync"
	"time"

	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/metrics"
)

type Worker struct {
	queue      *Queue
	processor  *Processor
	workers    int
	ctx        context.Context
	cancel     context.CancelFunc
	wg         sync.WaitGroup
	retryDelay time.Duration
}

func NewWorker(queue *Queue, processor *Processor, workers int) *Worker {
	ctx, cancel := context.WithCancel(context.Background())
	return &Worker{
		queue:      queue,
		processor:  processor,
		workers:    workers,
		ctx:        ctx,
		cancel:     cancel,
		retryDelay: 5 * time.Second,
	}
}

func (w *Worker) Start() {
	for i := 0; i < w.workers; i++ {
		w.wg.Add(1)
		go w.work()
	}
}

func (w *Worker) work() {
	defer w.wg.Done()

	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-w.ctx.Done():
			return
		case <-ticker.C:
			job, err := w.queue.ClaimJob(w.ctx)
			if err != nil || job == nil {
				continue
			}

			w.processJob(job)
		}
	}
}

func (w *Worker) processJob(job *db.Job) {
	result, err := w.processor.Process(w.ctx, job)

	status := "done"
	errorMsg := (*string)(nil)
	retryCount := job.RetryCount
	var completedAt *time.Time

	if err != nil {
		/* Use error classification to determine if job should be retried */
		shouldRetry := ShouldRetry(err, retryCount, job.MaxRetries)
		
		if !shouldRetry || retryCount >= job.MaxRetries {
			/* Non-retryable error or max retries reached */
			status = "failed"
			errStr := err.Error()
			errorMsg = &errStr
			now := time.Now()
			completedAt = &now
		} else {
			/* Retryable error - will be picked up again */
			retryCount++
			status = "queued"
			/* Don't set completedAt for retries */
		}
	} else {
		/* Success */
		now := time.Now()
		completedAt = &now
	}

	/* Record metrics */
	metrics.RecordJobProcessed(job.Type, status)

	/* Use proper time handling for UpdateJob */
	var completedAtVal *sql.NullTime
	if completedAt != nil {
		completedAtVal = &sql.NullTime{
			Time:  *completedAt,
			Valid: true,
		}
	}

	w.queue.UpdateJob(w.ctx, job.ID, status, result, errorMsg, retryCount, completedAtVal)
}

func (w *Worker) Stop() {
	w.cancel()
	w.wg.Wait()
}
