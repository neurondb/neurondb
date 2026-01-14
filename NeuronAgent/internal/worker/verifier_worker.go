/*-------------------------------------------------------------------------
 *
 * verifier_worker.go
 *    Background worker for verification queue processing
 *
 * Processes verification queue items, runs quality assurance checks,
 * and stores verification results.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/worker/verifier_worker.go
 *
 *-------------------------------------------------------------------------
 */

package worker

import (
	"context"
	"encoding/json"
	"time"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/agent"
	"github.com/neurondb/NeuronAgent/internal/db"
)

/* VerifierWorker processes verification queue */
type VerifierWorker struct {
	queries  *db.Queries
	runtime  *agent.Runtime
	interval time.Duration
	workers  int
}

/* NewVerifierWorker creates a new verifier worker */
func NewVerifierWorker(queries *db.Queries, runtime *agent.Runtime, interval time.Duration, workers int) *VerifierWorker {
	return &VerifierWorker{
		queries:  queries,
		runtime:  runtime,
		interval: interval,
		workers:  workers,
	}
}

/* Start starts the verifier worker */
func (v *VerifierWorker) Start(ctx context.Context) error {
	ticker := time.NewTicker(v.interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			/* Process verification queue */
			err := v.processQueue(ctx)
			if err != nil {
				/* Log error but continue - silently handle */
				_ = err
			}
		}
	}
}

/* QueueItem represents a verification queue item */
type QueueItem struct {
	ID            uuid.UUID  `db:"id"`
	SessionID     uuid.UUID  `db:"session_id"`
	OutputID      *uuid.UUID `db:"output_id"`
	OutputContent string     `db:"output_content"`
	Priority      string     `db:"priority"`
}

/* processQueue processes items from verification queue */
func (v *VerifierWorker) processQueue(ctx context.Context) error {
	/* Get pending items ordered by priority */
	query := `SELECT id, session_id, output_id, output_content, priority
		FROM neurondb_agent.verification_queue
		WHERE status = 'pending'
		ORDER BY 
			CASE priority
				WHEN 'high' THEN 1
				WHEN 'medium' THEN 2
				WHEN 'low' THEN 3
			END,
			created_at ASC
		LIMIT $1`

	var items []QueueItem
	err := v.queries.GetDB().SelectContext(ctx, &items, query, v.workers*10)
	if err != nil {
		return err
	}

	/* Process each item */
	for _, item := range items {
		err := v.processItem(ctx, item)
		if err != nil {
			/* Mark as failed - silently handle error */
			_ = err
			v.markFailed(ctx, item.ID)
		}
	}

	return nil
}

/* processItem processes a single verification queue item */
func (v *VerifierWorker) processItem(ctx context.Context, item QueueItem) error {
	/* Mark as processing */
	updateQuery := `UPDATE neurondb_agent.verification_queue
		SET status = 'processing', processed_at = NOW()
		WHERE id = $1`
	_, err := v.queries.GetDB().ExecContext(ctx, updateQuery, item.ID)
	if err != nil {
		return err
	}

	/* Get session to find agent */
	session, err := v.queries.GetSession(ctx, item.SessionID)
	if err != nil {
		return err
	}

	/* Create verification agent */
	verifier := agent.NewVerificationAgent(session.AgentID, v.runtime, v.queries)
	err = verifier.LoadRules(ctx)
	if err != nil {
		return err
	}

	/* Run verification */
	result, err := verifier.VerifyOutput(ctx, item.OutputContent, nil)
	if err != nil {
		return err
	}

	/* Store result */
	resultQuery := `INSERT INTO neurondb_agent.verification_results
		(queue_id, verifier_agent_id, passed, issues, suggestions, confidence)
		VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6)`

	issuesJSON, _ := json.Marshal(result.Issues)
	suggestionsJSON, _ := json.Marshal(result.Suggestions)

	_, err = v.queries.GetDB().ExecContext(ctx, resultQuery, item.ID, session.AgentID, result.Passed, issuesJSON, suggestionsJSON, result.Confidence)
	if err != nil {
		return err
	}

	/* Mark queue item as completed */
	completeQuery := `UPDATE neurondb_agent.verification_queue
		SET status = 'completed'
		WHERE id = $1`
	_, err = v.queries.GetDB().ExecContext(ctx, completeQuery, item.ID)
	if err != nil {
		return err
	}

	return nil
}

/* markFailed marks a queue item as failed */
func (v *VerifierWorker) markFailed(ctx context.Context, queueID uuid.UUID) {
	query := `UPDATE neurondb_agent.verification_queue
		SET status = 'failed'
		WHERE id = $1`
	v.queries.GetDB().ExecContext(ctx, query, queueID)
}
