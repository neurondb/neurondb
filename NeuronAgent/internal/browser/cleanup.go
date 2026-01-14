/*-------------------------------------------------------------------------
 *
 * cleanup.go
 *    Browser session cleanup worker for expired sessions
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/browser/cleanup.go
 *
 *-------------------------------------------------------------------------
 */

package browser

import (
	"context"
	"time"

	"github.com/neurondb/NeuronAgent/internal/db"
)

/* CleanupWorker periodically cleans up expired browser sessions */
type CleanupWorker struct {
	db       *db.DB
	driver   *Driver
	interval time.Duration
	maxAge   time.Duration
	ctx      context.Context
	cancel   context.CancelFunc
	done     chan struct{}
}

/* NewCleanupWorker creates a new browser session cleanup worker */
func NewCleanupWorker(database *db.DB, driver *Driver, interval, maxAge time.Duration) *CleanupWorker {
	ctx, cancel := context.WithCancel(context.Background())
	return &CleanupWorker{
		db:       database,
		driver:   driver,
		interval: interval,
		maxAge:   maxAge,
		ctx:      ctx,
		cancel:   cancel,
		done:     make(chan struct{}),
	}
}

/* Start starts the cleanup worker */
func (w *CleanupWorker) Start() {
	go w.run()
}

/* Stop stops the cleanup worker */
func (w *CleanupWorker) Stop() {
	w.cancel()
	<-w.done
}

func (w *CleanupWorker) run() {
	defer close(w.done)

	ticker := time.NewTicker(w.interval)
	defer ticker.Stop()

	/* Run immediately on start */
	w.cleanup()

	for {
		select {
		case <-w.ctx.Done():
			return
		case <-ticker.C:
			w.cleanup()
		}
	}
}

func (w *CleanupWorker) cleanup() {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	cutoffTime := time.Now().Add(-w.maxAge)

	/* Delete expired sessions from database */
	query := `
		DELETE FROM neurondb_agent.browser_sessions
		WHERE updated_at < $1 OR (expires_at IS NOT NULL AND expires_at < NOW())
		RETURNING session_id
	`

	rows, err := w.db.QueryContext(ctx, query, cutoffTime)
	if err != nil {
		return
	}
	defer rows.Close()

	/* Close browser contexts for deleted sessions */
	deletedCount := 0
	for rows.Next() {
		var sessionID string
		if err := rows.Scan(&sessionID); err != nil {
			continue
		}

		/* Close browser context if exists */
		if w.driver != nil {
			w.driver.CloseContext(sessionID)
		}

		deletedCount++
	}

	/* Clean up old screenshots (keep last 100 per session) */
	screenshotQuery := `
		DELETE FROM neurondb_agent.browser_snapshots
		WHERE id NOT IN (
			SELECT id FROM neurondb_agent.browser_snapshots
			ORDER BY created_at DESC
			LIMIT 100
		)
		AND created_at < $1
	`

	_, _ = w.db.ExecContext(ctx, screenshotQuery, cutoffTime)
}
