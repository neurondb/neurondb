/*-------------------------------------------------------------------------
 *
 * leader_election.go
 *    Leader election for distributed coordination
 *
 * Provides leader election using PostgreSQL advisory locks for coordination.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/distributed/leader_election.go
 *
 *-------------------------------------------------------------------------
 */

package distributed

import (
	"context"
	"hash/fnv"
	"sync"
	"time"

	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/metrics"
)

/* LeaderElection manages leader election */
type LeaderElection struct {
	nodeID   string
	queries  *db.Queries
	isLeader bool
	mu       sync.RWMutex
	stopChan chan struct{}
	lockID   int64
}

/* NewLeaderElection creates a new leader election instance */
func NewLeaderElection(nodeID string, queries *db.Queries) *LeaderElection {
	/* Use full 64-bit hash of node ID to avoid lock ID collisions between nodes */
	h := fnv.New64a()
	_, _ = h.Write([]byte(nodeID))
	lockID := int64(h.Sum64())
	return &LeaderElection{
		nodeID:   nodeID,
		queries:  queries,
		isLeader: false,
		lockID:   lockID,
		stopChan: make(chan struct{}),
	}
}

/* Start starts the leader election process */
func (le *LeaderElection) Start(ctx context.Context) error {
	go le.run(ctx)
	return nil
}

/* Stop stops the leader election process */
func (le *LeaderElection) Stop(ctx context.Context) {
	close(le.stopChan)

	/* Release lock if we're the leader */
	le.mu.Lock()
	if le.isLeader {
		le.releaseLock(ctx)
		le.isLeader = false
	}
	le.mu.Unlock()
}

/* run runs the leader election loop */
func (le *LeaderElection) run(ctx context.Context) {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-le.stopChan:
			return
		case <-ticker.C:
			le.tryAcquireLeadership(ctx)
		}
	}
}

/* tryAcquireLeadership attempts to acquire leadership */
func (le *LeaderElection) tryAcquireLeadership(ctx context.Context) {
	le.mu.Lock()
	defer le.mu.Unlock()

	if le.isLeader {
		/* Verify we still hold the lock */
		if !le.verifyLock(ctx) {
			le.isLeader = false
			metrics.InfoWithContext(ctx, "Lost leadership", map[string]interface{}{
				"node_id": le.nodeID,
			})
		}
		return
	}

	/* Try to acquire lock */
	if le.acquireLock(ctx) {
		le.isLeader = true
		metrics.InfoWithContext(ctx, "Acquired leadership", map[string]interface{}{
			"node_id": le.nodeID,
		})
	}
}

/* acquireLock attempts to acquire an advisory lock */
func (le *LeaderElection) acquireLock(ctx context.Context) bool {
	/* Use PostgreSQL advisory lock */
	query := `SELECT pg_try_advisory_lock($1)`

	var acquired bool
	err := le.queries.DB.GetContext(ctx, &acquired, query, le.lockID)
	if err != nil {
		metrics.WarnWithContext(ctx, "Failed to acquire advisory lock", map[string]interface{}{
			"node_id": le.nodeID,
			"error":   err.Error(),
		})
		return false
	}

	return acquired
}

/* verifyLock verifies we still hold the lock by checking pg_locks for our session only.
 * We do not use pg_try_advisory_lock here because in PostgreSQL, if the current session
 * already holds the lock, pg_try_advisory_lock returns true again, which would incorrectly
 * make us think we lost the lock and release it (split-brain). */
func (le *LeaderElection) verifyLock(ctx context.Context) bool {
	/* Check if this session holds the advisory lock (objid = lockID, granted, our pid) */
	query := `SELECT COUNT(*) > 0 FROM pg_locks WHERE locktype = 'advisory' AND objid = $1 AND granted = true AND pid = pg_backend_pid()`
	var stillHolding bool
	err := le.queries.DB.GetContext(ctx, &stillHolding, query, le.lockID)
	if err != nil {
		return false
	}
	return stillHolding
}

/* releaseLock releases the advisory lock */
func (le *LeaderElection) releaseLock(ctx context.Context) {
	query := `SELECT pg_advisory_unlock($1)`
	_, err := le.queries.DB.ExecContext(ctx, query, le.lockID)
	if err != nil {
		metrics.WarnWithContext(ctx, "Failed to release advisory lock", map[string]interface{}{
			"node_id": le.nodeID,
			"error":   err.Error(),
		})
	}
}

/* IsLeader returns whether this node is the leader */
func (le *LeaderElection) IsLeader() bool {
	le.mu.RLock()
	defer le.mu.RUnlock()
	return le.isLeader
}
