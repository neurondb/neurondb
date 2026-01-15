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
	"hash/crc32"
	"sync"
	"time"

	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/metrics"
)

/* LeaderElection manages leader election */
type LeaderElection struct {
	nodeID    string
	queries   *db.Queries
	isLeader  bool
	mu        sync.RWMutex
	stopChan  chan struct{}
	lockID    int64
}

/* NewLeaderElection creates a new leader election instance */
func NewLeaderElection(nodeID string, queries *db.Queries) *LeaderElection {
	/* Use node ID hash as lock ID */
	lockID := int64(crc32.ChecksumIEEE([]byte(nodeID)) % 1000000)
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

/* verifyLock verifies we still hold the lock */
func (le *LeaderElection) verifyLock(ctx context.Context) bool {
	query := `SELECT pg_try_advisory_lock($1)`
	
	var acquired bool
	err := le.queries.DB.GetContext(ctx, &acquired, query, le.lockID)
	if err != nil {
		return false
	}

	/* If we got the lock, we didn't have it before, so release it */
	if acquired {
		le.releaseLock(ctx)
		return false
	}

	/* Check if we hold the lock */
	query = `SELECT COUNT(*) FROM pg_locks WHERE locktype = 'advisory' AND objid = $1 AND granted = true`
	var count int
	err = le.queries.DB.GetContext(ctx, &count, query, le.lockID)
	if err != nil {
		return false
	}

	return count > 0
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

