/*-------------------------------------------------------------------------
 *
 * failover.go
 *    High availability and failover mechanisms
 *
 * Provides multi-instance replication, automatic failover with health checks,
 * and zero-downtime deployments.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/reliability/failover.go
 *
 *-------------------------------------------------------------------------
 */

package reliability

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/metrics"
)

/* FailoverManager manages failover for high availability */
type FailoverManager struct {
	queries      *db.Queries
	primaryNode  string
	replicaNodes []string
	currentNode  string
	healthChecker *HealthChecker
	mu           sync.RWMutex
	enabled      bool
}

/* HealthChecker checks health of nodes */
type HealthChecker struct {
	queries *db.Queries
}

/* NewFailoverManager creates a new failover manager */
func NewFailoverManager(queries *db.Queries, currentNode string) *FailoverManager {
	return &FailoverManager{
		queries:       queries,
		currentNode:   currentNode,
		replicaNodes:  make([]string, 0),
		healthChecker: NewHealthChecker(queries),
		enabled:       false,
	}
}

/* Enable enables failover */
func (fm *FailoverManager) Enable(ctx context.Context) error {
	fm.mu.Lock()
	defer fm.mu.Unlock()

	if fm.enabled {
		return nil
	}

	/* Discover primary and replica nodes */
	if err := fm.discoverNodes(ctx); err != nil {
		return fmt.Errorf("failover enable failed: node_discovery_error=true, error=%w", err)
	}

	fm.enabled = true

	/* Start health checking */
	go fm.healthCheckLoop(ctx)

	metrics.InfoWithContext(ctx, "Failover enabled", map[string]interface{}{
		"current_node": fm.currentNode,
		"primary_node": fm.primaryNode,
		"replica_count": len(fm.replicaNodes),
	})

	return nil
}

/* discoverNodes discovers primary and replica nodes */
func (fm *FailoverManager) discoverNodes(ctx context.Context) error {
	query := `SELECT node_id, role, status
		FROM neurondb_agent.cluster_nodes
		WHERE status = 'healthy'
		ORDER BY role, node_id`

	type NodeRow struct {
		NodeID string `db:"node_id"`
		Role   string `db:"role"`
		Status string `db:"status"`
	}

	var rows []NodeRow
	err := fm.queries.DB.SelectContext(ctx, &rows, query)
	if err != nil {
		return err
	}

	for _, row := range rows {
		if row.Role == "primary" {
			fm.primaryNode = row.NodeID
		} else if row.Role == "replica" {
			fm.replicaNodes = append(fm.replicaNodes, row.NodeID)
		}
	}

	return nil
}

/* healthCheckLoop periodically checks health of nodes */
func (fm *FailoverManager) healthCheckLoop(ctx context.Context) {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			fm.checkHealth(ctx)
		}
	}
}

/* checkHealth checks health of primary and replica nodes */
func (fm *FailoverManager) checkHealth(ctx context.Context) {
	fm.mu.RLock()
	primary := fm.primaryNode
	replicas := fm.replicaNodes
	fm.mu.RUnlock()

	/* Check primary health */
	if primary != "" {
		healthy, err := fm.healthChecker.CheckHealth(ctx, primary)
		if err != nil || !healthy {
			metrics.WarnWithContext(ctx, "Primary node unhealthy", map[string]interface{}{
				"node_id": primary,
				"error":   err,
			})
			/* Trigger failover */
			fm.triggerFailover(ctx, primary)
		}
	}

	/* Check replica health */
	for _, replica := range replicas {
		healthy, err := fm.healthChecker.CheckHealth(ctx, replica)
		if err != nil || !healthy {
			metrics.WarnWithContext(ctx, "Replica node unhealthy", map[string]interface{}{
				"node_id": replica,
				"error":   err,
			})
		}
	}
}

/* triggerFailover triggers failover to a replica */
func (fm *FailoverManager) triggerFailover(ctx context.Context, failedNode string) {
	fm.mu.Lock()
	defer fm.mu.Unlock()

	if fm.primaryNode != failedNode {
		/* Not the primary, no failover needed */
		return
	}

	/* Find healthy replica */
	for _, replica := range fm.replicaNodes {
		healthy, err := fm.healthChecker.CheckHealth(ctx, replica)
		if err == nil && healthy {
			/* Promote replica to primary */
			query := `UPDATE neurondb_agent.cluster_nodes
				SET role = 'primary', updated_at = NOW()
				WHERE node_id = $1`
			_, err := fm.queries.DB.ExecContext(ctx, query, replica)
			if err != nil {
				metrics.WarnWithContext(ctx, "Failed to promote replica to primary", map[string]interface{}{
					"replica_id": replica,
					"error":      err.Error(),
				})
				continue
			}

			/* Demote old primary */
			query = `UPDATE neurondb_agent.cluster_nodes
				SET role = 'replica', status = 'unhealthy', updated_at = NOW()
				WHERE node_id = $1`
			fm.queries.DB.ExecContext(ctx, query, failedNode)

			fm.primaryNode = replica
			metrics.InfoWithContext(ctx, "Failover completed", map[string]interface{}{
				"old_primary": failedNode,
				"new_primary": replica,
			})
			return
		}
	}

	metrics.ErrorWithContext(ctx, "Failover failed: no healthy replicas", fmt.Errorf("no healthy replicas available"), map[string]interface{}{
		"failed_node": failedNode,
	})
}

/* NewHealthChecker creates a new health checker */
func NewHealthChecker(queries *db.Queries) *HealthChecker {
	return &HealthChecker{
		queries: queries,
	}
}

/* CheckHealth checks health of a node */
func (hc *HealthChecker) CheckHealth(ctx context.Context, nodeID string) (bool, error) {
	query := `SELECT status, last_seen
		FROM neurondb_agent.cluster_nodes
		WHERE node_id = $1`

	type HealthRow struct {
		Status   string    `db:"status"`
		LastSeen time.Time `db:"last_seen"`
	}

	var row HealthRow
	err := hc.queries.DB.GetContext(ctx, &row, query, nodeID)
	if err != nil {
		return false, err
	}

	/* Check if node is stale */
	if time.Since(row.LastSeen) > 30*time.Second {
		return false, nil
	}

	return row.Status == "healthy", nil
}

