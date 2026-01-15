/*-------------------------------------------------------------------------
 *
 * balancer.go
 *    Load balancing and service discovery
 *
 * Provides intelligent load distribution based on agent type and
 * resource usage with health-based routing.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/distributed/balancer.go
 *
 *-------------------------------------------------------------------------
 */

package distributed

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/neurondb/NeuronAgent/internal/db"
)

/* LoadBalancer manages load distribution across nodes */
type LoadBalancer struct {
	queries      *db.Queries
	nodes        map[string]*NodeMetrics
	mu           sync.RWMutex
	strategy     LoadBalanceStrategy
	healthChecker *HealthChecker
}

/* NodeMetrics tracks node performance metrics */
type NodeMetrics struct {
	NodeID        string
	ActiveAgents  int
	CPUUsage      float64
	MemoryUsage   float64
	RequestRate   float64
	ErrorRate     float64
	LastUpdate    time.Time
	ResponseTime  time.Duration
}

/* LoadBalanceStrategy defines load balancing strategy */
type LoadBalanceStrategy string

const (
	StrategyRoundRobin    LoadBalanceStrategy = "round_robin"
	StrategyLeastConn     LoadBalanceStrategy = "least_connections"
	StrategyLeastLoad     LoadBalanceStrategy = "least_load"
	StrategyHealthBased   LoadBalanceStrategy = "health_based"
	StrategyConsistentHash LoadBalanceStrategy = "consistent_hash"
)

/* NewLoadBalancer creates a new load balancer */
func NewLoadBalancer(queries *db.Queries, strategy LoadBalanceStrategy) *LoadBalancer {
	if strategy == "" {
		strategy = StrategyHealthBased
	}

	return &LoadBalancer{
		queries:       queries,
		nodes:         make(map[string]*NodeMetrics),
		strategy:      strategy,
		healthChecker: NewHealthChecker(queries),
	}
}

/* SelectNode selects the best node for a request */
func (lb *LoadBalancer) SelectNode(ctx context.Context, agentType string, requirements map[string]interface{}) (string, error) {
	lb.mu.RLock()
	defer lb.mu.RUnlock()

	/* Filter healthy nodes */
	healthyNodes := lb.getHealthyNodes()

	if len(healthyNodes) == 0 {
		return "", fmt.Errorf("load balancer: no_healthy_nodes=true")
	}

	/* Apply strategy */
	switch lb.strategy {
	case StrategyRoundRobin:
		return lb.selectRoundRobin(healthyNodes), nil
	case StrategyLeastConn:
		return lb.selectLeastConnections(healthyNodes), nil
	case StrategyLeastLoad:
		return lb.selectLeastLoad(healthyNodes), nil
	case StrategyHealthBased:
		return lb.selectHealthBased(healthyNodes, requirements), nil
	case StrategyConsistentHash:
		return lb.selectConsistentHash(healthyNodes, requirements), nil
	default:
		return lb.selectHealthBased(healthyNodes, requirements), nil
	}
}

/* getHealthyNodes returns list of healthy nodes */
func (lb *LoadBalancer) getHealthyNodes() []*NodeMetrics {
	healthy := make([]*NodeMetrics, 0)
	now := time.Now()

	for _, node := range lb.nodes {
		/* Check if node is stale */
		if now.Sub(node.LastUpdate) > 30*time.Second {
			continue
		}

		/* Check error rate */
		if node.ErrorRate > 0.1 { // 10% error rate threshold
			continue
		}

		/* Check response time */
		if node.ResponseTime > 2*time.Second {
			continue
		}

		healthy = append(healthy, node)
	}

	return healthy
}

/* selectRoundRobin selects node using round-robin */
func (lb *LoadBalancer) selectRoundRobin(nodes []*NodeMetrics) string {
	if len(nodes) == 0 {
		return ""
	}
	/* Simple round-robin - in production use atomic counter */
	return nodes[0].NodeID
}

/* selectLeastConnections selects node with least active connections */
func (lb *LoadBalancer) selectLeastConnections(nodes []*NodeMetrics) string {
	if len(nodes) == 0 {
		return ""
	}

	bestNode := nodes[0]
	minConn := bestNode.ActiveAgents

	for _, node := range nodes[1:] {
		if node.ActiveAgents < minConn {
			minConn = node.ActiveAgents
			bestNode = node
		}
	}

	return bestNode.NodeID
}

/* selectLeastLoad selects node with least load */
func (lb *LoadBalancer) selectLeastLoad(nodes []*NodeMetrics) string {
	if len(nodes) == 0 {
		return ""
	}

	bestNode := nodes[0]
	minLoad := lb.calculateLoad(bestNode)

	for _, node := range nodes[1:] {
		load := lb.calculateLoad(node)
		if load < minLoad {
			minLoad = load
			bestNode = node
		}
	}

	return bestNode.NodeID
}

/* selectHealthBased selects node based on health metrics */
func (lb *LoadBalancer) selectHealthBased(nodes []*NodeMetrics, requirements map[string]interface{}) string {
	if len(nodes) == 0 {
		return ""
	}

	bestNode := nodes[0]
	bestScore := lb.calculateHealthScore(bestNode, requirements)

	for _, node := range nodes[1:] {
		score := lb.calculateHealthScore(node, requirements)
		if score > bestScore {
			bestScore = score
			bestNode = node
		}
	}

	return bestNode.NodeID
}

/* selectConsistentHash selects node using consistent hashing */
func (lb *LoadBalancer) selectConsistentHash(nodes []*NodeMetrics, requirements map[string]interface{}) string {
	/* Extract key from requirements for consistent hashing */
	key := ""
	if sessionID, ok := requirements["session_id"].(string); ok {
		key = sessionID
	}

	if key == "" {
		/* Fall back to round-robin */
		return lb.selectRoundRobin(nodes)
	}

	/* Use consistent hashing */
	ring := NewConsistentHashRing(150)
	for _, node := range nodes {
		ring.AddNode(node.NodeID)
	}

	return ring.GetNode(key)
}

/* calculateLoad calculates total load for a node */
func (lb *LoadBalancer) calculateLoad(node *NodeMetrics) float64 {
	/* Weighted combination of metrics */
	cpuWeight := 0.3
	memWeight := 0.3
	connWeight := 0.2
	rateWeight := 0.2

	load := node.CPUUsage*cpuWeight +
		node.MemoryUsage*memWeight +
		float64(node.ActiveAgents)/100.0*connWeight +
		node.RequestRate/1000.0*rateWeight

	return load
}

/* calculateHealthScore calculates health score for a node */
func (lb *LoadBalancer) calculateHealthScore(node *NodeMetrics, requirements map[string]interface{}) float64 {
	baseScore := 100.0

	/* Penalize high CPU usage */
	if node.CPUUsage > 0.8 {
		baseScore -= (node.CPUUsage - 0.8) * 50
	}

	/* Penalize high memory usage */
	if node.MemoryUsage > 0.8 {
		baseScore -= (node.MemoryUsage - 0.8) * 50
	}

	/* Penalize error rate */
	baseScore -= node.ErrorRate * 100

	/* Penalize slow response time */
	if node.ResponseTime > time.Second {
		penalty := float64(node.ResponseTime-time.Second) / float64(time.Second) * 20
		baseScore -= penalty
	}

	/* Bonus for low connection count */
	if node.ActiveAgents < 10 {
		baseScore += 10
	}

	if baseScore < 0 {
		return 0
	}

	return baseScore
}

/* UpdateNodeMetrics updates metrics for a node */
func (lb *LoadBalancer) UpdateNodeMetrics(ctx context.Context, nodeID string, metrics *NodeMetrics) {
	lb.mu.Lock()
	defer lb.mu.Unlock()

	metrics.LastUpdate = time.Now()
	lb.nodes[nodeID] = metrics
}

/* HealthChecker checks node health */
type HealthChecker struct {
	queries *db.Queries
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
		return false, fmt.Errorf("health check failed: node_not_found=true, error=%w", err)
	}

	/* Check if node is stale */
	if time.Since(row.LastSeen) > 30*time.Second {
		return false, nil
	}

	/* Check status */
	return row.Status == "healthy", nil
}

