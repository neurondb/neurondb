/*-------------------------------------------------------------------------
 *
 * coordinator.go
 *    Cluster coordination for horizontal scaling
 *
 * Implements cluster coordination for multiple NeuronAgent instances
 * with shared state and session affinity.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/cluster/coordinator.go
 *
 *-------------------------------------------------------------------------
 */

package cluster

import (
	"context"
	"sync"
	"time"

	"github.com/google/uuid"
)

// Coordinator manages cluster coordination for NeuronAgent
type Coordinator struct {
	nodeID      string
	nodes       map[string]*Node
	mu          sync.RWMutex
	sessionMap  map[uuid.UUID]string // Maps session ID to node ID
	lastUpdate  time.Time
}

// Node represents a cluster node
type Node struct {
	ID           string
	Address      string
	LastSeen     time.Time
	ActiveSessions int
	Status       string
}

// NewCoordinator creates a new cluster coordinator
func NewCoordinator(nodeID string) *Coordinator {
	return &Coordinator{
		nodeID:     nodeID,
		nodes:      make(map[string]*Node),
		sessionMap: make(map[uuid.UUID]string),
		lastUpdate: time.Now(),
	}
}

// RegisterNode registers a node in the cluster
func (c *Coordinator) RegisterNode(nodeID, address string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.nodes[nodeID] = &Node{
		ID:       nodeID,
		Address:  address,
		LastSeen: time.Now(),
		Status:   "active",
	}
	c.lastUpdate = time.Now()
}

// GetNodeForSession returns the node handling a session
func (c *Coordinator) GetNodeForSession(sessionID uuid.UUID) (string, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	nodeID, exists := c.sessionMap[sessionID]
	return nodeID, exists
}

// AssignSession assigns a session to a node
func (c *Coordinator) AssignSession(sessionID uuid.UUID, nodeID string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.sessionMap[sessionID] = nodeID
	if node, exists := c.nodes[nodeID]; exists {
		node.ActiveSessions++
	}
}

// GetAvailableNodes returns list of available nodes
func (c *Coordinator) GetAvailableNodes() []*Node {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var available []*Node
	now := time.Now()

	for _, node := range c.nodes {
		// Consider node available if seen within last 30 seconds
		if now.Sub(node.LastSeen) < 30*time.Second && node.Status == "active" {
			available = append(available, node)
		}
	}

	return available
}

// SelectNodeForSession selects the best node for a new session
func (c *Coordinator) SelectNodeForSession(ctx context.Context) (string, error) {
	available := c.GetAvailableNodes()

	if len(available) == 0 {
		return "", ErrNoAvailableNodes
	}

	// Simple round-robin or least-loaded selection
	// In production, would use more sophisticated load balancing
	bestNode := available[0]
	minSessions := bestNode.ActiveSessions

	for _, node := range available[1:] {
		if node.ActiveSessions < minSessions {
			bestNode = node
			minSessions = node.ActiveSessions
		}
	}

	return bestNode.ID, nil
}

var ErrNoAvailableNodes = &ClusterError{Message: "no available nodes in cluster"}

type ClusterError struct {
	Message string
}

func (e *ClusterError) Error() string {
	return e.Message
}



