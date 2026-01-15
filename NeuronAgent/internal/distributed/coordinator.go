/*-------------------------------------------------------------------------
 *
 * coordinator.go
 *    Distributed agent execution coordinator
 *
 * Provides distributed agent execution across multiple nodes with
 * session affinity, distributed memory, and leader election.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/distributed/coordinator.go
 *
 *-------------------------------------------------------------------------
 */

package distributed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"hash/crc32"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/agent"
	"github.com/neurondb/NeuronAgent/internal/config"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/metrics"
)

/* Coordinator manages distributed agent execution */
type Coordinator struct {
	nodeID        string
	queries       *db.Queries
	runtime       *agent.Runtime
	ring          *ConsistentHashRing
	leaderElection *LeaderElection
	memoryPubSub  *MemoryPubSub
	mu            sync.RWMutex
	nodes         map[string]*Node
	enabled       bool
	config        *config.DistributedConfig
	httpClient    *http.Client
}

/* Node represents a cluster node */
type Node struct {
	ID        string
	Address   string
	Port      int
	Status    NodeStatus
	LastSeen  time.Time
	Load      float64
	Capabilities []string
}

/* NodeStatus represents node health status */
type NodeStatus string

const (
	NodeStatusHealthy   NodeStatus = "healthy"
	NodeStatusUnhealthy NodeStatus = "unhealthy"
	NodeStatusUnknown   NodeStatus = "unknown"
)

/* ConsistentHashRing provides consistent hashing for session affinity */
type ConsistentHashRing struct {
	nodes   []string
	replicas int
	mu      sync.RWMutex
}

/* NewCoordinator creates a new distributed coordinator */
func NewCoordinator(nodeID string, queries *db.Queries, runtime *agent.Runtime, cfg *config.DistributedConfig) *Coordinator {
	if cfg == nil {
		cfg = &config.DistributedConfig{
			Enabled:     false,
			NodeAddress: "localhost",
			NodePort:    8080,
			RPCTimeout:  30 * time.Second,
		}
	}

	return &Coordinator{
		nodeID:        nodeID,
		queries:       queries,
		runtime:       runtime,
		ring:          NewConsistentHashRing(150), // 150 virtual nodes per physical node
		leaderElection: NewLeaderElection(nodeID, queries),
		memoryPubSub:   NewMemoryPubSub(queries),
		nodes:         make(map[string]*Node),
		enabled:       false,
		config:        cfg,
		httpClient: &http.Client{
			Timeout: cfg.RPCTimeout,
		},
	}
}

/* Enable enables distributed mode */
func (c *Coordinator) Enable(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.enabled {
		return nil
	}

	/* Register this node */
	if err := c.registerNode(ctx); err != nil {
		return fmt.Errorf("coordinator enable failed: node_registration_error=true, error=%w", err)
	}

	/* Start leader election */
	if err := c.leaderElection.Start(ctx); err != nil {
		return fmt.Errorf("coordinator enable failed: leader_election_start_error=true, error=%w", err)
	}

	/* Start memory pub-sub */
	if err := c.memoryPubSub.Start(ctx); err != nil {
		return fmt.Errorf("coordinator enable failed: memory_pubsub_start_error=true, error=%w", err)
	}

	/* Start node discovery */
	go c.discoverNodes(ctx)

	c.enabled = true
	metrics.InfoWithContext(ctx, "Distributed coordinator enabled", map[string]interface{}{
		"node_id": c.nodeID,
	})

	return nil
}

/* Disable disables distributed mode */
func (c *Coordinator) Disable(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if !c.enabled {
		return nil
	}

	c.leaderElection.Stop(ctx)
	c.memoryPubSub.Stop(ctx)
	c.enabled = false

	return nil
}

/* ExecuteAgent executes an agent on the appropriate node */
func (c *Coordinator) ExecuteAgent(ctx context.Context, sessionID uuid.UUID, userMessage string) (*agent.ExecutionState, error) {
	if !c.enabled {
		/* Fall back to local execution */
		return c.runtime.Execute(ctx, sessionID, userMessage)
	}

	/* Determine target node using consistent hashing */
	targetNodeID := c.ring.GetNode(sessionID.String())
	
	if targetNodeID == c.nodeID {
		/* Execute locally */
		return c.runtime.Execute(ctx, sessionID, userMessage)
	}

	/* Execute on remote node */
	return c.executeRemote(ctx, targetNodeID, sessionID, userMessage)
}

/* executeRemote executes agent on remote node */
func (c *Coordinator) executeRemote(ctx context.Context, nodeID string, sessionID uuid.UUID, userMessage string) (*agent.ExecutionState, error) {
	c.mu.RLock()
	node, exists := c.nodes[nodeID]
	c.mu.RUnlock()

	if !exists || node.Status != NodeStatusHealthy {
		/* Fall back to local execution if remote node unavailable */
		metrics.WarnWithContext(ctx, "Remote node unavailable, executing locally", map[string]interface{}{
			"node_id":    nodeID,
			"session_id": sessionID.String(),
		})
		return c.runtime.Execute(ctx, sessionID, userMessage)
	}

	/* Build RPC request */
	url := fmt.Sprintf("http://%s:%d/api/v1/sessions/%s/messages", node.Address, node.Port, sessionID.String())
	
	requestBody := map[string]interface{}{
		"content": userMessage,
	}

	jsonBody, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("rpc request marshal failed: error=%w", err)
	}

	/* Create HTTP request with timeout */
	reqCtx, cancel := context.WithTimeout(ctx, c.config.RPCTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(reqCtx, "POST", url, bytes.NewBuffer(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("rpc request creation failed: error=%w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	/* Add API key if available from context or config */
	/* For now, remote nodes would need to share API keys or use internal auth */

	/* Execute HTTP request with retry logic */
	var resp *http.Response
	maxRetries := 3
	for attempt := 0; attempt < maxRetries; attempt++ {
		resp, err = c.httpClient.Do(req)
		if err == nil {
			break
		}
		if attempt < maxRetries-1 {
			time.Sleep(time.Duration(attempt+1) * 100 * time.Millisecond)
		}
	}

	if err != nil {
		metrics.WarnWithContext(ctx, "RPC call failed, executing locally", map[string]interface{}{
			"node_id":    nodeID,
			"session_id": sessionID.String(),
			"error":      err.Error(),
		})
		return c.runtime.Execute(ctx, sessionID, userMessage)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		metrics.WarnWithContext(ctx, "RPC call returned error, executing locally", map[string]interface{}{
			"node_id":     nodeID,
			"session_id":  sessionID.String(),
			"status_code": resp.StatusCode,
			"response":    string(bodyBytes),
		})
		return c.runtime.Execute(ctx, sessionID, userMessage)
	}

	/* Parse response - API returns a map, convert to ExecutionState */
	var apiResponse map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&apiResponse); err != nil {
		metrics.WarnWithContext(ctx, "RPC response parse failed, executing locally", map[string]interface{}{
			"node_id":    nodeID,
			"session_id": sessionID.String(),
			"error":      err.Error(),
		})
		return c.runtime.Execute(ctx, sessionID, userMessage)
	}

	/* Convert API response to ExecutionState */
	executionState := &agent.ExecutionState{
		SessionID: sessionID,
	}

	if agentIDStr, ok := apiResponse["agent_id"].(string); ok {
		if agentID, err := uuid.Parse(agentIDStr); err == nil {
			executionState.AgentID = agentID
		}
	}

	if finalAnswer, ok := apiResponse["response"].(string); ok {
		executionState.FinalAnswer = finalAnswer
	}

	if tokensUsed, ok := apiResponse["tokens_used"].(float64); ok {
		executionState.TokensUsed = int(tokensUsed)
	}

	if toolCallsRaw, ok := apiResponse["tool_calls"].([]interface{}); ok {
		for _, tc := range toolCallsRaw {
			if tcMap, ok := tc.(map[string]interface{}); ok {
				toolCall := agent.ToolCall{}
				if id, ok := tcMap["id"].(string); ok {
					toolCall.ID = id
				}
				if name, ok := tcMap["name"].(string); ok {
					toolCall.Name = name
				}
				if args, ok := tcMap["arguments"].(map[string]interface{}); ok {
					toolCall.Arguments = args
				}
				executionState.ToolCalls = append(executionState.ToolCalls, toolCall)
			}
		}
	}

	if toolResultsRaw, ok := apiResponse["tool_results"].([]interface{}); ok {
		for _, tr := range toolResultsRaw {
			if trMap, ok := tr.(map[string]interface{}); ok {
				toolResult := agent.ToolResult{}
				if id, ok := trMap["tool_call_id"].(string); ok {
					toolResult.ToolCallID = id
				}
				if content, ok := trMap["content"].(string); ok {
					toolResult.Content = content
				}
				executionState.ToolResults = append(executionState.ToolResults, toolResult)
			}
		}
	}

	metrics.InfoWithContext(ctx, "RPC call succeeded", map[string]interface{}{
		"node_id":    nodeID,
		"session_id": sessionID.String(),
	})

	return executionState, nil
}

/* registerNode registers this node in the cluster */
func (c *Coordinator) registerNode(ctx context.Context) error {
	query := `INSERT INTO neurondb_agent.cluster_nodes
		(node_id, address, port, status, last_seen, capabilities, created_at)
		VALUES ($1, $2, $3, $4, $5, $6::jsonb, NOW())
		ON CONFLICT (node_id) DO UPDATE
		SET address = $2, port = $3, status = $4, last_seen = $5, capabilities = $6::jsonb`

	/* Get node address from config */
	address := c.config.NodeAddress
	port := c.config.NodePort

	_, err := c.queries.DB.ExecContext(ctx, query,
		c.nodeID,
		address,
		port,
		string(NodeStatusHealthy),
		time.Now(),
		[]string{"agent_execution", "memory", "tools"}, // Default capabilities
	)

	if err != nil {
		return fmt.Errorf("node registration failed: database_error=true, error=%w", err)
	}

	/* Add to local ring */
	c.ring.AddNode(c.nodeID)

	/* Add to local nodes map */
	c.mu.Lock()
	c.nodes[c.nodeID] = &Node{
		ID:          c.nodeID,
		Address:     address,
		Port:        port,
		Status:      NodeStatusHealthy,
		LastSeen:    time.Now(),
		Capabilities: []string{"agent_execution", "memory", "tools"},
	}
	c.mu.Unlock()

	return nil
}

/* discoverNodes periodically discovers other nodes in the cluster */
func (c *Coordinator) discoverNodes(ctx context.Context) {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			c.updateNodeList(ctx)
		}
	}
}

/* updateNodeList updates the list of available nodes */
func (c *Coordinator) updateNodeList(ctx context.Context) {
	query := `SELECT node_id, address, port, status, last_seen, capabilities
		FROM neurondb_agent.cluster_nodes
		WHERE last_seen > NOW() - INTERVAL '30 seconds'
		ORDER BY node_id`

	type NodeRow struct {
		NodeID      string    `db:"node_id"`
		Address     string    `db:"address"`
		Port        int       `db:"port"`
		Status      string    `db:"status"`
		LastSeen    time.Time `db:"last_seen"`
		Capabilities []string `db:"capabilities"`
	}

	var rows []NodeRow
	err := c.queries.DB.SelectContext(ctx, &rows, query)
	if err != nil {
		metrics.WarnWithContext(ctx, "Failed to discover nodes", map[string]interface{}{
			"error": err.Error(),
		})
		return
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	/* Update ring */
	c.ring.Clear()
	for _, row := range rows {
		if row.Status == string(NodeStatusHealthy) {
			c.ring.AddNode(row.NodeID)
			c.nodes[row.NodeID] = &Node{
				ID:          row.NodeID,
				Address:     row.Address,
				Port:        row.Port,
				Status:      NodeStatus(row.Status),
				LastSeen:    row.LastSeen,
				Capabilities: row.Capabilities,
			}
		}
	}

	/* Remove stale nodes */
	for nodeID, node := range c.nodes {
		if time.Since(node.LastSeen) > 30*time.Second {
			delete(c.nodes, nodeID)
		}
	}
}

/* NewConsistentHashRing creates a new consistent hash ring */
func NewConsistentHashRing(replicas int) *ConsistentHashRing {
	return &ConsistentHashRing{
		nodes:   make([]string, 0),
		replicas: replicas,
	}
}

/* AddNode adds a node to the hash ring */
func (r *ConsistentHashRing) AddNode(nodeID string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	/* Check if already exists */
	for _, n := range r.nodes {
		if n == nodeID {
			return
		}
	}

	r.nodes = append(r.nodes, nodeID)
}

/* RemoveNode removes a node from the hash ring */
func (r *ConsistentHashRing) RemoveNode(nodeID string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	for i, n := range r.nodes {
		if n == nodeID {
			r.nodes = append(r.nodes[:i], r.nodes[i+1:]...)
			return
		}
	}
}

/* Clear clears all nodes from the ring */
func (r *ConsistentHashRing) Clear() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.nodes = make([]string, 0)
}

/* GetNode returns the node responsible for a given key */
func (r *ConsistentHashRing) GetNode(key string) string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if len(r.nodes) == 0 {
		return ""
	}

	/* Use consistent hashing */
	hash := crc32.ChecksumIEEE([]byte(key))
	index := int(hash) % len(r.nodes)
	return r.nodes[index]
}

/* IsEnabled returns whether distributed mode is enabled */
func (c *Coordinator) IsEnabled() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.enabled
}

