/*-------------------------------------------------------------------------
 *
 * agent_router.go
 *    Agent routing and delegation
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/agent_router.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"context"
	"fmt"
	"strings"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
)

type AgentRouter struct {
	queries         *db.Queries
	subAgentManager *SubAgentManager
}

/* NewAgentRouter creates a new agent router */
func NewAgentRouter(queries *db.Queries) *AgentRouter {
	return &AgentRouter{queries: queries}
}

/* NewAgentRouterWithSubAgents creates a router with sub-agent support */
func NewAgentRouterWithSubAgents(queries *db.Queries, subAgentManager *SubAgentManager) *AgentRouter {
	return &AgentRouter{
		queries:         queries,
		subAgentManager: subAgentManager,
	}
}

/* RouteToAgent routes a task to the most appropriate agent */
func (r *AgentRouter) RouteToAgent(ctx context.Context, task string, availableAgents []uuid.UUID) (uuid.UUID, error) {
	if len(availableAgents) == 0 {
		return uuid.Nil, fmt.Errorf("agent routing failed: task_length=%d, no_agents_available=true", len(task))
	}

	/* Try sub-agent routing first if available */
	if r.subAgentManager != nil {
		/* Determine required specialization from task */
		specialization := r.determineSpecialization(task)
		if specialization != "" {
			agentID, err := r.subAgentManager.RouteTask(ctx, task, specialization)
			if err == nil && agentID != nil {
				/* Verify agent is in available list */
				for _, availableID := range availableAgents {
					if *agentID == availableID {
						return *agentID, nil
					}
				}
			}
		}
	}

	/* Fallback to simple routing based on agent names and descriptions */
	for _, agentID := range availableAgents {
		agent, err := r.queries.GetAgentByID(ctx, agentID)
		if err != nil {
			continue
		}

		/* Check if agent's name or description matches task keywords */
		if r.matchesTask(agent, task) {
			return agentID, nil
		}
	}

	/* Default to first available agent */
	return availableAgents[0], nil
}

/* determineSpecialization determines required specialization from task */
func (r *AgentRouter) determineSpecialization(task string) string {
	taskLower := strings.ToLower(task)

	/* Planning keywords */
	if strings.Contains(taskLower, "plan") || strings.Contains(taskLower, "strategy") || strings.Contains(taskLower, "design") {
		return "planning"
	}

	/* Research keywords */
	if strings.Contains(taskLower, "research") || strings.Contains(taskLower, "find") || strings.Contains(taskLower, "search") || strings.Contains(taskLower, "analyze") {
		return "research"
	}

	/* Coding keywords */
	if strings.Contains(taskLower, "code") || strings.Contains(taskLower, "program") || strings.Contains(taskLower, "script") || strings.Contains(taskLower, "function") {
		return "coding"
	}

	/* Execution keywords */
	if strings.Contains(taskLower, "execute") || strings.Contains(taskLower, "run") || strings.Contains(taskLower, "perform") || strings.Contains(taskLower, "do") {
		return "execution"
	}

	/* Analysis keywords */
	if strings.Contains(taskLower, "analysis") || strings.Contains(taskLower, "analyze") || strings.Contains(taskLower, "evaluate") {
		return "analysis"
	}

	return ""
}

/* matchesTask checks if an agent matches a task */
func (r *AgentRouter) matchesTask(agent *db.Agent, task string) bool {
	taskLower := strings.ToLower(task)
	nameLower := strings.ToLower(agent.Name)

	/* Check name keywords */
	keywords := []string{"code", "data", "research", "analysis", "sql", "http"}
	for _, keyword := range keywords {
		if strings.Contains(nameLower, keyword) && strings.Contains(taskLower, keyword) {
			return true
		}
	}

	/* Check description */
	if agent.Description != nil {
		descLower := strings.ToLower(*agent.Description)
		for _, keyword := range keywords {
			if strings.Contains(descLower, keyword) && strings.Contains(taskLower, keyword) {
				return true
			}
		}
	}

	return false
}

/* GetSpecializedAgents gets agents specialized for a task type */
func (r *AgentRouter) GetSpecializedAgents(ctx context.Context, taskType string) ([]uuid.UUID, error) {
	query := `SELECT id FROM neurondb_agent.agents
		WHERE name ILIKE $1 OR description ILIKE $1
		ORDER BY created_at DESC`

	var agents []struct {
		ID uuid.UUID `db:"id"`
	}

	err := r.queries.DB.SelectContext(ctx, &agents, query, "%"+taskType+"%")
	if err != nil {
		return nil, fmt.Errorf("specialized agents retrieval failed: task_type='%s', error=%w", taskType, err)
	}

	agentIDs := make([]uuid.UUID, len(agents))
	for i, agent := range agents {
		agentIDs[i] = agent.ID
	}

	return agentIDs, nil
}
