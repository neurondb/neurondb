/*-------------------------------------------------------------------------
 *
 * sub_agents.go
 *    Specialized sub-agents for task routing and coordination
 *
 * Provides specialized agent types (planning, research, coding, execution)
 * with automatic task routing and sub-agent coordination.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/sub_agents.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"context"
	"fmt"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
)

/* SubAgentManager manages specialized sub-agents */
type SubAgentManager struct {
	queries *db.Queries
	runtime *Runtime
}

/* SubAgent represents a specialized agent */
type SubAgent struct {
	ID             uuid.UUID
	AgentID        uuid.UUID
	Specialization string
	Capabilities   []string
	Config         map[string]interface{}
}

/* NewSubAgentManager creates a new sub-agent manager */
func NewSubAgentManager(queries *db.Queries, runtime *Runtime) *SubAgentManager {
	return &SubAgentManager{
		queries: queries,
		runtime: runtime,
	}
}

/* CreateSubAgent creates a specialized sub-agent */
func (m *SubAgentManager) CreateSubAgent(ctx context.Context, agentID uuid.UUID, specialization string, capabilities []string, config map[string]interface{}) (*SubAgent, error) {
	/* Validate specialization */
	validSpecializations := map[string]bool{
		"planning":  true,
		"research":  true,
		"coding":    true,
		"execution": true,
		"analysis":  true,
		"general":   true,
	}

	if !validSpecializations[specialization] {
		return nil, fmt.Errorf("sub-agent creation failed: invalid_specialization='%s'", specialization)
	}

	/* Check if agent exists */
	_, err := m.queries.GetAgentByID(ctx, agentID)
	if err != nil {
		return nil, fmt.Errorf("sub-agent creation failed: agent_not_found=true, agent_id='%s', error=%w", agentID.String(), err)
	}

	/* Insert or update specialization */
	query := `INSERT INTO neurondb_agent.agent_specializations
		(agent_id, specialization_type, capabilities, config, updated_at)
		VALUES ($1, $2, $3, $4, NOW())
		ON CONFLICT (agent_id) DO UPDATE SET
			specialization_type = EXCLUDED.specialization_type,
			capabilities = EXCLUDED.capabilities,
			config = EXCLUDED.config,
			updated_at = NOW()
		RETURNING id, agent_id, specialization_type, capabilities, config`

	subAgent := &SubAgent{}
	err = m.queries.DB.QueryRowContext(ctx, query,
		agentID, specialization, capabilities, db.FromMap(config),
	).Scan(
		&subAgent.ID, &subAgent.AgentID, &subAgent.Specialization,
		&subAgent.Capabilities, &subAgent.Config,
	)

	if err != nil {
		return nil, fmt.Errorf("sub-agent creation failed: agent_id='%s', specialization='%s', error=%w",
			agentID.String(), specialization, err)
	}

	return subAgent, nil
}

/* RouteTask routes a task to the appropriate sub-agent based on specialization */
func (m *SubAgentManager) RouteTask(ctx context.Context, task string, requiredSpecialization string) (*uuid.UUID, error) {
	/* Find agents with the required specialization */
	query := `SELECT agent_id FROM neurondb_agent.agent_specializations
		WHERE specialization_type = $1
		ORDER BY created_at ASC
		LIMIT 1`

	var agentID uuid.UUID
	err := m.queries.DB.QueryRowContext(ctx, query, requiredSpecialization).Scan(&agentID)
	if err != nil {
		return nil, fmt.Errorf("task routing failed: specialization='%s', no_agent_found=true, error=%w", requiredSpecialization, err)
	}

	return &agentID, nil
}

/* CoordinateSubAgents coordinates multiple sub-agents for a complex task */
func (m *SubAgentManager) CoordinateSubAgents(ctx context.Context, task string, specializations []string) (map[string]interface{}, error) {
	results := make(map[string]interface{})

	for _, spec := range specializations {
		agentID, err := m.RouteTask(ctx, task, spec)
		if err != nil {
			results[spec] = map[string]interface{}{
				"error": err.Error(),
			}
			continue
		}

		/* Execute task with routed agent */
		/* This would typically create a session and execute */
		results[spec] = map[string]interface{}{
			"agent_id": agentID.String(),
			"status":   "routed",
		}
	}

	return results, nil
}

/* GetSubAgentCapabilities retrieves capabilities for a specialization */
func (m *SubAgentManager) GetSubAgentCapabilities(ctx context.Context, specialization string) ([]*SubAgent, error) {
	query := `SELECT id, agent_id, specialization_type, capabilities, config
		FROM neurondb_agent.agent_specializations
		WHERE specialization_type = $1
		ORDER BY created_at ASC`

	rows, err := m.queries.DB.QueryContext(ctx, query, specialization)
	if err != nil {
		return nil, fmt.Errorf("capabilities retrieval failed: specialization='%s', error=%w", specialization, err)
	}
	defer rows.Close()

	var subAgents []*SubAgent
	for rows.Next() {
		subAgent := &SubAgent{}
		err := rows.Scan(
			&subAgent.ID, &subAgent.AgentID, &subAgent.Specialization,
			&subAgent.Capabilities, &subAgent.Config,
		)
		if err != nil {
			continue
		}
		subAgents = append(subAgents, subAgent)
	}

	return subAgents, nil
}

/* GetAgentSpecialization retrieves specialization for an agent */
func (m *SubAgentManager) GetAgentSpecialization(ctx context.Context, agentID uuid.UUID) (*SubAgent, error) {
	query := `SELECT id, agent_id, specialization_type, capabilities, config
		FROM neurondb_agent.agent_specializations
		WHERE agent_id = $1`

	subAgent := &SubAgent{}
	err := m.queries.DB.QueryRowContext(ctx, query, agentID).Scan(
		&subAgent.ID, &subAgent.AgentID, &subAgent.Specialization,
		&subAgent.Capabilities, &subAgent.Config,
	)

	if err != nil {
		return nil, fmt.Errorf("specialization retrieval failed: agent_id='%s', error=%w", agentID.String(), err)
	}

	return subAgent, nil
}
