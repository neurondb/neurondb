/*-------------------------------------------------------------------------
 *
 * agent_marketplace.go
 *    Agent marketplace system
 *
 * Provides agent templates, sharing, versioning, and performance metrics.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/marketplace/agent_marketplace.go
 *
 *-------------------------------------------------------------------------
 */

package marketplace

import (
	"context"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
)

/* AgentMarketplace manages agent marketplace */
type AgentMarketplace struct {
	queries *db.Queries
}

/* MarketplaceAgent represents an agent in the marketplace */
type MarketplaceAgent struct {
	ID            uuid.UUID
	Name          string
	Description   string
	Version       string
	Author        string
	Rating        float64
	Downloads     int
	SystemPrompt  string
	ModelName     string
	EnabledTools  []string
	Config        map[string]interface{}
	Performance   AgentPerformance
	Tags          []string
	CreatedAt     string
	UpdatedAt     string
}

/* AgentPerformance represents agent performance metrics */
type AgentPerformance struct {
	SuccessRate    float64
	AvgQuality     float64
	AvgResponseTime float64
	TotalExecutions int
}

/* NewAgentMarketplace creates a new agent marketplace */
func NewAgentMarketplace(queries *db.Queries) *AgentMarketplace {
	return &AgentMarketplace{
		queries: queries,
	}
}

/* PublishAgent publishes an agent to the marketplace */
func (am *AgentMarketplace) PublishAgent(ctx context.Context, agent *MarketplaceAgent) (uuid.UUID, error) {
	if agent.ID == uuid.Nil {
		agent.ID = uuid.New()
	}

	query := `INSERT INTO neurondb_agent.marketplace_agents
		(id, name, description, version, author, system_prompt, model_name, enabled_tools, config, tags, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8::text[], $9::jsonb, $10::text[], NOW(), NOW())
		ON CONFLICT (name, version) DO UPDATE
		SET description = $3, system_prompt = $6, model_name = $7, enabled_tools = $8::text[], config = $9::jsonb, tags = $10::text[], updated_at = NOW()`

	_, err := am.queries.DB.ExecContext(ctx, query,
		agent.ID,
		agent.Name,
		agent.Description,
		agent.Version,
		agent.Author,
		agent.SystemPrompt,
		agent.ModelName,
		agent.EnabledTools,
		agent.Config,
		agent.Tags,
	)

	return agent.ID, err
}

/* ListAgents lists agents in the marketplace */
func (am *AgentMarketplace) ListAgents(ctx context.Context, limit int, offset int) ([]*MarketplaceAgent, error) {
	if limit <= 0 {
		limit = 20
	}

	query := `SELECT id, name, description, version, author, rating, downloads, system_prompt, model_name, enabled_tools, config, tags, created_at, updated_at
		FROM neurondb_agent.marketplace_agents
		ORDER BY rating DESC, downloads DESC
		LIMIT $1 OFFSET $2`

	type AgentRow struct {
		ID           uuid.UUID              `db:"id"`
		Name         string                 `db:"name"`
		Description  string                 `db:"description"`
		Version      string                 `db:"version"`
		Author       string                 `db:"author"`
		Rating       float64                `db:"rating"`
		Downloads    int                    `db:"downloads"`
		SystemPrompt string                 `db:"system_prompt"`
		ModelName    string                 `db:"model_name"`
		EnabledTools []string               `db:"enabled_tools"`
		Config       map[string]interface{} `db:"config"`
		Tags         []string               `db:"tags"`
		CreatedAt    string                 `db:"created_at"`
		UpdatedAt    string                 `db:"updated_at"`
	}

	var rows []AgentRow
	err := am.queries.DB.SelectContext(ctx, &rows, query, limit, offset)
	if err != nil {
		return nil, err
	}

	agents := make([]*MarketplaceAgent, len(rows))
	for i, row := range rows {
		agents[i] = &MarketplaceAgent{
			ID:           row.ID,
			Name:         row.Name,
			Description:  row.Description,
			Version:      row.Version,
			Author:       row.Author,
			Rating:       row.Rating,
			Downloads:    row.Downloads,
			SystemPrompt: row.SystemPrompt,
			ModelName:    row.ModelName,
			EnabledTools: row.EnabledTools,
			Config:       row.Config,
			Tags:         row.Tags,
			CreatedAt:    row.CreatedAt,
			UpdatedAt:    row.UpdatedAt,
		}
	}

	return agents, nil
}

/* UpdatePerformance updates agent performance metrics */
func (am *AgentMarketplace) UpdatePerformance(ctx context.Context, agentID uuid.UUID, performance AgentPerformance) error {
	query := `UPDATE neurondb_agent.marketplace_agents
		SET performance = jsonb_build_object(
			'success_rate', $1,
			'avg_quality', $2,
			'avg_response_time', $3,
			'total_executions', $4
		),
		updated_at = NOW()
		WHERE id = $5`

	_, err := am.queries.DB.ExecContext(ctx, query,
		performance.SuccessRate,
		performance.AvgQuality,
		performance.AvgResponseTime,
		performance.TotalExecutions,
		agentID,
	)

	return err
}

