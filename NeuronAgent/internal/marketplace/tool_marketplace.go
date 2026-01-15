/*-------------------------------------------------------------------------
 *
 * tool_marketplace.go
 *    Tool marketplace system
 *
 * Provides tool discovery, sharing, versioning, and rating system.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/marketplace/tool_marketplace.go
 *
 *-------------------------------------------------------------------------
 */

package marketplace

import (
	"context"
	"fmt"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
)

/* ToolMarketplace manages tool marketplace */
type ToolMarketplace struct {
	queries *db.Queries
}

/* MarketplaceTool represents a tool in the marketplace */
type MarketplaceTool struct {
	ID          uuid.UUID
	Name        string
	Description string
	Version     string
	Author      string
	Rating      float64
	Downloads   int
	Schema      map[string]interface{}
	Code        string
	Tags        []string
	CreatedAt   string
	UpdatedAt   string
}

/* NewToolMarketplace creates a new tool marketplace */
func NewToolMarketplace(queries *db.Queries) *ToolMarketplace {
	return &ToolMarketplace{
		queries: queries,
	}
}

/* PublishTool publishes a tool to the marketplace */
func (tm *ToolMarketplace) PublishTool(ctx context.Context, tool *MarketplaceTool) (uuid.UUID, error) {
	if tool.ID == uuid.Nil {
		tool.ID = uuid.New()
	}

	query := `INSERT INTO neurondb_agent.marketplace_tools
		(id, name, description, version, author, schema, code, tags, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8::text[], NOW(), NOW())
		ON CONFLICT (name, version) DO UPDATE
		SET description = $3, schema = $6::jsonb, code = $7, tags = $8::text[], updated_at = NOW()`

	_, err := tm.queries.DB.ExecContext(ctx, query,
		tool.ID,
		tool.Name,
		tool.Description,
		tool.Version,
		tool.Author,
		tool.Schema,
		tool.Code,
		tool.Tags,
	)

	return tool.ID, err
}

/* ListTools lists tools in the marketplace */
func (tm *ToolMarketplace) ListTools(ctx context.Context, limit int, offset int) ([]*MarketplaceTool, error) {
	if limit <= 0 {
		limit = 20
	}

	query := `SELECT id, name, description, version, author, rating, downloads, schema, tags, created_at, updated_at
		FROM neurondb_agent.marketplace_tools
		ORDER BY rating DESC, downloads DESC
		LIMIT $1 OFFSET $2`

	type ToolRow struct {
		ID          uuid.UUID              `db:"id"`
		Name        string                 `db:"name"`
		Description string                 `db:"description"`
		Version     string                 `db:"version"`
		Author      string                 `db:"author"`
		Rating      float64                `db:"rating"`
		Downloads   int                    `db:"downloads"`
		Schema      map[string]interface{} `db:"schema"`
		Tags        []string               `db:"tags"`
		CreatedAt   string                 `db:"created_at"`
		UpdatedAt   string                 `db:"updated_at"`
	}

	var rows []ToolRow
	err := tm.queries.DB.SelectContext(ctx, &rows, query, limit, offset)
	if err != nil {
		return nil, err
	}

	tools := make([]*MarketplaceTool, len(rows))
	for i, row := range rows {
		tools[i] = &MarketplaceTool{
			ID:          row.ID,
			Name:        row.Name,
			Description: row.Description,
			Version:     row.Version,
			Author:      row.Author,
			Rating:      row.Rating,
			Downloads:   row.Downloads,
			Schema:      row.Schema,
			Tags:        row.Tags,
			CreatedAt:   row.CreatedAt,
			UpdatedAt:   row.UpdatedAt,
		}
	}

	return tools, nil
}

/* RateTool rates a tool */
func (tm *ToolMarketplace) RateTool(ctx context.Context, toolID uuid.UUID, userID string, rating float64, review string) error {
	if rating < 0 || rating > 5 {
		return fmt.Errorf("rating must be between 0 and 5")
	}

	query := `INSERT INTO neurondb_agent.tool_ratings
		(id, tool_id, user_id, rating, review, created_at)
		VALUES (gen_random_uuid(), $1, $2, $3, $4, NOW())
		ON CONFLICT (tool_id, user_id) DO UPDATE
		SET rating = $3, review = $4, updated_at = NOW()`

	_, err := tm.queries.DB.ExecContext(ctx, query, toolID, userID, rating, review)
	if err != nil {
		return err
	}

	/* Update tool average rating */
	updateQuery := `UPDATE neurondb_agent.marketplace_tools
		SET rating = (
			SELECT AVG(rating) FROM neurondb_agent.tool_ratings WHERE tool_id = $1
		)
		WHERE id = $1`

	_, err = tm.queries.DB.ExecContext(ctx, updateQuery, toolID)
	return err
}

