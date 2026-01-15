/*-------------------------------------------------------------------------
 *
 * marketplace.go
 *    Agent and tool marketplace for NeuronAgent
 *
 * Provides a marketplace for sharing and discovering agents, tools,
 * workflows, and plugins.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/marketplace/marketplace.go
 *
 *-------------------------------------------------------------------------
 */

package marketplace

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
)

/* Marketplace manages the agent and tool marketplace */
type Marketplace struct {
	queries *db.Queries
}

/* MarketplaceItem represents an item in the marketplace */
type MarketplaceItem struct {
	ID          uuid.UUID
	Name        string
	Description string
	Type        MarketplaceItemType
	Author      string
	Version     string
	Rating      float64
	Downloads   int64
	Price       float64
	Free        bool
	Tags        []string
	Metadata    map[string]interface{}
	CreatedAt   time.Time
	UpdatedAt   time.Time
}

/* MarketplaceItemType represents the type of marketplace item */
type MarketplaceItemType string

const (
	MarketplaceItemTypeAgent    MarketplaceItemType = "agent"
	MarketplaceItemTypeTool     MarketplaceItemType = "tool"
	MarketplaceItemTypeWorkflow MarketplaceItemType = "workflow"
	MarketplaceItemTypePlugin   MarketplaceItemType = "plugin"
)

/* NewMarketplace creates a new marketplace */
func NewMarketplace(queries *db.Queries) *Marketplace {
	return &Marketplace{
		queries: queries,
	}
}

/* PublishItem publishes an item to the marketplace */
func (m *Marketplace) PublishItem(ctx context.Context, item *MarketplaceItem) error {
	query := `INSERT INTO neurondb_agent.marketplace_items
		(id, name, description, type, author, version, rating, downloads, price, free, tags, metadata, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::text[], $12::jsonb, $13, $14)
		ON CONFLICT (name, version) DO UPDATE
		SET description = $3, rating = $7, metadata = $12::jsonb, updated_at = $14`

	_, err := m.queries.DB.ExecContext(ctx, query,
		item.ID,
		item.Name,
		item.Description,
		string(item.Type),
		item.Author,
		item.Version,
		item.Rating,
		item.Downloads,
		item.Price,
		item.Free,
		item.Tags,
		item.Metadata,
		item.CreatedAt,
		item.UpdatedAt,
	)

	if err != nil {
		return fmt.Errorf("marketplace publish failed: database_error=true, item_name='%s', error=%w", item.Name, err)
	}

	return nil
}

/* SearchItems searches marketplace items */
func (m *Marketplace) SearchItems(ctx context.Context, query string, itemType *MarketplaceItemType, tags []string, freeOnly bool, limit int, offset int) ([]*MarketplaceItem, error) {
	sqlQuery := `SELECT id, name, description, type, author, version, rating, downloads, price, free, tags, metadata, created_at, updated_at
		FROM neurondb_agent.marketplace_items
		WHERE ($1::text IS NULL OR name ILIKE '%' || $1 || '%' OR description ILIKE '%' || $1 || '%')
		AND ($2::text IS NULL OR type = $2)
		AND ($3::text[] IS NULL OR tags && $3)
		AND ($4::boolean IS NULL OR free = $4)
		ORDER BY rating DESC, downloads DESC
		LIMIT $5 OFFSET $6`

	var itemTypeStr *string
	if itemType != nil {
		typeStr := string(*itemType)
		itemTypeStr = &typeStr
	}

	type ItemRow struct {
		ID          uuid.UUID              `db:"id"`
		Name        string                 `db:"name"`
		Description string                 `db:"description"`
		Type        string                 `db:"type"`
		Author      string                 `db:"author"`
		Version     string                 `db:"version"`
		Rating      float64                `db:"rating"`
		Downloads   int64                  `db:"downloads"`
		Price       float64                `db:"price"`
		Free        bool                   `db:"free"`
		Tags        []string               `db:"tags"`
		Metadata    map[string]interface{} `db:"metadata"`
		CreatedAt   time.Time              `db:"created_at"`
		UpdatedAt   time.Time              `db:"updated_at"`
	}

	var rows []ItemRow
	err := m.queries.DB.SelectContext(ctx, &rows, sqlQuery, query, itemTypeStr, tags, freeOnly, limit, offset)
	if err != nil {
		return nil, fmt.Errorf("marketplace search failed: database_error=true, error=%w", err)
	}

	items := make([]*MarketplaceItem, 0, len(rows))
	for _, row := range rows {
		items = append(items, &MarketplaceItem{
			ID:          row.ID,
			Name:        row.Name,
			Description: row.Description,
			Type:        MarketplaceItemType(row.Type),
			Author:      row.Author,
			Version:     row.Version,
			Rating:      row.Rating,
			Downloads:   row.Downloads,
			Price:       row.Price,
			Free:        row.Free,
			Tags:        row.Tags,
			Metadata:    row.Metadata,
			CreatedAt:   row.CreatedAt,
			UpdatedAt:   row.UpdatedAt,
		})
	}

	return items, nil
}

/* GetItem gets a marketplace item by ID */
func (m *Marketplace) GetItem(ctx context.Context, itemID uuid.UUID) (*MarketplaceItem, error) {
	query := `SELECT id, name, description, type, author, version, rating, downloads, price, free, tags, metadata, created_at, updated_at
		FROM neurondb_agent.marketplace_items
		WHERE id = $1`

	type ItemRow struct {
		ID          uuid.UUID              `db:"id"`
		Name        string                 `db:"name"`
		Description string                 `db:"description"`
		Type        string                 `db:"type"`
		Author      string                 `db:"author"`
		Version     string                 `db:"version"`
		Rating      float64                `db:"rating"`
		Downloads   int64                  `db:"downloads"`
		Price       float64                `db:"price"`
		Free        bool                   `db:"free"`
		Tags        []string               `db:"tags"`
		Metadata    map[string]interface{} `db:"metadata"`
		CreatedAt   time.Time              `db:"created_at"`
		UpdatedAt   time.Time              `db:"updated_at"`
	}

	var row ItemRow
	err := m.queries.DB.GetContext(ctx, &row, query, itemID)
	if err != nil {
		return nil, fmt.Errorf("marketplace item retrieval failed: item_not_found=true, item_id='%s', error=%w", itemID.String(), err)
	}

	return &MarketplaceItem{
		ID:          row.ID,
		Name:        row.Name,
		Description: row.Description,
		Type:        MarketplaceItemType(row.Type),
		Author:      row.Author,
		Version:     row.Version,
		Rating:      row.Rating,
		Downloads:   row.Downloads,
		Price:       row.Price,
		Free:        row.Free,
		Tags:        row.Tags,
		Metadata:    row.Metadata,
		CreatedAt:   row.CreatedAt,
		UpdatedAt:   row.UpdatedAt,
	}, nil
}

/* RateItem rates a marketplace item */
func (m *Marketplace) RateItem(ctx context.Context, itemID uuid.UUID, userID uuid.UUID, rating float64) error {
	/* Store rating */
	query := `INSERT INTO neurondb_agent.marketplace_ratings
		(item_id, user_id, rating, created_at)
		VALUES ($1, $2, $3, $4)
		ON CONFLICT (item_id, user_id) DO UPDATE
		SET rating = $3, updated_at = $4`

	_, err := m.queries.DB.ExecContext(ctx, query, itemID, userID, rating, time.Now())
	if err != nil {
		return fmt.Errorf("marketplace rating failed: database_error=true, error=%w", err)
	}

	/* Update item rating */
	updateQuery := `UPDATE neurondb_agent.marketplace_items
		SET rating = (
			SELECT AVG(rating) FROM neurondb_agent.marketplace_ratings
			WHERE item_id = $1
		)
		WHERE id = $1`

	_, err = m.queries.DB.ExecContext(ctx, updateQuery, itemID)
	if err != nil {
		return fmt.Errorf("marketplace rating update failed: database_error=true, error=%w", err)
	}

	return nil
}

/* DownloadItem records a download of a marketplace item */
func (m *Marketplace) DownloadItem(ctx context.Context, itemID uuid.UUID) error {
	query := `UPDATE neurondb_agent.marketplace_items
		SET downloads = downloads + 1
		WHERE id = $1`

	_, err := m.queries.DB.ExecContext(ctx, query, itemID)
	if err != nil {
		return fmt.Errorf("marketplace download recording failed: database_error=true, error=%w", err)
	}

	return nil
}




