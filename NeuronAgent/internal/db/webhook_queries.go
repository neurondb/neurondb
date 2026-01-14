/*-------------------------------------------------------------------------
 *
 * webhook_queries.go
 *    Database queries for webhooks
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/db/webhook_queries.go
 *
 *-------------------------------------------------------------------------
 */

package db

import (
	"context"
	"database/sql"
	"fmt"

	"github.com/google/uuid"
	"github.com/lib/pq"
)

/* Webhook queries */
const (
	createWebhookQuery = `
		INSERT INTO neurondb_agent.webhooks 
		(url, events, secret, enabled, timeout_seconds, retry_count, metadata)
		VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
		RETURNING id, created_at, updated_at`

	getWebhookQuery = `SELECT * FROM neurondb_agent.webhooks WHERE id = $1`

	listWebhooksQuery = `SELECT * FROM neurondb_agent.webhooks ORDER BY created_at DESC`

	updateWebhookQuery = `
		UPDATE neurondb_agent.webhooks 
		SET url = $2, events = $3, secret = $4, enabled = $5, timeout_seconds = $6, retry_count = $7, metadata = $8::jsonb, updated_at = NOW()
		WHERE id = $1
		RETURNING updated_at`

	deleteWebhookQuery = `DELETE FROM neurondb_agent.webhooks WHERE id = $1`

	listWebhookDeliveriesQuery = `
		SELECT * FROM neurondb_agent.webhook_deliveries 
		WHERE webhook_id = $1 
		ORDER BY created_at DESC 
		LIMIT $2 OFFSET $3`
)

/* Webhook represents a webhook configuration */
type Webhook struct {
	ID             uuid.UUID      `db:"id"`
	URL            string         `db:"url"`
	Events         pq.StringArray `db:"events"`
	Secret         *string        `db:"secret"`
	Enabled        bool           `db:"enabled"`
	TimeoutSeconds int            `db:"timeout_seconds"`
	RetryCount     int            `db:"retry_count"`
	Metadata       JSONBMap       `db:"metadata"`
	CreatedAt      string         `db:"created_at"`
	UpdatedAt      string         `db:"updated_at"`
}

/* WebhookDelivery represents a webhook delivery */
type WebhookDelivery struct {
	ID           uuid.UUID `db:"id"`
	WebhookID    uuid.UUID `db:"webhook_id"`
	EventType    string    `db:"event_type"`
	Payload      JSONBMap  `db:"payload"`
	Status       string    `db:"status"`
	StatusCode   *int      `db:"status_code"`
	ResponseBody *string   `db:"response_body"`
	ErrorMessage *string   `db:"error_message"`
	AttemptCount int       `db:"attempt_count"`
	NextRetryAt  *string   `db:"next_retry_at"`
	CreatedAt    string    `db:"created_at"`
	DeliveredAt  *string   `db:"delivered_at"`
}

/* CreateWebhook creates a webhook */
func (q *Queries) CreateWebhook(ctx context.Context, webhook *Webhook) error {
	params := []interface{}{
		webhook.URL, webhook.Events, webhook.Secret, webhook.Enabled,
		webhook.TimeoutSeconds, webhook.RetryCount, webhook.Metadata,
	}
	err := q.DB.GetContext(ctx, webhook, createWebhookQuery, params...)
	if err != nil {
		return q.formatQueryError("INSERT", createWebhookQuery, len(params), "neurondb_agent.webhooks", err)
	}
	return nil
}

/* GetWebhook gets a webhook by ID */
func (q *Queries) GetWebhook(ctx context.Context, id uuid.UUID) (*Webhook, error) {
	var webhook Webhook
	err := q.DB.GetContext(ctx, &webhook, getWebhookQuery, id)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("webhook not found on %s: query='%s', webhook_id='%s', table='neurondb_agent.webhooks', error=%w",
			q.getConnInfoString(), getWebhookQuery, id.String(), err)
	}
	if err != nil {
		return nil, q.formatQueryError("SELECT", getWebhookQuery, 1, "neurondb_agent.webhooks", err)
	}
	return &webhook, nil
}

/* ListWebhooks lists all webhooks */
func (q *Queries) ListWebhooks(ctx context.Context) ([]Webhook, error) {
	var webhooks []Webhook
	err := q.DB.SelectContext(ctx, &webhooks, listWebhooksQuery)
	if err != nil {
		return nil, q.formatQueryError("SELECT", listWebhooksQuery, 0, "neurondb_agent.webhooks", err)
	}
	return webhooks, nil
}

/* UpdateWebhook updates a webhook */
func (q *Queries) UpdateWebhook(ctx context.Context, webhook *Webhook) error {
	params := []interface{}{
		webhook.ID, webhook.URL, webhook.Events, webhook.Secret, webhook.Enabled,
		webhook.TimeoutSeconds, webhook.RetryCount, webhook.Metadata,
	}
	err := q.DB.GetContext(ctx, webhook, updateWebhookQuery, params...)
	if err == sql.ErrNoRows {
		return fmt.Errorf("webhook not found on %s: query='%s', webhook_id='%s', table='neurondb_agent.webhooks', error=%w",
			q.getConnInfoString(), updateWebhookQuery, webhook.ID.String(), err)
	}
	if err != nil {
		return q.formatQueryError("UPDATE", updateWebhookQuery, len(params), "neurondb_agent.webhooks", err)
	}
	return nil
}

/* DeleteWebhook deletes a webhook */
func (q *Queries) DeleteWebhook(ctx context.Context, id uuid.UUID) error {
	result, err := q.DB.ExecContext(ctx, deleteWebhookQuery, id)
	if err != nil {
		return q.formatQueryError("DELETE", deleteWebhookQuery, 1, "neurondb_agent.webhooks", err)
	}
	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected for DELETE on %s: query='%s', webhook_id='%s', table='neurondb_agent.webhooks', error=%w",
			q.getConnInfoString(), deleteWebhookQuery, id.String(), err)
	}
	if rowsAffected == 0 {
		return fmt.Errorf("webhook not found on %s: query='%s', webhook_id='%s', table='neurondb_agent.webhooks', rows_affected=0",
			q.getConnInfoString(), deleteWebhookQuery, id.String())
	}
	return nil
}

/* ListWebhookDeliveries lists deliveries for a webhook */
func (q *Queries) ListWebhookDeliveries(ctx context.Context, webhookID uuid.UUID, limit, offset int) ([]WebhookDelivery, error) {
	var deliveries []WebhookDelivery
	params := []interface{}{webhookID, limit, offset}
	err := q.DB.SelectContext(ctx, &deliveries, listWebhookDeliveriesQuery, params...)
	if err != nil {
		return nil, q.formatQueryError("SELECT", listWebhookDeliveriesQuery, len(params), "neurondb_agent.webhook_deliveries", err)
	}
	return deliveries, nil
}
