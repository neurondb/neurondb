/*-------------------------------------------------------------------------
 *
 * webhook.go
 *    Webhook system for events and notifications
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/webhooks/webhook.go
 *
 *-------------------------------------------------------------------------
 */

package webhooks

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/jmoiron/sqlx"
)

/* EventType represents a webhook event type */
type EventType string

const (
	EventAgentCreated   EventType = "agent.created"
	EventAgentUpdated   EventType = "agent.updated"
	EventAgentDeleted   EventType = "agent.deleted"
	EventSessionCreated EventType = "session.created"
	EventMessageSent    EventType = "message.sent"
	EventToolExecuted   EventType = "tool.executed"
	EventJobCompleted   EventType = "job.completed"
	EventJobFailed      EventType = "job.failed"
	EventBudgetExceeded EventType = "budget.exceeded"
)

/* Webhook represents a webhook configuration */
type Webhook struct {
	ID             uuid.UUID              `db:"id"`
	URL            string                 `db:"url"`
	Events         []string               `db:"events"`
	Secret         *string                `db:"secret"`
	Enabled        bool                   `db:"enabled"`
	TimeoutSeconds int                    `db:"timeout_seconds"`
	RetryCount     int                    `db:"retry_count"`
	Metadata       map[string]interface{} `db:"metadata"`
	CreatedAt      time.Time              `db:"created_at"`
	UpdatedAt      time.Time              `db:"updated_at"`
}

/* WebhookDelivery represents a webhook delivery attempt */
type WebhookDelivery struct {
	ID           uuid.UUID              `db:"id"`
	WebhookID    uuid.UUID              `db:"webhook_id"`
	EventType    string                 `db:"event_type"`
	Payload      map[string]interface{} `db:"payload"`
	Status       string                 `db:"status"`
	StatusCode   *int                   `db:"status_code"`
	ResponseBody *string                `db:"response_body"`
	ErrorMessage *string                `db:"error_message"`
	AttemptCount int                    `db:"attempt_count"`
	NextRetryAt  *time.Time             `db:"next_retry_at"`
	CreatedAt    time.Time              `db:"created_at"`
	DeliveredAt  *time.Time             `db:"delivered_at"`
}

/* WebhookManager manages webhook delivery */
type WebhookManager struct {
	db      *sqlx.DB
	client  *http.Client
	workers int
	queue   chan *WebhookDelivery
	wg      sync.WaitGroup
	stop    chan struct{}
}

/* NewWebhookManager creates a new webhook manager */
func NewWebhookManager(db *sqlx.DB, workers int) *WebhookManager {
	return &WebhookManager{
		db:      db,
		client:  &http.Client{Timeout: 30 * time.Second},
		workers: workers,
		queue:   make(chan *WebhookDelivery, 1000),
		stop:    make(chan struct{}),
	}
}

/* Start starts the webhook delivery workers */
func (wm *WebhookManager) Start() {
	for i := 0; i < wm.workers; i++ {
		wm.wg.Add(1)
		go wm.worker()
	}
}

/* Stop stops the webhook delivery workers */
func (wm *WebhookManager) Stop() {
	close(wm.stop)
	wm.wg.Wait()
}

/* worker processes webhook deliveries */
func (wm *WebhookManager) worker() {
	defer wm.wg.Done()

	for {
		select {
		case <-wm.stop:
			return
		case delivery := <-wm.queue:
			wm.deliver(delivery)
		}
	}
}

/* Trigger triggers a webhook event */
func (wm *WebhookManager) Trigger(ctx context.Context, eventType EventType, payload map[string]interface{}) error {
	/* Get enabled webhooks for this event */
	query := `SELECT * FROM neurondb_agent.webhooks 
		WHERE enabled = true AND $1 = ANY(events)`

	var webhooks []Webhook
	err := wm.db.SelectContext(ctx, &webhooks, query, string(eventType))
	if err != nil {
		return fmt.Errorf("failed to get webhooks: %w", err)
	}

	/* Create deliveries for each webhook */
	for _, webhook := range webhooks {
		delivery := &WebhookDelivery{
			ID:           uuid.New(),
			WebhookID:    webhook.ID,
			EventType:    string(eventType),
			Payload:      payload,
			Status:       "pending",
			AttemptCount: 0,
		}

		/* Store delivery */
		insertQuery := `INSERT INTO neurondb_agent.webhook_deliveries 
			(id, webhook_id, event_type, payload, status, attempt_count, created_at)
			VALUES ($1, $2, $3, $4::jsonb, $5, $6, NOW())`
		_, err := wm.db.ExecContext(ctx, insertQuery,
			delivery.ID, delivery.WebhookID, delivery.EventType,
			delivery.Payload, delivery.Status, delivery.AttemptCount)
		if err != nil {
			continue /* Skip if insert fails */
		}

		/* Queue for delivery */
		select {
		case wm.queue <- delivery:
		default:
			/* Queue full, skip */
		}
	}

	return nil
}

/* deliver delivers a webhook */
func (wm *WebhookManager) deliver(delivery *WebhookDelivery) {
	/* Get webhook config */
	var webhook Webhook
	err := wm.db.GetContext(context.Background(), &webhook,
		`SELECT * FROM neurondb_agent.webhooks WHERE id = $1`, delivery.WebhookID)
	if err != nil {
		return
	}

	/* Prepare payload */
	payloadJSON, err := json.Marshal(delivery.Payload)
	if err != nil {
		errMsg := err.Error()
		wm.updateDeliveryStatus(delivery.ID, "failed", nil, nil, &errMsg)
		return
	}

	/* Create request */
	req, err := http.NewRequest("POST", webhook.URL, bytes.NewReader(payloadJSON))
	if err != nil {
		errMsg := err.Error()
		wm.updateDeliveryStatus(delivery.ID, "failed", nil, nil, &errMsg)
		return
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Webhook-Event", delivery.EventType)
	req.Header.Set("X-Webhook-Delivery-ID", delivery.ID.String())

	/* Add signature if secret exists */
	if webhook.Secret != nil && *webhook.Secret != "" {
		signature := wm.generateSignature(payloadJSON, *webhook.Secret)
		req.Header.Set("X-Webhook-Signature", signature)
	}

	/* Set timeout */
	client := &http.Client{Timeout: time.Duration(webhook.TimeoutSeconds) * time.Second}
	if webhook.TimeoutSeconds == 0 {
		client.Timeout = 30 * time.Second
	}

	/* Send request */
	resp, err := client.Do(req)
	delivery.AttemptCount++

	if err != nil {
		errMsg := err.Error()
		/* Retry if attempts left */
		if delivery.AttemptCount < webhook.RetryCount {
			nextRetry := time.Now().Add(time.Duration(delivery.AttemptCount) * time.Minute)
			wm.updateDeliveryStatus(delivery.ID, "retrying", nil, nil, &errMsg)
			wm.scheduleRetry(delivery.ID, nextRetry)
			return
		}
		wm.updateDeliveryStatus(delivery.ID, "failed", nil, nil, &errMsg)
		return
	}
	defer resp.Body.Close()

	/* Read response body */
	bodyBytes, _ := io.ReadAll(resp.Body)
	bodyStr := string(bodyBytes)

	/* Check status code */
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		now := time.Now()
		wm.updateDeliveryStatus(delivery.ID, "success", &resp.StatusCode, &bodyStr, nil)
		wm.markDelivered(delivery.ID, now)
	} else {
		/* Retry on failure */
		if delivery.AttemptCount < webhook.RetryCount {
			nextRetry := time.Now().Add(time.Duration(delivery.AttemptCount) * time.Minute)
			wm.updateDeliveryStatus(delivery.ID, "retrying", &resp.StatusCode, &bodyStr, nil)
			wm.scheduleRetry(delivery.ID, nextRetry)
		} else {
			wm.updateDeliveryStatus(delivery.ID, "failed", &resp.StatusCode, &bodyStr, nil)
		}
	}
}

/* updateDeliveryStatus updates delivery status */
func (wm *WebhookManager) updateDeliveryStatus(id uuid.UUID, status string, statusCode *int, responseBody *string, errorMessage *string) {
	query := `UPDATE neurondb_agent.webhook_deliveries 
		SET status = $2, status_code = $3, response_body = $4, error_message = $5, attempt_count = attempt_count + 1
		WHERE id = $1`
	wm.db.ExecContext(context.Background(), query, id, status, statusCode, responseBody, errorMessage)
}

/* scheduleRetry schedules a retry */
func (wm *WebhookManager) scheduleRetry(id uuid.UUID, nextRetry time.Time) {
	query := `UPDATE neurondb_agent.webhook_deliveries 
		SET next_retry_at = $2 WHERE id = $1`
	wm.db.ExecContext(context.Background(), query, id, nextRetry)
}

/* markDelivered marks delivery as delivered */
func (wm *WebhookManager) markDelivered(id uuid.UUID, deliveredAt time.Time) {
	query := `UPDATE neurondb_agent.webhook_deliveries 
		SET delivered_at = $2 WHERE id = $1`
	wm.db.ExecContext(context.Background(), query, id, deliveredAt)
}

/* generateSignature generates HMAC signature for webhook payload */
func (wm *WebhookManager) generateSignature(payload []byte, secret string) string {
	h := hmac.New(sha256.New, []byte(secret))
	h.Write(payload)
	return "sha256=" + hex.EncodeToString(h.Sum(nil))
}

/* VerifySignature verifies webhook signature */
func VerifySignature(payload []byte, signature, secret string) bool {
	h := hmac.New(sha256.New, []byte(secret))
	h.Write(payload)
	expected := "sha256=" + hex.EncodeToString(h.Sum(nil))
	return hmac.Equal([]byte(signature), []byte(expected))
}
