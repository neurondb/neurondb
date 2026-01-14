/*-------------------------------------------------------------------------
 *
 * webhooks.go
 *    Webhook system for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/webhooks/webhooks.go
 *
 *-------------------------------------------------------------------------
 */

package webhooks

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"
)

/* Webhook represents a webhook configuration */
type Webhook struct {
	ID        string
	URL       string
	Events    []string
	Secret    string
	Retries   int
	Timeout   time.Duration
	CreatedAt time.Time
}

/* WebhookEvent represents a webhook event */
type WebhookEvent struct {
	Type      string                 `json:"type"`
	Timestamp time.Time              `json:"timestamp"`
	Data      map[string]interface{} `json:"data"`
}

/* Manager manages webhooks */
type Manager struct {
	webhooks map[string]*Webhook
	client   *http.Client
	mu       sync.RWMutex
}

/* NewManager creates a new webhook manager */
func NewManager() *Manager {
	return &Manager{
		webhooks: make(map[string]*Webhook),
		client: &http.Client{
			Timeout: 10 * time.Second,
		},
	}
}

/* Register registers a webhook */
func (m *Manager) Register(webhook *Webhook) error {
	if webhook == nil {
		return fmt.Errorf("webhook cannot be nil")
	}
	if webhook.ID == "" {
		return fmt.Errorf("webhook ID cannot be empty")
	}
	if webhook.URL == "" {
		return fmt.Errorf("webhook URL cannot be empty")
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	m.webhooks[webhook.ID] = webhook
	return nil
}

/* Unregister unregisters a webhook */
func (m *Manager) Unregister(id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, exists := m.webhooks[id]; !exists {
		return fmt.Errorf("webhook not found: %s", id)
	}
	delete(m.webhooks, id)
	return nil
}

/* Trigger triggers webhooks for an event */
func (m *Manager) Trigger(ctx context.Context, eventType string, data map[string]interface{}) {
	if ctx == nil {
		return /* Cannot trigger without context */
	}
	if eventType == "" {
		return /* Event type is required */
	}

	m.mu.RLock()
	webhooks := make([]*Webhook, 0)
	for _, webhook := range m.webhooks {
		if webhook == nil {
			continue
		}
		for _, event := range webhook.Events {
			if event == eventType || event == "*" {
				webhooks = append(webhooks, webhook)
				break
			}
		}
	}
	m.mu.RUnlock()

	if len(webhooks) == 0 {
		return /* No webhooks registered for this event */
	}

	event := &WebhookEvent{
		Type:      eventType,
		Timestamp: time.Now(),
		Data:      data,
	}

	for _, webhook := range webhooks {
		/* Launch each webhook in a separate goroutine */
		go m.sendWebhook(ctx, webhook, event)
	}
}

/* sendWebhook sends a webhook with retry logic */
func (m *Manager) sendWebhook(ctx context.Context, webhook *Webhook, event *WebhookEvent) {
	if webhook == nil {
		return
	}
	if event == nil {
		return
	}
	if webhook.URL == "" {
		return
	}

	eventJSON, err := json.Marshal(event)
	if err != nil {
		return /* Silently fail - webhook payload couldn't be marshaled */
	}

	maxRetries := webhook.Retries
	if maxRetries < 0 {
		maxRetries = 0
	}
	if maxRetries > 10 {
		maxRetries = 10 /* Cap retries at 10 */
	}

	for attempt := 0; attempt <= maxRetries; attempt++ {
		/* Check context cancellation */
		select {
		case <-ctx.Done():
			return
		default:
		}

		req, err := http.NewRequestWithContext(ctx, http.MethodPost, webhook.URL, bytes.NewBuffer(eventJSON))
		if err != nil {
			if attempt < maxRetries {
				time.Sleep(time.Duration(attempt+1) * time.Second)
			}
			continue
		}

		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("User-Agent", "NeuronMCP-Webhook/1.0")
		if webhook.Secret != "" {
			req.Header.Set("X-Webhook-Secret", webhook.Secret)
		}

		/* Use webhook timeout if configured, otherwise use client default */
		client := m.client
		if webhook.Timeout > 0 {
			client = &http.Client{
				Timeout: webhook.Timeout,
			}
		}

		resp, err := client.Do(req)
		if err == nil {
			statusOK := resp.StatusCode >= 200 && resp.StatusCode < 300
			resp.Body.Close()
			if statusOK {
				return /* Success */
			}
		} else {
			/* Network error - will retry */
		}

		if resp != nil {
			resp.Body.Close()
		}

		/* Exponential backoff for retries */
		if attempt < maxRetries {
			backoff := time.Duration(attempt+1) * time.Second
			if backoff > 30*time.Second {
				backoff = 30 * time.Second /* Cap backoff at 30 seconds */
			}
			select {
			case <-ctx.Done():
				return
			case <-time.After(backoff):
			}
		}
	}
}

