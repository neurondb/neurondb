/*-------------------------------------------------------------------------
 *
 * webhook.go
 *    Webhook notification service
 *
 * Provides HTTP webhook notifications for task alerts and system events.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/notifications/webhook.go
 *
 *-------------------------------------------------------------------------
 */

package notifications

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

/* WebhookService provides webhook notification capabilities */
type WebhookService struct {
	httpClient *http.Client
	timeout    time.Duration
}

/* NewWebhookService creates a new webhook service */
func NewWebhookService(timeout time.Duration) *WebhookService {
	if timeout == 0 {
		timeout = 30 * time.Second
	}

	return &WebhookService{
		httpClient: &http.Client{
			Timeout: timeout,
		},
		timeout: timeout,
	}
}

/* SendWebhook sends a webhook notification */
func (w *WebhookService) SendWebhook(ctx context.Context, url string, payload map[string]interface{}) error {
	/* Validate URL */
	if url == "" {
		return fmt.Errorf("webhook URL is required")
	}

	/* Serialize payload */
	payloadJSON, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("webhook payload serialization failed: error=%w", err)
	}

	/* Create request */
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(payloadJSON))
	if err != nil {
		return fmt.Errorf("webhook request creation failed: url='%s', error=%w", url, err)
	}

	/* Set headers */
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "NeuronAgent/1.0")

	/* Send request */
	resp, err := w.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("webhook request failed: url='%s', error=%w", url, err)
	}
	defer resp.Body.Close()

	/* Check response status */
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("webhook request failed: url='%s', status_code=%d", url, resp.StatusCode)
	}

	return nil
}

/* SendWebhookWithHeaders sends a webhook with custom headers */
func (w *WebhookService) SendWebhookWithHeaders(ctx context.Context, url string, payload map[string]interface{}, headers map[string]string) error {
	/* Validate URL */
	if url == "" {
		return fmt.Errorf("webhook URL is required")
	}

	/* Serialize payload */
	payloadJSON, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("webhook payload serialization failed: error=%w", err)
	}

	/* Create request */
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(payloadJSON))
	if err != nil {
		return fmt.Errorf("webhook request creation failed: url='%s', error=%w", url, err)
	}

	/* Set default headers */
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "NeuronAgent/1.0")

	/* Add custom headers */
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	/* Send request */
	resp, err := w.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("webhook request failed: url='%s', error=%w", url, err)
	}
	defer resp.Body.Close()

	/* Check response status */
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("webhook request failed: url='%s', status_code=%d", url, resp.StatusCode)
	}

	return nil
}
