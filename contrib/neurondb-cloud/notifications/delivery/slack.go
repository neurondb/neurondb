package delivery

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
)

// SlackMessage holds a Slack notification payload.
type SlackMessage struct {
	Text     string
	Severity string // "info", "warn", "error"
	Channel  string // optional override
}

// SendSlack sends a message to Slack via webhook (env SLACK_WEBHOOK_URL).
func SendSlack(ctx context.Context, m SlackMessage) error {
	url := os.Getenv("SLACK_WEBHOOK_URL")
	if url == "" {
		return fmt.Errorf("SLACK_WEBHOOK_URL is required")
	}
	payload := map[string]interface{}{
		"text": m.Text,
	}
	if m.Severity == "error" {
		payload["attachments"] = []map[string]interface{}{{
			"color": "danger",
			"text":  m.Text,
		}}
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("slack webhook returned %d", resp.StatusCode)
	}
	return nil
}
