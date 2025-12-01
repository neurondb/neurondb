package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/neurondb/NeuronAgent/internal/db"
)

type HTTPTool struct {
	client  *http.Client
	allowed map[string]bool // URL allowlist
}

func NewHTTPTool() *HTTPTool {
	return &HTTPTool{
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
		allowed: make(map[string]bool),
	}
}

func (t *HTTPTool) Execute(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	url, ok := args["url"].(string)
	if !ok {
		return "", fmt.Errorf("url parameter is required and must be a string")
	}

	// Check allowlist if configured
	if len(t.allowed) > 0 && !t.allowed[url] {
		// Check if any allowed prefix matches
		allowed := false
		for allowedURL := range t.allowed {
			if strings.HasPrefix(url, allowedURL) {
				allowed = true
				break
			}
		}
		if !allowed {
			return "", fmt.Errorf("URL not in allowlist: %s", url)
		}
	}

	method := "GET"
	if m, ok := args["method"].(string); ok {
		method = strings.ToUpper(m)
	}

	// Create request
	req, err := http.NewRequestWithContext(ctx, method, url, nil)
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	// Add headers
	if headers, ok := args["headers"].(map[string]interface{}); ok {
		for k, v := range headers {
			if str, ok := v.(string); ok {
				req.Header.Set(k, str)
			}
		}
	}

	// Add body for POST/PUT
	if body, ok := args["body"].(string); ok && (method == "POST" || method == "PUT" || method == "PATCH") {
		req.Body = io.NopCloser(strings.NewReader(body))
		req.ContentLength = int64(len(body))
	}

	// Execute request
	resp, err := t.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	// Limit response size (1MB)
	limitedReader := io.LimitReader(resp.Body, 1024*1024)
	body, err := io.ReadAll(limitedReader)
	if err != nil {
		return "", fmt.Errorf("failed to read response: %w", err)
	}

	// Format response
	result := map[string]interface{}{
		"status_code": resp.StatusCode,
		"headers":     resp.Header,
		"body":        string(body),
	}

	jsonResult, err := json.Marshal(result)
	if err != nil {
		return "", fmt.Errorf("failed to marshal response: %w", err)
	}

	return string(jsonResult), nil
}

func (t *HTTPTool) Validate(args map[string]interface{}, schema map[string]interface{}) error {
	return ValidateArgs(args, schema)
}

