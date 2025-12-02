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
		argKeys := make([]string, 0, len(args))
		for k := range args {
			argKeys = append(argKeys, k)
		}
		return "", fmt.Errorf("HTTP tool execution failed: tool_name='%s', handler_type='http', args_count=%d, arg_keys=[%v], validation_error='url parameter is required and must be a string'",
			tool.Name, len(args), argKeys)
	}

	// Check allowlist if configured
	allowlistSize := len(t.allowed)
	if allowlistSize > 0 && !t.allowed[url] {
		// Check if any allowed prefix matches
		allowed := false
		for allowedURL := range t.allowed {
			if strings.HasPrefix(url, allowedURL) {
				allowed = true
				break
			}
		}
		if !allowed {
			return "", fmt.Errorf("HTTP tool execution failed: tool_name='%s', handler_type='http', url='%s', allowlist_size=%d, allowlist_check='failed', validation_error='URL not in allowlist'",
				tool.Name, url, allowlistSize)
		}
	}

	method := "GET"
	if m, ok := args["method"].(string); ok {
		method = strings.ToUpper(m)
	}
	
	headerCount := 0
	if headers, ok := args["headers"].(map[string]interface{}); ok {
		headerCount = len(headers)
	}
	
	bodySize := 0
	if body, ok := args["body"].(string); ok {
		bodySize = len(body)
	}

	// Create request
	req, err := http.NewRequestWithContext(ctx, method, url, nil)
	if err != nil {
		return "", fmt.Errorf("HTTP tool request creation failed: tool_name='%s', handler_type='http', method='%s', url='%s', headers_count=%d, body_size=%d, timeout=%v, error=%w",
			tool.Name, method, url, headerCount, bodySize, t.client.Timeout, err)
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
		return "", fmt.Errorf("HTTP tool request execution failed: tool_name='%s', handler_type='http', method='%s', url='%s', headers_count=%d, body_size=%d, timeout=%v, error=%w",
			tool.Name, method, url, headerCount, bodySize, t.client.Timeout, err)
	}
	defer resp.Body.Close()

	// Limit response size (1MB)
	maxResponseSize := 1024 * 1024
	limitedReader := io.LimitReader(resp.Body, int64(maxResponseSize))
	body, err := io.ReadAll(limitedReader)
	if err != nil {
		return "", fmt.Errorf("HTTP tool response reading failed: tool_name='%s', handler_type='http', method='%s', url='%s', response_status=%d, max_response_size=%d, error=%w",
			tool.Name, method, url, resp.StatusCode, maxResponseSize, err)
	}

	// Format response
	result := map[string]interface{}{
		"status_code": resp.StatusCode,
		"headers":     resp.Header,
		"body":        string(body),
	}

	jsonResult, err := json.Marshal(result)
	if err != nil {
		return "", fmt.Errorf("HTTP tool response marshaling failed: tool_name='%s', handler_type='http', method='%s', url='%s', response_status=%d, response_body_size=%d, error=%w",
			tool.Name, method, url, resp.StatusCode, len(body), err)
	}

	return string(jsonResult), nil
}

func (t *HTTPTool) Validate(args map[string]interface{}, schema map[string]interface{}) error {
	return ValidateArgs(args, schema)
}

