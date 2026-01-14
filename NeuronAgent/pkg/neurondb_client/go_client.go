/*-------------------------------------------------------------------------
 *
 * go_client.go
 *    Go SDK for NeuronAgent
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/pkg/neurondb_client/go_client.go
 *
 *-------------------------------------------------------------------------
 */

package neurondb_client

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/google/uuid"
)

/* Client is the main client for NeuronAgent API */
type Client struct {
	BaseURL    string
	APIKey     string
	HTTPClient *http.Client
}

/* NewClient creates a new NeuronAgent client */
func NewClient(baseURL, apiKey string) *Client {
	if baseURL == "" {
		baseURL = "http://localhost:8080"
	}

	return &Client{
		BaseURL: baseURL,
		APIKey:  apiKey,
		HTTPClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

/* Agent represents an agent */
type Agent struct {
	ID           uuid.UUID              `json:"id"`
	Name         string                 `json:"name"`
	Description  *string                `json:"description"`
	SystemPrompt string                 `json:"system_prompt"`
	ModelName    string                 `json:"model_name"`
	EnabledTools []string               `json:"enabled_tools"`
	Config       map[string]interface{} `json:"config"`
}

/* Session represents a session */
type Session struct {
	ID             uuid.UUID              `json:"id"`
	AgentID        uuid.UUID              `json:"agent_id"`
	ExternalUserID *string                `json:"external_user_id"`
	Metadata       map[string]interface{} `json:"metadata"`
}

/* Message represents a message */
type Message struct {
	ID         int64     `json:"id"`
	SessionID  uuid.UUID `json:"session_id"`
	Role       string    `json:"role"`
	Content    string    `json:"content"`
	TokenCount *int      `json:"token_count"`
}

/* CreateAgentRequest represents a request to create an agent */
type CreateAgentRequest struct {
	Name         string                 `json:"name"`
	Description  *string                `json:"description"`
	SystemPrompt string                 `json:"system_prompt"`
	ModelName    string                 `json:"model_name"`
	EnabledTools []string               `json:"enabled_tools"`
	Config       map[string]interface{} `json:"config"`
}

/* makeRequest makes an HTTP request */
func (c *Client) makeRequest(ctx context.Context, method, endpoint string, body interface{}) (*http.Response, error) {
	url := c.BaseURL + endpoint

	var reqBody io.Reader
	if body != nil {
		jsonData, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal request body: %w", err)
		}
		reqBody = bytes.NewReader(jsonData)
	}

	req, err := http.NewRequestWithContext(ctx, method, url, reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+c.APIKey)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	return c.HTTPClient.Do(req)
}

/* CreateAgent creates a new agent */
func (c *Client) CreateAgent(ctx context.Context, req CreateAgentRequest) (*Agent, error) {
	resp, err := c.makeRequest(ctx, "POST", "/api/v1/agents", req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusCreated {
		return nil, fmt.Errorf("failed to create agent: status %d", resp.StatusCode)
	}

	var agent Agent
	if err := json.NewDecoder(resp.Body).Decode(&agent); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &agent, nil
}

/* GetAgent gets an agent by ID */
func (c *Client) GetAgent(ctx context.Context, id uuid.UUID) (*Agent, error) {
	resp, err := c.makeRequest(ctx, "GET", fmt.Sprintf("/api/v1/agents/%s", id.String()), nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed to get agent: status %d", resp.StatusCode)
	}

	var agent Agent
	if err := json.NewDecoder(resp.Body).Decode(&agent); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &agent, nil
}

/* ListAgents lists all agents */
func (c *Client) ListAgents(ctx context.Context) ([]Agent, error) {
	resp, err := c.makeRequest(ctx, "GET", "/api/v1/agents", nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed to list agents: status %d", resp.StatusCode)
	}

	var agents []Agent
	if err := json.NewDecoder(resp.Body).Decode(&agents); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return agents, nil
}

/* CreateSession creates a new session */
func (c *Client) CreateSession(ctx context.Context, agentID uuid.UUID) (*Session, error) {
	req := map[string]interface{}{
		"agent_id": agentID,
	}

	resp, err := c.makeRequest(ctx, "POST", "/api/v1/sessions", req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusCreated {
		return nil, fmt.Errorf("failed to create session: status %d", resp.StatusCode)
	}

	var session Session
	if err := json.NewDecoder(resp.Body).Decode(&session); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &session, nil
}

/* SendMessage sends a message to an agent */
func (c *Client) SendMessage(ctx context.Context, sessionID uuid.UUID, content string) (string, error) {
	req := map[string]interface{}{
		"content": content,
		"role":    "user",
		"stream":  false,
	}

	resp, err := c.makeRequest(ctx, "POST", fmt.Sprintf("/api/v1/sessions/%s/messages", sessionID.String()), req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("failed to send message: status %d", resp.StatusCode)
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("failed to decode response: %w", err)
	}

	if response, ok := result["response"].(string); ok {
		return response, nil
	}

	return "", fmt.Errorf("response field not found in result")
}

/* GetMessages gets messages for a session */
func (c *Client) GetMessages(ctx context.Context, sessionID uuid.UUID) ([]Message, error) {
	resp, err := c.makeRequest(ctx, "GET", fmt.Sprintf("/api/v1/sessions/%s/messages", sessionID.String()), nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed to get messages: status %d", resp.StatusCode)
	}

	var messages []Message
	if err := json.NewDecoder(resp.Body).Decode(&messages); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return messages, nil
}
