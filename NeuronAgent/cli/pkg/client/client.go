/*-------------------------------------------------------------------------
 *
 * client.go
 *    HTTP client for NeuronAgent API
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/cli/pkg/client/client.go
 *
 *-------------------------------------------------------------------------
 */

package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/neurondb/NeuronAgent/cli/pkg/config"
)

type Client struct {
	baseURL    string
	apiKey     string
	httpClient *http.Client
}

type Agent struct {
	ID           string                 `json:"id"`
	Name         string                 `json:"name"`
	Description  string                 `json:"description,omitempty"`
	SystemPrompt string                 `json:"system_prompt,omitempty"`
	ModelName    string                 `json:"model_name,omitempty"`
	EnabledTools []string               `json:"enabled_tools,omitempty"`
	Config       map[string]interface{} `json:"config,omitempty"`
	CreatedAt    string                 `json:"created_at,omitempty"`
}

type Session struct {
	ID      string `json:"id"`
	AgentID string `json:"agent_id"`
}

type MessageResponse struct {
	Content string `json:"content"`
}

func NewClient(baseURL, apiKey string) *Client {
	return &Client{
		baseURL: baseURL,
		apiKey:  apiKey,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

func (c *Client) CreateAgent(agentConfig *config.AgentConfig) (*Agent, error) {
	reqBody := map[string]interface{}{
		"name":    agentConfig.Name,
		"profile": agentConfig.Profile,
	}

	if agentConfig.Description != "" {
		reqBody["description"] = agentConfig.Description
	}

	if agentConfig.SystemPrompt != "" {
		reqBody["system_prompt"] = agentConfig.SystemPrompt
	}

	if agentConfig.Model.Name != "" {
		reqBody["model_name"] = agentConfig.Model.Name
	}

	if len(agentConfig.Tools) > 0 {
		reqBody["enabled_tools"] = agentConfig.Tools
	}

	if len(agentConfig.Config) > 0 {
		reqBody["config"] = agentConfig.Config
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	resp, err := c.makeRequest("POST", "/api/v1/agents", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var agent Agent
	if err := json.NewDecoder(resp.Body).Decode(&agent); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &agent, nil
}

func (c *Client) GetAgent(agentID string) (*Agent, error) {
	resp, err := c.makeRequest("GET", fmt.Sprintf("/api/v1/agents/%s", agentID), nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var agent Agent
	if err := json.NewDecoder(resp.Body).Decode(&agent); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &agent, nil
}

func (c *Client) ListAgents() ([]Agent, error) {
	resp, err := c.makeRequest("GET", "/api/v1/agents", nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var agents []Agent
	if err := json.NewDecoder(resp.Body).Decode(&agents); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return agents, nil
}

func (c *Client) UpdateAgent(agentID string, agentConfig *config.AgentConfig) (*Agent, error) {
	reqBody := map[string]interface{}{}

	if agentConfig.Name != "" {
		reqBody["name"] = agentConfig.Name
	}

	if agentConfig.Description != "" {
		reqBody["description"] = agentConfig.Description
	}

	if agentConfig.SystemPrompt != "" {
		reqBody["system_prompt"] = agentConfig.SystemPrompt
	}

	if agentConfig.Model.Name != "" {
		reqBody["model_name"] = agentConfig.Model.Name
	}

	if len(agentConfig.Tools) > 0 {
		reqBody["enabled_tools"] = agentConfig.Tools
	}

	if len(agentConfig.Config) > 0 {
		reqBody["config"] = agentConfig.Config
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	resp, err := c.makeRequest("PUT", fmt.Sprintf("/api/v1/agents/%s", agentID), bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var agent Agent
	if err := json.NewDecoder(resp.Body).Decode(&agent); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &agent, nil
}

func (c *Client) DeleteAgent(agentID string) error {
	resp, err := c.makeRequest("DELETE", fmt.Sprintf("/api/v1/agents/%s", agentID), nil)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNoContent && resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("failed to delete agent: status %d, body: %s", resp.StatusCode, string(body))
	}

	return nil
}

func (c *Client) CreateSession(agentID string, metadata map[string]interface{}) (*Session, error) {
	reqBody := map[string]interface{}{
		"agent_id": agentID,
	}

	if metadata != nil {
		reqBody["metadata"] = metadata
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	resp, err := c.makeRequest("POST", "/api/v1/sessions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var session Session
	if err := json.NewDecoder(resp.Body).Decode(&session); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &session, nil
}

func (c *Client) SendMessage(sessionID, message string, stream bool) (*MessageResponse, error) {
	reqBody := map[string]interface{}{
		"content": message,
		"role":    "user",
		"stream":  stream,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	resp, err := c.makeRequest("POST", fmt.Sprintf("/api/v1/sessions/%s/messages", sessionID), bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var msgResp MessageResponse
	if err := json.NewDecoder(resp.Body).Decode(&msgResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &msgResp, nil
}

func (c *Client) makeRequest(method, path string, body io.Reader) (*http.Response, error) {
	url := c.baseURL + path
	req, err := http.NewRequest(method, url, body)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("API error: status %d, body: %s", resp.StatusCode, string(body))
	}

	return resp, nil
}

type Workflow struct {
	ID            string                 `json:"id"`
	Name          string                 `json:"name"`
	DAGDefinition map[string]interface{} `json:"dag_definition"`
	Status        string                 `json:"status"`
	CreatedAt     string                 `json:"created_at,omitempty"`
	UpdatedAt     string                 `json:"updated_at,omitempty"`
}

func (c *Client) CreateWorkflow(workflowConfig map[string]interface{}) (*Workflow, error) {
	body, err := json.Marshal(workflowConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	resp, err := c.makeRequest("POST", "/api/v1/workflows", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var workflow Workflow
	if err := json.NewDecoder(resp.Body).Decode(&workflow); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &workflow, nil
}

func (c *Client) ListWorkflows(limit, offset int) ([]Workflow, error) {
	path := fmt.Sprintf("/api/v1/workflows?limit=%d&offset=%d", limit, offset)
	resp, err := c.makeRequest("GET", path, nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var workflows []Workflow
	if err := json.NewDecoder(resp.Body).Decode(&workflows); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return workflows, nil
}

func (c *Client) GetWorkflow(workflowID string) (*Workflow, error) {
	resp, err := c.makeRequest("GET", fmt.Sprintf("/api/v1/workflows/%s", workflowID), nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var workflow Workflow
	if err := json.NewDecoder(resp.Body).Decode(&workflow); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &workflow, nil
}
