package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"github.com/google/uuid"
)

// NeuronAgentClient is a Go client for NeuronAgent API
type NeuronAgentClient struct {
	BaseURL string
	APIKey  string
	Client  *http.Client
}

// NewNeuronAgentClient creates a new client instance
func NewNeuronAgentClient(baseURL, apiKey string) *NeuronAgentClient {
	if baseURL == "" {
		baseURL = "http://localhost:8080"
	}
	if apiKey == "" {
		apiKey = os.Getenv("NEURONAGENT_API_KEY")
	}

	return &NeuronAgentClient{
		BaseURL: baseURL,
		APIKey:  apiKey,
		Client: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// Agent represents an agent configuration
type Agent struct {
	ID           uuid.UUID              `json:"id"`
	Name         string                 `json:"name"`
	Description  *string                `json:"description"`
	SystemPrompt string                 `json:"system_prompt"`
	ModelName    string                 `json:"model_name"`
	EnabledTools []string               `json:"enabled_tools"`
	Config       map[string]interface{} `json:"config"`
	CreatedAt    time.Time              `json:"created_at"`
	UpdatedAt    time.Time              `json:"updated_at"`
}

// Session represents a conversation session
type Session struct {
	ID             uuid.UUID              `json:"id"`
	AgentID        uuid.UUID              `json:"agent_id"`
	ExternalUserID *string                 `json:"external_user_id"`
	Metadata       map[string]interface{}  `json:"metadata"`
	CreatedAt      time.Time              `json:"created_at"`
	LastActivityAt time.Time              `json:"last_activity_at"`
}

// Message represents a message in a conversation
type Message struct {
	ID         int64                  `json:"id"`
	SessionID  uuid.UUID              `json:"session_id"`
	Role       string                 `json:"role"`
	Content    string                 `json:"content"`
	TokenCount *int                   `json:"token_count"`
	CreatedAt  time.Time              `json:"created_at"`
}

// CreateAgentRequest is the request to create an agent
type CreateAgentRequest struct {
	Name         string                 `json:"name"`
	Description  *string                `json:"description,omitempty"`
	SystemPrompt string                 `json:"system_prompt"`
	ModelName    string                 `json:"model_name"`
	EnabledTools []string               `json:"enabled_tools"`
	Config       map[string]interface{} `json:"config"`
}

// CreateSessionRequest is the request to create a session
type CreateSessionRequest struct {
	AgentID       uuid.UUID              `json:"agent_id"`
	ExternalUserID *string                `json:"external_user_id,omitempty"`
	Metadata      map[string]interface{}  `json:"metadata,omitempty"`
}

// SendMessageRequest is the request to send a message
type SendMessageRequest struct {
	Content  string                 `json:"content"`
	Role     string                 `json:"role"`
	Stream   bool                   `json:"stream"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// SendMessageResponse is the response from sending a message
type SendMessageResponse struct {
	SessionID   uuid.UUID `json:"session_id"`
	AgentID     uuid.UUID `json:"agent_id"`
	Response    string    `json:"response"`
	TokensUsed  int       `json:"tokens_used"`
	ToolCalls   []interface{} `json:"tool_calls"`
	ToolResults []interface{} `json:"tool_results"`
}

// makeRequest makes an authenticated HTTP request
func (c *NeuronAgentClient) makeRequest(method, path string, body interface{}) (*http.Response, error) {
	var reqBody io.Reader
	if body != nil {
		jsonData, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal request body: %w", err)
		}
		reqBody = bytes.NewBuffer(jsonData)
	}

	req, err := http.NewRequest(method, c.BaseURL+path, reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+c.APIKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.Client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}

	if resp.StatusCode >= 400 {
		bodyBytes, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(bodyBytes))
	}

	return resp, nil
}

// HealthCheck checks if the server is healthy
func (c *NeuronAgentClient) HealthCheck() error {
	resp, err := http.Get(c.BaseURL + "/health")
	if err != nil {
		return fmt.Errorf("health check failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("server unhealthy (status %d)", resp.StatusCode)
	}
	return nil
}

// CreateAgent creates a new agent
func (c *NeuronAgentClient) CreateAgent(req CreateAgentRequest) (*Agent, error) {
	resp, err := c.makeRequest("POST", "/api/v1/agents", req)
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

// GetAgent retrieves an agent by ID
func (c *NeuronAgentClient) GetAgent(id uuid.UUID) (*Agent, error) {
	resp, err := c.makeRequest("GET", "/api/v1/agents/"+id.String(), nil)
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

// ListAgents lists all agents
func (c *NeuronAgentClient) ListAgents() ([]Agent, error) {
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

// CreateSession creates a new session
func (c *NeuronAgentClient) CreateSession(req CreateSessionRequest) (*Session, error) {
	resp, err := c.makeRequest("POST", "/api/v1/sessions", req)
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

// SendMessage sends a message to the agent
func (c *NeuronAgentClient) SendMessage(sessionID uuid.UUID, req SendMessageRequest) (*SendMessageResponse, error) {
	resp, err := c.makeRequest("POST", "/api/v1/sessions/"+sessionID.String()+"/messages", req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var response SendMessageResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &response, nil
}

// GetMessages retrieves messages from a session
func (c *NeuronAgentClient) GetMessages(sessionID uuid.UUID, limit, offset int) ([]Message, error) {
	path := fmt.Sprintf("/api/v1/sessions/%s/messages?limit=%d&offset=%d", sessionID.String(), limit, offset)
	resp, err := c.makeRequest("GET", path, nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var messages []Message
	if err := json.NewDecoder(resp.Body).Decode(&messages); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return messages, nil
}

// Example usage
func main() {
	apiKey := os.Getenv("NEURONAGENT_API_KEY")
	if apiKey == "" {
		fmt.Println("Error: NEURONAGENT_API_KEY environment variable not set")
		fmt.Println("Set it with: export NEURONAGENT_API_KEY=your_api_key")
		os.Exit(1)
	}

	client := NewNeuronAgentClient("http://localhost:8080", apiKey)

	// Health check
	fmt.Println("Checking server health...")
	if err := client.HealthCheck(); err != nil {
		fmt.Printf("❌ Server health check failed: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("✅ Server is healthy")

	// Create an agent
	fmt.Println("\nCreating agent...")
	agent, err := client.CreateAgent(CreateAgentRequest{
		Name:         "go-example-agent",
		SystemPrompt: "You are a helpful assistant. Answer questions clearly and concisely.",
		ModelName:    "gpt-4",
		Description:  stringPtr("Example agent created from Go client"),
		EnabledTools: []string{"sql", "http"},
		Config: map[string]interface{}{
			"temperature": 0.7,
			"max_tokens":  1500,
		},
	})
	if err != nil {
		fmt.Printf("❌ Failed to create agent: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("✅ Agent created: %s (%s)\n", agent.Name, agent.ID)

	// Create a session
	fmt.Println("\nCreating session...")
	session, err := client.CreateSession(CreateSessionRequest{
		AgentID:       agent.ID,
		ExternalUserID: stringPtr("go-user-123"),
		Metadata: map[string]interface{}{
			"source": "go-example",
		},
	})
	if err != nil {
		fmt.Printf("❌ Failed to create session: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("✅ Session created: %s\n", session.ID)

	// Send messages
	messages := []string{
		"Hello! Can you introduce yourself?",
		"What can you help me with?",
	}

	for i, msg := range messages {
		fmt.Printf("\n💭 Sending message %d: %s\n", i+1, msg)
		response, err := client.SendMessage(session.ID, SendMessageRequest{
			Content: msg,
			Role:    "user",
			Stream:  false,
		})
		if err != nil {
			fmt.Printf("❌ Failed to send message: %v\n", err)
			continue
		}
		fmt.Printf("🤖 Agent response: %s\n", truncateString(response.Response, 200))
		fmt.Printf("   Tokens used: %d\n", response.TokensUsed)
		time.Sleep(1 * time.Second)
	}

	// Get conversation history
	fmt.Println("\n📜 Retrieving conversation history...")
	history, err := client.GetMessages(session.ID, 10, 0)
	if err != nil {
		fmt.Printf("❌ Failed to get messages: %v\n", err)
	} else {
		fmt.Printf("✅ Retrieved %d messages\n", len(history))
		for _, msg := range history {
			fmt.Printf("   [%s]: %s\n", msg.Role, truncateString(msg.Content, 80))
		}
	}

	fmt.Println("\n✅ Example completed successfully!")
}

// Helper functions
func stringPtr(s string) *string {
	return &s
}

func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

