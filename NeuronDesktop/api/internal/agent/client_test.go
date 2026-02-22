package agent

import (
	"context"
	"os"
	"testing"
	"time"
)

func getTestAgentEndpoint() string {
	endpoint := os.Getenv("TEST_AGENT_ENDPOINT")
	if endpoint == "" {
		endpoint = "http://localhost:8080"
	}
	return endpoint
}

func TestAgentClient_NewClient(t *testing.T) {
	client := NewClient("http://localhost:8080", "test-key")
	if client == nil {
		t.Error("NewClient() returned nil")
	}
	if client.baseURL != "http://localhost:8080" {
		t.Errorf("Expected baseURL http://localhost:8080, got %s", client.baseURL)
	}
	if client.apiKey != "test-key" {
		t.Errorf("Expected apiKey test-key, got %s", client.apiKey)
	}
}

func TestAgentClient_ListAgents(t *testing.T) {
	endpoint := getTestAgentEndpoint()
	client := NewClient(endpoint, "")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	agents, err := client.ListAgents(ctx)
	if err != nil {
		t.Logf("ListAgents failed (may be expected if agent not running): %v", err)
		return
	}

	// Verify agents structure
	for _, agent := range agents {
		if agent.ID == "" {
			t.Error("Agent ID should not be empty")
		}
		if agent.Name == "" {
			t.Error("Agent name should not be empty")
		}
	}
}

func TestAgentClient_ListModels(t *testing.T) {
	endpoint := getTestAgentEndpoint()
	client := NewClient(endpoint, "")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	models, err := client.ListModels(ctx)
	if err != nil {
		t.Skipf("ListModels failed (agent may not be running): %v", err)
	}
	if models == nil {
		t.Fatal("ListModels must not return nil")
	}
	_ = models
}

func TestAgentClient_CreateSession(t *testing.T) {
	endpoint := getTestAgentEndpoint()
	client := NewClient(endpoint, "")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req := CreateSessionRequest{
		AgentID: "test-agent-id",
	}

	session, err := client.CreateSession(ctx, req)
	if err != nil {
		t.Logf("CreateSession failed (may be expected if agent not running): %v", err)
		return
	}

	if session.ID == "" {
		t.Error("Session ID should not be empty")
	}
	if session.AgentID != req.AgentID {
		t.Errorf("Expected AgentID %s, got %s", req.AgentID, session.AgentID)
	}
}

func TestAgentClient_SendMessage(t *testing.T) {
	endpoint := getTestAgentEndpoint()
	client := NewClient(endpoint, "")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// First create a session
	createReq := CreateSessionRequest{
		AgentID: "test-agent-id",
	}
	session, err := client.CreateSession(ctx, createReq)
	if err != nil {
		t.Logf("CreateSession failed (may be expected if agent not running): %v", err)
		return
	}

	// Send a message
	msgReq := SendMessageRequest{
		Role:    "user",
		Content: "Hello, agent!",
		Stream:  false,
	}

	message, err := client.SendMessage(ctx, session.ID, msgReq)
	if err != nil {
		t.Logf("SendMessage failed (may be expected if agent not running): %v", err)
		return
	}

	if message.ID == "" {
		t.Error("Message ID should not be empty")
	}
	if message.Role != msgReq.Role {
		t.Errorf("Expected role %s, got %s", msgReq.Role, message.Role)
	}
	if message.Content != msgReq.Content {
		t.Errorf("Expected content %s, got %s", msgReq.Content, message.Content)
	}
}

func TestAgentClient_GetMessages(t *testing.T) {
	endpoint := getTestAgentEndpoint()
	client := NewClient(endpoint, "")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	messages, err := client.GetMessages(ctx, "test-session-id")
	if err != nil {
		t.Logf("GetMessages failed (may be expected if agent not running): %v", err)
		return
	}

	// Verify messages structure
	for _, msg := range messages {
		if msg.ID == "" {
			t.Error("Message ID should not be empty")
		}
		if msg.SessionID == "" {
			t.Error("Message SessionID should not be empty")
		}
		if msg.Role == "" {
			t.Error("Message role should not be empty")
		}
	}
}
