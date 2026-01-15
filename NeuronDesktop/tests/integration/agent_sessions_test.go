package integration

import (
	"context"
	"testing"

	testutil "github.com/neurondb/NeuronDesktop/api/internal/testing"
	"github.com/neurondb/NeuronDesktop/api/internal/agent"
)

func TestAgentIntegration_CreateSession(t *testing.T) {
	config := testutil.LoadIntegrationTestConfig()
	if config.SkipNeuronAgent {
		t.Skip("Skipping NeuronAgent tests (SKIP_NEURONAGENT=true)")
	}

	ctx := context.Background()
	client := testutil.CreateNeuronAgentTestClient(t, config.NeuronAgentURL, config.NeuronAgentKey)
	if client == nil {
		return
	}

	/* First, create or get an agent */
	agents, err := client.ListAgents(ctx)
	if err != nil {
		t.Fatalf("Failed to list agents: %v", err)
	}

	var agentID string
	if len(agents) == 0 {
		/* Create a test agent */
		createReq := agent.CreateAgentRequest{
			Name:         "test-agent-session",
			SystemPrompt: "You are a helpful assistant.",
			ModelName:    "gpt-4",
		}
		createdAgent, err := client.CreateAgent(ctx, createReq)
		if err != nil {
			t.Fatalf("Failed to create agent: %v", err)
		}
		agentID = createdAgent.ID
		defer client.DeleteAgent(ctx, agentID)
	} else {
		agentID = agents[0].ID
	}

	/* Create session */
	sessionReq := agent.CreateSessionRequest{
		AgentID:        agentID,
		ExternalUserID: "test-user",
		Metadata: map[string]interface{}{
			"test": true,
		},
	}

	session, err := client.CreateSession(ctx, sessionReq)
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}

	if session == nil {
		t.Fatal("Session is nil")
	}

	if session.AgentID != agentID {
		t.Errorf("Expected agent ID %s, got %s", agentID, session.AgentID)
	}
}

func TestAgentIntegration_GetSession(t *testing.T) {
	config := testutil.LoadIntegrationTestConfig()
	if config.SkipNeuronAgent {
		t.Skip("Skipping NeuronAgent tests (SKIP_NEURONAGENT=true)")
	}

	ctx := context.Background()
	client := testutil.CreateNeuronAgentTestClient(t, config.NeuronAgentURL, config.NeuronAgentKey)
	if client == nil {
		return
	}

	/* Get or create an agent */
	agents, err := client.ListAgents(ctx)
	if err != nil {
		t.Fatalf("Failed to list agents: %v", err)
	}

	if len(agents) == 0 {
		t.Skip("No agents available to test")
	}

	agentID := agents[0].ID

	/* Create session */
	sessionReq := agent.CreateSessionRequest{
		AgentID: agentID,
	}

	session, err := client.CreateSession(ctx, sessionReq)
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}

	/* Get session */
	retrievedSession, err := client.GetSession(ctx, session.ID)
	if err != nil {
		t.Fatalf("Failed to get session: %v", err)
	}

	if retrievedSession.ID != session.ID {
		t.Errorf("Expected session ID %s, got %s", session.ID, retrievedSession.ID)
	}
}

func TestAgentIntegration_ListSessions(t *testing.T) {
	config := testutil.LoadIntegrationTestConfig()
	if config.SkipNeuronAgent {
		t.Skip("Skipping NeuronAgent tests (SKIP_NEURONAGENT=true)")
	}

	ctx := context.Background()
	client := testutil.CreateNeuronAgentTestClient(t, config.NeuronAgentURL, config.NeuronAgentKey)
	if client == nil {
		return
	}

	/* Get or create an agent */
	agents, err := client.ListAgents(ctx)
	if err != nil {
		t.Fatalf("Failed to list agents: %v", err)
	}

	if len(agents) == 0 {
		t.Skip("No agents available to test")
	}

	agentID := agents[0].ID

	/* List sessions */
	sessions, err := client.ListSessions(ctx, agentID)
	if err != nil {
		t.Fatalf("Failed to list sessions: %v", err)
	}

	/* Verify sessions structure */
	for _, session := range sessions {
		if session.ID == "" {
			t.Error("Session missing ID")
		}
		if session.AgentID != agentID {
			t.Errorf("Expected agent ID %s, got %s", agentID, session.AgentID)
		}
	}
}

func TestAgentIntegration_SendMessage(t *testing.T) {
	config := testutil.LoadIntegrationTestConfig()
	if config.SkipNeuronAgent {
		t.Skip("Skipping NeuronAgent tests (SKIP_NEURONAGENT=true)")
	}

	ctx := context.Background()
	client := testutil.CreateNeuronAgentTestClient(t, config.NeuronAgentURL, config.NeuronAgentKey)
	if client == nil {
		return
	}

	/* Get or create an agent */
	agents, err := client.ListAgents(ctx)
	if err != nil {
		t.Fatalf("Failed to list agents: %v", err)
	}

	if len(agents) == 0 {
		t.Skip("No agents available to test")
	}

	agentID := agents[0].ID

	/* Create session */
	sessionReq := agent.CreateSessionRequest{
		AgentID: agentID,
	}

	session, err := client.CreateSession(ctx, sessionReq)
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}

	/* Send message */
	messageReq := agent.SendMessageRequest{
		Role:    "user",
		Content: "Hello, this is a test message",
		Stream:  false,
	}

	message, err := client.SendMessage(ctx, session.ID, messageReq)
	if err != nil {
		t.Fatalf("Failed to send message: %v", err)
	}

	if message == nil {
		t.Fatal("Message is nil")
	}

	if message.Role != messageReq.Role {
		t.Errorf("Expected role %s, got %s", messageReq.Role, message.Role)
	}
}

func TestAgentIntegration_GetMessages(t *testing.T) {
	config := testutil.LoadIntegrationTestConfig()
	if config.SkipNeuronAgent {
		t.Skip("Skipping NeuronAgent tests (SKIP_NEURONAGENT=true)")
	}

	ctx := context.Background()
	client := testutil.CreateNeuronAgentTestClient(t, config.NeuronAgentURL, config.NeuronAgentKey)
	if client == nil {
		return
	}

	/* Get or create an agent */
	agents, err := client.ListAgents(ctx)
	if err != nil {
		t.Fatalf("Failed to list agents: %v", err)
	}

	if len(agents) == 0 {
		t.Skip("No agents available to test")
	}

	agentID := agents[0].ID

	/* Create session */
	sessionReq := agent.CreateSessionRequest{
		AgentID: agentID,
	}

	session, err := client.CreateSession(ctx, sessionReq)
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}

	/* Send a message */
	messageReq := agent.SendMessageRequest{
		Role:    "user",
		Content: "Test message",
	}

	_, err = client.SendMessage(ctx, session.ID, messageReq)
	if err != nil {
		t.Fatalf("Failed to send message: %v", err)
	}

	/* Get messages */
	messages, err := client.GetMessages(ctx, session.ID)
	if err != nil {
		t.Fatalf("Failed to get messages: %v", err)
	}

	if len(messages) == 0 {
		t.Error("Expected at least one message")
	}

	/* Verify message structure */
	for _, msg := range messages {
		if msg.ID == "" {
			t.Error("Message missing ID")
		}
		if msg.SessionID != session.ID {
			t.Errorf("Expected session ID %s, got %s", session.ID, msg.SessionID)
		}
		if msg.Role == "" {
			t.Error("Message missing role")
		}
		if msg.Content == "" {
			t.Log("Message has empty content (might be expected)")
		}
	}
}








