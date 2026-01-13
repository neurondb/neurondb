package integration

import (
	"context"
	"testing"

	testutil "github.com/neurondb/NeuronDesktop/api/internal/testing"
	"github.com/neurondb/NeuronDesktop/api/internal/agent"
)

func TestAgentIntegration_Connection(t *testing.T) {
	config := testutil.LoadIntegrationTestConfig()
	if config.SkipNeuronAgent {
		t.Skip("Skipping NeuronAgent tests (SKIP_NEURONAGENT=true)")
	}

	ctx := context.Background()

	/* Verify connection */
	err := testutil.VerifyNeuronAgentConnection(ctx, config.NeuronAgentURL, config.NeuronAgentKey)
	if err != nil {
		t.Fatalf("Failed to connect to NeuronAgent: %v", err)
	}

	/* Create client */
	client := testutil.CreateNeuronAgentTestClient(t, config.NeuronAgentURL, config.NeuronAgentKey)
	if client == nil {
		return
	}
}

func TestAgentIntegration_ListAgents(t *testing.T) {
	config := testutil.LoadIntegrationTestConfig()
	if config.SkipNeuronAgent {
		t.Skip("Skipping NeuronAgent tests (SKIP_NEURONAGENT=true)")
	}

	ctx := context.Background()
	client := testutil.CreateNeuronAgentTestClient(t, config.NeuronAgentURL, config.NeuronAgentKey)
	if client == nil {
		return
	}

	/* List agents */
	agents, err := client.ListAgents(ctx)
	if err != nil {
		t.Fatalf("Failed to list agents: %v", err)
	}

	/* Verify agents structure */
	for _, agent := range agents {
		if agent.ID == "" {
			t.Error("Agent missing ID")
		}
		if agent.Name == "" {
			t.Error("Agent missing name")
		}
	}
}

func TestAgentIntegration_GetAgent(t *testing.T) {
	config := testutil.LoadIntegrationTestConfig()
	if config.SkipNeuronAgent {
		t.Skip("Skipping NeuronAgent tests (SKIP_NEURONAGENT=true)")
	}

	ctx := context.Background()
	client := testutil.CreateNeuronAgentTestClient(t, config.NeuronAgentURL, config.NeuronAgentKey)
	if client == nil {
		return
	}

	/* First, list agents to get an ID */
	agents, err := client.ListAgents(ctx)
	if err != nil {
		t.Fatalf("Failed to list agents: %v", err)
	}

	if len(agents) == 0 {
		t.Skip("No agents available to test")
	}

	/* Get first agent */
	agentID := agents[0].ID
	agent, err := client.GetAgent(ctx, agentID)
	if err != nil {
		t.Fatalf("Failed to get agent: %v", err)
	}

	if agent == nil {
		t.Fatal("Agent is nil")
	}

	if agent.ID != agentID {
		t.Errorf("Expected agent ID %s, got %s", agentID, agent.ID)
	}
}

func TestAgentIntegration_ErrorHandling(t *testing.T) {
	config := testutil.LoadIntegrationTestConfig()
	if config.SkipNeuronAgent {
		t.Skip("Skipping NeuronAgent tests (SKIP_NEURONAGENT=true)")
	}

	ctx := context.Background()
	client := testutil.CreateNeuronAgentTestClient(t, config.NeuronAgentURL, config.NeuronAgentKey)
	if client == nil {
		return
	}

	/* Try to get non-existent agent */
	_, err := client.GetAgent(ctx, "nonexistent-agent-id-xyz")
	if err == nil {
		t.Error("Expected error when getting non-existent agent")
	}
}







