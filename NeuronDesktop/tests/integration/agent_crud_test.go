package integration

import (
	"context"
	"testing"

	testutil "github.com/neurondb/NeuronDesktop/api/internal/testing"
	"github.com/neurondb/NeuronDesktop/api/internal/agent"
)

func TestAgentIntegration_CreateAgent(t *testing.T) {
	config := testutil.LoadIntegrationTestConfig()
	if config.SkipNeuronAgent {
		t.Skip("Skipping NeuronAgent tests (SKIP_NEURONAGENT=true)")
	}

	ctx := context.Background()
	client := testutil.CreateNeuronAgentTestClient(t, config.NeuronAgentURL, config.NeuronAgentKey)
	if client == nil {
		return
	}

	/* Create agent */
	createReq := agent.CreateAgentRequest{
		Name:         "test-agent-integration",
		Description:  "Test agent for integration testing",
		SystemPrompt: "You are a helpful assistant.",
		ModelName:    "gpt-4",
		EnabledTools: []string{"sql"},
		Config: map[string]interface{}{
			"temperature": 0.7,
			"max_tokens":  1000,
		},
	}

	createdAgent, err := client.CreateAgent(ctx, createReq)
	if err != nil {
		t.Fatalf("Failed to create agent: %v", err)
	}

	if createdAgent == nil {
		t.Fatal("Created agent is nil")
	}

	if createdAgent.Name != createReq.Name {
		t.Errorf("Expected agent name %s, got %s", createReq.Name, createdAgent.Name)
	}

	/* Cleanup */
	defer func() {
		if createdAgent.ID != "" {
			client.DeleteAgent(ctx, createdAgent.ID)
		}
	}()
}

func TestAgentIntegration_UpdateAgent(t *testing.T) {
	config := testutil.LoadIntegrationTestConfig()
	if config.SkipNeuronAgent {
		t.Skip("Skipping NeuronAgent tests (SKIP_NEURONAGENT=true)")
	}

	ctx := context.Background()
	client := testutil.CreateNeuronAgentTestClient(t, config.NeuronAgentURL, config.NeuronAgentKey)
	if client == nil {
		return
	}

	/* Create agent first */
	createReq := agent.CreateAgentRequest{
		Name:         "test-agent-update",
		Description:  "Test agent for update testing",
		SystemPrompt: "You are a helpful assistant.",
		ModelName:    "gpt-4",
	}

	createdAgent, err := client.CreateAgent(ctx, createReq)
	if err != nil {
		t.Fatalf("Failed to create agent: %v", err)
	}

	/* Cleanup */
	defer func() {
		if createdAgent.ID != "" {
			client.DeleteAgent(ctx, createdAgent.ID)
		}
	}()

	/* Update agent */
	updateReq := agent.UpdateAgentRequest{
		Description: "Updated description",
		Config: map[string]interface{}{
			"temperature": 0.9,
		},
	}

	updatedAgent, err := client.UpdateAgent(ctx, createdAgent.ID, updateReq)
	if err != nil {
		t.Fatalf("Failed to update agent: %v", err)
	}

	if updatedAgent.Description != updateReq.Description {
		t.Errorf("Expected description %s, got %s", updateReq.Description, updatedAgent.Description)
	}
}

func TestAgentIntegration_DeleteAgent(t *testing.T) {
	config := testutil.LoadIntegrationTestConfig()
	if config.SkipNeuronAgent {
		t.Skip("Skipping NeuronAgent tests (SKIP_NEURONAGENT=true)")
	}

	ctx := context.Background()
	client := testutil.CreateNeuronAgentTestClient(t, config.NeuronAgentURL, config.NeuronAgentKey)
	if client == nil {
		return
	}

	/* Create agent first */
	createReq := agent.CreateAgentRequest{
		Name:         "test-agent-delete",
		Description:  "Test agent for delete testing",
		SystemPrompt: "You are a helpful assistant.",
		ModelName:    "gpt-4",
	}

	createdAgent, err := client.CreateAgent(ctx, createReq)
	if err != nil {
		t.Fatalf("Failed to create agent: %v", err)
	}

	/* Delete agent */
	err = client.DeleteAgent(ctx, createdAgent.ID)
	if err != nil {
		t.Fatalf("Failed to delete agent: %v", err)
	}

	/* Verify agent is deleted */
	_, err = client.GetAgent(ctx, createdAgent.ID)
	if err == nil {
		t.Error("Expected error when getting deleted agent")
	}
}

func TestAgentIntegration_ListModels(t *testing.T) {
	config := testutil.LoadIntegrationTestConfig()
	if config.SkipNeuronAgent {
		t.Skip("Skipping NeuronAgent tests (SKIP_NEURONAGENT=true)")
	}

	ctx := context.Background()
	client := testutil.CreateNeuronAgentTestClient(t, config.NeuronAgentURL, config.NeuronAgentKey)
	if client == nil {
		return
	}

	/* List models */
	models, err := client.ListModels(ctx)
	if err != nil {
		/* Models endpoint might not be available, which is OK */
		t.Logf("Failed to list models (might not be available): %v", err)
		return
	}

	/* Verify models structure */
	for _, model := range models {
		if model.Name == "" {
			t.Error("Model missing name")
		}
	}
}







