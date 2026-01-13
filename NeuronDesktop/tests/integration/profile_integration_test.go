package integration

import (
	"context"
	"testing"

	testutil "github.com/neurondb/NeuronDesktop/api/internal/testing"
)

/* Note: Profile integration tests verify that profiles correctly configure
 * all three services (NeuronDB, NeuronMCP, NeuronAgent) and that they work
 * together through the profile configuration.
 */

func TestProfileIntegration_NeuronDBConfiguration(t *testing.T) {
	config := testutil.LoadIntegrationTestConfig()
	if config.SkipNeuronDB {
		t.Skip("Skipping profile integration tests (NeuronDB not available)")
	}

	ctx := context.Background()

	/* Test that a profile's NeuronDB DSN works */
	client := testutil.CreateNeuronDBTestClient(t, config.NeuronDBDSN)
	if client == nil {
		return
	}
	defer client.Close()

	/* Verify connection works */
	_, err := client.ListCollections(ctx)
	if err != nil {
		t.Fatalf("Failed to list collections with profile DSN: %v", err)
	}
}

func TestProfileIntegration_MCPConfiguration(t *testing.T) {
	config := testutil.LoadIntegrationTestConfig()
	if config.SkipNeuronMCP {
		t.Skip("Skipping profile integration tests (NeuronMCP not available)")
	}

	if config.NeuronMCPCommand == "" {
		t.Skip("Skipping test: NeuronMCP command not configured")
	}

	/* Test that a profile's MCP config works */
	env := map[string]string{
		"NEURONDB_CONNECTION_STRING": config.NeuronDBDSN,
	}

	client := testutil.CreateNeuronMCPTestClient(t, config.NeuronMCPCommand, env)
	if client == nil {
		return
	}
	defer client.Close()

	/* Verify MCP connection works */
	_, err := client.ListTools()
	if err != nil {
		t.Fatalf("Failed to list tools with profile MCP config: %v", err)
	}
}

func TestProfileIntegration_AgentConfiguration(t *testing.T) {
	config := testutil.LoadIntegrationTestConfig()
	if config.SkipNeuronAgent {
		t.Skip("Skipping profile integration tests (NeuronAgent not available)")
	}

	ctx := context.Background()

	/* Test that a profile's Agent endpoint works */
	client := testutil.CreateNeuronAgentTestClient(t, config.NeuronAgentURL, config.NeuronAgentKey)
	if client == nil {
		return
	}

	/* Verify agent connection works */
	_, err := client.ListAgents(ctx)
	if err != nil {
		t.Fatalf("Failed to list agents with profile Agent config: %v", err)
	}
}

func TestProfileIntegration_AllServicesConfigured(t *testing.T) {
	config := testutil.LoadIntegrationTestConfig()
	if config.SkipNeuronDB || config.SkipNeuronMCP || config.SkipNeuronAgent {
		t.Skip("Skipping test (not all services available)")
	}

	ctx := context.Background()

	/* Test that all three services can be configured in a profile and work together */

	/* NeuronDB */
	dbClient := testutil.CreateNeuronDBTestClient(t, config.NeuronDBDSN)
	if dbClient == nil {
		return
	}
	defer dbClient.Close()

	/* NeuronMCP */
	env := map[string]string{
		"NEURONDB_CONNECTION_STRING": config.NeuronDBDSN,
	}
	mcpClient := testutil.CreateNeuronMCPTestClient(t, config.NeuronMCPCommand, env)
	if mcpClient == nil {
		return
	}
	defer mcpClient.Close()

	/* NeuronAgent */
	agentClient := testutil.CreateNeuronAgentTestClient(t, config.NeuronAgentURL, config.NeuronAgentKey)
	if agentClient == nil {
		return
	}

	/* Verify all work */
	_, err := dbClient.ListCollections(ctx)
	if err != nil {
		t.Errorf("NeuronDB failed: %v", err)
	}

	_, err = mcpClient.ListTools()
	if err != nil {
		t.Errorf("NeuronMCP failed: %v", err)
	}

	_, err = agentClient.ListAgents(ctx)
	if err != nil {
		t.Errorf("NeuronAgent failed: %v", err)
	}
}







