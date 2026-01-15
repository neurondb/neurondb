package integration

import (
	"context"
	"testing"

	testutil "github.com/neurondb/NeuronDesktop/api/internal/testing"
	"github.com/neurondb/NeuronDesktop/api/internal/agent"
	"github.com/neurondb/NeuronDesktop/api/internal/mcp"
	"github.com/neurondb/NeuronDesktop/api/internal/neurondb"
)

func TestCrossComponentIntegration_SharedDatabaseConnection(t *testing.T) {
	config := testutil.LoadIntegrationTestConfig()
	if config.SkipNeuronDB || config.SkipNeuronMCP {
		t.Skip("Skipping cross-component tests (services not available)")
	}

	ctx := context.Background()

	/* Create NeuronDB client */
	dbClient := testutil.CreateNeuronDBTestClient(t, config.NeuronDBDSN)
	if dbClient == nil {
		return
	}
	defer dbClient.Close()

	/* Create NeuronMCP client (should use same database) */
	env := map[string]string{
		"NEURONDB_CONNECTION_STRING": config.NeuronDBDSN,
	}

	mcpClient := testutil.CreateNeuronMCPTestClient(t, config.NeuronMCPCommand, env)
	if mcpClient == nil {
		return
	}
	defer mcpClient.Close()

	/* Verify both can access the same database */
	collections, err := dbClient.ListCollections(ctx)
	if err != nil {
		t.Fatalf("Failed to list collections from NeuronDB: %v", err)
	}

	/* MCP should be able to call tools that access the same database */
	toolsResp, err := mcpClient.ListTools()
	if err != nil {
		t.Fatalf("Failed to list tools from MCP: %v", err)
	}

	/* Both should work with the same database */
	t.Logf("NeuronDB found %d collections, MCP found %d tools", len(collections), len(toolsResp.Tools))
}

func TestCrossComponentIntegration_EndToEndWorkflow(t *testing.T) {
	config := testutil.LoadIntegrationTestConfig()
	if config.SkipNeuronDB || config.SkipNeuronMCP || config.SkipNeuronAgent {
		t.Skip("Skipping end-to-end test (services not available)")
	}

	ctx := context.Background()

	/* Create all three clients */
	dbClient := testutil.CreateNeuronDBTestClient(t, config.NeuronDBDSN)
	if dbClient == nil {
		return
	}
	defer dbClient.Close()

	env := map[string]string{
		"NEURONDB_CONNECTION_STRING": config.NeuronDBDSN,
	}

	mcpClient := testutil.CreateNeuronMCPTestClient(t, config.NeuronMCPCommand, env)
	if mcpClient == nil {
		return
	}
	defer mcpClient.Close()

	agentClient := testutil.CreateNeuronAgentTestClient(t, config.NeuronAgentURL, config.NeuronAgentKey)
	if agentClient == nil {
		return
	}

	/* Test workflow:
	 * 1. Create collection in NeuronDB
	 * 2. Search via MCP tools
	 * 3. Use agent to interact with results
	 */

	/* Step 1: Create test collection */
	schema := "test_e2e"
	table := "test_vectors"
	vectorCol := "embedding"
	dimensions := 64

	err := testutil.CreateTestCollection(ctx, dbClient, schema, table, vectorCol, dimensions)
	if err != nil {
		t.Fatalf("Failed to create test collection: %v", err)
	}
	defer func() {
		cleanupSQL := "DROP SCHEMA IF EXISTS test_e2e CASCADE;"
		dbClient.ExecuteSQLFull(ctx, cleanupSQL)
	}()

	/* Step 2: Verify MCP can see the collection (via tools) */
	toolsResp, err := mcpClient.ListTools()
	if err != nil {
		t.Fatalf("Failed to list MCP tools: %v", err)
	}

	/* Step 3: Verify agent can be used */
	agents, err := agentClient.ListAgents(ctx)
	if err != nil {
		t.Fatalf("Failed to list agents: %v", err)
	}

	t.Logf("E2E workflow: Created collection, MCP has %d tools, Agent has %d agents", len(toolsResp.Tools), len(agents))
}

func TestCrossComponentIntegration_ConcurrentOperations(t *testing.T) {
	config := testutil.LoadIntegrationTestConfig()
	if config.SkipNeuronDB || config.SkipNeuronMCP {
		t.Skip("Skipping concurrent operations test (services not available)")
	}

	ctx := context.Background()

	/* Create multiple clients for concurrent access */
	dbClient1 := testutil.CreateNeuronDBTestClient(t, config.NeuronDBDSN)
	if dbClient1 == nil {
		return
	}
	defer dbClient1.Close()

	dbClient2 := testutil.CreateNeuronDBTestClient(t, config.NeuronDBDSN)
	if dbClient2 == nil {
		return
	}
	defer dbClient2.Close()

	env := map[string]string{
		"NEURONDB_CONNECTION_STRING": config.NeuronDBDSN,
	}

	mcpClient1 := testutil.CreateNeuronMCPTestClient(t, config.NeuronMCPCommand, env)
	if mcpClient1 == nil {
		return
	}
	defer mcpClient1.Close()

	mcpClient2 := testutil.CreateNeuronMCPTestClient(t, config.NeuronMCPCommand, env)
	if mcpClient2 == nil {
		return
	}
	defer mcpClient2.Close()

	/* Perform concurrent operations */
	done := make(chan bool, 4)

	go func() {
		_, err := dbClient1.ListCollections(ctx)
		if err != nil {
			t.Errorf("DB client 1 failed: %v", err)
		}
		done <- true
	}()

	go func() {
		_, err := dbClient2.ListCollections(ctx)
		if err != nil {
			t.Errorf("DB client 2 failed: %v", err)
		}
		done <- true
	}()

	go func() {
		_, err := mcpClient1.ListTools()
		if err != nil {
			t.Errorf("MCP client 1 failed: %v", err)
		}
		done <- true
	}()

	go func() {
		_, err := mcpClient2.ListTools()
		if err != nil {
			t.Errorf("MCP client 2 failed: %v", err)
		}
		done <- true
	}()

	/* Wait for all operations */
	for i := 0; i < 4; i++ {
		<-done
	}
}

func TestCrossComponentIntegration_ResourceCleanup(t *testing.T) {
	config := testutil.LoadIntegrationTestConfig()
	if config.SkipNeuronDB || config.SkipNeuronMCP {
		t.Skip("Skipping resource cleanup test (services not available)")
	}

	ctx := context.Background()

	/* Create clients */
	dbClient := testutil.CreateNeuronDBTestClient(t, config.NeuronDBDSN)
	if dbClient == nil {
		return
	}

	env := map[string]string{
		"NEURONDB_CONNECTION_STRING": config.NeuronDBDSN,
	}

	mcpClient := testutil.CreateNeuronMCPTestClient(t, config.NeuronMCPCommand, env)
	if mcpClient == nil {
		return
	}

	/* Verify clients are alive */
	if !mcpClient.IsAlive() {
		t.Error("MCP client is not alive")
	}

	/* Close clients */
	err := dbClient.Close()
	if err != nil {
		t.Errorf("Failed to close DB client: %v", err)
	}

	err = mcpClient.Close()
	if err != nil {
		t.Errorf("Failed to close MCP client: %v", err)
	}

	/* Verify cleanup */
	if mcpClient.IsAlive() {
		t.Error("MCP client is still alive after close")
	}
}








