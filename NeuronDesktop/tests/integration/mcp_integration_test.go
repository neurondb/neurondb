package integration

import (
	"context"
	"testing"
	"time"

	testutil "github.com/neurondb/NeuronDesktop/api/internal/testing"
	"github.com/neurondb/NeuronDesktop/api/internal/mcp"
)

func TestMCPIntegration_Connection(t *testing.T) {
	config := testutil.LoadIntegrationTestConfig()
	if config.SkipNeuronMCP {
		t.Skip("Skipping NeuronMCP tests (SKIP_NEURONMCP=true)")
	}

	if config.NeuronMCPCommand == "" {
		t.Skip("Skipping test: NeuronMCP command not configured")
	}

	ctx := context.Background()
	env := map[string]string{
		"NEURONDB_CONNECTION_STRING": config.NeuronDBDSN,
	}

	/* Verify connection */
	err := testutil.VerifyNeuronMCPConnection(ctx, config.NeuronMCPCommand, env)
	if err != nil {
		t.Fatalf("Failed to connect to NeuronMCP: %v", err)
	}

	/* Create client */
	client := testutil.CreateNeuronMCPTestClient(t, config.NeuronMCPCommand, env)
	if client == nil {
		return
	}
	defer client.Close()

	/* Verify client is alive */
	if !client.IsAlive() {
		t.Error("MCP client is not alive")
	}
}

func TestMCPIntegration_ProcessLifecycle(t *testing.T) {
	config := testutil.LoadIntegrationTestConfig()
	if config.SkipNeuronMCP {
		t.Skip("Skipping NeuronMCP tests (SKIP_NEURONMCP=true)")
	}

	if config.NeuronMCPCommand == "" {
		t.Skip("Skipping test: NeuronMCP command not configured")
	}

	env := map[string]string{
		"NEURONDB_CONNECTION_STRING": config.NeuronDBDSN,
	}

	/* Create client */
	client := testutil.CreateNeuronMCPTestClient(t, config.NeuronMCPCommand, env)
	if client == nil {
		return
	}

	/* Verify client is alive */
	if !client.IsAlive() {
		t.Error("MCP client is not alive after creation")
	}

	/* Close client */
	err := client.Close()
	if err != nil {
		t.Errorf("Failed to close MCP client: %v", err)
	}

	/* Wait a bit for process to terminate */
	time.Sleep(100 * time.Millisecond)

	/* Verify client is no longer alive */
	if client.IsAlive() {
		t.Error("MCP client is still alive after close")
	}
}

func TestMCPIntegration_Reconnection(t *testing.T) {
	config := testutil.LoadIntegrationTestConfig()
	if config.SkipNeuronMCP {
		t.Skip("Skipping NeuronMCP tests (SKIP_NEURONMCP=true)")
	}

	if config.NeuronMCPCommand == "" {
		t.Skip("Skipping test: NeuronMCP command not configured")
	}

	env := map[string]string{
		"NEURONDB_CONNECTION_STRING": config.NeuronDBDSN,
	}

	/* Create first client */
	client1 := testutil.CreateNeuronMCPTestClient(t, config.NeuronMCPCommand, env)
	if client1 == nil {
		return
	}
	defer client1.Close()

	/* Create second client (should work independently) */
	client2 := testutil.CreateNeuronMCPTestClient(t, config.NeuronMCPCommand, env)
	if client2 == nil {
		return
	}
	defer client2.Close()

	/* Both should be alive */
	if !client1.IsAlive() {
		t.Error("First MCP client is not alive")
	}
	if !client2.IsAlive() {
		t.Error("Second MCP client is not alive")
	}
}

func TestMCPIntegration_Initialization(t *testing.T) {
	config := testutil.LoadIntegrationTestConfig()
	if config.SkipNeuronMCP {
		t.Skip("Skipping NeuronMCP tests (SKIP_NEURONMCP=true)")
	}

	if config.NeuronMCPCommand == "" {
		t.Skip("Skipping test: NeuronMCP command not configured")
	}

	env := map[string]string{
		"NEURONDB_CONNECTION_STRING": config.NeuronDBDSN,
	}

	client := testutil.CreateNeuronMCPTestClient(t, config.NeuronMCPCommand, env)
	if client == nil {
		return
	}
	defer client.Close()

	/* Client should be initialized and ready after creation */
	if !client.IsAlive() {
		t.Error("MCP client is not alive after initialization")
	}
}







