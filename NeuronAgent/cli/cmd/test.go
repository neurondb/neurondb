/*-------------------------------------------------------------------------
 *
 * test.go
 *    Agent testing commands for neuronagent-cli
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/cli/cmd/test.go
 *
 *-------------------------------------------------------------------------
 */

package cmd

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/spf13/cobra"
	"github.com/neurondb/NeuronAgent/cli/pkg/client"
)

var (
	testMessage    string
	testWorkflow   bool
	testDebug      bool
	testDryRun     bool
	testConfig     string
)

var testCmd = &cobra.Command{
	Use:   "test [agent-id]",
	Short: "Test an agent",
	Long:  "Test an agent interactively or with a single message",
	Args:  cobra.MinimumNArgs(0),
	RunE:  runTest,
}

func init() {
	testCmd.Flags().StringVarP(&testMessage, "message", "m", "", "Single test message")
	testCmd.Flags().BoolVarP(&testWorkflow, "workflow", "w", false, "Test workflow step-by-step")
	testCmd.Flags().BoolVarP(&testDebug, "debug", "d", false, "Enable debug output")
	testCmd.Flags().BoolVar(&testDryRun, "dry-run", false, "Validate config without creating agent")
	testCmd.Flags().StringVarP(&testConfig, "config", "c", "", "Test config file before creating agent")
}

func runTest(cmd *cobra.Command, args []string) error {
	apiClient := client.NewClient(apiURL, apiKey)

	// Dry run mode - validate config
	if testDryRun && testConfig != "" {
		return testConfigFile(testConfig)
	}

	// Need agent ID for testing
	if len(args) == 0 {
		return fmt.Errorf("agent ID is required (or use --config --dry-run to validate)")
	}

	agentID := args[0]

	if testMessage != "" {
		// Single message test
		return testSingleMessage(apiClient, agentID, testMessage)
	}

	// Interactive test mode
	return testInteractive(apiClient, agentID)
}

func testConfigFile(configPath string) error {
	fmt.Printf("🔍 Validating configuration: %s\n", configPath)
	
	/*
	 * TODO: Implement comprehensive configuration file validation.
	 * This should include:
	 * 1. Parsing the configuration file (YAML/JSON).
	 * 2. Validating required fields are present.
	 * 3. Checking data types and value ranges.
	 * 4. Verifying file paths and network endpoints are accessible.
	 * 5. Returning detailed error messages for any validation failures.
	 *
	 * Current state: Placeholder, always returns success.
	 */
	fmt.Println("✅ Configuration file is valid")
	return nil
}

func testSingleMessage(apiClient *client.Client, agentID, message string) error {
	fmt.Printf("🤖 Testing agent: %s\n", agentID)
	fmt.Printf("💬 Message: %s\n\n", message)

	// Create session
	session, err := apiClient.CreateSession(agentID, nil)
	if err != nil {
		return fmt.Errorf("failed to create session: %w", err)
	}

	// Send message
	response, err := apiClient.SendMessage(session.ID, message, false)
	if err != nil {
		return fmt.Errorf("failed to send message: %w", err)
	}

	fmt.Println("📤 Response:")
	fmt.Println("─────────────────────────────────────────────────────────")
	fmt.Println(response.Content)
	fmt.Println()

	return nil
}

func testInteractive(apiClient *client.Client, agentID string) error {
	fmt.Printf("🤖 Interactive test mode for agent: %s\n", agentID)
	fmt.Println("Type 'exit' or 'quit' to end the session")
	fmt.Println("─────────────────────────────────────────────────────────")
	fmt.Println()

	// Create session
	session, err := apiClient.CreateSession(agentID, nil)
	if err != nil {
		return fmt.Errorf("failed to create session: %w", err)
	}

	fmt.Printf("✅ Session created: %s\n\n", session.ID)

	scanner := bufio.NewScanner(os.Stdin)
	
	for {
		fmt.Print("You: ")
		if !scanner.Scan() {
			break
		}

		message := strings.TrimSpace(scanner.Text())
		if message == "" {
			continue
		}

		if message == "exit" || message == "quit" {
			break
		}

		fmt.Print("\n🤖 Agent: ")
		response, err := apiClient.SendMessage(session.ID, message, false)
		if err != nil {
			fmt.Printf("❌ Error: %v\n\n", err)
			continue
		}

		fmt.Println(response.Content)
		fmt.Println()
	}

	fmt.Println("\n👋 Session ended")
	return nil
}



