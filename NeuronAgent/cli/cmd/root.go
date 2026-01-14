/*-------------------------------------------------------------------------
 *
 * root.go
 *    Root command and global flags for neuronagent-cli
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/cli/cmd/root.go
 *
 *-------------------------------------------------------------------------
 */

package cmd

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

var (
	apiURL       string
	apiKey       string
	outputFormat string
)

var rootCmd = &cobra.Command{
	Use:   "neuronagent-cli",
	Short: "NeuronAgent CLI - Easy agent creation and management",
	Long: `NeuronAgent CLI provides easy-to-use commands for creating, managing, and testing AI agents.

Features:
  - Interactive wizard for agent creation
  - Template-based agent creation
  - Workflow management
  - Agent testing and debugging
  - Comprehensive agent management

Examples:
  # Interactive agent creation
  neuronagent-cli create --interactive

  # Create agent from template
  neuronagent-cli create --template customer-support --name my-support-bot

  # Create agent from config file
  neuronagent-cli create --config agent.yaml

  # List templates
  neuronagent-cli template list

  # Test an agent
  neuronagent-cli test <agent-id>

  # List all agents
  neuronagent-cli list
`,
}

func init() {
	rootCmd.PersistentFlags().StringVar(&apiURL, "url", getEnvOrDefault("NEURONAGENT_URL", "http://localhost:8080"), "NeuronAgent API URL")
	rootCmd.PersistentFlags().StringVar(&apiKey, "key", getEnvOrDefault("NEURONAGENT_API_KEY", ""), "NeuronAgent API key (required)")
	rootCmd.PersistentFlags().StringVar(&outputFormat, "format", "text", "Output format (text, json)")

	// Add all subcommands
	rootCmd.AddCommand(createCmd)
	rootCmd.AddCommand(templateCmd)
	rootCmd.AddCommand(workflowCmd)
	rootCmd.AddCommand(testCmd)
	rootCmd.AddCommand(listCmd)
	rootCmd.AddCommand(showCmd)
	rootCmd.AddCommand(updateCmd)
	rootCmd.AddCommand(deleteCmd)
	rootCmd.AddCommand(cloneCmd)
}

func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func Execute() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}



