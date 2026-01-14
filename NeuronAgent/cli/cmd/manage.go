/*-------------------------------------------------------------------------
 *
 * manage.go
 *    Agent management commands for neuronagent-cli
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/cli/cmd/manage.go
 *
 *-------------------------------------------------------------------------
 */

package cmd

import (
	"fmt"

	"github.com/spf13/cobra"
	"github.com/neurondb/NeuronAgent/cli/pkg/client"
	"github.com/neurondb/NeuronAgent/cli/pkg/config"
)

var (
	listCmd = &cobra.Command{
		Use:   "list",
		Short: "List all agents",
		RunE:  listAgents,
	}

	showCmd = &cobra.Command{
		Use:   "show [agent-id]",
		Short: "Show agent details",
		Args:  cobra.ExactArgs(1),
		RunE:  showAgent,
	}

	updateCmd = &cobra.Command{
		Use:   "update [agent-id]",
		Short: "Update an agent",
		Args:  cobra.ExactArgs(1),
		RunE:  updateAgent,
	}

	deleteCmd = &cobra.Command{
		Use:   "delete [agent-id]",
		Short: "Delete an agent",
		Args:  cobra.ExactArgs(1),
		RunE:  deleteAgent,
	}

	cloneCmd = &cobra.Command{
		Use:   "clone [agent-id]",
		Short: "Clone an agent",
		Args:  cobra.ExactArgs(1),
		RunE:  cloneAgent,
	}

	updateConfig string
	cloneName    string
)

func init() {
	updateCmd.Flags().StringVarP(&updateConfig, "config", "c", "", "Path to agent configuration file")
	cloneCmd.Flags().StringVarP(&cloneName, "name", "n", "", "Name for cloned agent")
	cloneCmd.MarkFlagRequired("name")
}

func listAgents(cmd *cobra.Command, args []string) error {
	apiClient := client.NewClient(apiURL, apiKey)

	agents, err := apiClient.ListAgents()
	if err != nil {
		return fmt.Errorf("failed to list agents: %w", err)
	}

	if len(agents) == 0 {
		fmt.Println("No agents found")
		return nil
	}

	fmt.Println("\n🤖 Agents:")
	fmt.Println("─────────────────────────────────────────────────────────")
	for _, agent := range agents {
		fmt.Printf("  %-36s %s\n", agent.ID, agent.Name)
		if agent.Description != "" {
			fmt.Printf("    %s\n", agent.Description)
		}
		fmt.Printf("    Model: %s, Tools: %v\n", agent.ModelName, agent.EnabledTools)
	}
	fmt.Println()

	return nil
}

func showAgent(cmd *cobra.Command, args []string) error {
	agentID := args[0]
	apiClient := client.NewClient(apiURL, apiKey)

	agent, err := apiClient.GetAgent(agentID)
	if err != nil {
		return fmt.Errorf("failed to get agent: %w", err)
	}

	fmt.Printf("\n🤖 Agent: %s\n", agent.Name)
	fmt.Println("─────────────────────────────────────────────────────────")
	fmt.Printf("ID: %s\n", agent.ID)
	if agent.Description != "" {
		fmt.Printf("Description: %s\n", agent.Description)
	}
	fmt.Printf("Model: %s\n", agent.ModelName)
	fmt.Printf("Tools: %v\n", agent.EnabledTools)
	if len(agent.Config) > 0 {
		fmt.Printf("Config: %+v\n", agent.Config)
	}
	fmt.Println()

	return nil
}

func updateAgent(cmd *cobra.Command, args []string) error {
	agentID := args[0]
	
	if updateConfig == "" {
		return fmt.Errorf("--config flag is required for update")
	}

	apiClient := client.NewClient(apiURL, apiKey)

	fmt.Printf("📄 Loading configuration: %s\n", updateConfig)
	agentConfig, err := config.LoadAgentConfig(updateConfig)
	if err != nil {
		return fmt.Errorf("failed to load configuration: %w", err)
	}

	fmt.Printf("🔄 Updating agent: %s\n", agentID)
	agent, err := apiClient.UpdateAgent(agentID, agentConfig)
	if err != nil {
		return fmt.Errorf("failed to update agent: %w", err)
	}

	fmt.Printf("✅ Agent updated successfully!\n")
	fmt.Printf("Name: %s\n", agent.Name)
	return nil
}

func deleteAgent(cmd *cobra.Command, args []string) error {
	agentID := args[0]
	apiClient := client.NewClient(apiURL, apiKey)

	fmt.Printf("🗑️  Deleting agent: %s\n", agentID)
	
	if err := apiClient.DeleteAgent(agentID); err != nil {
		return fmt.Errorf("failed to delete agent: %w", err)
	}

	fmt.Println("✅ Agent deleted successfully")
	return nil
}

func cloneAgent(cmd *cobra.Command, args []string) error {
	agentID := args[0]
	apiClient := client.NewClient(apiURL, apiKey)

	fmt.Printf("📋 Fetching agent: %s\n", agentID)
	sourceAgent, err := apiClient.GetAgent(agentID)
	if err != nil {
		return fmt.Errorf("failed to get agent: %w", err)
	}

	agentConfig := &config.AgentConfig{
		Name:        cloneName,
		Description: sourceAgent.Description + " (cloned)",
		Model: config.ModelConfig{
			Name: sourceAgent.ModelName,
		},
		Tools: sourceAgent.EnabledTools,
		Config: sourceAgent.Config,
	}

	fmt.Printf("🚀 Creating cloned agent: %s\n", cloneName)
	agent, err := apiClient.CreateAgent(agentConfig)
	if err != nil {
		return fmt.Errorf("failed to create cloned agent: %w", err)
	}

	fmt.Printf("✅ Agent cloned successfully!\n")
	fmt.Printf("ID: %s\n", agent.ID)
	fmt.Printf("Name: %s\n", agent.Name)
	return nil
}



