/*-------------------------------------------------------------------------
 *
 * create.go
 *    Agent creation commands for neuronagent-cli
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/cli/cmd/create.go
 *
 *-------------------------------------------------------------------------
 */

package cmd

import (
	"fmt"

	"github.com/neurondb/NeuronAgent/cli/pkg/client"
	"github.com/neurondb/NeuronAgent/cli/pkg/config"
	"github.com/neurondb/NeuronAgent/cli/pkg/templates"
	"github.com/neurondb/NeuronAgent/cli/pkg/wizard"
	"github.com/spf13/cobra"
)

var (
	createName        string
	createProfile     string
	createTools       []string
	createModel       string
	createConfig      string
	createTemplate    string
	createWorkflow    string
	createInteractive bool
)

var createCmd = &cobra.Command{
	Use:   "create",
	Short: "Create a new agent",
	Long:  "Create a new agent using interactive wizard, template, or configuration file",
	RunE:  runCreate,
}

func init() {
	createCmd.Flags().BoolVarP(&createInteractive, "interactive", "i", false, "Start interactive wizard")
	createCmd.Flags().StringVarP(&createName, "name", "n", "", "Agent name")
	createCmd.Flags().StringVarP(&createProfile, "profile", "p", "", "Agent profile (general-assistant, research-assistant, data-analyst, workflow-agent)")
	createCmd.Flags().StringSliceVarP(&createTools, "tools", "t", []string{}, "Comma-separated list of tools (sql, http, code, shell, browser)")
	createCmd.Flags().StringVarP(&createModel, "model", "m", "", "Model name (e.g., gpt-4)")
	createCmd.Flags().StringVarP(&createConfig, "config", "c", "", "Path to agent configuration file (YAML or JSON)")
	createCmd.Flags().StringVarP(&createTemplate, "template", "T", "", "Template name to use for creation")
	createCmd.Flags().StringVarP(&createWorkflow, "workflow", "w", "", "Path to workflow definition file (YAML)")
}

func runCreate(cmd *cobra.Command, args []string) error {
	apiClient := client.NewClient(apiURL, apiKey)

	/* Interactive mode */
	if createInteractive {
		return wizard.RunWizard(apiClient)
	}

	/* Template mode */
	if createTemplate != "" {
		return createFromTemplate(apiClient, createTemplate)
	}

	/* Config file mode */
	if createConfig != "" {
		return createFromConfig(apiClient, createConfig)
	}

	if createName == "" {
		return fmt.Errorf("agent name is required (use --name or --interactive)")
	}

	agentConfig := &config.AgentConfig{
		Name:    createName,
		Profile: createProfile,
		Model: config.ModelConfig{
			Name: createModel,
		},
		Tools: createTools,
	}

	/* Load workflow if provided */
	if createWorkflow != "" {
		workflow, err := config.LoadWorkflow(createWorkflow)
		if err != nil {
			return fmt.Errorf("failed to load workflow: %w", err)
		}
		agentConfig.Workflow = workflow
	}

	agent, err := apiClient.CreateAgent(agentConfig)
	if err != nil {
		return fmt.Errorf("failed to create agent: %w", err)
	}

	fmt.Printf("✅ Agent created successfully!\n")
	fmt.Printf("ID: %s\n", agent.ID)
	fmt.Printf("Name: %s\n", agent.Name)
	return nil
}

func createFromTemplate(apiClient *client.Client, templateName string) error {
	fmt.Printf("📋 Loading template: %s\n", templateName)

	template, err := templates.LoadTemplate(templateName)
	if err != nil {
		return fmt.Errorf("failed to load template: %w", err)
	}

	fmt.Printf("✅ Template loaded: %s\n", template.Name)
	fmt.Printf("Description: %s\n", template.Description)

	/* If name is provided, use it; otherwise use template name */
	agentName := createName
	if agentName == "" {
		agentName = template.Name + "-instance"
	}

	agentConfig := template.ToAgentConfig()
	agentConfig.Name = agentName

	agent, err := apiClient.CreateAgent(agentConfig)
	if err != nil {
		return fmt.Errorf("failed to create agent from template: %w", err)
	}

	fmt.Printf("✅ Agent created from template!\n")
	fmt.Printf("ID: %s\n", agent.ID)
	fmt.Printf("Name: %s\n", agent.Name)
	return nil
}

func createFromConfig(apiClient *client.Client, configPath string) error {
	fmt.Printf("📄 Loading configuration from: %s\n", configPath)

	agentConfig, err := config.LoadAgentConfig(configPath)
	if err != nil {
		return fmt.Errorf("failed to load configuration: %w", err)
	}

	agent, err := apiClient.CreateAgent(agentConfig)
	if err != nil {
		return fmt.Errorf("failed to create agent: %w", err)
	}

	fmt.Printf("✅ Agent created from configuration!\n")
	fmt.Printf("ID: %s\n", agent.ID)
	fmt.Printf("Name: %s\n", agent.Name)
	return nil
}
