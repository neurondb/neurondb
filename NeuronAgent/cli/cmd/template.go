/*-------------------------------------------------------------------------
 *
 * template.go
 *    Template management commands for neuronagent-cli
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/cli/cmd/template.go
 *
 *-------------------------------------------------------------------------
 */

package cmd

import (
	"fmt"

	"github.com/neurondb/NeuronAgent/cli/pkg/client"
	"github.com/neurondb/NeuronAgent/cli/pkg/templates"
	"github.com/spf13/cobra"
)

var templateListCmd = &cobra.Command{
	Use:   "list",
	Short: "List available templates",
	RunE:  listTemplates,
}

var templateShowCmd = &cobra.Command{
	Use:   "show [name]",
	Short: "Show template details",
	Args:  cobra.ExactArgs(1),
	RunE:  showTemplate,
}

var templateSearchCmd = &cobra.Command{
	Use:   "search [query]",
	Short: "Search templates",
	Args:  cobra.ExactArgs(1),
	RunE:  searchTemplates,
}

var templateDeployCmd = &cobra.Command{
	Use:   "deploy [name]",
	Short: "Deploy agent from template",
	Args:  cobra.ExactArgs(1),
	RunE:  deployTemplate,
}

var templateSaveCmd = &cobra.Command{
	Use:   "save [agent-id]",
	Short: "Save agent as template",
	Args:  cobra.ExactArgs(1),
	RunE:  saveTemplate,
}

var templateCmd = &cobra.Command{
	Use:   "template",
	Short: "Manage agent templates",
	Long:  "List, show, search, deploy, and save agent templates",
}

var deployName string

func init() {
	templateDeployCmd.Flags().StringVarP(&deployName, "name", "n", "", "Agent name (default: template-name-instance)")
	templateDeployCmd.MarkFlagRequired("name")

	templateCmd.AddCommand(templateListCmd)
	templateCmd.AddCommand(templateShowCmd)
	templateCmd.AddCommand(templateSearchCmd)
	templateCmd.AddCommand(templateDeployCmd)
	templateCmd.AddCommand(templateSaveCmd)
}

func listTemplates(cmd *cobra.Command, args []string) error {
	tmplList, err := templates.ListTemplates()
	if err != nil {
		return fmt.Errorf("failed to list templates: %w", err)
	}

	if len(tmplList) == 0 {
		fmt.Println("No templates available")
		return nil
	}

	fmt.Println("\n📋 Available Templates:")
	fmt.Println("─────────────────────────────────────────────────────────")

	for _, tmpl := range tmplList {
		fmt.Printf("  %-30s %s\n", tmpl.Name, tmpl.Description)
		fmt.Printf("    Category: %s\n", tmpl.Category)
	}
	fmt.Println()

	return nil
}

func showTemplate(cmd *cobra.Command, args []string) error {
	templateName := args[0]

	tmpl, err := templates.LoadTemplate(templateName)
	if err != nil {
		return fmt.Errorf("failed to load template: %w", err)
	}

	fmt.Printf("\n📋 Template: %s\n", tmpl.Name)
	fmt.Println("─────────────────────────────────────────────────────────")
	fmt.Printf("Description: %s\n", tmpl.Description)
	fmt.Printf("Category: %s\n", tmpl.Category)
	fmt.Printf("Profile: %s\n", tmpl.Profile)

	if len(tmpl.Tools) > 0 {
		fmt.Printf("Tools: %v\n", tmpl.Tools)
	}

	if tmpl.Model.Name != "" {
		fmt.Printf("Model: %s\n", tmpl.Model.Name)
	}

	if tmpl.Workflow != nil && len(tmpl.Workflow.Steps) > 0 {
		fmt.Printf("\nWorkflow Steps: %d\n", len(tmpl.Workflow.Steps))
		for i, step := range tmpl.Workflow.Steps {
			fmt.Printf("  %d. %s (%s)\n", i+1, step.Name, step.Type)
		}
	}

	fmt.Println()
	return nil
}

func searchTemplates(cmd *cobra.Command, args []string) error {
	query := args[0]

	tmplList, err := templates.SearchTemplates(query)
	if err != nil {
		return fmt.Errorf("failed to search templates: %w", err)
	}

	if len(tmplList) == 0 {
		fmt.Printf("No templates found matching: %s\n", query)
		return nil
	}

	fmt.Printf("\n🔍 Templates matching '%s':\n", query)
	fmt.Println("─────────────────────────────────────────────────────────")

	for _, tmpl := range tmplList {
		fmt.Printf("  %-30s %s\n", tmpl.Name, tmpl.Description)
	}
	fmt.Println()

	return nil
}

func deployTemplate(cmd *cobra.Command, args []string) error {
	templateName := args[0]
	apiClient := client.NewClient(apiURL, apiKey)

	fmt.Printf("📋 Loading template: %s\n", templateName)

	template, err := templates.LoadTemplate(templateName)
	if err != nil {
		return fmt.Errorf("failed to load template: %w", err)
	}

	agentConfig := template.ToAgentConfig()
	agentConfig.Name = deployName

	fmt.Printf("🚀 Creating agent: %s\n", deployName)
	agent, err := apiClient.CreateAgent(agentConfig)
	if err != nil {
		return fmt.Errorf("failed to create agent: %w", err)
	}

	fmt.Printf("✅ Agent deployed successfully!\n")
	fmt.Printf("ID: %s\n", agent.ID)
	fmt.Printf("Name: %s\n", agent.Name)
	return nil
}

func saveTemplate(cmd *cobra.Command, args []string) error {
	agentID := args[0]
	apiClient := client.NewClient(apiURL, apiKey)

	fmt.Printf("📥 Fetching agent: %s\n", agentID)

	agent, err := apiClient.GetAgent(agentID)
	if err != nil {
		return fmt.Errorf("failed to get agent: %w", err)
	}

	fmt.Print("Template name: ")
	var templateName string
	fmt.Scanln(&templateName)

	if templateName == "" {
		return fmt.Errorf("template name is required")
	}

	template := templates.AgentToTemplate(agent, templateName)

	if err := templates.SaveTemplate(template); err != nil {
		return fmt.Errorf("failed to save template: %w", err)
	}

	fmt.Printf("✅ Template saved: %s\n", templateName)
	return nil
}
