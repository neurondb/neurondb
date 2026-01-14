/*-------------------------------------------------------------------------
 *
 * workflow.go
 *    Workflow management commands for neuronagent-cli
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/cli/cmd/workflow.go
 *
 *-------------------------------------------------------------------------
 */

package cmd

import (
	"encoding/json"
	"fmt"

	"github.com/neurondb/NeuronAgent/cli/pkg/client"
	"github.com/neurondb/NeuronAgent/cli/pkg/config"
	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"
)

var workflowCreateCmd = &cobra.Command{
	Use:   "create [file]",
	Short: "Create workflow from YAML file",
	Args:  cobra.ExactArgs(1),
	RunE:  runCreateWorkflow,
}

var workflowListCmd = &cobra.Command{
	Use:   "list",
	Short: "List workflows",
	RunE:  listWorkflows,
}

var workflowShowCmd = &cobra.Command{
	Use:   "show [id]",
	Short: "Show workflow details",
	Args:  cobra.ExactArgs(1),
	RunE:  showWorkflow,
}

var workflowValidateCmd = &cobra.Command{
	Use:   "validate [file]",
	Short: "Validate workflow definition",
	Args:  cobra.ExactArgs(1),
	RunE:  validateWorkflow,
}

var workflowExportCmd = &cobra.Command{
	Use:   "export [id]",
	Short: "Export workflow to YAML",
	Args:  cobra.ExactArgs(1),
	RunE:  exportWorkflow,
}

var workflowTemplatesCmd = &cobra.Command{
	Use:   "templates",
	Short: "List workflow templates",
	RunE:  listWorkflowTemplates,
}

var workflowCmd = &cobra.Command{
	Use:   "workflow",
	Short: "Manage workflows",
	Long:  "Create, list, validate, and export workflows",
}

func init() {
	workflowCmd.AddCommand(workflowCreateCmd)
	workflowCmd.AddCommand(workflowListCmd)
	workflowCmd.AddCommand(workflowShowCmd)
	workflowCmd.AddCommand(workflowValidateCmd)
	workflowCmd.AddCommand(workflowExportCmd)
	workflowCmd.AddCommand(workflowTemplatesCmd)
}

func runCreateWorkflow(cmd *cobra.Command, args []string) error {
	filePath := args[0]

	if apiKey == "" {
		return fmt.Errorf("API key is required. Set NEURONAGENT_API_KEY environment variable or use --key flag")
	}

	fmt.Printf("📄 Loading workflow from: %s\n", filePath)

	workflowConfig, err := config.LoadWorkflow(filePath)
	if err != nil {
		return fmt.Errorf("failed to load workflow: %w", err)
	}

	fmt.Printf("✅ Workflow loaded: %s\n", workflowConfig.Name)
	fmt.Printf("Steps: %d\n", len(workflowConfig.Steps))

	/* Convert workflow config to API format */
	dagDefinition := map[string]interface{}{
		"name":        workflowConfig.Name,
		"description": workflowConfig.Description,
		"type":        workflowConfig.Type,
		"steps":       workflowConfig.Steps,
		"triggers":    workflowConfig.Triggers,
	}

	reqBody := map[string]interface{}{
		"name":           workflowConfig.Name,
		"dag_definition": dagDefinition,
		"status":         "active",
	}

	/* Create workflow via API */
	apiClient := client.NewClient(apiURL, apiKey)
	workflow, err := apiClient.CreateWorkflow(reqBody)
	if err != nil {
		return fmt.Errorf("failed to create workflow: %w", err)
	}

	if outputFormat == "json" {
		jsonData, _ := json.MarshalIndent(workflow, "", "  ")
		fmt.Println(string(jsonData))
	} else {
		fmt.Printf("✅ Workflow created successfully!\n")
		fmt.Printf("ID: %s\n", workflow.ID)
		fmt.Printf("Name: %s\n", workflow.Name)
		fmt.Printf("Status: %s\n", workflow.Status)
	}

	return nil
}

func listWorkflows(cmd *cobra.Command, args []string) error {
	if apiKey == "" {
		return fmt.Errorf("API key is required. Set NEURONAGENT_API_KEY environment variable or use --key flag")
	}

	apiClient := client.NewClient(apiURL, apiKey)
	workflows, err := apiClient.ListWorkflows(100, 0)
	if err != nil {
		return fmt.Errorf("failed to list workflows: %w", err)
	}

	if outputFormat == "json" {
		jsonData, _ := json.MarshalIndent(workflows, "", "  ")
		fmt.Println(string(jsonData))
	} else {
		if len(workflows) == 0 {
			fmt.Println("No workflows found")
			return nil
		}

		fmt.Println("\n📋 Workflows:")
		fmt.Println("─────────────────────────────────────────────────────────")
		for _, wf := range workflows {
			fmt.Printf("  %-36s %-30s %s\n", wf.ID, wf.Name, wf.Status)
		}
		fmt.Println()
	}

	return nil
}

func showWorkflow(cmd *cobra.Command, args []string) error {
	workflowID := args[0]

	if apiKey == "" {
		return fmt.Errorf("API key is required. Set NEURONAGENT_API_KEY environment variable or use --key flag")
	}

	apiClient := client.NewClient(apiURL, apiKey)
	workflow, err := apiClient.GetWorkflow(workflowID)
	if err != nil {
		return fmt.Errorf("failed to get workflow: %w", err)
	}

	if outputFormat == "json" {
		jsonData, _ := json.MarshalIndent(workflow, "", "  ")
		fmt.Println(string(jsonData))
	} else {
		fmt.Printf("\n📋 Workflow Details:\n")
		fmt.Println("─────────────────────────────────────────────────────────")
		fmt.Printf("ID:          %s\n", workflow.ID)
		fmt.Printf("Name:        %s\n", workflow.Name)
		fmt.Printf("Status:      %s\n", workflow.Status)
		if workflow.CreatedAt != "" {
			fmt.Printf("Created At:  %s\n", workflow.CreatedAt)
		}
		if workflow.UpdatedAt != "" {
			fmt.Printf("Updated At:  %s\n", workflow.UpdatedAt)
		}
		if workflow.DAGDefinition != nil {
			fmt.Printf("\nDAG Definition:\n")
			jsonData, _ := json.MarshalIndent(workflow.DAGDefinition, "  ", "  ")
			fmt.Println(string(jsonData))
		}
		fmt.Println()
	}

	return nil
}

func validateWorkflow(cmd *cobra.Command, args []string) error {
	filePath := args[0]

	fmt.Printf("🔍 Validating workflow: %s\n", filePath)

	workflow, err := config.LoadWorkflow(filePath)
	if err != nil {
		return fmt.Errorf("failed to load workflow: %w", err)
	}

	if err := config.ValidateWorkflow(workflow); err != nil {
		return fmt.Errorf("workflow validation failed: %w", err)
	}

	fmt.Println("✅ Workflow is valid!")
	fmt.Printf("Name: %s\n", workflow.Name)
	fmt.Printf("Steps: %d\n", len(workflow.Steps))

	return nil
}

func exportWorkflow(cmd *cobra.Command, args []string) error {
	workflowID := args[0]

	if apiKey == "" {
		return fmt.Errorf("API key is required. Set NEURONAGENT_API_KEY environment variable or use --key flag")
	}

	apiClient := client.NewClient(apiURL, apiKey)
	workflow, err := apiClient.GetWorkflow(workflowID)
	if err != nil {
		return fmt.Errorf("failed to get workflow: %w", err)
	}

	/* Convert workflow to YAML format */
	var workflowConfig config.WorkflowConfig

	/* DAGDefinition is already map[string]interface{} */
	if workflow.DAGDefinition != nil {
		dagDef := workflow.DAGDefinition
		if name, ok := dagDef["name"].(string); ok {
			workflowConfig.Name = name
		}
		if desc, ok := dagDef["description"].(string); ok {
			workflowConfig.Description = desc
		}
		if typ, ok := dagDef["type"].(string); ok {
			workflowConfig.Type = typ
		}

		/* Convert steps */
		if steps, ok := dagDef["steps"].([]interface{}); ok {
			for _, stepData := range steps {
				if stepMap, ok := stepData.(map[string]interface{}); ok {
					step := config.WorkflowStep{}
					if id, ok := stepMap["id"].(string); ok {
						step.ID = id
					}
					if name, ok := stepMap["name"].(string); ok {
						step.Name = name
					}
					if typ, ok := stepMap["type"].(string); ok {
						step.Type = typ
					}
					if deps, ok := stepMap["depends_on"].([]interface{}); ok {
						step.DependsOn = make([]string, len(deps))
						for i, dep := range deps {
							if depStr, ok := dep.(string); ok {
								step.DependsOn[i] = depStr
							}
						}
					}
					if cfg, ok := stepMap["config"].(map[string]interface{}); ok {
						step.Config = cfg
					}
					workflowConfig.Steps = append(workflowConfig.Steps, step)
				}
			}
		}
	}

	/* Output YAML */
	yamlData, err := yaml.Marshal(&workflowConfig)
	if err != nil {
		return fmt.Errorf("failed to marshal workflow to YAML: %w", err)
	}

	fmt.Print(string(yamlData))

	return nil
}

func listWorkflowTemplates(cmd *cobra.Command, args []string) error {
	templates := config.GetWorkflowTemplates()

	if len(templates) == 0 {
		fmt.Println("No workflow templates available")
		return nil
	}

	fmt.Println("\n📋 Workflow Templates:")
	fmt.Println("─────────────────────────────────────────────────────────")

	for _, tmpl := range templates {
		fmt.Printf("  %-30s %s\n", tmpl, "Pre-built workflow template")
	}
	fmt.Println()

	return nil
}
