/*-------------------------------------------------------------------------
 *
 * wizard.go
 *    Interactive wizard for agent creation
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/cli/pkg/wizard/wizard.go
 *
 *-------------------------------------------------------------------------
 */

package wizard

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/neurondb/NeuronAgent/cli/pkg/client"
	"github.com/neurondb/NeuronAgent/cli/pkg/config"
)

func RunWizard(apiClient *client.Client) error {
	fmt.Println("\n✨ NeuronAgent Creation Wizard")
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println()

	agentConfig := &config.AgentConfig{}

	// Step 1: Basic Information
	if err := stepBasicInfo(agentConfig); err != nil {
		return err
	}

	// Step 2: Agent Profile
	if err := stepProfile(agentConfig); err != nil {
		return err
	}

	// Step 3: Tools Selection
	if err := stepTools(agentConfig); err != nil {
		return err
	}

	// Step 4: Workflow Configuration
	if err := stepWorkflow(agentConfig); err != nil {
		return err
	}

	// Step 5: Memory Settings
	if err := stepMemory(agentConfig); err != nil {
		return err
	}

	// Step 6: Review & Create
	return stepReview(apiClient, agentConfig)
}

func stepBasicInfo(cfg *config.AgentConfig) error {
	fmt.Println("📝 Step 1: Basic Information")
	fmt.Println("───────────────────────────────────────────────────────────")

	reader := bufio.NewReader(os.Stdin)

	fmt.Print("Agent name: ")
	name, _ := reader.ReadString('\n')
	cfg.Name = strings.TrimSpace(name)
	if cfg.Name == "" {
		return fmt.Errorf("agent name is required")
	}

	fmt.Print("Description (optional): ")
	desc, _ := reader.ReadString('\n')
	cfg.Description = strings.TrimSpace(desc)

	fmt.Print("System prompt (optional): ")
	prompt, _ := reader.ReadString('\n')
	cfg.SystemPrompt = strings.TrimSpace(prompt)

	fmt.Println()
	return nil
}

func stepProfile(cfg *config.AgentConfig) error {
	fmt.Println("🎭 Step 2: Agent Profile")
	fmt.Println("───────────────────────────────────────────────────────────")
	fmt.Println("Select a profile:")
	fmt.Println("  1. General Assistant")
	fmt.Println("  2. Research Assistant")
	fmt.Println("  3. Data Analyst")
	fmt.Println("  4. Workflow Agent")
	fmt.Println("  5. Custom (skip)")
	fmt.Print("\nChoice [1-5]: ")

	reader := bufio.NewReader(os.Stdin)
	choice, _ := reader.ReadString('\n')
	choice = strings.TrimSpace(choice)

	profiles := map[string]string{
		"1": "general-assistant",
		"2": "research-assistant",
		"3": "data-analyst",
		"4": "workflow-agent",
		"5": "",
	}

	if profile, ok := profiles[choice]; ok {
		cfg.Profile = profile
	}

	fmt.Println()
	return nil
}

func stepTools(cfg *config.AgentConfig) error {
	fmt.Println("🔧 Step 3: Tools Selection")
	fmt.Println("───────────────────────────────────────────────────────────")
	fmt.Println("Available tools:")
	tools := []string{"sql", "http", "code", "shell", "browser"}
	
	for i, tool := range tools {
		fmt.Printf("  %d. %s\n", i+1, strings.ToUpper(tool))
	}
	fmt.Print("\nSelect tools (comma-separated numbers, e.g., 1,2,3): ")

	reader := bufio.NewReader(os.Stdin)
	input, _ := reader.ReadString('\n')
	input = strings.TrimSpace(input)

	if input != "" {
		selected := []string{}
		parts := strings.Split(input, ",")
		for _, part := range parts {
			idx, err := strconv.Atoi(strings.TrimSpace(part))
			if err == nil && idx >= 1 && idx <= len(tools) {
				selected = append(selected, tools[idx-1])
			}
		}
		cfg.Tools = selected
	}

	fmt.Println()
	return nil
}

func stepWorkflow(cfg *config.AgentConfig) error {
	fmt.Println("⚙️  Step 4: Workflow Configuration")
	fmt.Println("───────────────────────────────────────────────────────────")
	fmt.Print("Create a workflow? (y/n): ")

	reader := bufio.NewReader(os.Stdin)
	response, _ := reader.ReadString('\n')
	response = strings.TrimSpace(strings.ToLower(response))

	if response == "y" || response == "yes" {
		fmt.Println("Workflow creation via wizard not yet fully implemented")
		fmt.Println("You can add a workflow later using --workflow flag")
	}

	fmt.Println()
	return nil
}

func stepMemory(cfg *config.AgentConfig) error {
	fmt.Println("🧠 Step 5: Memory Settings")
	fmt.Println("───────────────────────────────────────────────────────────")

	reader := bufio.NewReader(os.Stdin)

	fmt.Print("Enable long-term memory? (y/n) [n]: ")
	response, _ := reader.ReadString('\n')
	response = strings.TrimSpace(strings.ToLower(response))
	cfg.Memory.Enabled = response == "y" || response == "yes"

	if cfg.Memory.Enabled {
		fmt.Print("Enable hierarchical memory? (y/n) [n]: ")
		response, _ := reader.ReadString('\n')
		response = strings.TrimSpace(strings.ToLower(response))
		cfg.Memory.Hierarchical = response == "y" || response == "yes"
	}

	fmt.Println()
	return nil
}

func stepReview(apiClient *client.Client, cfg *config.AgentConfig) error {
	fmt.Println("📋 Step 6: Review & Create")
	fmt.Println("───────────────────────────────────────────────────────────")
	fmt.Printf("Name: %s\n", cfg.Name)
	if cfg.Description != "" {
		fmt.Printf("Description: %s\n", cfg.Description)
	}
	if cfg.Profile != "" {
		fmt.Printf("Profile: %s\n", cfg.Profile)
	}
	if len(cfg.Tools) > 0 {
		fmt.Printf("Tools: %v\n", cfg.Tools)
	}
	if cfg.Memory.Enabled {
		fmt.Printf("Memory: Enabled (Hierarchical: %v)\n", cfg.Memory.Hierarchical)
	}
	fmt.Println()

	fmt.Print("Create agent? (y/n): ")
	reader := bufio.NewReader(os.Stdin)
	response, _ := reader.ReadString('\n')
	response = strings.TrimSpace(strings.ToLower(response))

	if response != "y" && response != "yes" {
		fmt.Println("Cancelled")
		return nil
	}

	fmt.Println("\n🚀 Creating agent...")
	agent, err := apiClient.CreateAgent(cfg)
	if err != nil {
		return fmt.Errorf("failed to create agent: %w", err)
	}

	fmt.Println("\n✅ Agent created successfully!")
	fmt.Printf("ID: %s\n", agent.ID)
	fmt.Printf("Name: %s\n", agent.Name)
	return nil
}



