/*-------------------------------------------------------------------------
 *
 * config.go
 *    Configuration file handling for agent and workflow definitions
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/cli/pkg/config/config.go
 *
 *-------------------------------------------------------------------------
 */

package config

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

type AgentConfig struct {
	Name         string                 `yaml:"name" json:"name"`
	Description  string                 `yaml:"description,omitempty" json:"description,omitempty"`
	Profile      string                 `yaml:"profile,omitempty" json:"profile,omitempty"`
	SystemPrompt string                 `yaml:"system_prompt,omitempty" json:"system_prompt,omitempty"`
	Model        ModelConfig            `yaml:"model,omitempty" json:"model,omitempty"`
	Tools        []string               `yaml:"tools,omitempty" json:"tools,omitempty"`
	Config       map[string]interface{} `yaml:"config,omitempty" json:"config,omitempty"`
	Memory       MemoryConfig           `yaml:"memory,omitempty" json:"memory,omitempty"`
	Workflow     *WorkflowConfig        `yaml:"workflow,omitempty" json:"workflow,omitempty"`
}

type ModelConfig struct {
	Name        string                 `yaml:"name" json:"name"`
	Temperature float64                `yaml:"temperature,omitempty" json:"temperature,omitempty"`
	MaxTokens   int                    `yaml:"max_tokens,omitempty" json:"max_tokens,omitempty"`
	TopP        float64                `yaml:"top_p,omitempty" json:"top_p,omitempty"`
	Config      map[string]interface{} `yaml:"config,omitempty" json:"config,omitempty"`
}

type MemoryConfig struct {
	Enabled         bool   `yaml:"enabled,omitempty" json:"enabled,omitempty"`
	Hierarchical    bool   `yaml:"hierarchical,omitempty" json:"hierarchical,omitempty"`
	RetentionDays   int    `yaml:"retention_days,omitempty" json:"retention_days,omitempty"`
	VectorDimension int    `yaml:"vector_dimension,omitempty" json:"vector_dimension,omitempty"`
}

type WorkflowConfig struct {
	Name        string          `yaml:"name" json:"name"`
	Description string          `yaml:"description,omitempty" json:"description,omitempty"`
	Type        string          `yaml:"type" json:"type"`
	Steps       []WorkflowStep  `yaml:"steps" json:"steps"`
	Triggers    []WorkflowTrigger `yaml:"triggers,omitempty" json:"triggers,omitempty"`
}

type WorkflowStep struct {
	ID          string                 `yaml:"id" json:"id"`
	Name        string                 `yaml:"name" json:"name"`
	Type        string                 `yaml:"type" json:"type"`
	DependsOn   []string               `yaml:"depends_on,omitempty" json:"depends_on,omitempty"`
	Config      map[string]interface{} `yaml:"config,omitempty" json:"config,omitempty"`
	OnError     string                 `yaml:"on_error,omitempty" json:"on_error,omitempty"`
	RetryConfig *RetryConfig           `yaml:"retry_config,omitempty" json:"retry_config,omitempty"`
}

type RetryConfig struct {
	MaxAttempts int    `yaml:"max_attempts" json:"max_attempts"`
	Backoff     string `yaml:"backoff" json:"backoff"`
}

type WorkflowTrigger struct {
	Type    string                 `yaml:"type" json:"type"`
	Cron    string                 `yaml:"cron,omitempty" json:"cron,omitempty"`
	Path    string                 `yaml:"path,omitempty" json:"path,omitempty"`
	Config  map[string]interface{} `yaml:"config,omitempty" json:"config,omitempty"`
}

func LoadAgentConfig(path string) (*AgentConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	var config AgentConfig
	if err := yaml.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse YAML: %w", err)
	}

	if config.Name == "" {
		return nil, fmt.Errorf("agent name is required")
	}

	return &config, nil
}

func LoadWorkflow(path string) (*WorkflowConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	var workflow WorkflowConfig
	if err := yaml.Unmarshal(data, &workflow); err != nil {
		return nil, fmt.Errorf("failed to parse YAML: %w", err)
	}

	if err := ValidateWorkflow(&workflow); err != nil {
		return nil, err
	}

	return &workflow, nil
}

func ValidateWorkflow(workflow *WorkflowConfig) error {
	if workflow.Name == "" {
		return fmt.Errorf("workflow name is required")
	}

	if len(workflow.Steps) == 0 {
		return fmt.Errorf("workflow must have at least one step")
	}

	// Check for cycles in dependencies
	stepMap := make(map[string]bool)
	for _, step := range workflow.Steps {
		if step.ID == "" {
			return fmt.Errorf("step ID is required")
		}
		if stepMap[step.ID] {
			return fmt.Errorf("duplicate step ID: %s", step.ID)
		}
		stepMap[step.ID] = true
	}

	// Validate dependencies
	for _, step := range workflow.Steps {
		for _, dep := range step.DependsOn {
			if !stepMap[dep] {
				return fmt.Errorf("step %s depends on unknown step: %s", step.ID, dep)
			}
		}
	}

	// Simple cycle detection (DFS)
	visited := make(map[string]bool)
	recStack := make(map[string]bool)

	var dfs func(stepID string) bool
	dfs = func(stepID string) bool {
		visited[stepID] = true
		recStack[stepID] = true

		var step *WorkflowStep
		for i := range workflow.Steps {
			if workflow.Steps[i].ID == stepID {
				step = &workflow.Steps[i]
				break
			}
		}

		if step != nil {
			for _, dep := range step.DependsOn {
				if !visited[dep] {
					if dfs(dep) {
						return true
					}
				} else if recStack[dep] {
					return true
				}
			}
		}

		recStack[stepID] = false
		return false
	}

	for _, step := range workflow.Steps {
		if !visited[step.ID] {
			if dfs(step.ID) {
				return fmt.Errorf("workflow contains a cycle in dependencies")
			}
		}
	}

	return nil
}

func GetWorkflowTemplates() []string {
	return []string{
		"data-pipeline",
		"customer-support",
		"document-qa",
		"code-reviewer",
		"research-assistant",
		"report-generator",
	}
}



