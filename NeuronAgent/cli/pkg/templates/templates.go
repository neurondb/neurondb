/*-------------------------------------------------------------------------
 *
 * templates.go
 *    Template management for agent creation
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/cli/pkg/templates/templates.go
 *
 *-------------------------------------------------------------------------
 */

package templates

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/neurondb/NeuronAgent/cli/pkg/client"
	"github.com/neurondb/NeuronAgent/cli/pkg/config"
	"gopkg.in/yaml.v3"
)

type Template struct {
	Name         string                 `yaml:"name" json:"name"`
	Description  string                 `yaml:"description" json:"description"`
	Category     string                 `yaml:"category" json:"category"`
	Profile      string                 `yaml:"profile,omitempty" json:"profile,omitempty"`
	SystemPrompt string                 `yaml:"system_prompt,omitempty" json:"system_prompt,omitempty"`
	Model        config.ModelConfig     `yaml:"model,omitempty" json:"model,omitempty"`
	Tools        []string               `yaml:"tools,omitempty" json:"tools,omitempty"`
	Config       map[string]interface{} `yaml:"config,omitempty" json:"config,omitempty"`
	Workflow     *config.WorkflowConfig `yaml:"workflow,omitempty" json:"workflow,omitempty"`
}

func getTemplatesDir() string {
	/* Check if templates directory exists next to CLI binary */
	if exe, err := os.Executable(); err == nil {
		dir := filepath.Dir(exe)
		templatesDir := filepath.Join(dir, "templates")
		if _, err := os.Stat(templatesDir); err == nil {
			return templatesDir
		}
	}

	/* Fallback to cli/templates in source */
	cliDir := filepath.Join("cli", "templates")
	if _, err := os.Stat(cliDir); err == nil {
		return cliDir
	}

	return "templates"
}

func ListTemplates() ([]Template, error) {
	templatesDir := getTemplatesDir()

	var templates []Template

	err := filepath.Walk(templatesDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if info.IsDir() || !strings.HasSuffix(path, ".yaml") && !strings.HasSuffix(path, ".yml") {
			return nil
		}

		tmpl, err := loadTemplateFile(path)
		if err != nil {
			return err
		}

		templates = append(templates, *tmpl)
		return nil
	})

	if err != nil && !os.IsNotExist(err) {
		return nil, fmt.Errorf("failed to list templates: %w", err)
	}

	return templates, nil
}

func LoadTemplate(name string) (*Template, error) {
	templatesDir := getTemplatesDir()

	/* Try different paths */
	paths := []string{
		filepath.Join(templatesDir, name+".yaml"),
		filepath.Join(templatesDir, name+".yml"),
		filepath.Join(templatesDir, "agents", name+".yaml"),
		filepath.Join(templatesDir, "agents", name+".yml"),
	}

	for _, path := range paths {
		if tmpl, err := loadTemplateFile(path); err == nil {
			return tmpl, nil
		}
	}

	return nil, fmt.Errorf("template not found: %s", name)
}

func loadTemplateFile(path string) (*Template, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var tmpl Template
	if err := yaml.Unmarshal(data, &tmpl); err != nil {
		return nil, fmt.Errorf("failed to parse template: %w", err)
	}

	return &tmpl, nil
}

func SearchTemplates(query string) ([]Template, error) {
	all, err := ListTemplates()
	if err != nil {
		return nil, err
	}

	query = strings.ToLower(query)
	var results []Template

	for _, tmpl := range all {
		if strings.Contains(strings.ToLower(tmpl.Name), query) ||
			strings.Contains(strings.ToLower(tmpl.Description), query) ||
			strings.Contains(strings.ToLower(tmpl.Category), query) {
			results = append(results, tmpl)
		}
	}

	return results, nil
}

func (t *Template) ToAgentConfig() *config.AgentConfig {
	return &config.AgentConfig{
		Name:         t.Name,
		Description:  t.Description,
		Profile:      t.Profile,
		SystemPrompt: t.SystemPrompt,
		Model:        t.Model,
		Tools:        t.Tools,
		Config:       t.Config,
		Workflow:     t.Workflow,
	}
}

func AgentToTemplate(agent *client.Agent, templateName string) *Template {
	tmpl := &Template{
		Name:         templateName,
		Description:  agent.Description,
		Profile:      "", // Would need to be determined from agent config
		SystemPrompt: agent.SystemPrompt,
		Model: config.ModelConfig{
			Name: agent.ModelName,
		},
		Tools:  agent.EnabledTools,
		Config: agent.Config,
	}

	// Extract config values if present
	if agent.Config != nil {
		if temp, ok := agent.Config["temperature"].(float64); ok {
			tmpl.Model.Temperature = temp
		}
		if maxTok, ok := agent.Config["max_tokens"].(float64); ok {
			tmpl.Model.MaxTokens = int(maxTok)
		}
	}

	return tmpl
}

func SaveTemplate(tmpl *Template) error {
	templatesDir := getTemplatesDir()

	// Create directory if it doesn't exist
	if err := os.MkdirAll(templatesDir, 0755); err != nil {
		return fmt.Errorf("failed to create templates directory: %w", err)
	}

	path := filepath.Join(templatesDir, tmpl.Name+".yaml")

	data, err := yaml.Marshal(tmpl)
	if err != nil {
		return fmt.Errorf("failed to marshal template: %w", err)
	}

	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("failed to write template file: %w", err)
	}

	return nil
}
