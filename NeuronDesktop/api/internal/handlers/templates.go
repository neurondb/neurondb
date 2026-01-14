/*-------------------------------------------------------------------------
 *
 * templates.go
 *    Template management handlers for NeuronDesktop API
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronDesktop/api/internal/handlers/templates.go
 *
 *-------------------------------------------------------------------------
 */

package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronDesktop/api/internal/agent"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
	"github.com/neurondb/NeuronDesktop/api/internal/logging"
	"gopkg.in/yaml.v3"
)

/* TemplateHandlers handles agent template endpoints */
type TemplateHandlers struct {
	queries *db.Queries
	logger  *logging.Logger
}

/* NewTemplateHandlers creates new template handlers */
func NewTemplateHandlers(queries *db.Queries, logger *logging.Logger) *TemplateHandlers {
	return &TemplateHandlers{
		queries: queries,
		logger:  logger,
	}
}

type Template struct {
	ID           string                 `json:"id"`
	Name         string                 `json:"name"`
	Description  string                 `json:"description"`
	Category     string                 `json:"category"`
	Configuration map[string]interface{} `json:"configuration"`
	Workflow     map[string]interface{} `json:"workflow,omitempty"`
}

/* ListTemplates lists all available agent templates */
func (h *TemplateHandlers) ListTemplates(w http.ResponseWriter, r *http.Request) {
	templates, err := h.loadTemplates()
	if err != nil {
		h.logger.Error("Failed to load templates", err, nil)
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	WriteSuccess(w, templates, http.StatusOK)
}

/* GetTemplate gets a specific template by ID */
func (h *TemplateHandlers) GetTemplate(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	templateID := vars["id"]

	template, err := h.loadTemplate(templateID)
	if err != nil {
		if os.IsNotExist(err) {
			WriteError(w, r, http.StatusNotFound, fmt.Errorf("template not found"), nil)
			return
		}
		h.logger.Error("Failed to load template", err, nil)
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	WriteSuccess(w, template, http.StatusOK)
}

/* DeployTemplate creates an agent from a template */
func (h *TemplateHandlers) DeployTemplate(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["profile_id"]
	templateID := vars["id"]

	var req struct {
		Name string `json:"name"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid request body"), nil)
		return
	}

	/* Load template */
	template, err := h.loadTemplate(templateID)
	if err != nil {
		if os.IsNotExist(err) {
			WriteError(w, r, http.StatusNotFound, fmt.Errorf("template not found"), nil)
			return
		}
		h.logger.Error("Failed to load template", err, nil)
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	/* Convert template to agent creation request */
	agentConfig := h.templateToAgentConfig(template, req.Name)

	/* Get profile to get agent endpoint */
	profile, err := h.queries.GetProfile(r.Context(), profileID)
	if err != nil {
		h.logger.Error("Failed to get profile", err, nil)
		WriteError(w, r, http.StatusNotFound, fmt.Errorf("profile not found"), nil)
		return
	}

	if profile.AgentEndpoint == "" {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("agent endpoint not configured for this profile"), nil)
		return
	}

	/* Create agent via NeuronAgent API */
	agentClient := agent.NewClient(profile.AgentEndpoint, profile.AgentAPIKey)
	createdAgent, err := agentClient.CreateAgent(r.Context(), agentConfig)
	if err != nil {
		h.logger.Error("Failed to create agent from template", err, nil)
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	WriteSuccess(w, createdAgent, http.StatusCreated)
}

func (h *TemplateHandlers) loadTemplates() ([]Template, error) {
	templateDir := h.getTemplateDir()
	var templates []Template

	err := filepath.Walk(templateDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if info.IsDir() || (!strings.HasSuffix(path, ".yaml") && !strings.HasSuffix(path, ".yml")) {
			return nil
		}

		template, err := h.loadTemplateFromFile(path)
		if err != nil {
			h.logger.Warn("Failed to load template file", map[string]interface{}{"path": path, "error": err.Error()})
			return nil // Continue with other files
		}

		templates = append(templates, *template)
		return nil
	})

	if err != nil && !os.IsNotExist(err) {
		return nil, err
	}

	return templates, nil
}

func (h *TemplateHandlers) loadTemplate(templateID string) (*Template, error) {
	templateDir := h.getTemplateDir()
	
	/* Try different paths */
	paths := []string{
		filepath.Join(templateDir, templateID+".yaml"),
		filepath.Join(templateDir, templateID+".yml"),
		filepath.Join(templateDir, "agents", templateID+".yaml"),
		filepath.Join(templateDir, "agents", templateID+".yml"),
	}

	for _, path := range paths {
		if template, err := h.loadTemplateFromFile(path); err == nil {
			return template, nil
		}
	}

	return nil, fmt.Errorf("template not found: %s", templateID)
}

func (h *TemplateHandlers) loadTemplateFromFile(path string) (*Template, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var templateData map[string]interface{}
	/* Try YAML first (templates are YAML) */
	if err := yaml.Unmarshal(data, &templateData); err != nil {
		/* Fallback to JSON */
		if err2 := json.Unmarshal(data, &templateData); err2 != nil {
			return nil, fmt.Errorf("failed to parse template (YAML: %v, JSON: %v)", err, err2)
		}
	}

	template := &Template{
		ID:          filepath.Base(path[:len(path)-len(filepath.Ext(path))]),
		Configuration: templateData,
	}

	if name, ok := templateData["name"].(string); ok {
		template.Name = name
	}
	if desc, ok := templateData["description"].(string); ok {
		template.Description = desc
	}
	if cat, ok := templateData["category"].(string); ok {
		template.Category = cat
	}
	if workflow, ok := templateData["workflow"].(map[string]interface{}); ok {
		template.Workflow = workflow
	}

	return template, nil
}

func (h *TemplateHandlers) getTemplateDir() string {
	/* Try multiple locations */
	possibleDirs := []string{
		"templates",
		"../../NeuronAgent/cli/templates",
		"/usr/local/share/neuronagent/templates",
	}

	for _, dir := range possibleDirs {
		if _, err := os.Stat(dir); err == nil {
			return dir
		}
	}

	/* Default */
	return "templates"
}

func (h *TemplateHandlers) templateToAgentConfig(template *Template, name string) agent.CreateAgentRequest {
	req := agent.CreateAgentRequest{
		Name: name,
	}

	/* Copy relevant fields from template configuration */
	tmplConfig := template.Configuration

	if desc, ok := tmplConfig["description"].(string); ok {
		req.Description = desc
	}
	if prompt, ok := tmplConfig["system_prompt"].(string); ok {
		req.SystemPrompt = prompt
	}
	if model, ok := tmplConfig["model"].(map[string]interface{}); ok {
		if modelName, ok := model["name"].(string); ok {
			req.ModelName = modelName
		}
	}
	if tools, ok := tmplConfig["tools"].([]interface{}); ok {
		toolStrings := make([]string, 0, len(tools))
		for _, t := range tools {
			if toolStr, ok := t.(string); ok {
				toolStrings = append(toolStrings, toolStr)
			}
		}
		req.EnabledTools = toolStrings
	}
	if cfg, ok := tmplConfig["config"].(map[string]interface{}); ok {
		req.Config = cfg
	}

	return req
}

