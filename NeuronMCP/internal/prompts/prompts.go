/*-------------------------------------------------------------------------
 *
 * prompts.go
 *    Prompt management for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/prompts/prompts.go
 *
 *-------------------------------------------------------------------------
 */

package prompts

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/jackc/pgx/v5"
	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

/* Prompt represents a prompt template */
type Prompt struct {
	ID          int                    `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Template    string                 `json:"template"`
	Variables   []VariableDefinition   `json:"variables"`
	Category    *string                `json:"category,omitempty"`
	Tags        []string               `json:"tags,omitempty"`
	IsDefault   bool                   `json:"isDefault"`
	CreatedAt   string                 `json:"createdAt"`
	UpdatedAt   string                 `json:"updatedAt"`
	CreatedBy   string                 `json:"createdBy"`
}

/* VariableDefinition defines a prompt variable */
type VariableDefinition struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Required    bool   `json:"required"`
	Default     string `json:"default,omitempty"`
}

/* Manager manages prompts */
type Manager struct {
	db     *database.Database
	logger *logging.Logger
}

/* NewManager creates a new prompt manager */
func NewManager(db *database.Database, logger *logging.Logger) *Manager {
	return &Manager{
		db:     db,
		logger: logger,
	}
}

/* ListPrompts lists all available prompts */
func (m *Manager) ListPrompts(ctx context.Context) ([]Prompt, error) {
	if m == nil {
		return nil, fmt.Errorf("prompts: manager instance is nil")
	}
	if m.db == nil {
		return nil, fmt.Errorf("prompts: database instance is not initialized")
	}
	if ctx == nil {
		return nil, fmt.Errorf("prompts: context cannot be nil")
	}

	query := `
		SELECT 
			prompt_id,
			prompt_name,
			description,
			template,
			variables,
			category,
			tags,
			is_default,
			created_at,
			updated_at,
			created_by
		FROM neurondb.prompts
		ORDER BY prompt_name
	`

	rows, err := m.db.Query(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("prompts: failed to query prompts: error=%w", err)
	}
	defer rows.Close()

	var prompts []Prompt
	for rows.Next() {
		var p Prompt
		var variablesJSON []byte
		var category *string
		var tags []string

		err := rows.Scan(
			&p.ID,
			&p.Name,
			&p.Description,
			&p.Template,
			&variablesJSON,
			&category,
			&tags,
			&p.IsDefault,
			&p.CreatedAt,
			&p.UpdatedAt,
			&p.CreatedBy,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan prompt: %w", err)
		}

		if len(variablesJSON) > 0 {
			if err := json.Unmarshal(variablesJSON, &p.Variables); err != nil {
				if m.logger != nil {
					m.logger.Warn("Failed to unmarshal variables", map[string]interface{}{
						"prompt_id":       p.ID,
						"prompt_name":     p.Name,
						"variables_json": string(variablesJSON),
						"error":           err.Error(),
					})
				}
				/* Continue with empty variables rather than failing */
				p.Variables = []VariableDefinition{}
			}
		}

		p.Category = category
		p.Tags = tags

		prompts = append(prompts, p)
	}

	return prompts, rows.Err()
}

/* GetPrompt retrieves a prompt by name */
func (m *Manager) GetPrompt(ctx context.Context, name string) (*Prompt, error) {
	if m == nil {
		return nil, fmt.Errorf("prompts: manager instance is nil")
	}
	if m.db == nil {
		return nil, fmt.Errorf("prompts: database instance is not initialized")
	}
	if ctx == nil {
		return nil, fmt.Errorf("prompts: context cannot be nil")
	}
	if name == "" {
		return nil, fmt.Errorf("prompts: prompt name cannot be empty")
	}

	/* Validate prompt name format and length */
	if len(name) > 200 {
		return nil, fmt.Errorf("prompts: prompt name too long: name_length=%d, max_length=200, name='%s'", len(name), name[:min(50, len(name))])
	}

	/* Validate prompt name doesn't contain control characters */
	for _, char := range name {
		if char < 32 && char != 9 && char != 10 && char != 13 { /* Allow tab, newline, carriage return */
			return nil, fmt.Errorf("prompts: prompt name contains invalid control characters: name='%s'", name)
		}
	}

	query := `
		SELECT 
			prompt_id,
			prompt_name,
			description,
			template,
			variables,
			category,
			tags,
			is_default,
			created_at,
			updated_at,
			created_by
		FROM neurondb.prompts
		WHERE prompt_name = $1
	`

	var p Prompt
	var variablesJSON []byte
	var category *string
	var tags []string

	err := m.db.QueryRow(ctx, query, name).Scan(
		&p.ID,
		&p.Name,
		&p.Description,
		&p.Template,
		&variablesJSON,
		&category,
		&tags,
		&p.IsDefault,
		&p.CreatedAt,
		&p.UpdatedAt,
		&p.CreatedBy,
	)
	if err != nil {
		if err == pgx.ErrNoRows {
			return nil, fmt.Errorf("prompts: prompt not found: prompt_name='%s'", name)
		}
		return nil, fmt.Errorf("prompts: failed to get prompt: prompt_name='%s', error=%w", name, err)
	}

	if len(variablesJSON) > 0 {
		if err := json.Unmarshal(variablesJSON, &p.Variables); err != nil {
			if m.logger != nil {
				m.logger.Warn("Failed to unmarshal variables", map[string]interface{}{
					"prompt_id":       p.ID,
					"prompt_name":     p.Name,
					"variables_json":  string(variablesJSON),
					"error":           err.Error(),
				})
			}
			/* Continue with empty variables rather than failing */
			p.Variables = []VariableDefinition{}
		}
	}

	p.Category = category
	p.Tags = tags

	return &p, nil
}

/* RenderPrompt renders a prompt template with provided variables */
func (m *Manager) RenderPrompt(ctx context.Context, name string, variables map[string]string) (string, error) {
	if m == nil {
		return "", fmt.Errorf("prompts: manager instance is nil")
	}
	if m.db == nil {
		return "", fmt.Errorf("prompts: database instance is not initialized")
	}
	if ctx == nil {
		return "", fmt.Errorf("prompts: context cannot be nil")
	}
	if name == "" {
		return "", fmt.Errorf("prompts: prompt name cannot be empty")
	}

	/* Validate variables map */
	if variables == nil {
		variables = make(map[string]string)
	}

	prompt, err := m.GetPrompt(ctx, name)
	if err != nil {
		return "", fmt.Errorf("prompts: failed to get prompt: prompt_name='%s', error=%w", name, err)
	}

	if prompt == nil {
		return "", fmt.Errorf("prompts: GetPrompt returned nil: prompt_name='%s'", name)
	}

	if prompt.Template == "" {
		return "", fmt.Errorf("prompts: prompt template is empty: prompt_name='%s', prompt_id=%d", name, prompt.ID)
	}

	rendered, err := RenderTemplate(prompt.Template, prompt.Variables, variables)
	if err != nil {
		return "", fmt.Errorf("prompts: failed to render template: prompt_name='%s', prompt_id=%d, variable_count=%d, error=%w", name, prompt.ID, len(variables), err)
	}

	return rendered, nil
}

