/*-------------------------------------------------------------------------
 *
 * prompt_handlers.go
 *    MCP prompt handlers
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/prompts/prompt_handlers.go
 *
 *-------------------------------------------------------------------------
 */

package prompts

import (
	"context"
	"encoding/json"
	"fmt"
)

/* ListPromptsRequest represents a prompts/list request */
type ListPromptsRequest struct {
	Method string `json:"method"`
}

/* GetPromptRequest represents a prompts/get request */
type GetPromptRequest struct {
	Name      string            `json:"name"`
	Variables map[string]string `json:"variables,omitempty"`
}

/* PromptDefinition represents a prompt in MCP format */
type PromptDefinition struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Arguments   []PromptArgument       `json:"arguments,omitempty"`
}

/* PromptArgument represents a prompt argument */
type PromptArgument struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Required    bool   `json:"required"`
}

/* ListPromptsResponse represents a prompts/list response */
type ListPromptsResponse struct {
	Prompts []PromptDefinition `json:"prompts"`
}

/* GetPromptResponse represents a prompts/get response */
type GetPromptResponse struct {
	Description string                 `json:"description,omitempty"`
	Messages    []PromptMessage        `json:"messages"`
}

/* PromptMessage represents a message in a prompt */
type PromptMessage struct {
	Role    string                 `json:"role"`
	Content PromptMessageContent   `json:"content"`
}

/* PromptMessageContent represents message content */
type PromptMessageContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

/* HandleListPrompts handles the prompts/list request */
func (m *Manager) HandleListPrompts(ctx context.Context, params json.RawMessage) (interface{}, error) {
	if ctx == nil {
		return nil, fmt.Errorf("context cannot be nil")
	}

	prompts, err := m.ListPrompts(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to list prompts: %w", err)
	}

	mcpPrompts := make([]PromptDefinition, len(prompts))
	for i, p := range prompts {
		args := make([]PromptArgument, len(p.Variables))
		for j, v := range p.Variables {
			args[j] = PromptArgument{
				Name:        v.Name,
				Description: v.Description,
				Required:    v.Required,
			}
		}

		mcpPrompts[i] = PromptDefinition{
			Name:        p.Name,
			Description: p.Description,
			Arguments:   args,
		}
	}

	return ListPromptsResponse{Prompts: mcpPrompts}, nil
}

/* HandleGetPrompt handles the prompts/get request */
func (m *Manager) HandleGetPrompt(ctx context.Context, params json.RawMessage) (interface{}, error) {
	if ctx == nil {
		return nil, fmt.Errorf("context cannot be nil")
	}

	if len(params) == 0 {
		return nil, fmt.Errorf("request parameters are required")
	}

	var req GetPromptRequest
	if err := json.Unmarshal(params, &req); err != nil {
		return nil, fmt.Errorf("failed to parse prompts/get request: %w", err)
	}

	if req.Name == "" {
		return nil, fmt.Errorf("prompt name is required and cannot be empty")
	}

	prompt, err := m.GetPrompt(ctx, req.Name)
	if err != nil {
		return nil, fmt.Errorf("failed to get prompt: %w", err)
	}

	/* Render template with variables if provided */
	renderedText := prompt.Template
	if len(req.Variables) > 0 {
		rendered, err := m.RenderPrompt(ctx, req.Name, req.Variables)
		if err != nil {
			return nil, fmt.Errorf("failed to render prompt: %w", err)
		}
		renderedText = rendered
	}

	/* Create response with rendered prompt */
	response := GetPromptResponse{
		Description: prompt.Description,
		Messages: []PromptMessage{
			{
				Role: "user",
				Content: PromptMessageContent{
					Type: "text",
					Text: renderedText,
				},
			},
		},
	}

	return response, nil
}

