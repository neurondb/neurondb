/*-------------------------------------------------------------------------
 *
 * handlers.go
 *    Elicitation MCP handlers
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/elicitation/handlers.go
 *
 *-------------------------------------------------------------------------
 */

package elicitation

import (
	"context"
	"encoding/json"
	"fmt"
)

/* RequestPromptRequest represents a prompts/request request */
type RequestPromptRequest struct {
	Message string                 `json:"message"`
	Type    string                 `json:"type,omitempty"` /* text, confirm, choice, number */
	Options []string               `json:"options,omitempty"`
	Default interface{}            `json:"default,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

/* RespondPromptRequest represents a prompts/respond request */
type RespondPromptRequest struct {
	RequestID string      `json:"requestId"`
	Value     interface{} `json:"value"`
}

/* HandleRequestPrompt handles the prompts/request request */
func (m *Manager) HandleRequestPrompt(ctx context.Context, params json.RawMessage) (interface{}, error) {
	if ctx == nil {
		return nil, fmt.Errorf("context cannot be nil")
	}

	if len(params) == 0 {
		return nil, fmt.Errorf("request parameters are required")
	}

	var req RequestPromptRequest
	if err := json.Unmarshal(params, &req); err != nil {
		return nil, fmt.Errorf("failed to parse prompts/request: %w", err)
	}

	if req.Message == "" {
		return nil, fmt.Errorf("message is required")
	}

	/* Build options */
	var options []PromptOption
	if len(req.Options) > 0 {
		options = append(options, WithOptions(req.Options...))
	}
	if req.Default != nil {
		options = append(options, WithDefault(req.Default))
	}
	if req.Metadata != nil {
		options = append(options, WithMetadata(req.Metadata))
	}

	/* Create prompt request */
	promptType := req.Type
	if promptType == "" {
		promptType = "text"
	}

	request, err := m.RequestPrompt(ctx, req.Message, promptType, options...)
	if err != nil {
		return nil, fmt.Errorf("failed to create prompt request: %w", err)
	}

	return request, nil
}

/* HandleRespondPrompt handles the prompts/respond request */
func (m *Manager) HandleRespondPrompt(ctx context.Context, params json.RawMessage) (interface{}, error) {
	if ctx == nil {
		return nil, fmt.Errorf("context cannot be nil")
	}

	if len(params) == 0 {
		return nil, fmt.Errorf("request parameters are required")
	}

	var req RespondPromptRequest
	if err := json.Unmarshal(params, &req); err != nil {
		return nil, fmt.Errorf("failed to parse prompts/respond: %w", err)
	}

	if req.RequestID == "" {
		return nil, fmt.Errorf("requestId is required")
	}

	if err := m.RespondToPrompt(ctx, req.RequestID, req.Value); err != nil {
		return nil, fmt.Errorf("failed to record response: %w", err)
	}

	return map[string]interface{}{"success": true}, nil
}
