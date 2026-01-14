/*-------------------------------------------------------------------------
 *
 * sampling_handlers.go
 *    MCP sampling handlers
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/sampling/sampling_handlers.go
 *
 *-------------------------------------------------------------------------
 */

package sampling

import (
	"context"
	"encoding/json"
	"fmt"
)

/* CreateMessageRequest represents a sampling/createMessage request */
type CreateMessageRequest struct {
	Messages    []Message              `json:"messages"`
	Model       string                 `json:"model,omitempty"`
	Temperature *float64               `json:"temperature,omitempty"`
	MaxTokens   *int                   `json:"maxTokens,omitempty"`
	TopP        *float64               `json:"topP,omitempty"`
	Stream      bool                   `json:"stream,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

/* CreateMessageResponse represents a sampling/createMessage response */
type CreateMessageResponse struct {
	Content   string                 `json:"content"`
	Model     string                 `json:"model"`
	StopReason string                `json:"stopReason,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

/* HandleCreateMessage handles the sampling/createMessage request */
func (m *Manager) HandleCreateMessage(ctx context.Context, params json.RawMessage) (interface{}, error) {
	if ctx == nil {
		return nil, fmt.Errorf("context cannot be nil")
	}

	if len(params) == 0 {
		return nil, fmt.Errorf("request parameters are required")
	}

	var req CreateMessageRequest
	if err := json.Unmarshal(params, &req); err != nil {
		return nil, fmt.Errorf("failed to parse sampling/createMessage request: %w", err)
	}

	if len(req.Messages) == 0 {
		return nil, fmt.Errorf("messages array is required and cannot be empty")
	}

	/* Validate message count */
	if len(req.Messages) > 100 {
		return nil, fmt.Errorf("messages array exceeds maximum size of 100: received %d messages", len(req.Messages))
	}

	samplingReq := SamplingRequest{
		Messages:    req.Messages,
		Model:       req.Model,
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
		TopP:        req.TopP,
		Stream:      req.Stream,
		Metadata:    req.Metadata,
	}

	if req.Stream {
		/* For streaming, we'd need to handle it differently */
		/* For now, return a non-streaming response */
		response, err := m.CreateMessage(ctx, samplingReq)
		if err != nil {
			return nil, fmt.Errorf("failed to create message: %w", err)
		}

		return CreateMessageResponse{
			Content:    response.Content,
			Model:      response.Model,
			StopReason: response.StopReason,
			Metadata:   response.Metadata,
		}, nil
	}

	response, err := m.CreateMessage(ctx, samplingReq)
	if err != nil {
		return nil, fmt.Errorf("failed to create message: %w", err)
	}

	return CreateMessageResponse{
		Content:    response.Content,
		Model:      response.Model,
		StopReason: response.StopReason,
		Metadata:   response.Metadata,
	}, nil
}

