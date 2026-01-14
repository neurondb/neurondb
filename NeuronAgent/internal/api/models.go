/*-------------------------------------------------------------------------
 *
 * models.go
 *    API request and response models for NeuronAgent
 *
 * Defines data structures for API requests, responses, and internal
 * data transfer objects used throughout the API handlers.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/api/models.go
 *
 *-------------------------------------------------------------------------
 */

package api

import (
	"time"

	"github.com/google/uuid"
)

/* Request DTOs */

type CreateAgentRequest struct {
	Name         string                 `json:"name"`
	Description  *string                `json:"description"`
	SystemPrompt string                 `json:"system_prompt"`
	ModelName    string                 `json:"model_name"`
	MemoryTable  *string                `json:"memory_table"`
	EnabledTools []string               `json:"enabled_tools"`
	Config       map[string]interface{} `json:"config"`
}

type CreateSessionRequest struct {
	AgentID        uuid.UUID              `json:"agent_id"`
	ExternalUserID *string                `json:"external_user_id"`
	Metadata       map[string]interface{} `json:"metadata"`
}

type SendMessageRequest struct {
	Role     string                 `json:"role"`
	Content  string                 `json:"content"`
	Stream   bool                   `json:"stream"`
	Metadata map[string]interface{} `json:"metadata"`
}

/* Response DTOs */

type AgentResponse struct {
	ID           uuid.UUID              `json:"id"`
	Name         string                 `json:"name"`
	Description  *string                `json:"description"`
	SystemPrompt string                 `json:"system_prompt"`
	ModelName    string                 `json:"model_name"`
	MemoryTable  *string                `json:"memory_table"`
	EnabledTools []string               `json:"enabled_tools"`
	Config       map[string]interface{} `json:"config"`
	CreatedAt    time.Time              `json:"created_at"`
	UpdatedAt    time.Time              `json:"updated_at"`
}

type SessionResponse struct {
	ID             uuid.UUID              `json:"id"`
	AgentID        uuid.UUID              `json:"agent_id"`
	ExternalUserID *string                `json:"external_user_id"`
	Metadata       map[string]interface{} `json:"metadata"`
	CreatedAt      time.Time              `json:"created_at"`
	LastActivityAt time.Time              `json:"last_activity_at"`
}

type MessageResponse struct {
	ID         int64                  `json:"id"`
	SessionID  uuid.UUID              `json:"session_id"`
	Role       string                 `json:"role"`
	Content    string                 `json:"content"`
	ToolName   *string                `json:"tool_name"`
	ToolCallID *string                `json:"tool_call_id"`
	TokenCount *int                   `json:"token_count"`
	Metadata   map[string]interface{} `json:"metadata"`
	CreatedAt  time.Time              `json:"created_at"`
}

type ErrorResponse struct {
	Error   string `json:"error"`
	Message string `json:"message,omitempty"`
	Code    int    `json:"code"`
}
