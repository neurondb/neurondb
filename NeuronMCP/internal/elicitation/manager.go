/*-------------------------------------------------------------------------
 *
 * manager.go
 *    Elicitation manager for NeuronMCP
 *
 * Manages server-initiated user prompts (elicitation) for interactive workflows.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/elicitation/manager.go
 *
 *-------------------------------------------------------------------------
 */

package elicitation

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/neurondb/NeuronMCP/internal/logging"
)

/* PromptRequest represents a request for user input */
type PromptRequest struct {
	ID          string                 `json:"id"`
	Message     string                 `json:"message"`
	Type        string                 `json:"type"` /* text, confirm, choice, number */
	Options     []string               `json:"options,omitempty"` /* For choice type */
	Default     interface{}            `json:"default,omitempty"`
	Required    bool                   `json:"required"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt   time.Time              `json:"createdAt"`
	ExpiresAt   *time.Time             `json:"expiresAt,omitempty"`
}

/* PromptResponse represents a user response to a prompt */
type PromptResponse struct {
	RequestID string      `json:"requestId"`
	Value     interface{} `json:"value"`
	Timestamp time.Time   `json:"timestamp"`
}

/* PromptSession represents an active prompt session */
type PromptSession struct {
	ID        string
	Request   *PromptRequest
	Response  *PromptResponse
	Status    string /* pending, responded, expired, cancelled */
	CreatedAt time.Time
	UpdatedAt time.Time
}

/* Manager manages elicitation prompts */
type Manager struct {
	mu       sync.RWMutex
	sessions map[string]*PromptSession
	logger   *logging.Logger
}

/* NewManager creates a new elicitation manager */
func NewManager(logger *logging.Logger) *Manager {
	return &Manager{
		sessions: make(map[string]*PromptSession),
		logger:   logger,
	}
}

/* RequestPrompt creates a new prompt request */
func (m *Manager) RequestPrompt(ctx context.Context, message string, promptType string, options ...PromptOption) (*PromptRequest, error) {
	if message == "" {
		return nil, fmt.Errorf("prompt message cannot be empty")
	}
	if promptType == "" {
		promptType = "text" /* Default to text */
	}

	/* Validate prompt type */
	validTypes := map[string]bool{
		"text":    true,
		"confirm": true,
		"choice":  true,
		"number":  true,
	}
	if !validTypes[promptType] {
		return nil, fmt.Errorf("invalid prompt type: %s (valid types: text, confirm, choice, number)", promptType)
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	requestID := fmt.Sprintf("prompt-%d", time.Now().UnixNano())
	request := &PromptRequest{
		ID:        requestID,
		Message:   message,
		Type:      promptType,
		Required:  true,
		CreatedAt: time.Now(),
	}

	/* Apply options */
	for _, opt := range options {
		opt(request)
	}

	/* Create session */
	session := &PromptSession{
		ID:        requestID,
		Request:   request,
		Status:    "pending",
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	m.sessions[requestID] = session

	if m.logger != nil {
		m.logger.Info("Prompt request created", map[string]interface{}{
			"request_id": requestID,
			"type":       promptType,
			"message":    message,
		})
	}

	return request, nil
}

/* RespondToPrompt records a user response to a prompt */
func (m *Manager) RespondToPrompt(ctx context.Context, requestID string, value interface{}) error {
	if requestID == "" {
		return fmt.Errorf("request ID cannot be empty")
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	session, exists := m.sessions[requestID]
	if !exists {
		return fmt.Errorf("prompt request not found: %s", requestID)
	}

	if session.Status != "pending" {
		return fmt.Errorf("prompt request is not pending: status=%s", session.Status)
	}

	/* Validate response based on prompt type */
	if err := validateResponse(session.Request, value); err != nil {
		return fmt.Errorf("invalid response: %w", err)
	}

	/* Create response */
	response := &PromptResponse{
		RequestID: requestID,
		Value:     value,
		Timestamp: time.Now(),
	}

	session.Response = response
	session.Status = "responded"
	session.UpdatedAt = time.Now()

	if m.logger != nil {
		m.logger.Info("Prompt response received", map[string]interface{}{
			"request_id": requestID,
			"type":       session.Request.Type,
		})
	}

	return nil
}

/* GetSession retrieves a prompt session */
func (m *Manager) GetSession(requestID string) (*PromptSession, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	session, exists := m.sessions[requestID]
	if !exists {
		return nil, fmt.Errorf("prompt session not found: %s", requestID)
	}

	return session, nil
}

/* CancelPrompt cancels a pending prompt */
func (m *Manager) CancelPrompt(requestID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	session, exists := m.sessions[requestID]
	if !exists {
		return fmt.Errorf("prompt session not found: %s", requestID)
	}

	if session.Status != "pending" {
		return fmt.Errorf("prompt is not pending: status=%s", session.Status)
	}

	session.Status = "cancelled"
	session.UpdatedAt = time.Now()

	return nil
}

/* CleanupExpiredSessions removes expired sessions */
func (m *Manager) CleanupExpiredSessions() {
	m.mu.Lock()
	defer m.mu.Unlock()

	now := time.Now()
	for id, session := range m.sessions {
		if session.Request.ExpiresAt != nil && now.After(*session.Request.ExpiresAt) {
			if session.Status == "pending" {
				session.Status = "expired"
				session.UpdatedAt = now
			}
		}
		/* Remove old sessions (older than 1 hour) */
		if now.Sub(session.CreatedAt) > time.Hour {
			delete(m.sessions, id)
		}
	}
}

/* validateResponse validates a response based on prompt type */
func validateResponse(request *PromptRequest, value interface{}) error {
	switch request.Type {
	case "confirm":
		if _, ok := value.(bool); !ok {
			return fmt.Errorf("confirm prompt requires boolean value")
		}
	case "choice":
		if str, ok := value.(string); ok {
			valid := false
			for _, opt := range request.Options {
				if opt == str {
					valid = true
					break
				}
			}
			if !valid {
				return fmt.Errorf("value must be one of: %v", request.Options)
			}
		} else {
			return fmt.Errorf("choice prompt requires string value")
		}
	case "number":
		switch value.(type) {
		case int, int64, float64, float32:
			/* Valid number types */
		default:
			return fmt.Errorf("number prompt requires numeric value")
		}
	case "text":
		if _, ok := value.(string); !ok {
			return fmt.Errorf("text prompt requires string value")
		}
	}
	return nil
}

/* PromptOption is a function that configures a prompt request */
type PromptOption func(*PromptRequest)

/* WithOptions sets choice options */
func WithOptions(options ...string) PromptOption {
	return func(r *PromptRequest) {
		r.Options = options
	}
}

/* WithDefault sets a default value */
func WithDefault(value interface{}) PromptOption {
	return func(r *PromptRequest) {
		r.Default = value
	}
}

/* WithRequired sets whether the prompt is required */
func WithRequired(required bool) PromptOption {
	return func(r *PromptRequest) {
		r.Required = required
	}
}

/* WithExpiration sets an expiration time */
func WithExpiration(duration time.Duration) PromptOption {
	return func(r *PromptRequest) {
		expiresAt := time.Now().Add(duration)
		r.ExpiresAt = &expiresAt
	}
}

/* WithMetadata adds metadata to the prompt */
func WithMetadata(metadata map[string]interface{}) PromptOption {
	return func(r *PromptRequest) {
		if r.Metadata == nil {
			r.Metadata = make(map[string]interface{})
		}
		for k, v := range metadata {
			r.Metadata[k] = v
		}
	}
}
