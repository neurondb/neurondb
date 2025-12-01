package api

import (
	"fmt"
	"net/http"

	"github.com/neurondb/NeuronAgent/internal/utils"
)

// ValidateCreateAgentRequest validates CreateAgentRequest
func ValidateCreateAgentRequest(req *CreateAgentRequest) error {
	if err := utils.ValidateRequiredWithError(req.Name, "name"); err != nil {
		return err
	}
	if err := utils.ValidateRequiredWithError(req.SystemPrompt, "system_prompt"); err != nil {
		return err
	}
	if err := utils.ValidateRequiredWithError(req.ModelName, "model_name"); err != nil {
		return err
	}
	if !utils.ValidateLength(req.Name, 1, 100) {
		return fmt.Errorf("name must be between 1 and 100 characters")
	}
	if !utils.ValidateMinLength(req.SystemPrompt, 10) {
		return fmt.Errorf("system_prompt must be at least 10 characters")
	}
	return nil
}

// ValidateCreateSessionRequest validates CreateSessionRequest
func ValidateCreateSessionRequest(req *CreateSessionRequest) error {
	// AgentID is required (UUID validation happens in handler)
	return nil
}

// ValidateSendMessageRequest validates SendMessageRequest
func ValidateSendMessageRequest(req *SendMessageRequest) error {
	if err := utils.ValidateRequiredWithError(req.Content, "content"); err != nil {
		return err
	}
	if !utils.ValidateIn(req.Role, "user", "system") {
		return fmt.Errorf("role must be 'user' or 'system'")
	}
	if !utils.ValidateMinLength(req.Content, 1) {
		return fmt.Errorf("content must not be empty")
	}
	return nil
}

// ValidateAndRespond validates a request and responds with error if invalid
func ValidateAndRespond(w http.ResponseWriter, validator func() error) bool {
	if err := validator(); err != nil {
		respondError(w, NewError(http.StatusBadRequest, "validation failed", err))
		return false
	}
	return true
}

