/*-------------------------------------------------------------------------
 *
 * validation.go
 *    Comprehensive request validation utilities for NeuronAgent API
 *
 * Provides validation functions for API requests including body size
 * limits, UUID validation, input sanitization, pagination, and more.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/api/validation.go
 *
 *-------------------------------------------------------------------------
 */

package api

import (
	"fmt"
	"net/http"
	"regexp"
	"strings"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/utils"
)

/* ValidateCreateAgentRequest validates CreateAgentRequest with comprehensive checks */
func ValidateCreateAgentRequest(req *CreateAgentRequest) error {
	/* Name validation */
	if err := utils.ValidateRequiredWithError(req.Name, "name"); err != nil {
		return err
	}
	if !utils.ValidateLength(req.Name, 1, 100) {
		return fmt.Errorf("name must be between 1 and 100 characters")
	}
	/* Validate name format (alphanumeric, dash, underscore) */
	nameRegex := regexp.MustCompile(`^[a-zA-Z0-9_-]+$`)
	if !nameRegex.MatchString(req.Name) {
		return fmt.Errorf("name must contain only alphanumeric characters, dashes, and underscores")
	}

	/* System prompt validation */
	if err := utils.ValidateRequiredWithError(req.SystemPrompt, "system_prompt"); err != nil {
		return err
	}
	if !utils.ValidateMinLength(req.SystemPrompt, 10) {
		return fmt.Errorf("system_prompt must be at least 10 characters")
	}
	if !utils.ValidateMaxLength(req.SystemPrompt, 50000) {
		return fmt.Errorf("system_prompt must not exceed 50000 characters")
	}

	/* Model name validation */
	if err := utils.ValidateRequiredWithError(req.ModelName, "model_name"); err != nil {
		return err
	}
	if !utils.ValidateLength(req.ModelName, 1, 100) {
		return fmt.Errorf("model_name must be between 1 and 100 characters")
	}

	/* Description validation (optional) */
	if req.Description != nil && *req.Description != "" {
		if !utils.ValidateMaxLength(*req.Description, 1000) {
			return fmt.Errorf("description must not exceed 1000 characters")
		}
	}

	/* Memory table validation (optional) */
	if req.MemoryTable != nil && *req.MemoryTable != "" {
		if !utils.ValidateLength(*req.MemoryTable, 1, 100) {
			return fmt.Errorf("memory_table must be between 1 and 100 characters")
		}
		/* Validate table name format */
		tableNameRegex := regexp.MustCompile(`^[a-zA-Z][a-zA-Z0-9_]*$`)
		if !tableNameRegex.MatchString(*req.MemoryTable) {
			return fmt.Errorf("memory_table must be a valid table name (start with letter, alphanumeric and underscores)")
		}
	}

	/* Enabled tools validation */
	if req.EnabledTools == nil {
		return fmt.Errorf("enabled_tools is required")
	}
	if len(req.EnabledTools) == 0 {
		return fmt.Errorf("enabled_tools must contain at least one tool")
	}
	if len(req.EnabledTools) > 50 {
		return fmt.Errorf("enabled_tools must not exceed 50 tools")
	}
	/* Validate tool names */
	validTools := map[string]bool{
		"sql":           true,
		"http":          true,
		"code":          true,
		"shell":         true,
		"vector":        true,
		"rag":           true,
		"analytics":     true,
		"hybrid_search": true,
		"reranking":     true,
		"ml":            true,
		"browser":       true,
	}
	for _, tool := range req.EnabledTools {
		if !validTools[strings.ToLower(tool)] {
			/* Allow custom tools (not in predefined list) */
			if !utils.ValidateLength(tool, 1, 50) {
				return fmt.Errorf("invalid tool name: %s (must be 1-50 characters)", tool)
			}
		}
	}

	/* Config validation (optional) */
	if req.Config != nil {
		/* Validate config size (max 10KB when serialized) */
		configStr := fmt.Sprintf("%v", req.Config)
		if len(configStr) > 10000 {
			return fmt.Errorf("config is too large (max 10KB)")
		}
	}

	return nil
}

/* ValidateCreateSessionRequest validates CreateSessionRequest */
func ValidateCreateSessionRequest(req *CreateSessionRequest) error {
	/* AgentID is required (UUID validation happens in handler) */
	if req.AgentID == uuid.Nil {
		return fmt.Errorf("agent_id is required")
	}

	/* External user ID validation (optional) */
	if req.ExternalUserID != nil && *req.ExternalUserID != "" {
		if !utils.ValidateLength(*req.ExternalUserID, 1, 255) {
			return fmt.Errorf("external_user_id must be between 1 and 255 characters")
		}
	}

	/* Metadata validation (optional) */
	if req.Metadata != nil {
		/* Validate metadata size */
		metadataStr := fmt.Sprintf("%v", req.Metadata)
		if len(metadataStr) > 5000 {
			return fmt.Errorf("metadata is too large (max 5KB)")
		}
	}

	return nil
}

/* ValidateSendMessageRequest validates SendMessageRequest with comprehensive checks */
func ValidateSendMessageRequest(req *SendMessageRequest) error {
	/* Content validation */
	if err := utils.ValidateRequiredWithError(req.Content, "content"); err != nil {
		return err
	}
	if !utils.ValidateMinLength(req.Content, 1) {
		return fmt.Errorf("content must not be empty")
	}
	if !utils.ValidateMaxLength(req.Content, 100000) {
		return fmt.Errorf("content must not exceed 100000 characters")
	}

	/* Role validation */
	if !utils.ValidateIn(req.Role, "user", "system", "assistant") {
		return fmt.Errorf("role must be 'user', 'system', or 'assistant'")
	}

	/* Metadata validation (optional) */
	if req.Metadata != nil {
		/* Validate metadata size */
		metadataStr := fmt.Sprintf("%v", req.Metadata)
		if len(metadataStr) > 5000 {
			return fmt.Errorf("metadata is too large (max 5KB)")
		}
	}

	return nil
}

/* ValidatePaginationParams validates pagination query parameters */
func ValidatePaginationParams(limit, offset int) error {
	if limit < 1 {
		return fmt.Errorf("limit must be at least 1")
	}
	if limit > 1000 {
		return fmt.Errorf("limit must not exceed 1000")
	}
	if offset < 0 {
		return fmt.Errorf("offset must be non-negative")
	}
	if offset > 100000 {
		return fmt.Errorf("offset must not exceed 100000")
	}
	return nil
}

/* ValidateDateRange validates date range parameters */
func ValidateDateRange(startDate, endDate string) error {
	if startDate != "" {
		/* Validate RFC3339 format */
		dateRegex := regexp.MustCompile(`^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(Z|[\+\-]\d{2}:\d{2})$`)
		if !dateRegex.MatchString(startDate) {
			return fmt.Errorf("start_date must be in RFC3339 format (e.g., 2024-01-01T00:00:00Z)")
		}
	}
	if endDate != "" {
		dateRegex := regexp.MustCompile(`^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(Z|[\+\-]\d{2}:\d{2})$`)
		if !dateRegex.MatchString(endDate) {
			return fmt.Errorf("end_date must be in RFC3339 format (e.g., 2024-01-01T00:00:00Z)")
		}
	}
	if startDate != "" && endDate != "" {
		/* Validate start_date < end_date (would need to parse, but basic check here) */
		if strings.Compare(startDate, endDate) > 0 {
			return fmt.Errorf("start_date must be before end_date")
		}
	}
	return nil
}

/* ValidateSearchQuery validates search query parameters */
func ValidateSearchQuery(query string, maxLength int) error {
	if query == "" {
		return fmt.Errorf("search query cannot be empty")
	}
	if len(query) > maxLength {
		return fmt.Errorf("search query must not exceed %d characters", maxLength)
	}
	/* Check for potentially dangerous patterns */
	dangerousPatterns := []string{
		"<script",
		"javascript:",
		"onerror=",
		"onload=",
	}
	lowerQuery := strings.ToLower(query)
	for _, pattern := range dangerousPatterns {
		if strings.Contains(lowerQuery, pattern) {
			return fmt.Errorf("search query contains potentially dangerous content")
		}
	}
	return nil
}

/* ValidateAndRespond validates a request and responds with error if invalid */
func ValidateAndRespond(w http.ResponseWriter, validator func() error) bool {
	if err := validator(); err != nil {
		respondError(w, NewError(http.StatusBadRequest, "validation failed", err))
		return false
	}
	return true
}
