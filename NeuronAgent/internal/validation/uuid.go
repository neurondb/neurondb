/*-------------------------------------------------------------------------
 *
 * uuid.go
 *    UUID validation for NeuronAgent
 *
 * Provides comprehensive UUID validation functions.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/validation/uuid.go
 *
 *-------------------------------------------------------------------------
 */

package validation

import (
	"fmt"
	"strings"

	"github.com/google/uuid"
)

/* ValidateUUID validates a UUID string format */
func ValidateUUID(s, fieldName string) error {
	if s == "" {
		return fmt.Errorf("%s cannot be empty", fieldName)
	}

	s = strings.ToLower(strings.TrimSpace(s))
	_, err := uuid.Parse(s)
	if err != nil {
		return fmt.Errorf("%s has invalid UUID format: %w", fieldName, err)
	}

	return nil
}

/* ValidateUUIDRequired validates a UUID and ensures it's not empty */
func ValidateUUIDRequired(s, fieldName string) error {
	if s == "" {
		return fmt.Errorf("%s is required and cannot be empty", fieldName)
	}
	return ValidateUUID(s, fieldName)
}

/* ParseUUID parses a UUID string and returns error if invalid */
func ParseUUID(s, fieldName string) (uuid.UUID, error) {
	if err := ValidateUUID(s, fieldName); err != nil {
		return uuid.Nil, err
	}
	return uuid.Parse(s)
}


