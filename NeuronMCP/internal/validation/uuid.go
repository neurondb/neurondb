/*-------------------------------------------------------------------------
 *
 * uuid.go
 *    UUID validation for NeuronMCP
 *
 * Provides comprehensive UUID validation functions.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/validation/uuid.go
 *
 *-------------------------------------------------------------------------
 */

package validation

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/google/uuid"
)

var (
	uuidRegex = regexp.MustCompile(`^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`)
)

/* ValidateUUID validates a UUID string format */
func ValidateUUID(s string) error {
	if s == "" {
		return fmt.Errorf("UUID cannot be empty")
	}
	
	s = strings.ToLower(strings.TrimSpace(s))
	if !uuidRegex.MatchString(s) {
		return fmt.Errorf("invalid UUID format: %s (expected format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)", s)
	}
	
	_, err := uuid.Parse(s)
	if err != nil {
		return fmt.Errorf("invalid UUID: %w", err)
	}
	
	return nil
}

/* ValidateUUIDRequired validates a UUID and ensures it's not empty */
func ValidateUUIDRequired(s, fieldName string) error {
	if s == "" {
		return fmt.Errorf("%s is required and cannot be empty", fieldName)
	}
	return ValidateUUID(s)
}

/* ParseUUID parses a UUID string and returns error if invalid */
func ParseUUID(s string) (uuid.UUID, error) {
	if err := ValidateUUID(s); err != nil {
		return uuid.Nil, err
	}
	return uuid.Parse(s)
}



