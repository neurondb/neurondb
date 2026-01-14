/*-------------------------------------------------------------------------
 *
 * uuid.go
 *    UUID utility functions for NeuronAgent
 *
 * Provides UUID generation and parsing utilities for creating unique
 * identifiers throughout the application.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/utils/uuid.go
 *
 *-------------------------------------------------------------------------
 */

package utils

import (
	"github.com/google/uuid"
)

/* GenerateUUID generates a new UUID */
func GenerateUUID() uuid.UUID {
	return uuid.New()
}

/* GenerateUUIDString generates a new UUID as string */
func GenerateUUIDString() string {
	return uuid.New().String()
}

/* ParseUUID parses a UUID string */
func ParseUUID(s string) (uuid.UUID, error) {
	return uuid.Parse(s)
}

/* IsValidUUID checks if a string is a valid UUID */
func IsValidUUID(s string) bool {
	_, err := uuid.Parse(s)
	return err == nil
}

/* MustParseUUID parses a UUID string or panics */
func MustParseUUID(s string) uuid.UUID {
	return uuid.MustParse(s)
}
