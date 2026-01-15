/*-------------------------------------------------------------------------
 *
 * identifier.go
 *    PostgreSQL identifier validation and escaping for NeuronAgent
 *
 * Provides utilities for safely handling SQL identifiers (table names,
 * column names) to prevent SQL injection attacks.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/pkg/neurondb/identifier.go
 *
 *-------------------------------------------------------------------------
 */

package neurondb

import (
	"fmt"
	"regexp"
	"strings"
)

var (
	/* SQL identifier regex: alphanumeric, underscore, dollar sign, must start with letter or underscore */
	sqlIdentifierRegex = regexp.MustCompile(`^[a-zA-Z_][a-zA-Z0-9_$]*$`)

	/* Dangerous SQL keywords that should not appear in user input */
	dangerousKeywords = map[string]bool{
		"DROP": true, "TRUNCATE": true, "DELETE": true, "UPDATE": true,
		"INSERT": true, "ALTER": true, "CREATE": true, "GRANT": true,
		"REVOKE": true, "EXECUTE": true, "CALL": true, "COPY": true,
		"VACUUM": true, "ANALYZE": true, "REINDEX": true, "CLUSTER": true,
	}
)

/* ValidateSQLIdentifier validates a SQL identifier (table, column name) */
func ValidateSQLIdentifier(identifier, fieldName string) error {
	if identifier == "" {
		return fmt.Errorf("%s cannot be empty", fieldName)
	}

	identifier = strings.TrimSpace(identifier)

	/* Check length (PostgreSQL limit is 63 bytes, but we'll be more conservative) */
	if len(identifier) > 63 {
		return fmt.Errorf("%s exceeds maximum length of 63 characters: %s", fieldName, identifier)
	}

	/* Check for SQL identifier format */
	if !sqlIdentifierRegex.MatchString(identifier) {
		return fmt.Errorf("%s contains invalid characters: %s (must start with letter/underscore, followed by alphanumeric/underscore/dollar)", fieldName, identifier)
	}

	/* Check for reserved keywords (case-insensitive) */
	upperIdentifier := strings.ToUpper(identifier)
	if dangerousKeywords[upperIdentifier] {
		return fmt.Errorf("%s is a reserved SQL keyword and cannot be used: %s", fieldName, identifier)
	}

	return nil
}

/* EscapeSQLIdentifier escapes a SQL identifier for safe use in SQL queries */
/* This wraps the identifier in double quotes and escapes any embedded quotes */
func EscapeSQLIdentifier(identifier string) string {
	/* Replace double quotes with double double quotes and wrap in quotes */
	return fmt.Sprintf(`"%s"`, strings.ReplaceAll(identifier, `"`, `""`))
}

