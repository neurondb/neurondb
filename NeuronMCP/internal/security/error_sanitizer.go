/*-------------------------------------------------------------------------
 *
 * error_sanitizer.go
 *    Error message sanitization for NeuronMCP
 *
 * Provides utilities to sanitize error messages and remove sensitive
 * information like passwords, connection strings, and API keys.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/security/error_sanitizer.go
 *
 *-------------------------------------------------------------------------
 */

package security

import (
	"fmt"
	"regexp"
	"strings"
)

var (
	/* Patterns to detect and sanitize sensitive information */
	passwordPattern     = regexp.MustCompile(`(?i)(password|pwd|passwd)\s*[=:]\s*[^\s,;\)]+`)
	connectionStringPattern = regexp.MustCompile(`(?i)(postgresql://|postgres://|mysql://)[^'"\s]+`)
	apiKeyPattern      = regexp.MustCompile(`(?i)(api[_-]?key|apikey|token)\s*[=:]\s*[^\s,;\)]+`)
	secretPattern      = regexp.MustCompile(`(?i)(secret|private[_-]?key)\s*[=:]\s*[^\s,;\)]+`)
)

/* SanitizeError sanitizes an error message to remove sensitive information */
func SanitizeError(err error) error {
	if err == nil {
		return nil
	}
	
	errMsg := err.Error()
	sanitized := SanitizeString(errMsg)
	
	/* If sanitization changed the message, return a new error */
	if sanitized != errMsg {
		return fmt.Errorf("%s", sanitized)
	}
	
	return err
}

/* SanitizeString sanitizes a string to remove sensitive information */
func SanitizeString(s string) string {
	if s == "" {
		return s
	}
	
	result := s
	
	/* Remove passwords */
	result = passwordPattern.ReplaceAllString(result, "[password redacted]")
	
	/* Remove connection strings (but preserve the protocol part for debugging) */
	result = connectionStringPattern.ReplaceAllStringFunc(result, func(match string) string {
		/* Extract protocol */
		if strings.HasPrefix(strings.ToLower(match), "postgresql://") {
			return "postgresql://[connection string redacted]"
		}
		if strings.HasPrefix(strings.ToLower(match), "postgres://") {
			return "postgres://[connection string redacted]"
		}
		if strings.HasPrefix(strings.ToLower(match), "mysql://") {
			return "mysql://[connection string redacted]"
		}
		return "[connection string redacted]"
	})
	
	/* Remove API keys */
	result = apiKeyPattern.ReplaceAllString(result, "[api key redacted]")
	
	/* Remove secrets */
	result = secretPattern.ReplaceAllString(result, "[secret redacted]")
	
	return result
}

/* SanitizeErrorWithContext sanitizes an error and includes safe context */
func SanitizeErrorWithContext(err error, safeContext map[string]interface{}) error {
	if err == nil {
		return nil
	}
	
	sanitizedErr := SanitizeError(err)
	
	/* Add safe context if provided */
	if len(safeContext) > 0 {
		contextStr := formatSafeContext(safeContext)
		return fmt.Errorf("%s (context: %s)", sanitizedErr.Error(), contextStr)
	}
	
	return sanitizedErr
}

/* formatSafeContext formats safe context information */
func formatSafeContext(ctx map[string]interface{}) string {
	var parts []string
	for k, v := range ctx {
		/* Only include non-sensitive keys */
		safeKeys := map[string]bool{
			"host": true, "port": true, "database": true, "user": true,
			"attempt": true, "max_retries": true, "timeout": true,
		}
		if safeKeys[strings.ToLower(k)] {
			parts = append(parts, fmt.Sprintf("%s=%v", k, v))
		}
	}
	return strings.Join(parts, ", ")
}
