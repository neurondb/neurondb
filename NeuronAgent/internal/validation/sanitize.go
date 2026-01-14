/*-------------------------------------------------------------------------
 *
 * sanitize.go
 *    Input sanitization functions for NeuronAgent
 *
 * Provides input sanitization to prevent injection attacks and ensure
 * data safety.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/validation/sanitize.go
 *
 *-------------------------------------------------------------------------
 */

package validation

import (
	"fmt"
	"html"
	"regexp"
	"strings"
)

/* SanitizeString sanitizes a string input */
func SanitizeString(input string) string {
	/* Trim whitespace */
	output := strings.TrimSpace(input)
	
	/* Escape HTML entities */
	output = html.EscapeString(output)
	
	return output
}

/* SanitizeSQLIdentifier sanitizes SQL identifier to prevent injection */
func SanitizeSQLIdentifier(input string) string {
	/* Remove any characters that aren't alphanumeric, underscore, or dash */
	reg := regexp.MustCompile(`[^a-zA-Z0-9_-]`)
	return reg.ReplaceAllString(input, "")
}

/* SanitizeFilename sanitizes a filename to prevent path traversal */
func SanitizeFilename(input string) string {
	/* Remove path separators and dangerous characters */
	reg := regexp.MustCompile(`[^a-zA-Z0-9._-]`)
	output := reg.ReplaceAllString(input, "")
	
	/* Remove leading dots to prevent hidden files */
	output = strings.TrimLeft(output, ".")
	
	/* Limit length */
	if len(output) > 255 {
		output = output[:255]
	}
	
	return output
}

/* SanitizeURL sanitizes a URL input */
func SanitizeURL(input string) string {
	/* Basic URL validation - remove dangerous protocols */
	lower := strings.ToLower(strings.TrimSpace(input))
	if strings.HasPrefix(lower, "javascript:") ||
		strings.HasPrefix(lower, "data:") ||
		strings.HasPrefix(lower, "vbscript:") {
		return ""
	}
	
	return strings.TrimSpace(input)
}

/* SanitizeEmail sanitizes an email address */
func SanitizeEmail(input string) string {
	/* Basic email sanitization */
	output := strings.TrimSpace(strings.ToLower(input))
	
	/* Remove any whitespace */
	output = strings.ReplaceAll(output, " ", "")
	output = strings.ReplaceAll(output, "\t", "")
	output = strings.ReplaceAll(output, "\n", "")
	output = strings.ReplaceAll(output, "\r", "")
	
	return output
}

/* SanitizeJSON sanitizes JSON input by validating structure */
func SanitizeJSON(input string) (string, error) {
	/* Basic check - ensure it's valid JSON structure */
	/* In production, use proper JSON parser and re-serialize */
	trimmed := strings.TrimSpace(input)
	if !strings.HasPrefix(trimmed, "{") && !strings.HasPrefix(trimmed, "[") {
		return "", fmt.Errorf("invalid JSON structure")
	}
	return trimmed, nil
}

