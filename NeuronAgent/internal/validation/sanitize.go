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
	"path/filepath"
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

/* Safe SQL identifier pattern: must start with letter or underscore, then alphanumeric or underscore */
var safeIdentifierRe = regexp.MustCompile(`^[a-zA-Z_][a-zA-Z0-9_]*$`)

/* QuoteIdentifier validates and double-quote-wraps a PostgreSQL identifier to prevent SQL injection.
 * Returns the quoted identifier or error if the name does not match ^[a-zA-Z_][a-zA-Z0-9_]*$ */
func QuoteIdentifier(name string) (string, error) {
	trimmed := strings.TrimSpace(name)
	if trimmed == "" {
		return "", fmt.Errorf("identifier cannot be empty")
	}
	if !safeIdentifierRe.MatchString(trimmed) {
		return "", fmt.Errorf("invalid SQL identifier: only [a-zA-Z_][a-zA-Z0-9_]* allowed")
	}
	/* PostgreSQL: double quotes escape identifiers; double-quote any existing " in name */
	escaped := strings.ReplaceAll(trimmed, `"`, `""`)
	return `"` + escaped + `"`, nil
}

/* SanitizeSQLIdentifier sanitizes SQL identifier to prevent injection */
func SanitizeSQLIdentifier(input string) string {
	/* Remove any characters that aren't alphanumeric, underscore, or dash */
	reg := regexp.MustCompile(`[^a-zA-Z0-9_-]`)
	return reg.ReplaceAllString(input, "")
}

/* SafeVirtualPath validates and normalizes a virtual path so it cannot escape root.
 * Root is treated as "/". Returns the cleaned path or error if path escapes. */
func SafeVirtualPath(path string) (string, error) {
	if path == "" {
		return "/", nil
	}
	cleaned := filepath.Clean(filepath.Join("/", strings.TrimPrefix(path, "/")))
	if !strings.HasPrefix(cleaned, "/") || strings.HasPrefix(cleaned, "/..") || cleaned == ".." {
		return "", fmt.Errorf("path escapes virtual root: %s", path)
	}
	return cleaned, nil
}

/* SafePathUnderBase resolves path against baseDir and returns the resolved path only if it
 * remains under baseDir (no symlink escape). Use for real filesystem access.
 * baseDir must be an absolute path. */
func SafePathUnderBase(baseDir, path string) (string, error) {
	baseAbs, err := filepath.Abs(baseDir)
	if err != nil {
		return "", fmt.Errorf("base directory invalid: %w", err)
	}
	joined := filepath.Join(baseAbs, path)
	resolved, err := filepath.EvalSymlinks(joined)
	if err != nil {
		return "", fmt.Errorf("path resolution failed: %w", err)
	}
	resolved, err = filepath.Abs(resolved)
	if err != nil {
		return "", fmt.Errorf("path invalid: %w", err)
	}
	if resolved != baseAbs && !strings.HasPrefix(resolved, baseAbs+string(filepath.Separator)) {
		return "", fmt.Errorf("path escapes base directory")
	}
	return resolved, nil
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
