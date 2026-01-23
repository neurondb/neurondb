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
	passwordPattern     = regexp.MustCompile(`(?i)(password|pwd|passwd|pass)\s*[=:]\s*[^\s,;\)]+`)
	connectionStringPattern = regexp.MustCompile(`(?i)(postgresql://|postgres://|mysql://|mongodb://|redis://|http://|https://)[^'"\s]+`)
	apiKeyPattern      = regexp.MustCompile(`(?i)(api[_-]?key|apikey|token|bearer|auth[_-]?token|access[_-]?token)\s*[=:]\s*[^\s,;\)]+`)
	secretPattern      = regexp.MustCompile(`(?i)(secret|private[_-]?key|private[_-]?key|session[_-]?key|encryption[_-]?key)\s*[=:]\s*[^\s,;\)]+`)
	/* Additional patterns for common sensitive data */
	emailPattern       = regexp.MustCompile(`(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b`)
	creditCardPattern  = regexp.MustCompile(`\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b`)
	ssnPattern         = regexp.MustCompile(`\b\d{3}-\d{2}-\d{4}\b`)
	/* AWS/GCP/Azure credential patterns */
	awsKeyPattern      = regexp.MustCompile(`(?i)(aws[_-]?(access[_-]?key|secret[_-]?key|session[_-]?token))\s*[=:]\s*[^\s,;\)]+`)
	gcpKeyPattern      = regexp.MustCompile(`(?i)(gcp[_-]?(key|credentials|service[_-]?account))\s*[=:]\s*[^\s,;\)]+`)
	azureKeyPattern    = regexp.MustCompile(`(?i)(azure[_-]?(key|secret|connection[_-]?string))\s*[=:]\s*[^\s,;\)]+`)
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
	
	/* Remove passwords (highest priority - most common) */
	result = passwordPattern.ReplaceAllString(result, "[password redacted]")
	
	/* Remove connection strings (but preserve the protocol part for debugging) */
	result = connectionStringPattern.ReplaceAllStringFunc(result, func(match string) string {
		/* Extract protocol */
		matchLower := strings.ToLower(match)
		if strings.HasPrefix(matchLower, "postgresql://") {
			return "postgresql://[connection string redacted]"
		}
		if strings.HasPrefix(matchLower, "postgres://") {
			return "postgres://[connection string redacted]"
		}
		if strings.HasPrefix(matchLower, "mysql://") {
			return "mysql://[connection string redacted]"
		}
		if strings.HasPrefix(matchLower, "mongodb://") {
			return "mongodb://[connection string redacted]"
		}
		if strings.HasPrefix(matchLower, "redis://") {
			return "redis://[connection string redacted]"
		}
		return "[connection string redacted]"
	})
	
	/* Remove API keys and tokens */
	result = apiKeyPattern.ReplaceAllString(result, "[api key redacted]")
	
	/* Remove secrets and private keys */
	result = secretPattern.ReplaceAllString(result, "[secret redacted]")
	
	/* Remove AWS credentials */
	result = awsKeyPattern.ReplaceAllString(result, "[aws credential redacted]")
	
	/* Remove GCP credentials */
	result = gcpKeyPattern.ReplaceAllString(result, "[gcp credential redacted]")
	
	/* Remove Azure credentials */
	result = azureKeyPattern.ReplaceAllString(result, "[azure credential redacted]")
	
	/* Remove email addresses (may contain sensitive info) */
	result = emailPattern.ReplaceAllString(result, "[email redacted]")
	
	/* Remove credit card numbers */
	result = creditCardPattern.ReplaceAllString(result, "[credit card redacted]")
	
	/* Remove SSN */
	result = ssnPattern.ReplaceAllString(result, "[SSN redacted]")
	
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
