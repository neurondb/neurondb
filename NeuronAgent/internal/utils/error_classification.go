/*-------------------------------------------------------------------------
 *
 * error_classification.go
 *    Shared error classification utilities
 *
 * Provides centralized error classification logic for retry decisions
 * and error handling across the codebase.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/utils/error_classification.go
 *
 *-------------------------------------------------------------------------
 */

package utils

import (
	"strings"
)

/* ErrorType classifies errors for retry logic */
type ErrorType int

const (
	ErrorTypeRetryable ErrorType = iota
	ErrorTypeNonRetryable
	ErrorTypeTimeout
	ErrorTypeRateLimit
)

/* ClassifyError classifies an error as retryable or non-retryable */
func ClassifyError(err error) ErrorType {
	if err == nil {
		return ErrorTypeNonRetryable
	}

	errStr := err.Error()
	errLower := strings.ToLower(errStr)

	/* Check for timeout errors first */
	if strings.Contains(errLower, "timeout") || strings.Contains(errLower, "deadline exceeded") {
		return ErrorTypeTimeout
	}

	/* Check for rate limit errors */
	if strings.Contains(errLower, "rate limit") || strings.Contains(errLower, "too many") {
		return ErrorTypeRateLimit
	}

	/* Non-retryable errors: validation, authentication, authorization, not found */
	nonRetryablePatterns := []string{
		"validation",
		"invalid",
		"unauthorized",
		"forbidden",
		"not found",
		"does not exist",
		"already exists",
		"duplicate",
		"malformed",
		"parse error",
		"syntax error",
	}

	/* Retryable errors: network, timeout, temporary, rate limit, server errors */
	retryablePatterns := []string{
		"connection",
		"temporary",
		"temporarily",
		"server error",
		"internal error",
		"context canceled",
		"network",
		"unavailable",
		"busy",
		"locked",
	}

	/* Check for non-retryable patterns first */
	for _, pattern := range nonRetryablePatterns {
		if strings.Contains(errLower, pattern) {
			return ErrorTypeNonRetryable
		}
	}

	/* Check for retryable patterns */
	for _, pattern := range retryablePatterns {
		if strings.Contains(errLower, pattern) {
			return ErrorTypeRetryable
		}
	}

	/* Default to retryable for unknown errors (conservative approach) */
	return ErrorTypeRetryable
}

/* IsRetryable checks if an error is retryable */
func IsRetryable(err error) bool {
	errorType := ClassifyError(err)
	return errorType == ErrorTypeRetryable || errorType == ErrorTypeTimeout || errorType == ErrorTypeRateLimit
}
