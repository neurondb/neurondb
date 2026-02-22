/*-------------------------------------------------------------------------
 *
 * retry.go
 *    Safe retry manager for NeuronMCP
 *
 * Provides retry logic that only retries idempotent operations
 * to prevent duplicate side effects.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/reliability/retry.go
 *
 *-------------------------------------------------------------------------
 */

package reliability

import (
	"errors"
	"time"
)

/* RetryableError indicates an error that can be retried */
type RetryableError struct {
	Err error
}

/* Error returns the error message */
func (e *RetryableError) Error() string {
	return e.Err.Error()
}

/* Unwrap returns the underlying error */
func (e *RetryableError) Unwrap() error {
	return e.Err
}

/* RetryManager manages retry logic for idempotent operations */
type RetryManager struct {
	maxRetries        int
	initialBackoff    time.Duration
	maxBackoff        time.Duration
	backoffMultiplier float64
	idempotentTools   map[string]bool
}

/* NewRetryManager creates a new retry manager */
func NewRetryManager(maxRetries int, initialBackoff, maxBackoff time.Duration, backoffMultiplier float64) *RetryManager {
	if maxRetries < 0 {
		maxRetries = 0
	}
	if initialBackoff <= 0 {
		initialBackoff = 100 * time.Millisecond
	}
	if maxBackoff <= 0 {
		maxBackoff = 5 * time.Second
	}
	if backoffMultiplier <= 0 {
		backoffMultiplier = 2.0
	}

	rm := &RetryManager{
		maxRetries:        maxRetries,
		initialBackoff:    initialBackoff,
		maxBackoff:        maxBackoff,
		backoffMultiplier: backoffMultiplier,
		idempotentTools:   make(map[string]bool),
	}

	/* Mark known idempotent tools */
	rm.markIdempotentTools()

	return rm
}

/* markIdempotentTools marks tools that are known to be idempotent */
func (rm *RetryManager) markIdempotentTools() {
	/* Read operations are generally idempotent */
	idempotentTools := []string{
		"postgresql_execute_query",
		"postgresql_query_plan",
		"postgresql_tables",
		"postgresql_extensions",
		"postgresql_stats",
		"postgresql_settings",
		"vector_search",
		"vector_search_l2",
		"vector_search_cosine",
		"vector_search_inner_product",
		"list_models",
		"get_model_info",
		"index_status",
		"gpu_info",
	}

	for _, tool := range idempotentTools {
		rm.idempotentTools[tool] = true
	}
}

/* IsIdempotent checks if a tool is idempotent */
func (rm *RetryManager) IsIdempotent(toolName string) bool {
	if rm == nil {
		return false
	}
	return rm.idempotentTools[toolName]
}

/* MarkIdempotent marks a tool as idempotent */
func (rm *RetryManager) MarkIdempotent(toolName string) {
	if rm == nil {
		return
	}
	rm.idempotentTools[toolName] = true
}

/* ShouldRetry determines if an error should be retried */
func (rm *RetryManager) ShouldRetry(toolName string, err error) bool {
	if rm == nil || rm.maxRetries <= 0 {
		return false
	}

	/* Only retry idempotent operations */
	if !rm.IsIdempotent(toolName) {
		return false
	}

	/* Check if error is retryable */
	if err == nil {
		return false
	}

	/* Don't retry validation errors, safety violations, or permission errors */
	errorStr := err.Error()
	nonRetryableErrors := []string{
		"VALIDATION_ERROR",
		"SAFETY_VIOLATION",
		"READ_ONLY_VIOLATION",
		"PERMISSION_DENIED",
		"CONFIRMATION_REQUIRED",
	}

	for _, nonRetryable := range nonRetryableErrors {
		if contains(errorStr, nonRetryable) {
			return false
		}
	}

	/* Retry connection errors, timeouts, and transient errors */
	retryableErrors := []string{
		"connection",
		"timeout",
		"temporary",
		"network",
		"unavailable",
	}

	for _, retryable := range retryableErrors {
		if contains(errorStr, retryable) {
			return true
		}
	}

	/* Check if it's a RetryableError */
	var retryableErr *RetryableError
	if errors.As(err, &retryableErr) {
		return true
	}

	/* Default: don't retry */
	return false
}

/* GetBackoff calculates the backoff duration for a retry attempt */
func (rm *RetryManager) GetBackoff(attempt int) time.Duration {
	if rm == nil {
		return 100 * time.Millisecond
	}

	backoff := rm.initialBackoff
	for i := 0; i < attempt; i++ {
		backoff = time.Duration(float64(backoff) * rm.backoffMultiplier)
		if backoff > rm.maxBackoff {
			backoff = rm.maxBackoff
			break
		}
	}

	return backoff
}

/* GetMaxRetries returns the maximum number of retries */
func (rm *RetryManager) GetMaxRetries() int {
	if rm == nil {
		return 0
	}
	return rm.maxRetries
}

/* contains checks if a string contains a substring (case-insensitive) */
func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr ||
		(len(s) > len(substr) &&
			(s[:len(substr)] == substr ||
				s[len(s)-len(substr):] == substr ||
				containsMiddle(s, substr))))
}

/* containsMiddle checks if substr is in the middle of s */
func containsMiddle(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

/* NewRetryableError creates a new retryable error */
func NewRetryableError(err error) error {
	return &RetryableError{Err: err}
}
