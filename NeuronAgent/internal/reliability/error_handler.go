/*-------------------------------------------------------------------------
 *
 * error_handler.go
 *    Advanced error handling with retry and recovery
 *
 * Provides intelligent retry with exponential backoff, error classification,
 * recovery strategies, and dead letter queues.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/reliability/error_handler.go
 *
 *-------------------------------------------------------------------------
 */

package reliability

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/metrics"
	"github.com/neurondb/NeuronAgent/internal/utils"
)

/* ErrorHandler provides advanced error handling */
type ErrorHandler struct {
	queries      *db.Queries
	maxRetries   int
	baseDelay    time.Duration
	maxDelay     time.Duration
	retryableErrors map[string]bool
}

/* ErrorType represents error classification */
type ErrorType string

const (
	ErrorTypeTransient ErrorType = "transient" // Retryable
	ErrorTypePermanent ErrorType = "permanent" // Not retryable
	ErrorTypeTimeout   ErrorType = "timeout"   // Retryable with backoff
	ErrorTypeRateLimit ErrorType = "rate_limit" // Retryable with longer delay
)

/* NewErrorHandler creates a new error handler */
func NewErrorHandler(queries *db.Queries) *ErrorHandler {
	retryableErrors := map[string]bool{
		"connection":     true,
		"timeout":        true,
		"rate_limit":     true,
		"temporary":       true,
		"unavailable":    true,
		"network":        true,
	}

	return &ErrorHandler{
		queries:         queries,
		maxRetries:      3,
		baseDelay:       100 * time.Millisecond,
		maxDelay:        30 * time.Second,
		retryableErrors: retryableErrors,
	}
}

/* RetryWithBackoff retries a function with exponential backoff */
func (eh *ErrorHandler) RetryWithBackoff(ctx context.Context, fn func() error) error {
	var lastErr error

	for attempt := 0; attempt < eh.maxRetries; attempt++ {
		err := fn()
		if err == nil {
			return nil
		}

		lastErr = err

		/* Check if error is retryable */
		if !eh.isRetryable(err) {
			return err
		}

		/* Don't retry on last attempt */
		if attempt == eh.maxRetries-1 {
			break
		}

		/* Calculate delay with exponential backoff */
		delay := eh.calculateDelay(attempt, err)

		/* Wait before retry */
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(delay):
			/* Continue to next attempt */
		}

		metrics.InfoWithContext(ctx, "Retrying after error", map[string]interface{}{
			"attempt": attempt + 1,
			"max_retries": eh.maxRetries,
			"delay_ms": delay.Milliseconds(),
			"error": err.Error(),
		})
	}

	return fmt.Errorf("max retries exceeded: last_error=%w", lastErr)
}

/* isRetryable checks if an error is retryable */
/* Uses shared classification logic from utils package */
func (eh *ErrorHandler) isRetryable(err error) bool {
	return utils.IsRetryable(err)
}

/* calculateDelay calculates retry delay with exponential backoff */
func (eh *ErrorHandler) calculateDelay(attempt int, err error) time.Duration {
	/* Base exponential backoff */
	delay := time.Duration(math.Pow(2, float64(attempt))) * eh.baseDelay

	/* Adjust for rate limit errors */
	if contains(err.Error(), "rate_limit") {
		delay *= 2
	}

	/* Cap at max delay */
	if delay > eh.maxDelay {
		delay = eh.maxDelay
	}

	return delay
}

/* ClassifyError classifies an error type */
/* Uses shared classification logic from utils package */
func (eh *ErrorHandler) ClassifyError(err error) ErrorType {
	errorType := utils.ClassifyError(err)
	switch errorType {
	case utils.ErrorTypeTimeout:
		return ErrorTypeTimeout
	case utils.ErrorTypeRateLimit:
		return ErrorTypeRateLimit
	case utils.ErrorTypeRetryable:
		return ErrorTypeTransient
	default:
		return ErrorTypePermanent
	}
}

/* StoreDeadLetter stores failed operation in dead letter queue */
func (eh *ErrorHandler) StoreDeadLetter(ctx context.Context, operation string, payload interface{}, err error) error {
	query := `INSERT INTO neurondb_agent.dead_letter_queue
		(id, operation, payload, error, created_at)
		VALUES (gen_random_uuid(), $1, $2::jsonb, $3, NOW())`

	payloadJSON, marshalErr := json.Marshal(payload)
	if marshalErr != nil {
		return fmt.Errorf("dead letter storage failed: json_marshal_error=true, error=%w", marshalErr)
	}

	/* Handle nil error - convert to empty string */
	errorStr := ""
	if err != nil {
		errorStr = err.Error()
	}

	_, dbErr := eh.queries.DB.ExecContext(ctx, query, operation, payloadJSON, errorStr)
	return dbErr
}

/* contains checks if string contains substring (case-insensitive) */
func contains(s, substr string) bool {
	return strings.Contains(strings.ToLower(s), strings.ToLower(substr))
}

