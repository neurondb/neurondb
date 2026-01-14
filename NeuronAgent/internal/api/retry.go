/*-------------------------------------------------------------------------
 *
 * retry.go
 *    Retry logic utilities for API handlers
 *
 * Provides retry mechanisms for transient failures with exponential
 * backoff and configurable retry policies.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/api/retry.go
 *
 *-------------------------------------------------------------------------
 */

package api

import (
	"context"
	"fmt"
	"strings"
	"time"
)

/* RetryConfig defines retry configuration */
type RetryConfig struct {
	MaxAttempts int
	InitialDelay time.Duration
	MaxDelay     time.Duration
	Multiplier   float64
	IsRetryable  func(error) bool
}

/* DefaultRetryConfig returns default retry configuration */
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxAttempts:  3,
		InitialDelay: 100 * time.Millisecond,
		MaxDelay:     5 * time.Second,
		Multiplier:   2.0,
		IsRetryable:  IsRetryableError,
	}
}

/* IsRetryableError checks if an error is retryable */
func IsRetryableError(err error) bool {
	if err == nil {
		return false
	}
	
	errStr := err.Error()
	
	/* Check for transient errors */
	retryablePatterns := []string{
		"timeout",
		"connection",
		"temporary",
		"network",
		"503",
		"502",
		"504",
		"429", /* Rate limit - might be retryable */
	}
	
	for _, pattern := range retryablePatterns {
		if strings.Contains(strings.ToLower(errStr), pattern) {
			return true
		}
	}
	
	return false
}

/* Retry executes a function with retry logic */
func Retry(ctx context.Context, config RetryConfig, fn func() error) error {
	var lastErr error
	delay := config.InitialDelay
	
	for attempt := 0; attempt < config.MaxAttempts; attempt++ {
		/* Check context cancellation */
		if ctx.Err() != nil {
			return ctx.Err()
		}
		
		/* Execute function */
		err := fn()
		if err == nil {
			return nil /* Success */
		}
		
		lastErr = err
		
		/* Check if error is retryable */
		if !config.IsRetryable(err) {
			return err /* Not retryable, return immediately */
		}
		
		/* Don't sleep after last attempt */
		if attempt < config.MaxAttempts-1 {
			/* Wait before retry */
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(delay):
				/* Exponential backoff */
				delay = time.Duration(float64(delay) * config.Multiplier)
				if delay > config.MaxDelay {
					delay = config.MaxDelay
				}
			}
		}
	}
	
	return fmt.Errorf("operation failed after %d attempts: %w", config.MaxAttempts, lastErr)
}

/* RetryWithResult executes a function with retry logic and returns result */
func RetryWithResult[T any](ctx context.Context, config RetryConfig, fn func() (T, error)) (T, error) {
	var zero T
	var lastErr error
	delay := config.InitialDelay
	
	for attempt := 0; attempt < config.MaxAttempts; attempt++ {
		/* Check context cancellation */
		if ctx.Err() != nil {
			return zero, ctx.Err()
		}
		
		/* Execute function */
		result, err := fn()
		if err == nil {
			return result, nil /* Success */
		}
		
		lastErr = err
		
		/* Check if error is retryable */
		if !config.IsRetryable(err) {
			return zero, err /* Not retryable, return immediately */
		}
		
		/* Don't sleep after last attempt */
		if attempt < config.MaxAttempts-1 {
			/* Wait before retry */
			select {
			case <-ctx.Done():
				return zero, ctx.Err()
			case <-time.After(delay):
				/* Exponential backoff */
				delay = time.Duration(float64(delay) * config.Multiplier)
				if delay > config.MaxDelay {
					delay = config.MaxDelay
				}
			}
		}
	}
	
	return zero, fmt.Errorf("operation failed after %d attempts: %w", config.MaxAttempts, lastErr)
}

