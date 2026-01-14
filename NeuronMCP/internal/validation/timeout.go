/*-------------------------------------------------------------------------
 *
 * timeout.go
 *    Timeout validation for NeuronMCP
 *
 * Provides timeout validation and context deadline checking.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/validation/timeout.go
 *
 *-------------------------------------------------------------------------
 */

package validation

import (
	"context"
	"fmt"
	"time"
)

/* ValidateContextTimeout validates that context has sufficient time remaining */
func ValidateContextTimeout(ctx context.Context, minTimeRemaining time.Duration) error {
	deadline, ok := ctx.Deadline()
	if !ok {
		return nil // No deadline set, assume infinite time
	}
	
	remaining := time.Until(deadline)
	if remaining < minTimeRemaining {
		return fmt.Errorf("context deadline too soon: %v remaining, minimum %v required", remaining, minTimeRemaining)
	}
	
	return nil
}

/* ValidateTimeoutValue validates a timeout duration value */
func ValidateTimeoutValue(timeout time.Duration, min, max time.Duration, fieldName string) error {
	if timeout < min {
		return fmt.Errorf("%s timeout %v is less than minimum %v", fieldName, timeout, min)
	}
	if timeout > max {
		return fmt.Errorf("%s timeout %v exceeds maximum %v", fieldName, timeout, max)
	}
	return nil
}

/* CheckContextDeadline checks if context deadline has been exceeded */
func CheckContextDeadline(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return fmt.Errorf("context deadline exceeded: %w", ctx.Err())
	default:
		return nil
	}
}

/* ValidateContextDeadline validates that context has not expired */
func ValidateContextDeadline(ctx context.Context) error {
	return CheckContextDeadline(ctx)
}

