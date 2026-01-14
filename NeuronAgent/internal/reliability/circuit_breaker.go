/*-------------------------------------------------------------------------
 *
 * circuit_breaker.go
 *    Circuit breaker pattern for preventing cascade failures
 *
 * Implements circuit breaker to prevent calling failing services
 * and allow time for recovery.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/reliability/circuit_breaker.go
 *
 *-------------------------------------------------------------------------
 */

package reliability

import (
	"context"
	"fmt"
	"sync"
	"time"
)

/* CircuitState represents the state of a circuit breaker */
type CircuitState string

const (
	StateClosed   CircuitState = "closed"   /* Normal operation */
	StateOpen     CircuitState = "open"     /* Failing, reject requests */
	StateHalfOpen CircuitState = "half-open" /* Testing if service recovered */
)

/* CircuitBreaker implements circuit breaker pattern */
type CircuitBreaker struct {
	name              string
	failureThreshold  int
	successThreshold  int
	timeout           time.Duration
	currentState      CircuitState
	failureCount      int
	successCount      int
	lastFailureTime   time.Time
	mu                sync.RWMutex
}

/* NewCircuitBreaker creates a new circuit breaker */
func NewCircuitBreaker(name string, failureThreshold, successThreshold int, timeout time.Duration) *CircuitBreaker {
	return &CircuitBreaker{
		name:             name,
		failureThreshold: failureThreshold,
		successThreshold: successThreshold,
		timeout:          timeout,
		currentState:     StateClosed,
	}
}

/* Execute executes a function with circuit breaker protection */
func (cb *CircuitBreaker) Execute(ctx context.Context, fn func() error) error {
	cb.mu.Lock()

	/* Check circuit state */
	state := cb.currentState
	cb.mu.Unlock()

	switch state {
	case StateOpen:
		/* Check if timeout has passed */
		cb.mu.Lock()
		if time.Since(cb.lastFailureTime) >= cb.timeout {
			cb.currentState = StateHalfOpen
			cb.successCount = 0
			state = StateHalfOpen
		} else {
			cb.mu.Unlock()
			return fmt.Errorf("circuit breaker open: service='%s'", cb.name)
		}
		cb.mu.Unlock()

	case StateHalfOpen:
		/* Allow request to test if service recovered */
		break

	case StateClosed:
		/* Normal operation */
		break
	}

	/* Execute function */
	err := fn()

	cb.mu.Lock()
	defer cb.mu.Unlock()

	if err != nil {
		/* Record failure */
		cb.failureCount++
		cb.lastFailureTime = time.Now()

		if cb.currentState == StateHalfOpen {
			/* Test failed, open circuit */
			cb.currentState = StateOpen
			cb.failureCount = 0
		} else if cb.failureCount >= cb.failureThreshold {
			/* Too many failures, open circuit */
			cb.currentState = StateOpen
		}

		return err
	}

	/* Record success */
	cb.successCount++
	cb.failureCount = 0

	if cb.currentState == StateHalfOpen {
		if cb.successCount >= cb.successThreshold {
			/* Service recovered, close circuit */
			cb.currentState = StateClosed
			cb.successCount = 0
		}
	}

	return nil
}

/* GetState returns the current circuit breaker state */
func (cb *CircuitBreaker) GetState() CircuitState {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return cb.currentState
}

/* Reset resets the circuit breaker to closed state */
func (cb *CircuitBreaker) Reset() {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.currentState = StateClosed
	cb.failureCount = 0
	cb.successCount = 0
}



