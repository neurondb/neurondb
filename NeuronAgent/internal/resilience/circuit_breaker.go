/*-------------------------------------------------------------------------
 *
 * circuit_breaker.go
 *    Circuit breaker pattern for external service calls
 *
 * Implements circuit breaker pattern to handle failures gracefully
 * and prevent cascading failures in NeuronAgent.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/resilience/circuit_breaker.go
 *
 *-------------------------------------------------------------------------
 */

package resilience

import (
	"context"
	"errors"
	"sync"
	"time"
)

// CircuitState represents the state of the circuit breaker
type CircuitState int

const (
	StateClosed   CircuitState = iota // Normal operation
	StateOpen                         // Circuit is open, failing fast
	StateHalfOpen                     // Testing if service recovered
)

// CircuitBreaker implements the circuit breaker pattern
type CircuitBreaker struct {
	mu               sync.RWMutex
	state            CircuitState
	failureCount     int
	successCount     int
	lastFailureTime  time.Time
	lastSuccessTime  time.Time
	openDuration     time.Duration
	failureThreshold int
	successThreshold int
	timeout          time.Duration
	onStateChange    func(name string, from, to CircuitState)
	name             string
}

// CircuitBreakerConfig configures a circuit breaker
type CircuitBreakerConfig struct {
	Name             string
	FailureThreshold int           // Number of failures before opening
	SuccessThreshold int           // Number of successes before closing
	OpenDuration     time.Duration // How long to stay open
	Timeout          time.Duration // Timeout for operations
	OnStateChange    func(name string, from, to CircuitState)
}

// NewCircuitBreaker creates a new circuit breaker
func NewCircuitBreaker(config CircuitBreakerConfig) *CircuitBreaker {
	if config.FailureThreshold <= 0 {
		config.FailureThreshold = 5
	}
	if config.SuccessThreshold <= 0 {
		config.SuccessThreshold = 2
	}
	if config.OpenDuration == 0 {
		config.OpenDuration = 60 * time.Second
	}
	if config.Timeout == 0 {
		config.Timeout = 30 * time.Second
	}

	return &CircuitBreaker{
		state:            StateClosed,
		failureThreshold: config.FailureThreshold,
		successThreshold: config.SuccessThreshold,
		openDuration:     config.OpenDuration,
		timeout:          config.Timeout,
		onStateChange:    config.OnStateChange,
		name:             config.Name,
	}
}

// Execute runs a function through the circuit breaker
func (cb *CircuitBreaker) Execute(ctx context.Context, fn func() error) error {
	// Check if we can proceed
	if !cb.canProceed() {
		return errors.New("circuit breaker is open")
	}

	// Create context with timeout
	ctx, cancel := context.WithTimeout(ctx, cb.timeout)
	defer cancel()

	// Execute the function
	err := fn()

	// Record result
	if err != nil {
		cb.recordFailure()
		return err
	}

	cb.recordSuccess()
	return nil
}

// canProceed checks if the circuit breaker allows execution
func (cb *CircuitBreaker) canProceed() bool {
	cb.mu.RLock()
	defer cb.mu.RUnlock()

	switch cb.state {
	case StateClosed:
		return true
	case StateOpen:
		// Check if we should transition to half-open
		if time.Since(cb.lastFailureTime) >= cb.openDuration {
			cb.mu.RUnlock()
			cb.mu.Lock()
			if cb.state == StateOpen && time.Since(cb.lastFailureTime) >= cb.openDuration {
				cb.transitionTo(StateHalfOpen)
			}
			cb.mu.Unlock()
			cb.mu.RLock()
			return cb.state == StateHalfOpen
		}
		return false
	case StateHalfOpen:
		return true
	default:
		return false
	}
}

// recordFailure records a failure and updates state
func (cb *CircuitBreaker) recordFailure() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	if cb.state == StateOpen {
		return /* Already open, avoid redundant updates */
	}
	cb.failureCount++
	cb.lastFailureTime = time.Now()

	switch cb.state {
	case StateClosed:
		if cb.failureCount >= cb.failureThreshold {
			cb.transitionTo(StateOpen)
		}
	case StateHalfOpen:
		// Any failure in half-open state opens the circuit
		cb.transitionTo(StateOpen)
	}
}

// recordSuccess records a success and updates state
func (cb *CircuitBreaker) recordSuccess() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.successCount++
	cb.lastSuccessTime = time.Now()

	switch cb.state {
	case StateHalfOpen:
		if cb.successCount >= cb.successThreshold {
			cb.transitionTo(StateClosed)
			cb.resetCounters()
		}
	case StateClosed:
		// Reset failure count on success
		if cb.failureCount > 0 {
			cb.failureCount = 0
		}
	}
}

// transitionTo transitions the circuit breaker to a new state
func (cb *CircuitBreaker) transitionTo(newState CircuitState) {
	oldState := cb.state
	cb.state = newState

	if cb.onStateChange != nil {
		cb.onStateChange(cb.name, oldState, newState)
	}

	// Reset counters when transitioning
	if newState == StateClosed {
		cb.resetCounters()
	} else if newState == StateHalfOpen {
		cb.successCount = 0
	}
}

// resetCounters resets failure and success counters
func (cb *CircuitBreaker) resetCounters() {
	cb.failureCount = 0
	cb.successCount = 0
}

// GetState returns the current state of the circuit breaker
func (cb *CircuitBreaker) GetState() CircuitState {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return cb.state
}

// GetStats returns statistics about the circuit breaker
func (cb *CircuitBreaker) GetStats() map[string]interface{} {
	cb.mu.RLock()
	defer cb.mu.RUnlock()

	return map[string]interface{}{
		"state":             cb.state,
		"failure_count":     cb.failureCount,
		"success_count":     cb.successCount,
		"last_failure":      cb.lastFailureTime,
		"last_success":      cb.lastSuccessTime,
		"failure_threshold": cb.failureThreshold,
		"success_threshold": cb.successThreshold,
	}
}
