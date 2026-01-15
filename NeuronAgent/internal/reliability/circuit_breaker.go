/*-------------------------------------------------------------------------
 *
 * circuit_breaker.go
 *    Circuit breaker pattern for resilience
 *
 * Provides circuit breakers for LLM calls, tool execution, and database
 * operations with automatic fallback strategies.
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

	"github.com/neurondb/NeuronAgent/internal/metrics"
)

/* CircuitBreaker implements the circuit breaker pattern */
type CircuitBreaker struct {
	name          string
	maxFailures   int
	resetTimeout  time.Duration
	state         CircuitState
	failureCount  int
	lastFailure   time.Time
	mu            sync.RWMutex
	onStateChange func(name string, from, to CircuitState)
}

/* CircuitState represents circuit breaker state */
type CircuitState string

const (
	StateClosed   CircuitState = "closed"   // Normal operation
	StateOpen     CircuitState = "open"     // Failing, reject requests
	StateHalfOpen CircuitState = "half_open" // Testing if service recovered
)

/* NewCircuitBreaker creates a new circuit breaker */
func NewCircuitBreaker(name string, maxFailures int, resetTimeout time.Duration) *CircuitBreaker {
	return &CircuitBreaker{
		name:         name,
		maxFailures:  maxFailures,
		resetTimeout: resetTimeout,
		state:        StateClosed,
		failureCount: 0,
	}
}

/* Execute executes a function with circuit breaker protection */
func (cb *CircuitBreaker) Execute(ctx context.Context, fn func() error) error {
	cb.mu.Lock()
	state := cb.state
	cb.mu.Unlock()

	/* Check if circuit is open */
	if state == StateOpen {
		/* Check if reset timeout has passed */
		cb.mu.Lock()
		if time.Since(cb.lastFailure) >= cb.resetTimeout {
			cb.state = StateHalfOpen
			cb.failureCount = 0
			state = StateHalfOpen
			cb.notifyStateChange(StateOpen, StateHalfOpen)
		}
		cb.mu.Unlock()

		if state == StateOpen {
			return fmt.Errorf("circuit breaker open: service=%s", cb.name)
		}
	}

	/* Execute function */
	err := fn()

	cb.mu.Lock()
	defer cb.mu.Unlock()

	if err != nil {
		cb.failureCount++
		cb.lastFailure = time.Now()

		if cb.failureCount >= cb.maxFailures {
			if cb.state != StateOpen {
				cb.notifyStateChange(cb.state, StateOpen)
			}
			cb.state = StateOpen
		} else if cb.state == StateHalfOpen {
			/* Failed in half-open state, go back to open */
			cb.notifyStateChange(StateHalfOpen, StateOpen)
			cb.state = StateOpen
		}
	} else {
		/* Success - reset failure count */
		if cb.state == StateHalfOpen {
			/* Success in half-open state, close the circuit */
			cb.notifyStateChange(StateHalfOpen, StateClosed)
			cb.state = StateClosed
		}
		cb.failureCount = 0
	}

	return err
}

/* notifyStateChange notifies about state change */
func (cb *CircuitBreaker) notifyStateChange(from, to CircuitState) {
	if cb.onStateChange != nil {
		cb.onStateChange(cb.name, from, to)
	}

	metrics.InfoWithContext(context.Background(), "Circuit breaker state changed", map[string]interface{}{
		"circuit": cb.name,
		"from":    string(from),
		"to":      string(to),
	})
}

/* GetState returns current circuit breaker state */
func (cb *CircuitBreaker) GetState() CircuitState {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return cb.state
}

/* SetStateChangeCallback sets callback for state changes */
func (cb *CircuitBreaker) SetStateChangeCallback(callback func(name string, from, to CircuitState)) {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.onStateChange = callback
}

/* CircuitBreakerManager manages multiple circuit breakers */
type CircuitBreakerManager struct {
	breakers map[string]*CircuitBreaker
	mu       sync.RWMutex
}

/* NewCircuitBreakerManager creates a new circuit breaker manager */
func NewCircuitBreakerManager() *CircuitBreakerManager {
	return &CircuitBreakerManager{
		breakers: make(map[string]*CircuitBreaker),
	}
}

/* GetOrCreate gets or creates a circuit breaker */
func (cbm *CircuitBreakerManager) GetOrCreate(name string, maxFailures int, resetTimeout time.Duration) *CircuitBreaker {
	cbm.mu.Lock()
	defer cbm.mu.Unlock()

	if breaker, exists := cbm.breakers[name]; exists {
		return breaker
	}

	breaker := NewCircuitBreaker(name, maxFailures, resetTimeout)
	cbm.breakers[name] = breaker
	return breaker
}

/* Get gets a circuit breaker */
func (cbm *CircuitBreakerManager) Get(name string) (*CircuitBreaker, bool) {
	cbm.mu.RLock()
	defer cbm.mu.RUnlock()

	breaker, exists := cbm.breakers[name]
	return breaker, exists
}
