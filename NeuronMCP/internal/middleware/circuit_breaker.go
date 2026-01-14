/*-------------------------------------------------------------------------
 *
 * circuit_breaker.go
 *    Circuit breaker middleware for fault tolerance and resilience
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/middleware/circuit_breaker.go
 *
 *-------------------------------------------------------------------------
 */

package middleware

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/neurondb/NeuronMCP/internal/logging"
)

/* CircuitBreakerState represents the state of the circuit breaker */
type CircuitBreakerState int

const (
	StateClosed CircuitBreakerState = iota
	StateOpen
	StateHalfOpen
)

func (s CircuitBreakerState) String() string {
	switch s {
	case StateClosed:
		return "closed"
	case StateOpen:
		return "open"
	case StateHalfOpen:
		return "half-open"
	default:
		return "unknown"
	}
}

/* CircuitBreakerStats tracks circuit breaker statistics */
type CircuitBreakerStats struct {
	TotalRequests     int64
	SuccessfulRequests int64
	FailedRequests    int64
	RejectedRequests  int64
	StateChanges      int64
	LastStateChange   time.Time
	CurrentState      CircuitBreakerState
	ErrorRate         float64
	LastError         string
	LastErrorTime     time.Time
}

/* CircuitBreakerMiddleware implements circuit breaker pattern for resilience */
type CircuitBreakerMiddleware struct {
	logger              *logging.Logger
	mu                  sync.RWMutex
	state               CircuitBreakerState
	failureThreshold    int
	successThreshold    int
	timeout             time.Duration
	failureCount        int
	successCount        int
	lastFailureTime     time.Time
	lastStateChange     time.Time
	stats               *CircuitBreakerStats
	perToolBreakers     map[string]*CircuitBreakerState
	perToolStats        map[string]*CircuitBreakerStats
	enablePerToolBreaker bool
}

/* CircuitBreakerConfig configures circuit breaker behavior */
type CircuitBreakerConfig struct {
	FailureThreshold     int           // Number of consecutive failures to open circuit
	SuccessThreshold     int           // Number of consecutive successes to close circuit
	Timeout              time.Duration // Time to wait before attempting half-open
	EnablePerToolBreaker bool          // Enable per-tool circuit breakers
}

/* NewCircuitBreakerMiddleware creates a new circuit breaker middleware */
func NewCircuitBreakerMiddleware(logger *logging.Logger, config CircuitBreakerConfig) *CircuitBreakerMiddleware {
	if config.FailureThreshold == 0 {
		config.FailureThreshold = 5
	}
	if config.SuccessThreshold == 0 {
		config.SuccessThreshold = 2
	}
	if config.Timeout == 0 {
		config.Timeout = 60 * time.Second
	}

	return &CircuitBreakerMiddleware{
		logger:               logger,
		state:                StateClosed,
		failureThreshold:     config.FailureThreshold,
		successThreshold:     config.SuccessThreshold,
		timeout:              config.Timeout,
		lastStateChange:      time.Now(),
		stats:                &CircuitBreakerStats{CurrentState: StateClosed},
		perToolBreakers:      make(map[string]*CircuitBreakerState),
		perToolStats:         make(map[string]*CircuitBreakerStats),
		enablePerToolBreaker: config.EnablePerToolBreaker,
	}
}

/* Execute enforces circuit breaker pattern */
func (m *CircuitBreakerMiddleware) Execute(ctx context.Context, params map[string]interface{}, next MiddlewareFunc) (interface{}, error) {
	toolName := "global"
	if name, ok := params["_tool_name"].(string); ok {
		toolName = name
	}

	/* Check circuit breaker state */
	if !m.canExecute(toolName) {
		m.recordRejection(toolName)
		return nil, fmt.Errorf("circuit breaker is open for tool '%s': too many failures, retrying after %v", toolName, m.timeout)
	}

	/* Execute request */
	result, err := next(ctx, params)

	/* Record success or failure */
	if err != nil {
		m.recordFailure(toolName, err)
		return result, err
	}

	m.recordSuccess(toolName)
	return result, nil
}

/* canExecute checks if request can be executed based on circuit breaker state */
func (m *CircuitBreakerMiddleware) canExecute(toolName string) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()

	currentState := m.state
	if m.enablePerToolBreaker {
		if state, exists := m.perToolBreakers[toolName]; exists {
			currentState = *state
		}
	}

	switch currentState {
	case StateClosed:
		return true
	case StateOpen:
		/* Check if timeout has elapsed */
		if time.Since(m.lastStateChange) >= m.timeout {
			/* Transition to half-open */
			m.mu.RUnlock()
			m.mu.Lock()
			m.transitionTo(StateHalfOpen, toolName)
			m.mu.Unlock()
			m.mu.RLock()
			return true
		}
		return false
	case StateHalfOpen:
		return true
	default:
		return false
	}
}

/* recordSuccess records a successful request */
func (m *CircuitBreakerMiddleware) recordSuccess(toolName string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.stats.TotalRequests++
	m.stats.SuccessfulRequests++
	m.updateErrorRate()

	if m.enablePerToolBreaker {
		if _, exists := m.perToolStats[toolName]; !exists {
			m.perToolStats[toolName] = &CircuitBreakerStats{CurrentState: StateClosed}
		}
		m.perToolStats[toolName].TotalRequests++
		m.perToolStats[toolName].SuccessfulRequests++
	}

	currentState := m.state
	if m.enablePerToolBreaker {
		if state, exists := m.perToolBreakers[toolName]; exists {
			currentState = *state
		}
	}

	if currentState == StateHalfOpen {
		m.successCount++
		if m.successCount >= m.successThreshold {
			m.transitionTo(StateClosed, toolName)
			m.successCount = 0
			m.failureCount = 0
		}
	} else {
		m.failureCount = 0 // Reset failure count on success
	}
}

/* recordFailure records a failed request */
func (m *CircuitBreakerMiddleware) recordFailure(toolName string, err error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.stats.TotalRequests++
	m.stats.FailedRequests++
	m.stats.LastError = err.Error()
	m.stats.LastErrorTime = time.Now()
	m.updateErrorRate()

	if m.enablePerToolBreaker {
		if _, exists := m.perToolStats[toolName]; !exists {
			m.perToolStats[toolName] = &CircuitBreakerStats{CurrentState: StateClosed}
		}
		m.perToolStats[toolName].TotalRequests++
		m.perToolStats[toolName].FailedRequests++
		m.perToolStats[toolName].LastError = err.Error()
		m.perToolStats[toolName].LastErrorTime = time.Now()
	}

	m.failureCount++
	m.successCount = 0
	m.lastFailureTime = time.Now()

	currentState := m.state
	if m.enablePerToolBreaker {
		if state, exists := m.perToolBreakers[toolName]; exists {
			currentState = *state
		}
	}

	if currentState == StateHalfOpen {
		/* Immediately open on failure in half-open state */
		m.transitionTo(StateOpen, toolName)
		m.failureCount = 0
	} else if m.failureCount >= m.failureThreshold {
		m.transitionTo(StateOpen, toolName)
	}
}

/* recordRejection records a rejected request */
func (m *CircuitBreakerMiddleware) recordRejection(toolName string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.stats.TotalRequests++
	m.stats.RejectedRequests++

	if m.enablePerToolBreaker {
		if _, exists := m.perToolStats[toolName]; !exists {
			m.perToolStats[toolName] = &CircuitBreakerStats{CurrentState: StateClosed}
		}
		m.perToolStats[toolName].TotalRequests++
		m.perToolStats[toolName].RejectedRequests++
	}
}

/* transitionTo transitions circuit breaker to a new state */
func (m *CircuitBreakerMiddleware) transitionTo(newState CircuitBreakerState, toolName string) {
	oldState := m.state
	if m.enablePerToolBreaker {
		if state, exists := m.perToolBreakers[toolName]; exists {
			oldState = *state
		}
	}

	if oldState == newState {
		return
	}

	m.logger.Info("Circuit breaker state change", map[string]interface{}{
		"tool":      toolName,
		"old_state": oldState.String(),
		"new_state": newState.String(),
		"failures":  m.failureCount,
		"successes": m.successCount,
	})

	if m.enablePerToolBreaker {
		m.perToolBreakers[toolName] = &newState
		if stats, exists := m.perToolStats[toolName]; exists {
			stats.CurrentState = newState
			stats.StateChanges++
			stats.LastStateChange = time.Now()
		}
	} else {
		m.state = newState
	}

	m.stats.StateChanges++
	m.stats.LastStateChange = time.Now()
	m.stats.CurrentState = newState
	m.lastStateChange = time.Now()
}

/* updateErrorRate calculates current error rate */
func (m *CircuitBreakerMiddleware) updateErrorRate() {
	if m.stats.TotalRequests > 0 {
		m.stats.ErrorRate = float64(m.stats.FailedRequests) / float64(m.stats.TotalRequests)
	}
}

/* GetStats returns circuit breaker statistics */
func (m *CircuitBreakerMiddleware) GetStats() *CircuitBreakerStats {
	m.mu.RLock()
	defer m.mu.RUnlock()

	statsCopy := *m.stats
	return &statsCopy
}

/* GetToolStats returns statistics for a specific tool */
func (m *CircuitBreakerMiddleware) GetToolStats(toolName string) *CircuitBreakerStats {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if stats, exists := m.perToolStats[toolName]; exists {
		statsCopy := *stats
		return &statsCopy
	}
	return nil
}

/* GetAllToolStats returns statistics for all tools */
func (m *CircuitBreakerMiddleware) GetAllToolStats() map[string]*CircuitBreakerStats {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make(map[string]*CircuitBreakerStats)
	for toolName, stats := range m.perToolStats {
		statsCopy := *stats
		result[toolName] = &statsCopy
	}
	return result
}

/* Reset resets the circuit breaker to closed state */
func (m *CircuitBreakerMiddleware) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.state = StateClosed
	m.failureCount = 0
	m.successCount = 0
	m.perToolBreakers = make(map[string]*CircuitBreakerState)

	m.logger.Info("Circuit breaker reset to closed state", nil)
}

/* Name returns the middleware name */
func (m *CircuitBreakerMiddleware) Name() string {
	return "circuit_breaker"
}



