/*-------------------------------------------------------------------------
 *
 * retry.go
 *    Retry middleware with exponential backoff and circuit breaker
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/middleware/builtin/retry.go
 *
 *-------------------------------------------------------------------------
 */

package builtin

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/middleware"
	"github.com/neurondb/NeuronMCP/internal/reliability"
)

/* RetryConfig holds retry configuration */
type RetryConfig struct {
	Enabled         bool
	MaxRetries      int
	InitialBackoff  time.Duration
	MaxBackoff      time.Duration
	BackoffMultiplier float64
	CircuitBreaker  *CircuitBreakerConfig
}

/* CircuitBreakerConfig holds circuit breaker configuration */
type CircuitBreakerConfig struct {
	Enabled          bool
	FailureThreshold int
	SuccessThreshold int
	Timeout          time.Duration
}

/* CircuitBreakerState represents circuit breaker state */
type CircuitBreakerState int

const (
	CircuitBreakerClosed CircuitBreakerState = iota
	CircuitBreakerOpen
	CircuitBreakerHalfOpen
)

/* CircuitBreaker implements circuit breaker pattern */
type CircuitBreaker struct {
	config          *CircuitBreakerConfig
	state           CircuitBreakerState
	failureCount    int
	successCount    int
	lastFailureTime time.Time
	mu              sync.RWMutex
}

/* NewCircuitBreaker creates a new circuit breaker */
func NewCircuitBreaker(config *CircuitBreakerConfig) *CircuitBreaker {
	return &CircuitBreaker{
		config: config,
		state:  CircuitBreakerClosed,
	}
}

/* Allow checks if request is allowed */
func (cb *CircuitBreaker) Allow() bool {
	cb.mu.RLock()
	defer cb.mu.RUnlock()

	switch cb.state {
	case CircuitBreakerClosed:
		return true
	case CircuitBreakerOpen:
		if time.Since(cb.lastFailureTime) > cb.config.Timeout {
			cb.mu.RUnlock()
			cb.mu.Lock()
			cb.state = CircuitBreakerHalfOpen
			cb.successCount = 0
			cb.mu.Unlock()
			cb.mu.RLock()
			return true
		}
		return false
	case CircuitBreakerHalfOpen:
		return true
	}
	return false
}

/* RecordSuccess records a successful request */
func (cb *CircuitBreaker) RecordSuccess() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	if cb.state == CircuitBreakerHalfOpen {
		cb.successCount++
		if cb.successCount >= cb.config.SuccessThreshold {
			cb.state = CircuitBreakerClosed
			cb.failureCount = 0
		}
	}
}

/* RecordFailure records a failed request */
func (cb *CircuitBreaker) RecordFailure() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.failureCount++
	cb.lastFailureTime = time.Now()

	if cb.state == CircuitBreakerHalfOpen {
		cb.state = CircuitBreakerOpen
		cb.successCount = 0
	} else if cb.failureCount >= cb.config.FailureThreshold {
		cb.state = CircuitBreakerOpen
	}
}

/* RetryMiddleware provides retry logic */
type RetryMiddleware struct {
	config          *RetryConfig
	retryManager    *reliability.RetryManager
	logger          *logging.Logger
	circuitBreakers map[string]*CircuitBreaker
	mu              sync.RWMutex
}

/* NewRetryMiddleware creates a new retry middleware */
func NewRetryMiddleware(config *RetryConfig, logger *logging.Logger) middleware.Middleware {
	var retryManager *reliability.RetryManager
	if config.Enabled {
		retryManager = reliability.NewRetryManager(
			config.MaxRetries,
			config.InitialBackoff,
			config.MaxBackoff,
			config.BackoffMultiplier,
		)
	}
	rm := &RetryMiddleware{
		config:          config,
		retryManager:    retryManager,
		logger:          logger,
		circuitBreakers: make(map[string]*CircuitBreaker),
	}

	if config.CircuitBreaker != nil && config.CircuitBreaker.Enabled {
		/* Initialize circuit breakers */
	}

	return rm
}

/* Name returns the middleware name */
func (m *RetryMiddleware) Name() string {
	return "retry"
}

/* Order returns the middleware order */
func (m *RetryMiddleware) Order() int {
	return 5
}

/* Enabled returns whether the middleware is enabled */
func (m *RetryMiddleware) Enabled() bool {
	return true /* Retry is always enabled if configured */
}

/* Execute handles retry logic */
func (m *RetryMiddleware) Execute(ctx context.Context, req *middleware.MCPRequest, next middleware.Handler) (*middleware.MCPResponse, error) {
	if !m.config.Enabled {
		return next(ctx, req)
	}

	/* Check circuit breaker */
	if m.config.CircuitBreaker != nil && m.config.CircuitBreaker.Enabled {
		key := req.Method
		cb := m.getCircuitBreaker(key)
		if !cb.Allow() {
			return &middleware.MCPResponse{
				Content: []middleware.ContentBlock{
					{Type: "text", Text: "Circuit breaker is open"},
				},
				IsError: true,
			}, nil
		}
	}

	/* Check if tool is idempotent before retrying */
	toolName := ""
	if req.Method == "tools/call" {
		if name, ok := req.Params["name"].(string); ok {
			toolName = name
		}
	}

	/* Only retry if tool is idempotent */
	if toolName != "" && m.retryManager != nil && !m.retryManager.IsIdempotent(toolName) {
		/* Not idempotent, execute once without retry */
		return next(ctx, req)
	}

	/* Retry with exponential backoff */
	var lastErr error
	maxRetries := m.config.MaxRetries
	if m.retryManager != nil {
		maxRetries = m.retryManager.GetMaxRetries()
	}

	for attempt := 0; attempt <= maxRetries; attempt++ {
		backoff := m.config.InitialBackoff
		if m.retryManager != nil && attempt > 0 {
			backoff = m.retryManager.GetBackoff(attempt - 1)
		}
		if attempt > 0 {
			/* Wait before retry */
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(backoff):
			}
		}

		resp, err := next(ctx, req)
		if err == nil && !resp.IsError {
			/* Success */
			if m.config.CircuitBreaker != nil && m.config.CircuitBreaker.Enabled {
				key := req.Method
				cb := m.getCircuitBreaker(key)
				cb.RecordSuccess()
			}
			return resp, nil
		}

		lastErr = err
		if resp != nil && resp.IsError {
			lastErr = fmt.Errorf("request failed: %v", resp.Content)
		}

		/* Check if error should be retried */
		if m.retryManager != nil && toolName != "" {
			if !m.retryManager.ShouldRetry(toolName, lastErr) {
				/* Error is not retryable, return immediately */
				if m.config.CircuitBreaker != nil && m.config.CircuitBreaker.Enabled {
					key := req.Method
					cb := m.getCircuitBreaker(key)
					cb.RecordFailure()
				}
				return resp, err
			}
		}

		/* Record failure for circuit breaker */
		if m.config.CircuitBreaker != nil && m.config.CircuitBreaker.Enabled {
			key := req.Method
			cb := m.getCircuitBreaker(key)
			cb.RecordFailure()
		}
	}

	return &middleware.MCPResponse{
		Content: []middleware.ContentBlock{
			{Type: "text", Text: fmt.Sprintf("Request failed after %d retries: %v", maxRetries, lastErr)},
		},
		IsError: true,
		Metadata: map[string]interface{}{
			"error_code": "RETRY_EXHAUSTED",
			"retries":    maxRetries,
		},
	}, nil
}

/* getCircuitBreaker gets or creates a circuit breaker */
func (m *RetryMiddleware) getCircuitBreaker(key string) *CircuitBreaker {
	m.mu.RLock()
	cb, exists := m.circuitBreakers[key]
	m.mu.RUnlock()

	if !exists {
		m.mu.Lock()
		cb, exists = m.circuitBreakers[key]
		if !exists {
			cb = NewCircuitBreaker(m.config.CircuitBreaker)
			m.circuitBreakers[key] = cb
		}
		m.mu.Unlock()
	}

	return cb
}

