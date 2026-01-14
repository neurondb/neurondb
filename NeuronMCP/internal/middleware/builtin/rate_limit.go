/*-------------------------------------------------------------------------
 *
 * rate_limit.go
 *    Rate limiting middleware for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/middleware/builtin/rate_limit.go
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
)

/* RateLimitConfig holds rate limiting configuration */
type RateLimitConfig struct {
	Enabled         bool
	RequestsPerMin  int
	RequestsPerHour int
	BurstSize       int
	PerUser         bool
	PerTool         bool
	PerEndpoint     bool
	EndpointLimits  map[string]int /* Custom limits per endpoint (requests per minute) */
	ToolLimits      map[string]int /* Custom limits per tool (requests per minute) */
}

/* TokenBucket implements a token bucket rate limiter */
type TokenBucket struct {
	capacity     int
	tokens       int
	refillRate   int  /* tokens per second */
	lastRefill   time.Time
	mu           sync.Mutex
}

/* NewTokenBucket creates a new token bucket */
func NewTokenBucket(capacity, refillRate int) *TokenBucket {
	return &TokenBucket{
		capacity:   capacity,
		tokens:     capacity,
		refillRate: refillRate,
		lastRefill: time.Now(),
	}
}

/* Allow checks if a request is allowed */
func (tb *TokenBucket) Allow() bool {
	tb.mu.Lock()
	defer tb.mu.Unlock()

	/* Refill tokens */
	now := time.Now()
	elapsed := now.Sub(tb.lastRefill)
	tokensToAdd := int(elapsed.Seconds()) * tb.refillRate
	if tokensToAdd > 0 {
		tb.tokens = min(tb.capacity, tb.tokens+tokensToAdd)
		tb.lastRefill = now
	}

	/* Check if we have tokens */
	if tb.tokens > 0 {
		tb.tokens--
		return true
	}

	return false
}

/* RateLimitMiddleware provides rate limiting */
type RateLimitMiddleware struct {
	config        *RateLimitConfig
	logger        *logging.Logger
	globalBucket  *TokenBucket
	userBuckets     map[string]*TokenBucket
	toolBuckets     map[string]*TokenBucket
	endpointBuckets map[string]*TokenBucket
	mu            sync.RWMutex
}

/* NewRateLimitMiddleware creates a new rate limiting middleware */
func NewRateLimitMiddleware(config *RateLimitConfig, logger *logging.Logger) middleware.Middleware {
	rl := &RateLimitMiddleware{
		config:          config,
		logger:          logger,
		userBuckets:     make(map[string]*TokenBucket),
		toolBuckets:     make(map[string]*TokenBucket),
		endpointBuckets: make(map[string]*TokenBucket),
	}

	if config.Enabled {
		/* Create global bucket */
		requestsPerSec := config.RequestsPerMin / 60
		if requestsPerSec < 1 {
			requestsPerSec = 1
		}
		rl.globalBucket = NewTokenBucket(config.BurstSize, requestsPerSec)
	}

	return rl
}

/* Name returns the middleware name */
func (m *RateLimitMiddleware) Name() string {
	return "rate_limit"
}

/* Order returns the middleware order */
func (m *RateLimitMiddleware) Order() int {
	return 10
}

/* Enabled returns whether the middleware is enabled */
func (m *RateLimitMiddleware) Enabled() bool {
	return m.config.Enabled
}

/* Execute handles rate limiting */
func (m *RateLimitMiddleware) Execute(ctx context.Context, req *middleware.MCPRequest, next middleware.Handler) (*middleware.MCPResponse, error) {
	if !m.config.Enabled {
		return next(ctx, req)
	}

	/* Check global rate limit */
	if !m.globalBucket.Allow() {
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: "Rate limit exceeded"},
			},
			IsError: true,
		}, nil
	}

	/* Check per-user rate limit */
	if m.config.PerUser {
		user := m.getUser(ctx)
		if user != "" {
			bucket := m.getUserBucket(user)
			if !bucket.Allow() {
				return &middleware.MCPResponse{
					Content: []middleware.ContentBlock{
						{Type: "text", Text: fmt.Sprintf("Rate limit exceeded for user: %s", user)},
					},
					IsError: true,
				}, nil
			}
		}
	}

	/* Check per-tool rate limit */
	if m.config.PerTool {
		if toolName := m.getToolName(req); toolName != "" {
			bucket := m.getToolBucket(toolName)
			if !bucket.Allow() {
				return &middleware.MCPResponse{
					Content: []middleware.ContentBlock{
						{Type: "text", Text: fmt.Sprintf("Rate limit exceeded for tool: %s", toolName)},
					},
					IsError: true,
				}, nil
			}
		}
	}

	/* Check per-endpoint rate limit */
	if m.config.PerEndpoint {
		endpoint := req.Method
		bucket := m.getEndpointBucket(endpoint)
		if !bucket.Allow() {
			return &middleware.MCPResponse{
				Content: []middleware.ContentBlock{
					{Type: "text", Text: fmt.Sprintf("Rate limit exceeded for endpoint: %s", endpoint)},
				},
				IsError: true,
			}, nil
		}
	}

	return next(ctx, req)
}

/* getUser gets user from context */
func (m *RateLimitMiddleware) getUser(ctx context.Context) string {
	if user, ok := ctx.Value("user").(string); ok {
		return user
	}
	return ""
}

/* getToolName gets tool name from request */
func (m *RateLimitMiddleware) getToolName(req *middleware.MCPRequest) string {
	if req.Method == "tools/call" {
		if req.Params != nil {
			if name, ok := req.Params["name"].(string); ok {
				return name
			}
		}
	}
	return ""
}

/* getUserBucket gets or creates a user bucket */
func (m *RateLimitMiddleware) getUserBucket(user string) *TokenBucket {
	m.mu.RLock()
	bucket, exists := m.userBuckets[user]
	m.mu.RUnlock()

	if !exists {
		m.mu.Lock()
		bucket, exists = m.userBuckets[user]
		if !exists {
			requestsPerSec := m.config.RequestsPerMin / 60
			if requestsPerSec < 1 {
				requestsPerSec = 1
			}
			bucket = NewTokenBucket(m.config.BurstSize, requestsPerSec)
			m.userBuckets[user] = bucket
		}
		m.mu.Unlock()
	}

	return bucket
}

/* getToolBucket gets or creates a tool bucket */
func (m *RateLimitMiddleware) getToolBucket(tool string) *TokenBucket {
	m.mu.RLock()
	bucket, exists := m.toolBuckets[tool]
	m.mu.RUnlock()

	if !exists {
		m.mu.Lock()
		bucket, exists = m.toolBuckets[tool]
		if !exists {
			/* Use custom limit if configured, otherwise use default */
			requestsPerMin := m.config.RequestsPerMin
			if m.config.ToolLimits != nil {
				if customLimit, ok := m.config.ToolLimits[tool]; ok {
					requestsPerMin = customLimit
				}
			}
			requestsPerSec := requestsPerMin / 60
			if requestsPerSec < 1 {
				requestsPerSec = 1
			}
			bucket = NewTokenBucket(m.config.BurstSize, requestsPerSec)
			m.toolBuckets[tool] = bucket
		}
		m.mu.Unlock()
	}

	return bucket
}

/* getEndpointBucket gets or creates an endpoint bucket */
func (m *RateLimitMiddleware) getEndpointBucket(endpoint string) *TokenBucket {
	m.mu.RLock()
	bucket, exists := m.endpointBuckets[endpoint]
	m.mu.RUnlock()

	if !exists {
		m.mu.Lock()
		bucket, exists = m.endpointBuckets[endpoint]
		if !exists {
			/* Use custom limit if configured, otherwise use default */
			requestsPerMin := m.config.RequestsPerMin
			if m.config.EndpointLimits != nil {
				if customLimit, ok := m.config.EndpointLimits[endpoint]; ok {
					requestsPerMin = customLimit
				}
			}
			requestsPerSec := requestsPerMin / 60
			if requestsPerSec < 1 {
				requestsPerSec = 1
			}
			bucket = NewTokenBucket(m.config.BurstSize, requestsPerSec)
			m.endpointBuckets[endpoint] = bucket
		}
		m.mu.Unlock()
	}

	return bucket
}

/* min returns the minimum of two integers */
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

