/*-------------------------------------------------------------------------
 *
 * rate_limiter.go
 *    Adaptive rate limiting middleware with token bucket algorithm
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/middleware/rate_limiter.go
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

/* TokenBucket implements the token bucket algorithm */
type TokenBucket struct {
	tokens         float64
	capacity       float64
	refillRate     float64
	lastRefillTime time.Time
	mu             sync.Mutex
}

/* NewTokenBucket creates a new token bucket */
func NewTokenBucket(capacity, refillRate float64) *TokenBucket {
	return &TokenBucket{
		tokens:         capacity,
		capacity:       capacity,
		refillRate:     refillRate,
		lastRefillTime: time.Now(),
	}
}

/* TryConsume attempts to consume tokens */
func (tb *TokenBucket) TryConsume(tokens float64) bool {
	tb.mu.Lock()
	defer tb.mu.Unlock()

	/* Refill tokens based on elapsed time */
	now := time.Now()
	elapsed := now.Sub(tb.lastRefillTime).Seconds()
	tb.tokens = min(tb.capacity, tb.tokens+elapsed*tb.refillRate)
	tb.lastRefillTime = now

	/* Try to consume tokens */
	if tb.tokens >= tokens {
		tb.tokens -= tokens
		return true
	}
	return false
}

/* GetTokens returns current token count */
func (tb *TokenBucket) GetTokens() float64 {
	tb.mu.Lock()
	defer tb.mu.Unlock()
	return tb.tokens
}

func min(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

/* RateLimiterMiddleware implements adaptive rate limiting */
type RateLimiterMiddleware struct {
	logger           *logging.Logger
	globalBucket     *TokenBucket
	userBuckets      map[string]*TokenBucket
	toolBuckets      map[string]*TokenBucket
	mu               sync.RWMutex
	stats            *RateLimiterStats
	enableAdaptive   bool
	enablePerUser    bool
	enablePerTool    bool
	userCapacity     float64
	userRefillRate   float64
	toolCapacity     float64
	toolRefillRate   float64
}

/* RateLimiterStats tracks rate limiter statistics */
type RateLimiterStats struct {
	TotalRequests     int64
	AcceptedRequests  int64
	RejectedRequests  int64
	LastRejectionTime time.Time
	CurrentLoad       float64
}

/* RateLimiterConfig configures rate limiter behavior */
type RateLimiterConfig struct {
	GlobalCapacity   float64 // Global token bucket capacity
	GlobalRefillRate float64 // Global tokens per second
	UserCapacity     float64 // Per-user token bucket capacity
	UserRefillRate   float64 // Per-user tokens per second
	ToolCapacity     float64 // Per-tool token bucket capacity
	ToolRefillRate   float64 // Per-tool tokens per second
	EnableAdaptive   bool    // Enable adaptive rate limiting
	EnablePerUser    bool    // Enable per-user rate limiting
	EnablePerTool    bool    // Enable per-tool rate limiting
}

/* NewRateLimiterMiddleware creates a new rate limiter middleware */
func NewRateLimiterMiddleware(logger *logging.Logger, config RateLimiterConfig) *RateLimiterMiddleware {
	if config.GlobalCapacity == 0 {
		config.GlobalCapacity = 1000
	}
	if config.GlobalRefillRate == 0 {
		config.GlobalRefillRate = 100 // 100 requests per second
	}
	if config.UserCapacity == 0 {
		config.UserCapacity = 100
	}
	if config.UserRefillRate == 0 {
		config.UserRefillRate = 10 // 10 requests per second per user
	}
	if config.ToolCapacity == 0 {
		config.ToolCapacity = 50
	}
	if config.ToolRefillRate == 0 {
		config.ToolRefillRate = 5 // 5 requests per second per tool
	}

	return &RateLimiterMiddleware{
		logger:         logger,
		globalBucket:   NewTokenBucket(config.GlobalCapacity, config.GlobalRefillRate),
		userBuckets:    make(map[string]*TokenBucket),
		toolBuckets:    make(map[string]*TokenBucket),
		stats:          &RateLimiterStats{},
		enableAdaptive: config.EnableAdaptive,
		enablePerUser:  config.EnablePerUser,
		enablePerTool:  config.EnablePerTool,
		userCapacity:   config.UserCapacity,
		userRefillRate: config.UserRefillRate,
		toolCapacity:   config.ToolCapacity,
		toolRefillRate: config.ToolRefillRate,
	}
}

/* Execute enforces rate limiting */
func (m *RateLimiterMiddleware) Execute(ctx context.Context, params map[string]interface{}, next MiddlewareFunc) (interface{}, error) {
	m.mu.Lock()
	m.stats.TotalRequests++
	m.mu.Unlock()

	/* Calculate cost (can be based on tool complexity) */
	cost := m.calculateCost(params)

	/* Check global rate limit */
	if !m.globalBucket.TryConsume(cost) {
		m.recordRejection()
		return nil, fmt.Errorf("rate limit exceeded: global limit reached, please try again later")
	}

	/* Check per-user rate limit */
	if m.enablePerUser {
		if userID, ok := params["_user_id"].(string); ok && userID != "" {
			userBucket := m.getUserBucket(userID)
			if !userBucket.TryConsume(cost) {
				m.recordRejection()
				return nil, fmt.Errorf("rate limit exceeded: user '%s' has exceeded their rate limit", userID)
			}
		}
	}

	/* Check per-tool rate limit */
	if m.enablePerTool {
		if toolName, ok := params["_tool_name"].(string); ok && toolName != "" {
			toolBucket := m.getToolBucket(toolName)
			if !toolBucket.TryConsume(cost) {
				m.recordRejection()
				return nil, fmt.Errorf("rate limit exceeded: tool '%s' is currently rate limited", toolName)
			}
		}
	}

	/* Execute request */
	result, err := next(ctx, params)

	m.mu.Lock()
	m.stats.AcceptedRequests++
	m.stats.CurrentLoad = float64(m.stats.AcceptedRequests) / float64(m.stats.TotalRequests)
	m.mu.Unlock()

	return result, err
}

/* calculateCost calculates request cost based on complexity */
func (m *RateLimiterMiddleware) calculateCost(params map[string]interface{}) float64 {
	if !m.enableAdaptive {
		return 1.0
	}

	cost := 1.0
	toolName := ""
	if name, ok := params["_tool_name"].(string); ok {
		toolName = name
	}

	/* Expensive operations cost more tokens */
	switch toolName {
	case "load_dataset":
		cost = 50.0
	case "train_model":
		cost = 100.0
	case "create_hnsw_index":
		cost = 20.0
	case "vector_search":
		if limit, ok := params["limit"].(float64); ok && limit > 100 {
			cost = 5.0
		} else {
			cost = 2.0
		}
	case "generate_embeddings":
		if texts, ok := params["texts"].([]interface{}); ok {
			cost = float64(len(texts)) * 0.5
		} else {
			cost = 3.0
		}
	case "hybrid_search":
		cost = 3.0
	case "cluster_data":
		cost = 10.0
	default:
		cost = 1.0
	}

	/* Apply batch size multiplier */
	if batchSize, ok := params["batch_size"].(float64); ok && batchSize > 100 {
		cost *= (batchSize / 100.0)
	}

	return cost
}

/* getUserBucket gets or creates a user token bucket */
func (m *RateLimiterMiddleware) getUserBucket(userID string) *TokenBucket {
	m.mu.RLock()
	bucket, exists := m.userBuckets[userID]
	m.mu.RUnlock()

	if !exists {
		m.mu.Lock()
		bucket, exists = m.userBuckets[userID]
		if !exists {
			bucket = NewTokenBucket(m.userCapacity, m.userRefillRate)
			m.userBuckets[userID] = bucket
		}
		m.mu.Unlock()
	}

	return bucket
}

/* getToolBucket gets or creates a tool token bucket */
func (m *RateLimiterMiddleware) getToolBucket(toolName string) *TokenBucket {
	m.mu.RLock()
	bucket, exists := m.toolBuckets[toolName]
	m.mu.RUnlock()

	if !exists {
		m.mu.Lock()
		bucket, exists = m.toolBuckets[toolName]
		if !exists {
			bucket = NewTokenBucket(m.toolCapacity, m.toolRefillRate)
			m.toolBuckets[toolName] = bucket
		}
		m.mu.Unlock()
	}

	return bucket
}

/* recordRejection records a rejected request */
func (m *RateLimiterMiddleware) recordRejection() {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.stats.RejectedRequests++
	m.stats.LastRejectionTime = time.Now()
}

/* GetStats returns rate limiter statistics */
func (m *RateLimiterMiddleware) GetStats() *RateLimiterStats {
	m.mu.RLock()
	defer m.mu.RUnlock()

	statsCopy := *m.stats
	return &statsCopy
}

/* GetGlobalTokens returns current global token count */
func (m *RateLimiterMiddleware) GetGlobalTokens() float64 {
	return m.globalBucket.GetTokens()
}

/* Name returns the middleware name */
func (m *RateLimiterMiddleware) Name() string {
	return "rate_limiter"
}



