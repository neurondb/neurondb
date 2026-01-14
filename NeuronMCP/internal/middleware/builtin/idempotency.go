/*-------------------------------------------------------------------------
 *
 * idempotency.go
 *    Idempotency key middleware
 *
 * Handles idempotency keys to ensure idempotent request processing.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/middleware/builtin/idempotency.go
 *
 *-------------------------------------------------------------------------
 */

package builtin

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sync"
	"time"

	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/middleware"
)

/* IdempotencyStore stores idempotency key results */
type IdempotencyStore interface {
	Get(key string) (*IdempotencyResult, bool)
	Set(key string, result *IdempotencyResult, ttl time.Duration)
}

/* IdempotencyResult represents a cached idempotency result */
type IdempotencyResult struct {
	Response *middleware.MCPResponse
	ExpiresAt time.Time
}

/* MemoryIdempotencyStore is an in-memory idempotency store */
type MemoryIdempotencyStore struct {
	mu    sync.RWMutex
	store map[string]*IdempotencyResult
}

/* NewMemoryIdempotencyStore creates a new memory idempotency store */
func NewMemoryIdempotencyStore() *MemoryIdempotencyStore {
	store := &MemoryIdempotencyStore{
		store: make(map[string]*IdempotencyResult),
	}

	/* Start cleanup goroutine */
	go store.cleanup()

	return store
}

/* Get retrieves an idempotency result */
func (s *MemoryIdempotencyStore) Get(key string) (*IdempotencyResult, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	result, exists := s.store[key]
	if !exists {
		return nil, false
	}

	/* Check if expired */
	if time.Now().After(result.ExpiresAt) {
		return nil, false
	}

	return result, true
}

/* Set stores an idempotency result */
func (s *MemoryIdempotencyStore) Set(key string, result *IdempotencyResult, ttl time.Duration) {
	s.mu.Lock()
	defer s.mu.Unlock()

	result.ExpiresAt = time.Now().Add(ttl)
	s.store[key] = result
}

/* cleanup removes expired entries */
func (s *MemoryIdempotencyStore) cleanup() {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		s.mu.Lock()
		now := time.Now()
		for key, result := range s.store {
			if now.After(result.ExpiresAt) {
				delete(s.store, key)
			}
		}
		s.mu.Unlock()
	}
}

/* IdempotencyMiddleware handles idempotency keys */
type IdempotencyMiddleware struct {
	logger  *logging.Logger
	store   IdempotencyStore
	enabled bool
}

/* NewIdempotencyMiddleware creates a new idempotency middleware */
func NewIdempotencyMiddleware(logger *logging.Logger, enabled bool) *IdempotencyMiddleware {
	var store IdempotencyStore
	if enabled {
		store = NewMemoryIdempotencyStore()
	}

	return &IdempotencyMiddleware{
		logger:  logger,
		store:   store,
		enabled: enabled,
	}
}

/* Name returns the middleware name */
func (m *IdempotencyMiddleware) Name() string {
	return "idempotency"
}

/* Order returns the execution order (after validation, before execution) */
func (m *IdempotencyMiddleware) Order() int {
	return 18
}

/* Enabled returns whether the middleware is enabled */
func (m *IdempotencyMiddleware) Enabled() bool {
	return m.enabled && m.store != nil
}

/* Execute handles idempotency */
func (m *IdempotencyMiddleware) Execute(ctx context.Context, req *middleware.MCPRequest, next middleware.Handler) (*middleware.MCPResponse, error) {
	if !m.Enabled() {
		return next(ctx, req)
	}

	/* Extract idempotency key from request */
	idempotencyKey := ""
	if req.Method == "tools/call" {
		if key, ok := req.Params["idempotencyKey"].(string); ok && key != "" {
			idempotencyKey = key
		}
	}

	/* If no idempotency key, proceed normally */
	if idempotencyKey == "" {
		return next(ctx, req)
	}

	/* Generate normalized key (method + key) */
	normalizedKey := m.normalizeKey(req.Method, idempotencyKey)

	/* Check if we've seen this request before */
	if result, exists := m.store.Get(normalizedKey); exists {
		m.logger.Debug(fmt.Sprintf("Returning cached response for idempotency key: %s", idempotencyKey), nil)
		return result.Response, nil
	}

	/* Execute request */
	resp, err := next(ctx, req)
	if err != nil {
		return nil, err
	}

	/* Cache successful responses only */
	if resp != nil && !resp.IsError {
		result := &IdempotencyResult{
			Response: resp,
		}
		m.store.Set(normalizedKey, result, 24*time.Hour) // Cache for 24 hours
	}

	return resp, nil
}

/* normalizeKey creates a normalized key from method and idempotency key */
func (m *IdempotencyMiddleware) normalizeKey(method, key string) string {
	combined := fmt.Sprintf("%s:%s", method, key)
	hash := sha256.Sum256([]byte(combined))
	return hex.EncodeToString(hash[:])
}

