/*-------------------------------------------------------------------------
 *
 * idempotency.go
 *    Idempotency key cache for NeuronMCP
 *
 * Provides in-memory caching of tool execution results by idempotency key
 * to support idempotent tool calls.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/cache/idempotency.go
 *
 *-------------------------------------------------------------------------
 */

package cache

import (
	"sync"
	"time"

	"github.com/neurondb/NeuronMCP/pkg/mcp"
)

/* IdempotencyCacheEntry represents a cached idempotency result */
type IdempotencyCacheEntry struct {
	Result    *mcp.ToolResult
	Timestamp time.Time
	ExpiresAt time.Time
}

/* IdempotencyCache provides caching for idempotency keys */
type IdempotencyCache struct {
	entries map[string]*IdempotencyCacheEntry
	mu      sync.RWMutex
	ttl     time.Duration
	cleanupInterval time.Duration
	stopCleanup     chan struct{}
}

/* NewIdempotencyCache creates a new idempotency cache */
func NewIdempotencyCache(ttl time.Duration) *IdempotencyCache {
	cache := &IdempotencyCache{
		entries:         make(map[string]*IdempotencyCacheEntry),
		ttl:              ttl,
		cleanupInterval: time.Minute * 5, /* Clean up expired entries every 5 minutes */
		stopCleanup:     make(chan struct{}),
	}
	
	/* Start background cleanup goroutine */
	go cache.cleanup()
	
	return cache
}

/* Get retrieves a cached result by idempotency key */
func (c *IdempotencyCache) Get(key string) (*mcp.ToolResult, bool) {
	if key == "" {
		return nil, false
	}
	
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	entry, exists := c.entries[key]
	if !exists {
		return nil, false
	}
	
	/* Check if entry has expired */
	if time.Now().After(entry.ExpiresAt) {
		/* Entry expired, but we'll let cleanup handle it */
		return nil, false
	}
	
	return entry.Result, true
}

/* Set stores a result with an idempotency key */
func (c *IdempotencyCache) Set(key string, result *mcp.ToolResult) {
	if key == "" {
		return
	}
	
	c.mu.Lock()
	defer c.mu.Unlock()
	
	now := time.Now()
	c.entries[key] = &IdempotencyCacheEntry{
		Result:    result,
		Timestamp: now,
		ExpiresAt: now.Add(c.ttl),
	}
}

/* Delete removes an entry from the cache */
func (c *IdempotencyCache) Delete(key string) {
	if key == "" {
		return
	}
	
	c.mu.Lock()
	defer c.mu.Unlock()
	
	delete(c.entries, key)
}

/* Clear removes all entries from the cache */
func (c *IdempotencyCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	c.entries = make(map[string]*IdempotencyCacheEntry)
}

/* cleanup periodically removes expired entries */
func (c *IdempotencyCache) cleanup() {
	ticker := time.NewTicker(c.cleanupInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			c.cleanupExpired()
		case <-c.stopCleanup:
			return
		}
	}
}

/* cleanupExpired removes expired entries */
func (c *IdempotencyCache) cleanupExpired() {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	now := time.Now()
	for key, entry := range c.entries {
		if now.After(entry.ExpiresAt) {
			delete(c.entries, key)
		}
	}
}

/* Close stops the cleanup goroutine and clears the cache */
func (c *IdempotencyCache) Close() {
	close(c.stopCleanup)
	c.Clear()
}

/* Size returns the number of entries in the cache */
func (c *IdempotencyCache) Size() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	return len(c.entries)
}

