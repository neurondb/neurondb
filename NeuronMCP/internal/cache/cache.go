/*-------------------------------------------------------------------------
 *
 * cache.go
 *    Caching layer for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/cache/cache.go
 *
 *-------------------------------------------------------------------------
 */

package cache

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

/* CacheEntry represents a cache entry */
type CacheEntry struct {
	Value     interface{}
	ExpiresAt time.Time
}

/* IsExpired checks if the entry is expired */
func (e *CacheEntry) IsExpired() bool {
	return time.Now().After(e.ExpiresAt)
}

/* Cache is the cache interface */
type Cache interface {
	Get(ctx context.Context, key string) (interface{}, bool)
	Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error
	Delete(ctx context.Context, key string) error
	Clear(ctx context.Context) error
}

/* MemoryCache is an in-memory cache implementation */
type MemoryCache struct {
	entries map[string]*CacheEntry
	mu      sync.RWMutex
}

/* NewMemoryCache creates a new memory cache */
func NewMemoryCache() *MemoryCache {
	cache := &MemoryCache{
		entries: make(map[string]*CacheEntry),
	}

	/* Start cleanup goroutine */
	go cache.cleanup()

	return cache
}

/* Get retrieves a value from cache */
func (c *MemoryCache) Get(ctx context.Context, key string) (interface{}, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	entry, exists := c.entries[key]
	if !exists {
		return nil, false
	}

	if entry.IsExpired() {
		delete(c.entries, key)
		return nil, false
	}

	return entry.Value, true
}

/* Set stores a value in cache */
func (c *MemoryCache) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.entries[key] = &CacheEntry{
		Value:     value,
		ExpiresAt: time.Now().Add(ttl),
	}

	return nil
}

/* Delete removes a value from cache */
func (c *MemoryCache) Delete(ctx context.Context, key string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	delete(c.entries, key)
	return nil
}

/* Clear removes all entries from cache */
func (c *MemoryCache) Clear(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.entries = make(map[string]*CacheEntry)
	return nil
}

/* cleanup periodically removes expired entries */
func (c *MemoryCache) cleanup() {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		c.mu.Lock()
		for key, entry := range c.entries {
			if entry.IsExpired() {
				delete(c.entries, key)
			}
		}
		c.mu.Unlock()
	}
}

/* GenerateCacheKey generates a cache key from parameters */
func GenerateCacheKey(prefix string, params map[string]interface{}) string {
	/* Serialize params to JSON */
	paramsJSON, _ := json.Marshal(params)
	
	/* Create hash */
	hash := sha256.Sum256(append([]byte(prefix), paramsJSON...))
	hashStr := hex.EncodeToString(hash[:])
	
	return fmt.Sprintf("%s:%s", prefix, hashStr[:16])
}












