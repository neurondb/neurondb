/*-------------------------------------------------------------------------
 *
 * manager.go
 *    Caching layer for responses, embeddings, and tool results
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/cache/manager.go
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

/* CacheManager manages caching for various data types */
type CacheManager struct {
	responses   *TTLCache
	embeddings  *TTLCache
	toolResults *TTLCache
	mu          sync.RWMutex
}

/* TTLCache is a time-to-live cache */
type TTLCache struct {
	items      map[string]*CacheItem
	mu         sync.RWMutex
	defaultTTL time.Duration
	maxSize    int
}

/* CacheItem represents a cached item */
type CacheItem struct {
	Value       interface{}
	ExpiresAt   time.Time
	AccessCount int64
	LastAccess  time.Time
}

/* NewCacheManager creates a new cache manager */
func NewCacheManager(responseTTL, embeddingTTL, toolResultTTL time.Duration, maxSize int) *CacheManager {
	return &CacheManager{
		responses:   NewTTLCache(responseTTL, maxSize),
		embeddings:  NewTTLCache(embeddingTTL, maxSize),
		toolResults: NewTTLCache(toolResultTTL, maxSize),
	}
}

/* NewTTLCache creates a new TTL cache */
func NewTTLCache(defaultTTL time.Duration, maxSize int) *TTLCache {
	cache := &TTLCache{
		items:      make(map[string]*CacheItem),
		defaultTTL: defaultTTL,
		maxSize:    maxSize,
	}
	/* Start cleanup goroutine */
	go cache.cleanup()
	return cache
}

/* Get retrieves a value from cache */
func (c *TTLCache) Get(key string) (interface{}, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	item, exists := c.items[key]
	if !exists {
		return nil, false
	}

	/* Check if expired */
	if time.Now().After(item.ExpiresAt) {
		return nil, false
	}

	/* Update access stats */
	item.AccessCount++
	item.LastAccess = time.Now()

	return item.Value, true
}

/* Set stores a value in cache */
func (c *TTLCache) Set(key string, value interface{}, ttl time.Duration) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	/* Check if we need to evict */
	if len(c.items) >= c.maxSize {
		c.evictLRU()
	}

	if ttl == 0 {
		ttl = c.defaultTTL
	}

	c.items[key] = &CacheItem{
		Value:       value,
		ExpiresAt:   time.Now().Add(ttl),
		AccessCount: 0,
		LastAccess:  time.Now(),
	}

	return nil
}

/* Delete removes a key from cache */
func (c *TTLCache) Delete(key string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.items, key)
}

/* Clear clears all items from cache */
func (c *TTLCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.items = make(map[string]*CacheItem)
}

/* Size returns the number of items in cache */
func (c *TTLCache) Size() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.items)
}

/* evictLRU evicts the least recently used item */
func (c *TTLCache) evictLRU() {
	if len(c.items) == 0 {
		return
	}

	var oldestKey string
	var oldestTime time.Time = time.Now()

	for key, item := range c.items {
		if item.LastAccess.Before(oldestTime) {
			oldestTime = item.LastAccess
			oldestKey = key
		}
	}

	if oldestKey != "" {
		delete(c.items, oldestKey)
	}
}

/* cleanup periodically removes expired items */
func (c *TTLCache) cleanup() {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		c.mu.Lock()
		now := time.Now()
		for key, item := range c.items {
			if now.After(item.ExpiresAt) {
				delete(c.items, key)
			}
		}
		c.mu.Unlock()
	}
}

/* CacheResponse caches an agent response */
func (cm *CacheManager) CacheResponse(ctx context.Context, key string, response interface{}, ttl time.Duration) error {
	return cm.responses.Set(key, response, ttl)
}

/* GetCachedResponse retrieves a cached response */
func (cm *CacheManager) GetCachedResponse(ctx context.Context, key string) (interface{}, bool) {
	return cm.responses.Get(key)
}

/* CacheEmbedding caches an embedding */
func (cm *CacheManager) CacheEmbedding(ctx context.Context, text string, embedding []float32, ttl time.Duration) error {
	key := hashText(text)
	return cm.embeddings.Set(key, embedding, ttl)
}

/* GetCachedEmbedding retrieves a cached embedding */
func (cm *CacheManager) GetCachedEmbedding(ctx context.Context, text string) ([]float32, bool) {
	key := hashText(text)
	value, ok := cm.embeddings.Get(key)
	if !ok {
		return nil, false
	}
	embedding, ok := value.([]float32)
	return embedding, ok
}

/* CacheToolResult caches a tool execution result */
func (cm *CacheManager) CacheToolResult(ctx context.Context, toolName string, args map[string]interface{}, result string, ttl time.Duration) error {
	key := hashToolCall(toolName, args)
	return cm.toolResults.Set(key, result, ttl)
}

/* GetCachedToolResult retrieves a cached tool result */
func (cm *CacheManager) GetCachedToolResult(ctx context.Context, toolName string, args map[string]interface{}) (string, bool) {
	key := hashToolCall(toolName, args)
	value, ok := cm.toolResults.Get(key)
	if !ok {
		return "", false
	}
	result, ok := value.(string)
	return result, ok
}

/* ClearAll clears all caches */
func (cm *CacheManager) ClearAll() {
	cm.responses.Clear()
	cm.embeddings.Clear()
	cm.toolResults.Clear()
}

/* GetStats returns cache statistics */
func (cm *CacheManager) GetStats() map[string]interface{} {
	return map[string]interface{}{
		"responses_size":    cm.responses.Size(),
		"embeddings_size":   cm.embeddings.Size(),
		"tool_results_size": cm.toolResults.Size(),
	}
}

/* DeleteResponse deletes a response from cache */
func (cm *CacheManager) DeleteResponse(key string) {
	cm.responses.Delete(key)
}

/* DeleteEmbedding deletes an embedding from cache */
func (cm *CacheManager) DeleteEmbedding(key string) {
	cm.embeddings.Delete(key)
}

/* DeleteToolResult deletes a tool result from cache */
func (cm *CacheManager) DeleteToolResult(key string) {
	cm.toolResults.Delete(key)
}

/* Helper functions for hashing */
func hashText(text string) string {
	hash := sha256.Sum256([]byte(text))
	return hex.EncodeToString(hash[:])
}

func hashToolCall(toolName string, args map[string]interface{}) string {
	data := map[string]interface{}{
		"tool": toolName,
		"args": args,
	}
	jsonData, err := json.Marshal(data)
	if err != nil {
		/* Fallback to simple concatenation */
		return fmt.Sprintf("%s:%v", toolName, args)
	}
	hash := sha256.Sum256(jsonData)
	return hex.EncodeToString(hash[:])
}
