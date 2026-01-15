/*-------------------------------------------------------------------------
 *
 * query_cache.go
 *    Query Result Caching for NeuronMCP
 *
 * Provides intelligent query result caching with invalidation strategies.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/cache/query_cache.go
 *
 *-------------------------------------------------------------------------
 */

package cache

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sync"
	"time"
)

/* QueryCache provides query result caching */
type QueryCache struct {
	cache    map[string]*QueryCacheEntry
	mu       sync.RWMutex
	ttl      time.Duration
	maxSize  int
	eviction string
}

/* QueryCacheEntry represents a cached query result */
type QueryCacheEntry struct {
	Result    interface{}
	Timestamp time.Time
	TTL       time.Duration
	Hits      int64
}

/* NewQueryCache creates a new query cache */
func NewQueryCache(ttl time.Duration, maxSize int) *QueryCache {
	return &QueryCache{
		cache:   make(map[string]*QueryCacheEntry),
		ttl:     ttl,
		maxSize: maxSize,
	}
}

/* Get gets a cached result */
func (qc *QueryCache) Get(ctx context.Context, query string, params []interface{}) (interface{}, bool) {
	key := qc.generateKey(query, params)

	qc.mu.RLock()
	defer qc.mu.RUnlock()

	entry, exists := qc.cache[key]
	if !exists {
		return nil, false
	}

	/* Check if expired */
	if time.Since(entry.Timestamp) > entry.TTL {
		delete(qc.cache, key)
		return nil, false
	}

	entry.Hits++
	return entry.Result, true
}

/* Set sets a cached result */
func (qc *QueryCache) Set(ctx context.Context, query string, params []interface{}, result interface{}, ttl time.Duration) {
	key := qc.generateKey(query, params)

	qc.mu.Lock()
	defer qc.mu.Unlock()

	/* Evict if at capacity */
	if len(qc.cache) >= qc.maxSize {
		qc.evict()
	}

	if ttl == 0 {
		ttl = qc.ttl
	}

	qc.cache[key] = &QueryCacheEntry{
		Result:    result,
		Timestamp: time.Now(),
		TTL:       ttl,
		Hits:      0,
	}
}

/* Invalidate invalidates cache entries */
func (qc *QueryCache) Invalidate(ctx context.Context, pattern string) {
	qc.mu.Lock()
	defer qc.mu.Unlock()

	for key := range qc.cache {
		if pattern == "" || key == pattern {
			delete(qc.cache, key)
		}
	}
}

/* generateKey generates a cache key from query and params */
func (qc *QueryCache) generateKey(query string, params []interface{}) string {
	keyData := map[string]interface{}{
		"query":  query,
		"params": params,
	}

	keyJSON, _ := json.Marshal(keyData)
	hash := sha256.Sum256(keyJSON)
	return hex.EncodeToString(hash[:])
}

/* evict evicts entries based on eviction strategy */
func (qc *QueryCache) evict() {
	/* Simple LRU - evict oldest */
	oldestKey := ""
	oldestTime := time.Now()

	for key, entry := range qc.cache {
		if entry.Timestamp.Before(oldestTime) {
			oldestTime = entry.Timestamp
			oldestKey = key
		}
	}

	if oldestKey != "" {
		delete(qc.cache, oldestKey)
	}
}

/* GetStats gets cache statistics */
func (qc *QueryCache) GetStats() map[string]interface{} {
	qc.mu.RLock()
	defer qc.mu.RUnlock()

	totalHits := int64(0)
	for _, entry := range qc.cache {
		totalHits += entry.Hits
	}

	return map[string]interface{}{
		"size":       len(qc.cache),
		"max_size":   qc.maxSize,
		"total_hits": totalHits,
		"hit_rate":   float64(totalHits) / float64(len(qc.cache)+1),
	}
}

