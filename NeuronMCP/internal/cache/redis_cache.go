/*-------------------------------------------------------------------------
 *
 * redis_cache.go
 *    Redis cache backend for NeuronMCP
 *
 * Provides distributed caching using Redis for multi-instance deployments.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/cache/redis_cache.go
 *
 *-------------------------------------------------------------------------
 */

package cache

import (
	"context"
	"encoding/json"
	"time"

	"github.com/redis/go-redis/v9"
)

/* RedisCache is a Redis-backed cache implementation */
type RedisCache struct {
	client   *redis.Client
	fallback *MemoryCache
	enabled  bool
}

/* NewRedisCache creates a new Redis cache */
/* If redisURL is empty or Redis is unavailable, falls back to memory cache */
func NewRedisCache(redisURL string) (*RedisCache, error) {
	fallback := NewMemoryCache()
	if redisURL == "" {
		return &RedisCache{fallback: fallback, enabled: false}, nil
	}
	opts, err := redis.ParseURL(redisURL)
	if err != nil {
		return &RedisCache{fallback: fallback, enabled: false}, nil
	}
	client := redis.NewClient(opts)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	err = client.Ping(ctx).Err()
	cancel()
	if err != nil {
		_ = client.Close()
		return &RedisCache{fallback: fallback, enabled: false}, nil
	}
	return &RedisCache{client: client, fallback: fallback, enabled: true}, nil
}

/* Get retrieves a value from Redis cache */
func (r *RedisCache) Get(ctx context.Context, key string) (interface{}, bool) {
	if !r.enabled {
		return r.fallback.Get(ctx, key)
	}
	val, err := r.client.Get(ctx, key).Bytes()
	if err == redis.Nil {
		return nil, false
	}
	if err != nil {
		return r.fallback.Get(ctx, key)
	}
	var out interface{}
	if err := json.Unmarshal(val, &out); err != nil {
		return nil, false
	}
	return out, true
}

/* Set stores a value in Redis cache */
func (r *RedisCache) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
	if !r.enabled {
		return r.fallback.Set(ctx, key, value, ttl)
	}
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	return r.client.Set(ctx, key, data, ttl).Err()
}

/* Delete removes a value from Redis cache */
func (r *RedisCache) Delete(ctx context.Context, key string) error {
	if !r.enabled {
		return r.fallback.Delete(ctx, key)
	}
	return r.client.Del(ctx, key).Err()
}

/* Clear removes all entries from Redis cache (FLUSHDB) */
func (r *RedisCache) Clear(ctx context.Context) error {
	if !r.enabled {
		return r.fallback.Clear(ctx)
	}
	return r.client.FlushDB(ctx).Err()
}

/* MultiLevelCache provides a multi-level cache (memory + Redis) */
type MultiLevelCache struct {
	l1 *MemoryCache /* L1: In-memory cache (fast, local) */
	l2 Cache        /* L2: Redis cache (distributed) */
}

/* NewMultiLevelCache creates a new multi-level cache */
func NewMultiLevelCache(redisURL string) (*MultiLevelCache, error) {
	redisCache, err := NewRedisCache(redisURL)
	if err != nil {
		/* If Redis fails, use memory cache only */
		return &MultiLevelCache{
			l1: NewMemoryCache(),
			l2: nil,
		}, nil
	}

	return &MultiLevelCache{
		l1: NewMemoryCache(),
		l2: redisCache,
	}, nil
}

/* Get retrieves from L1 first, then L2 */
func (m *MultiLevelCache) Get(ctx context.Context, key string) (interface{}, bool) {
	/* Try L1 first */
	if value, found := m.l1.Get(ctx, key); found {
		return value, true
	}

	/* Try L2 if available */
	if m.l2 != nil {
		if value, found := m.l2.Get(ctx, key); found {
			/* Promote to L1 */
			m.l1.Set(ctx, key, value, 5*time.Minute)
			return value, true
		}
	}

	return nil, false
}

/* Set stores in both L1 and L2 */
func (m *MultiLevelCache) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
	/* Set in L1 */
	if err := m.l1.Set(ctx, key, value, ttl); err != nil {
		return err
	}

	/* Set in L2 if available */
	if m.l2 != nil {
		return m.l2.Set(ctx, key, value, ttl)
	}

	return nil
}

/* Delete removes from both levels */
func (m *MultiLevelCache) Delete(ctx context.Context, key string) error {
	m.l1.Delete(ctx, key)
	if m.l2 != nil {
		return m.l2.Delete(ctx, key)
	}
	return nil
}

/* Clear clears both levels */
func (m *MultiLevelCache) Clear(ctx context.Context) error {
	m.l1.Clear(ctx)
	if m.l2 != nil {
		return m.l2.Clear(ctx)
	}
	return nil
}

/* QueryExecutor runs a warm query and returns the result to be cached */
type QueryExecutor interface {
	Execute(ctx context.Context, query CacheWarmQuery) (interface{}, error)
}

/* CacheWarmer provides cache warming functionality */
type CacheWarmer struct {
	cache    Cache
	executor QueryExecutor
}

/* NewCacheWarmer creates a new cache warmer. If executor is non-nil, WarmCache will execute each query and store the result. */
func NewCacheWarmer(cache Cache, executor QueryExecutor) *CacheWarmer {
	return &CacheWarmer{cache: cache, executor: executor}
}

/* WarmCache runs each warm query (when executor is set), then stores the result in the cache with the query TTL */
func (w *CacheWarmer) WarmCache(ctx context.Context, queries []CacheWarmQuery) error {
	ttl := 5 * time.Minute
	for _, query := range queries {
		key := GenerateCacheKey(query.Prefix, query.Params)
		if w.executor != nil {
			result, err := w.executor.Execute(ctx, query)
			if err != nil {
				continue
			}
			if qttl := query.TTL; qttl > 0 {
				ttl = qttl
			}
			_ = w.cache.Set(ctx, key, result, ttl)
		}
		_ = key
	}
	return nil
}

/* CacheWarmQuery represents a query to warm the cache */
type CacheWarmQuery struct {
	Prefix string
	Params map[string]interface{}
	TTL    time.Duration
}
