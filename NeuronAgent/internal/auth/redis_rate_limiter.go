/*-------------------------------------------------------------------------
 *
 * redis_rate_limiter.go
 *    Redis-backed rate limiter for NeuronAgent (optional, when REDIS_URL is set)
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/auth/redis_rate_limiter.go
 *
 *-------------------------------------------------------------------------
 */

package auth

import (
	"context"
	"fmt"
	"time"

	"github.com/neurondb/NeuronAgent/internal/metrics"
	"github.com/redis/go-redis/v9"
)

const rateLimitKeyPrefix = "neurondb_agent:ratelimit:"
const rateLimitWindow = 1 * time.Minute

/* RedisRateLimiter uses Redis for distributed rate limiting */
type RedisRateLimiter struct {
	client *redis.Client
}

/* NewRedisRateLimiter creates a Redis-backed rate limiter; returns nil if url is empty or connection fails */
func NewRedisRateLimiter(redisURL string) (RateLimiterInterface, error) {
	if redisURL == "" {
		return nil, nil
	}
	opts, err := redis.ParseURL(redisURL)
	if err != nil {
		return nil, fmt.Errorf("redis rate limiter: parse URL: %w", err)
	}
	client := redis.NewClient(opts)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if err := client.Ping(ctx).Err(); err != nil {
		_ = client.Close()
		return nil, fmt.Errorf("redis rate limiter: ping: %w", err)
	}
	return &RedisRateLimiter{client: client}, nil
}

/* incrWithExpireScript sets EXPIRE only when key is new (INCR returns 1) */
var incrWithExpireScript = redis.NewScript(`
	local c = redis.call('INCR', KEYS[1])
	if c == 1 then
		redis.call('EXPIRE', KEYS[1], ARGV[1])
	end
	return c
`)

/* CheckLimit checks and increments rate limit in Redis (fixed 1-minute window) */
func (r *RedisRateLimiter) CheckLimit(keyID string, limitPerMin int) bool {
	key := rateLimitKeyPrefix + keyID
	ctx := context.Background()

	count, err := incrWithExpireScript.Run(ctx, r.client, []string{key}, "60").Int64()
	if err != nil {
		metrics.RecordRateLimitAllowed(keyID)
		return true
	}

	if count <= int64(limitPerMin) {
		metrics.RecordRateLimitAllowed(keyID)
		return true
	}
	metrics.RecordRateLimitRejected(keyID)
	return false
}

/* Close closes the Redis client (call on shutdown) */
func (r *RedisRateLimiter) Close() error {
	if r.client != nil {
		return r.client.Close()
	}
	return nil
}
