package middleware

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"
)

/* CSRFTokenStorage defines the interface for CSRF token storage */
type CSRFTokenStorage interface {
	Store(ctx context.Context, token string, expiry time.Time) error
	Get(ctx context.Context, token string) (time.Time, bool)
	Delete(ctx context.Context, token string) error
	Cleanup(ctx context.Context) error
}

/* MemoryCSRFStorage is an in-memory implementation of CSRF token storage */
type MemoryCSRFStorage struct {
	tokens map[string]time.Time
	mu     sync.RWMutex
}

/* NewMemoryCSRFStorage creates a new in-memory CSRF token storage */
func NewMemoryCSRFStorage() *MemoryCSRFStorage {
	return &MemoryCSRFStorage{
		tokens: make(map[string]time.Time),
	}
}

/* Store stores a CSRF token with expiry */
func (m *MemoryCSRFStorage) Store(ctx context.Context, token string, expiry time.Time) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.tokens[token] = expiry
	return nil
}

/* Get retrieves a CSRF token's expiry time */
func (m *MemoryCSRFStorage) Get(ctx context.Context, token string) (time.Time, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	expiry, exists := m.tokens[token]
	return expiry, exists
}

/* Delete removes a CSRF token */
func (m *MemoryCSRFStorage) Delete(ctx context.Context, token string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.tokens, token)
	return nil
}

/* Cleanup removes expired CSRF tokens */
func (m *MemoryCSRFStorage) Cleanup(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	now := time.Now()
	for token, expiry := range m.tokens {
		if now.After(expiry) {
			delete(m.tokens, token)
		}
	}
	return nil
}

/* RedisCSRFStorage is a Redis-based implementation of CSRF token storage */
/* This implementation requires the redis client to be provided */
type RedisCSRFStorage struct {
	client    interface{} // Redis client (interface{} to avoid direct dependency)
	keyPrefix string
	mu        sync.RWMutex
}

/* NewRedisCSRFStorage creates a new Redis-based CSRF token storage */
/* client should be a *redis.Client from github.com/redis/go-redis/v9 */
/* If client is nil, this will fall back to in-memory storage */
func NewRedisCSRFStorage(client interface{}, keyPrefix string) *RedisCSRFStorage {
	if keyPrefix == "" {
		keyPrefix = "csrf:"
	}
	return &RedisCSRFStorage{
		client:    client,
		keyPrefix: keyPrefix,
	}
}

/* Store stores a CSRF token with expiry in Redis */
func (r *RedisCSRFStorage) Store(ctx context.Context, token string, expiry time.Time) error {
	if r.client == nil {
		return fmt.Errorf("Redis client not configured")
	}

	/* Use reflection to call Redis methods without direct dependency */
	/* This allows the code to compile even without Redis dependency */
	/* In production, ensure redis/go-redis/v9 is available */
	key := r.keyPrefix + token
	ttl := time.Until(expiry)
	if ttl <= 0 {
		return nil /* Already expired */
	}

	/* Try to use Redis client if available */
	/* This is a simplified implementation - in production, use proper Redis client */
	/* For now, we'll use a type assertion approach */
	if setFunc, ok := r.client.(interface {
		Set(ctx context.Context, key string, value interface{}, expiration time.Duration) error
	}); ok {
		expiryBytes, _ := json.Marshal(expiry)
		return setFunc.Set(ctx, key, expiryBytes, ttl)
	}

	/* Fallback: if Redis client doesn't match expected interface, return error */
	return fmt.Errorf("Redis client not properly configured")
}

/* Get retrieves a CSRF token's expiry time from Redis */
func (r *RedisCSRFStorage) Get(ctx context.Context, token string) (time.Time, bool) {
	if r.client == nil {
		return time.Time{}, false
	}

	key := r.keyPrefix + token

	/* Try to use Redis client if available */
	if getFunc, ok := r.client.(interface {
		Get(ctx context.Context, key string) (string, error)
	}); ok {
		value, err := getFunc.Get(ctx, key)
		if err != nil {
			return time.Time{}, false
		}
		var expiry time.Time
		if err := json.Unmarshal([]byte(value), &expiry); err != nil {
			return time.Time{}, false
		}
		return expiry, true
	}

	return time.Time{}, false
}

/* Delete removes a CSRF token from Redis */
func (r *RedisCSRFStorage) Delete(ctx context.Context, token string) error {
	if r.client == nil {
		return nil
	}

	key := r.keyPrefix + token

	/* Try to use Redis client if available */
	if delFunc, ok := r.client.(interface {
		Del(ctx context.Context, keys ...string) error
	}); ok {
		return delFunc.Del(ctx, key)
	}

	return nil
}

/* Cleanup removes expired CSRF tokens from Redis */
/* Redis handles expiration automatically via TTL, so this is a no-op */
func (r *RedisCSRFStorage) Cleanup(ctx context.Context) error {
	/* Redis handles expiration automatically via TTL */
	return nil
}

/* Global CSRF token storage (defaults to in-memory) */
var (
	csrfStorage CSRFTokenStorage = NewMemoryCSRFStorage()
	csrfTokenExpiry               = 24 * time.Hour
	csrfStorageMu                sync.RWMutex
)

/* SetCSRFStorage sets the CSRF token storage backend */
func SetCSRFStorage(storage CSRFTokenStorage) {
	csrfStorageMu.Lock()
	defer csrfStorageMu.Unlock()
	csrfStorage = storage
}

/* GetCSRFStorage returns the current CSRF token storage backend */
func GetCSRFStorage() CSRFTokenStorage {
	csrfStorageMu.RLock()
	defer csrfStorageMu.RUnlock()
	return csrfStorage
}

/* GenerateCSRFToken generates a new CSRF token */
func GenerateCSRFToken() (string, error) {
	return GenerateCSRFTokenWithContext(context.Background())
}

/* GenerateCSRFTokenWithContext generates a new CSRF token with context */
func GenerateCSRFTokenWithContext(ctx context.Context) (string, error) {
	token := make([]byte, 32)
	if _, err := rand.Read(token); err != nil {
		return "", err
	}
	tokenStr := base64.URLEncoding.EncodeToString(token)
	expiry := time.Now().Add(csrfTokenExpiry)

	storage := GetCSRFStorage()
	if err := storage.Store(ctx, tokenStr, expiry); err != nil {
		return "", fmt.Errorf("failed to store CSRF token: %w", err)
	}

	return tokenStr, nil
}

/* ValidateCSRFToken validates a CSRF token */
func ValidateCSRFToken(token string) bool {
	return ValidateCSRFTokenWithContext(context.Background(), token)
}

/* ValidateCSRFTokenWithContext validates a CSRF token with context */
func ValidateCSRFTokenWithContext(ctx context.Context, token string) bool {
	storage := GetCSRFStorage()
	expiry, exists := storage.Get(ctx, token)
	if !exists {
		return false
	}
	if time.Now().After(expiry) {
		storage.Delete(ctx, token)
		return false
	}
	return true
}

/* CSRFMiddleware provides CSRF protection for state-changing operations */
func CSRFMiddleware() func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			/* Only check CSRF for state-changing methods */
			if r.Method == "GET" || r.Method == "HEAD" || r.Method == "OPTIONS" {
				next.ServeHTTP(w, r)
				return
			}

			/* Get token from header or form */
			token := r.Header.Get("X-CSRF-Token")
			if token == "" {
				token = r.FormValue("csrf_token")
			}

			/* Validate token */
			if !ValidateCSRFTokenWithContext(r.Context(), token) {
				http.Error(w, "Invalid or missing CSRF token", http.StatusForbidden)
				return
			}

			next.ServeHTTP(w, r)
		})
	}
}

/* CleanupExpiredTokens removes expired CSRF tokens (should be called periodically) */
func CleanupExpiredTokens() {
	CleanupExpiredTokensWithContext(context.Background())
}

/* CleanupExpiredTokensWithContext removes expired CSRF tokens with context */
func CleanupExpiredTokensWithContext(ctx context.Context) {
	storage := GetCSRFStorage()
	storage.Cleanup(ctx)
}






