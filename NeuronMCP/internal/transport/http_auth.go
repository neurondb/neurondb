/*-------------------------------------------------------------------------
 *
 * http_auth.go
 *    HTTP authentication middleware for NeuronMCP
 *
 * Provides bearer token and API key authentication for HTTP transport.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/transport/http_auth.go
 *
 *-------------------------------------------------------------------------
 */

package transport

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"
)

/* APIKey represents an API key with scoped permissions */
type APIKey struct {
	ID          string
	KeyHash     string
	UserID      string
	Scopes      []string
	RateLimit   int /* Requests per minute */
	ExpiresAt   *time.Time
	CreatedAt   time.Time
	LastUsed    *time.Time
}

/* APIKeyStore manages API keys */
type APIKeyStore struct {
	mu    sync.RWMutex
	keys  map[string]*APIKey /* Keyed by key hash */
	byID  map[string]*APIKey /* Keyed by key ID */
}

/* NewAPIKeyStore creates a new API key store */
func NewAPIKeyStore() *APIKeyStore {
	return &APIKeyStore{
		keys: make(map[string]*APIKey),
		byID: make(map[string]*APIKey),
	}
}

/* AddAPIKey adds an API key to the store */
func (s *APIKeyStore) AddAPIKey(key *APIKey) error {
	if key == nil {
		return fmt.Errorf("API key cannot be nil")
	}
	if key.ID == "" {
		return fmt.Errorf("API key ID cannot be empty")
	}
	if key.KeyHash == "" {
		return fmt.Errorf("API key hash cannot be empty")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	s.keys[key.KeyHash] = key
	s.byID[key.ID] = key
	return nil
}

/* GetAPIKeyByHash retrieves an API key by hash */
func (s *APIKeyStore) GetAPIKeyByHash(hash string) (*APIKey, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	key, exists := s.keys[hash]
	if !exists {
		return nil, fmt.Errorf("API key not found")
	}

	/* Check expiration */
	if key.ExpiresAt != nil && time.Now().After(*key.ExpiresAt) {
		return nil, fmt.Errorf("API key has expired")
	}

	/* Update last used */
	now := time.Now()
	key.LastUsed = &now

	return key, nil
}

/* HashAPIKey hashes an API key for storage */
func HashAPIKey(apiKey string) string {
	hash := sha256.Sum256([]byte(apiKey))
	return hex.EncodeToString(hash[:])
}

/* HTTPAuthMiddleware provides HTTP authentication */
type HTTPAuthMiddleware struct {
	apiKeyStore    *APIKeyStore
	bearerTokens   map[string]*BearerToken
	mu             sync.RWMutex
	enabled        bool
	requireAuth    bool
}

/* BearerToken represents a bearer token */
type BearerToken struct {
	Token     string
	UserID    string
	Scopes    []string
	ExpiresAt *time.Time
}

/* NewHTTPAuthMiddleware creates a new HTTP auth middleware */
func NewHTTPAuthMiddleware() *HTTPAuthMiddleware {
	return &HTTPAuthMiddleware{
		apiKeyStore:  NewAPIKeyStore(),
		bearerTokens: make(map[string]*BearerToken),
		enabled:      false, /* Disabled by default */
		requireAuth:  false,
	}
}

/* Enable enables authentication */
func (m *HTTPAuthMiddleware) Enable(requireAuth bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.enabled = true
	m.requireAuth = requireAuth
}

/* AddBearerToken adds a bearer token */
func (m *HTTPAuthMiddleware) AddBearerToken(token *BearerToken) error {
	if token == nil {
		return fmt.Errorf("token cannot be nil")
	}
	if token.Token == "" {
		return fmt.Errorf("token cannot be empty")
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	m.bearerTokens[token.Token] = token
	return nil
}

/* AuthenticateRequest authenticates an HTTP request */
func (m *HTTPAuthMiddleware) AuthenticateRequest(r *http.Request) (*AuthResult, error) {
	m.mu.RLock()
	enabled := m.enabled
	requireAuth := m.requireAuth
	m.mu.RUnlock()

	if !enabled {
		/* Auth disabled - allow all requests */
		return &AuthResult{
			Authenticated: true,
			AuthMethod:    "none",
		}, nil
	}

	/* Try bearer token first */
	authHeader := r.Header.Get("Authorization")
	if authHeader != "" {
		if strings.HasPrefix(authHeader, "Bearer ") {
			token := strings.TrimPrefix(authHeader, "Bearer ")
			return m.authenticateBearer(token)
		}
	}

	/* Try API key */
	apiKey := r.Header.Get("X-API-Key")
	if apiKey == "" {
		apiKey = r.URL.Query().Get("api_key")
	}
	if apiKey != "" {
		return m.authenticateAPIKey(apiKey)
	}

	/* Try custom header */
	customKey := r.Header.Get("X-NeuronDB-Key")
	if customKey != "" {
		return m.authenticateAPIKey(customKey)
	}

	if requireAuth {
		return nil, fmt.Errorf("authentication required: provide Bearer token or API key")
	}

	/* Auth not required - allow unauthenticated */
	return &AuthResult{
		Authenticated: false,
		AuthMethod:    "none",
	}, nil
}

/* authenticateBearer authenticates a bearer token */
func (m *HTTPAuthMiddleware) authenticateBearer(token string) (*AuthResult, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	bearerToken, exists := m.bearerTokens[token]
	if !exists {
		return nil, fmt.Errorf("invalid bearer token")
	}

	/* Check expiration */
	if bearerToken.ExpiresAt != nil && time.Now().After(*bearerToken.ExpiresAt) {
		return nil, fmt.Errorf("bearer token has expired")
	}

	return &AuthResult{
		Authenticated: true,
		AuthMethod:    "bearer",
		UserID:        bearerToken.UserID,
		Scopes:        bearerToken.Scopes,
	}, nil
}

/* authenticateAPIKey authenticates an API key */
func (m *HTTPAuthMiddleware) authenticateAPIKey(apiKey string) (*AuthResult, error) {
	if apiKey == "" {
		return nil, fmt.Errorf("API key is empty")
	}

	keyHash := HashAPIKey(apiKey)
	key, err := m.apiKeyStore.GetAPIKeyByHash(keyHash)
	if err != nil {
		return nil, fmt.Errorf("invalid API key: %w", err)
	}

	return &AuthResult{
		Authenticated: true,
		AuthMethod:    "api_key",
		UserID:        key.UserID,
		Scopes:        key.Scopes,
		APIKeyID:      key.ID,
	}, nil
}

/* AuthResult represents authentication result */
type AuthResult struct {
	Authenticated bool
	AuthMethod    string /* bearer, api_key, none */
	UserID        string
	Scopes        []string
	APIKeyID      string
}

/* HasScope checks if the auth result has a specific scope */
func (a *AuthResult) HasScope(scope string) bool {
	if !a.Authenticated {
		return false
	}
	for _, s := range a.Scopes {
		if s == scope || s == "*" {
			return true
		}
	}
	return false
}

/* RateLimiter provides rate limiting per API key */
type RateLimiter struct {
	mu         sync.RWMutex
	rateLimits map[string]*RateLimitEntry
}

/* RateLimitEntry tracks rate limit for a key */
type RateLimitEntry struct {
	Requests   []time.Time
	Limit      int
	Window     time.Duration
	LastReset  time.Time
}

/* NewRateLimiter creates a new rate limiter */
func NewRateLimiter() *RateLimiter {
	return &RateLimiter{
		rateLimits: make(map[string]*RateLimitEntry),
	}
}

/* CheckRateLimit checks if a request is within rate limit */
func (rl *RateLimiter) CheckRateLimit(keyID string, limit int, window time.Duration) (bool, error) {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	entry, exists := rl.rateLimits[keyID]
	if !exists {
		entry = &RateLimitEntry{
			Requests:  make([]time.Time, 0),
			Limit:     limit,
			Window:    window,
			LastReset: time.Now(),
		}
		rl.rateLimits[keyID] = entry
	}

	now := time.Now()

	/* Remove old requests outside window */
	cutoff := now.Add(-window)
	validRequests := []time.Time{}
	for _, reqTime := range entry.Requests {
		if reqTime.After(cutoff) {
			validRequests = append(validRequests, reqTime)
		}
	}
	entry.Requests = validRequests

	/* Check if limit exceeded */
	if len(entry.Requests) >= limit {
		return false, fmt.Errorf("rate limit exceeded: %d requests in %v", limit, window)
	}

	/* Add current request */
	entry.Requests = append(entry.Requests, now)
	return true, nil
}

/* CleanupRateLimits removes old rate limit entries */
func (rl *RateLimiter) CleanupRateLimits() {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	now := time.Now()
	for keyID, entry := range rl.rateLimits {
		/* Remove entries older than 1 hour */
		if now.Sub(entry.LastReset) > time.Hour {
			delete(rl.rateLimits, keyID)
		}
	}
}
