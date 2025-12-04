package auth

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"fmt"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
)

type APIKeyManager struct {
	queries *db.Queries
}

func NewAPIKeyManager(queries *db.Queries) *APIKeyManager {
	return &APIKeyManager{queries: queries}
}

// GenerateAPIKey generates a new API key
func (m *APIKeyManager) GenerateAPIKey(ctx context.Context, organizationID, userID *string, rateLimit int, roles []string) (string, *db.APIKey, error) {
	// Generate random key (32 bytes = 44 base64 chars)
	keyBytes := make([]byte, 32)
	if _, err := rand.Read(keyBytes); err != nil {
		return "", nil, fmt.Errorf("failed to generate key: %w", err)
	}

	key := base64.URLEncoding.EncodeToString(keyBytes)
	keyPrefix := GetKeyPrefix(key)
	keyHash, err := HashAPIKey(key)
	if err != nil {
		return "", nil, fmt.Errorf("failed to hash key: %w", err)
	}

	apiKey := &db.APIKey{
		KeyHash:         keyHash,
		KeyPrefix:       keyPrefix,
		OrganizationID:  organizationID,
		UserID:          userID,
		RateLimitPerMin: rateLimit,
		Roles:           roles,
		Metadata:        make(db.JSONBMap), // Initialize empty metadata
	}

	if err := m.queries.CreateAPIKey(ctx, apiKey); err != nil {
		return "", nil, fmt.Errorf("failed to create API key: %w", err)
	}

	return key, apiKey, nil
}

// ValidateAPIKey validates an API key and returns the key record
func (m *APIKeyManager) ValidateAPIKey(ctx context.Context, key string) (*db.APIKey, error) {
	prefix := GetKeyPrefix(key)
	fmt.Printf("[AUTH] ValidateAPIKey: prefix=%s, key_len=%d\n", prefix, len(key))

	// Find key by prefix
	apiKey, err := m.queries.GetAPIKeyByPrefix(ctx, prefix)
	if err != nil {
		fmt.Printf("[AUTH] GetAPIKeyByPrefix failed: prefix=%s, error=%v\n", prefix, err)
		return nil, fmt.Errorf("API key lookup failed: prefix=%s, error=%w", prefix, err)
	}
	fmt.Printf("[AUTH] GetAPIKeyByPrefix succeeded: prefix=%s, hash=%s\n", apiKey.KeyPrefix, apiKey.KeyHash[:30])

	// Verify key
	if !VerifyAPIKey(key, apiKey.KeyHash) {
		fmt.Printf("[AUTH] Key verification failed: prefix=%s\n", prefix)
		return nil, fmt.Errorf("invalid API key: key verification failed")
	}
	fmt.Printf("[AUTH] Key verification succeeded: prefix=%s\n", prefix)

	// Update last used
	_ = m.queries.UpdateAPIKeyLastUsed(ctx, apiKey.ID)

	return apiKey, nil
}

// DeleteAPIKey deletes an API key
func (m *APIKeyManager) DeleteAPIKey(ctx context.Context, id uuid.UUID) error {
	return m.queries.DeleteAPIKey(ctx, id)
}

