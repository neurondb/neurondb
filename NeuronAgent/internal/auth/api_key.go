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
	}

	if err := m.queries.CreateAPIKey(ctx, apiKey); err != nil {
		return "", nil, fmt.Errorf("failed to create API key: %w", err)
	}

	return key, apiKey, nil
}

// ValidateAPIKey validates an API key and returns the key record
func (m *APIKeyManager) ValidateAPIKey(ctx context.Context, key string) (*db.APIKey, error) {
	prefix := GetKeyPrefix(key)

	// Find key by prefix
	apiKey, err := m.queries.GetAPIKeyByPrefix(ctx, prefix)
	if err != nil {
		return nil, fmt.Errorf("API key not found")
	}

	// Verify key
	if !VerifyAPIKey(key, apiKey.KeyHash) {
		return nil, fmt.Errorf("invalid API key")
	}

	// Update last used
	_ = m.queries.UpdateAPIKeyLastUsed(ctx, apiKey.ID)

	return apiKey, nil
}

// DeleteAPIKey deletes an API key
func (m *APIKeyManager) DeleteAPIKey(ctx context.Context, id uuid.UUID) error {
	return m.queries.DeleteAPIKey(ctx, id)
}

