/*-------------------------------------------------------------------------
 *
 * api_key.go
 *    Database operations
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/auth/api_key.go
 *
 *-------------------------------------------------------------------------
 */

package auth

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"fmt"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/metrics"
)

type APIKeyManager struct {
	queries *db.Queries
}

func NewAPIKeyManager(queries *db.Queries) *APIKeyManager {
	return &APIKeyManager{queries: queries}
}

/* GenerateAPIKey generates a new API key */
func (m *APIKeyManager) GenerateAPIKey(ctx context.Context, organizationID, userID *string, rateLimit int, roles []string) (string, *db.APIKey, error) {
	/* Generate random key (32 bytes = 44 base64 chars) */
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
		Metadata:        make(db.JSONBMap), /* Initialize empty metadata */
	}

	if err := m.queries.CreateAPIKey(ctx, apiKey); err != nil {
		return "", nil, fmt.Errorf("failed to create API key: %w", err)
	}

	return key, apiKey, nil
}

/* ValidateAPIKey validates an API key and returns the key record */
func (m *APIKeyManager) ValidateAPIKey(ctx context.Context, key string) (*db.APIKey, error) {
	prefix := GetKeyPrefix(key)
	metrics.DebugWithContext(ctx, "Validating API key", map[string]interface{}{
		"key_prefix": prefix,
		"key_length": len(key),
	})

	/* Find key by prefix */
	apiKey, err := m.queries.GetAPIKeyByPrefix(ctx, prefix)
	if err != nil {
		metrics.WarnWithContext(ctx, "API key lookup failed", map[string]interface{}{
			"key_prefix": prefix,
			"error":      err.Error(),
		})
		return nil, fmt.Errorf("API key lookup failed: prefix=%s, error=%w", prefix, err)
	}
	metrics.DebugWithContext(ctx, "API key found in database", map[string]interface{}{
		"key_prefix": apiKey.KeyPrefix,
		"key_id":     apiKey.ID.String(),
	})

	/* Verify key */
	if !VerifyAPIKey(key, apiKey.KeyHash) {
		metrics.WarnWithContext(ctx, "API key verification failed", map[string]interface{}{
			"key_prefix": prefix,
		})
		return nil, fmt.Errorf("invalid API key: key verification failed")
	}
	metrics.DebugWithContext(ctx, "API key verification succeeded", map[string]interface{}{
		"key_prefix": prefix,
		"key_id":     apiKey.ID.String(),
	})

	/* Update last used */
	_ = m.queries.UpdateAPIKeyLastUsed(ctx, apiKey.ID)

	return apiKey, nil
}

/* DeleteAPIKey deletes an API key */
func (m *APIKeyManager) DeleteAPIKey(ctx context.Context, id uuid.UUID) error {
	return m.queries.DeleteAPIKey(ctx, id)
}
