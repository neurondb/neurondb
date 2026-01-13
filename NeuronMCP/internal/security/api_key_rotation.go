/*-------------------------------------------------------------------------
 *
 * api_key_rotation.go
 *    API key rotation and expiration management
 *
 * Implements API key rotation, expiration, and lifecycle management
 * as specified in Phase 2.1.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/security/api_key_rotation.go
 *
 *-------------------------------------------------------------------------
 */

package security

import (
	"crypto/rand"
	"encoding/base64"
	"fmt"
	"time"
)

/* APIKey represents an API key */
type APIKey struct {
	ID          string
	Key         string
	UserID      string
	CreatedAt   time.Time
	ExpiresAt   *time.Time
	LastUsedAt  *time.Time
	RotatedAt   *time.Time
	Description string
	Active      bool
}

/* APIKeyManager manages API keys */
type APIKeyManager struct {
	keys map[string]*APIKey /* key -> APIKey */
}

/* NewAPIKeyManager creates a new API key manager */
func NewAPIKeyManager() *APIKeyManager {
	return &APIKeyManager{
		keys: make(map[string]*APIKey),
	}
}

/* GenerateAPIKey generates a new API key */
func (m *APIKeyManager) GenerateAPIKey(userID, description string, expiresInDays *int) (*APIKey, error) {
	/* Generate random key */
	keyBytes := make([]byte, 32)
	if _, err := rand.Read(keyBytes); err != nil {
		return nil, fmt.Errorf("failed to generate key: %w", err)
	}

	key := base64.URLEncoding.EncodeToString(keyBytes)
	keyID := fmt.Sprintf("key_%d", time.Now().UnixNano())

	apiKey := &APIKey{
		ID:          keyID,
		Key:         key,
		UserID:      userID,
		CreatedAt:   time.Now(),
		Description: description,
		Active:      true,
	}

	if expiresInDays != nil {
		expiresAt := time.Now().Add(time.Duration(*expiresInDays) * 24 * time.Hour)
		apiKey.ExpiresAt = &expiresAt
	}

	m.keys[key] = apiKey
	return apiKey, nil
}

/* RotateAPIKey rotates an existing API key */
func (m *APIKeyManager) RotateAPIKey(oldKey string) (*APIKey, error) {
	oldAPIKey, exists := m.keys[oldKey]
	if !exists {
		return nil, fmt.Errorf("API key not found")
	}

	if !oldAPIKey.Active {
		return nil, fmt.Errorf("API key is not active")
	}

	/* Generate new key */
	newKey, err := m.GenerateAPIKey(oldAPIKey.UserID, oldAPIKey.Description+" (rotated)", nil)
	if err != nil {
		return nil, err
	}

	/* Deactivate old key */
	now := time.Now()
	oldAPIKey.Active = false
	oldAPIKey.RotatedAt = &now

	/* Copy expiration from old key */
	if oldAPIKey.ExpiresAt != nil {
		newKey.ExpiresAt = oldAPIKey.ExpiresAt
	}

	return newKey, nil
}

/* ValidateAPIKey validates an API key */
func (m *APIKeyManager) ValidateAPIKey(key string) (*APIKey, error) {
	apiKey, exists := m.keys[key]
	if !exists {
		return nil, fmt.Errorf("invalid API key")
	}

	if !apiKey.Active {
		return nil, fmt.Errorf("API key is not active")
	}

	/* Check expiration */
	if apiKey.ExpiresAt != nil && time.Now().After(*apiKey.ExpiresAt) {
		return nil, fmt.Errorf("API key has expired")
	}

	/* Update last used */
	now := time.Now()
	apiKey.LastUsedAt = &now

	return apiKey, nil
}

/* RevokeAPIKey revokes an API key */
func (m *APIKeyManager) RevokeAPIKey(key string) error {
	apiKey, exists := m.keys[key]
	if !exists {
		return fmt.Errorf("API key not found")
	}

	apiKey.Active = false
	return nil
}

/* ListAPIKeys lists all API keys for a user */
func (m *APIKeyManager) ListAPIKeys(userID string) []*APIKey {
	keys := []*APIKey{}
	for _, key := range m.keys {
		if key.UserID == userID {
			keys = append(keys, key)
		}
	}
	return keys
}

/* GetExpiringKeys returns keys expiring within the specified days */
func (m *APIKeyManager) GetExpiringKeys(days int) []*APIKey {
	keys := []*APIKey{}
	threshold := time.Now().Add(time.Duration(days) * 24 * time.Hour)

	for _, key := range m.keys {
		if key.Active && key.ExpiresAt != nil && key.ExpiresAt.Before(threshold) {
			keys = append(keys, key)
		}
	}

	return keys
}



