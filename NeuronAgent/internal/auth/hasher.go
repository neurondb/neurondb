/*-------------------------------------------------------------------------
 *
 * hasher.go
 *    Cryptographic hashing utilities for NeuronAgent authentication
 *
 * Provides bcrypt-based hashing functions for API key storage and
 * verification with configurable cost parameters.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/auth/hasher.go
 *
 *-------------------------------------------------------------------------
 */

package auth

import (
	"golang.org/x/crypto/bcrypt"
)

/* bcryptCost is set to 14 (16,384 rounds) for improved security against modern attacks */
/* Cost 12 (4,096 rounds) was previously used but is no longer sufficient for production */
/* Cost 14 is the recommended minimum as of 2024-2026 */
const bcryptCost = 14

/* HashAPIKey hashes an API key using bcrypt */
func HashAPIKey(key string) (string, error) {
	hash, err := bcrypt.GenerateFromPassword([]byte(key), bcryptCost)
	if err != nil {
		return "", err
	}
	return string(hash), nil
}

/* VerifyAPIKey verifies an API key against its hash */
func VerifyAPIKey(key, hash string) bool {
	err := bcrypt.CompareHashAndPassword([]byte(hash), []byte(key))
	return err == nil
}

/* GetKeyPrefix returns the first 8 characters of a key for identification */
func GetKeyPrefix(key string) string {
	if len(key) < 8 {
		return key
	}
	return key[:8]
}
