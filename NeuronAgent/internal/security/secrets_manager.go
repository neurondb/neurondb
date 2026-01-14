/*-------------------------------------------------------------------------
 *
 * secrets_manager.go
 *    Secrets management for secure storage of sensitive data
 *
 * Provides secure storage and retrieval of secrets like API keys,
 * passwords, and tokens.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/security/secrets_manager.go
 *
 *-------------------------------------------------------------------------
 */

package security

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
)

/* SecretsManager manages secrets securely */
type SecretsManager struct {
	db         *db.DB
	queries    *db.Queries
	encryption *Encryption
}

/* NewSecretsManager creates a new secrets manager */
func NewSecretsManager(database *db.DB, queries *db.Queries, encryption *Encryption) *SecretsManager {
	return &SecretsManager{
		db:         database,
		queries:    queries,
		encryption: encryption,
	}
}

/* StoreSecret stores a secret securely */
func (sm *SecretsManager) StoreSecret(ctx context.Context, name string, value string, metadata map[string]interface{}) (uuid.UUID, error) {
	/* Encrypt value */
	encryptedValue, err := sm.encryption.EncryptString(value)
	if err != nil {
		return uuid.Nil, fmt.Errorf("secret storage failed: encryption_error=true, error=%w", err)
	}

	/* Store in database */
	query := `INSERT INTO neurondb_agent.secrets
		(id, name, value, metadata, created_at)
		VALUES ($1, $2, $3, $4::jsonb, $5)
		RETURNING id`

	id := uuid.New()
	_, err = sm.db.DB.ExecContext(ctx, query, id, name, encryptedValue, metadata, time.Now())
	if err != nil {
		return uuid.Nil, fmt.Errorf("secret storage failed: database_error=true, error=%w", err)
	}

	return id, nil
}

/* GetSecret retrieves a secret */
func (sm *SecretsManager) GetSecret(ctx context.Context, name string) (string, error) {
	query := `SELECT value FROM neurondb_agent.secrets WHERE name = $1`

	var encryptedValue string
	err := sm.db.DB.GetContext(ctx, &encryptedValue, query, name)
	if err != nil {
		return "", fmt.Errorf("secret retrieval failed: secret_not_found=true, name='%s', error=%w", name, err)
	}

	/* Decrypt value */
	value, err := sm.encryption.DecryptString(encryptedValue)
	if err != nil {
		return "", fmt.Errorf("secret retrieval failed: decryption_error=true, error=%w", err)
	}

	return value, nil
}

/* DeleteSecret deletes a secret */
func (sm *SecretsManager) DeleteSecret(ctx context.Context, name string) error {
	query := `DELETE FROM neurondb_agent.secrets WHERE name = $1`

	result, err := sm.db.DB.ExecContext(ctx, query, name)
	if err != nil {
		return fmt.Errorf("secret deletion failed: database_error=true, error=%w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("secret deletion failed: rows_affected_check_error=true, error=%w", err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("secret deletion failed: secret_not_found=true, name='%s'", name)
	}

	return nil
}

/* ListSecrets lists all secret names */
func (sm *SecretsManager) ListSecrets(ctx context.Context) ([]string, error) {
	query := `SELECT name FROM neurondb_agent.secrets ORDER BY name`

	var names []string
	err := sm.db.DB.SelectContext(ctx, &names, query)
	if err != nil {
		return nil, fmt.Errorf("secret listing failed: database_error=true, error=%w", err)
	}

	return names, nil
}

