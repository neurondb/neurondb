/*-------------------------------------------------------------------------
 *
 * encryption.go
 *    Encryption support for data at rest and in transit
 *
 * Provides encryption capabilities for sensitive data storage and
 * secure communication.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/security/encryption.go
 *
 *-------------------------------------------------------------------------
 */

package security

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"io"

	"golang.org/x/crypto/pbkdf2"
)

/* Encryption provides encryption and decryption capabilities */
type Encryption struct {
	key []byte
}

/* NewEncryption creates a new encryption instance */
func NewEncryption(secretKey string) (*Encryption, error) {
	/* Derive key from secret using PBKDF2 */
	salt := []byte("neurondb-encryption-salt") /* In production, use random salt */
	key := pbkdf2.Key([]byte(secretKey), salt, 4096, 32, sha256.New)

	return &Encryption{
		key: key,
	}, nil
}

/* Encrypt encrypts data */
func (e *Encryption) Encrypt(plaintext []byte) ([]byte, error) {
	block, err := aes.NewCipher(e.key)
	if err != nil {
		return nil, fmt.Errorf("encryption failed: cipher_creation_error=true, error=%w", err)
	}

	/* Create GCM */
	aesGCM, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("encryption failed: gcm_creation_error=true, error=%w", err)
	}

	/* Create nonce */
	nonce := make([]byte, aesGCM.NonceSize())
	if _, err = io.ReadFull(rand.Reader, nonce); err != nil {
		return nil, fmt.Errorf("encryption failed: nonce_creation_error=true, error=%w", err)
	}

	/* Encrypt */
	ciphertext := aesGCM.Seal(nonce, nonce, plaintext, nil)
	return ciphertext, nil
}

/* Decrypt decrypts data */
func (e *Encryption) Decrypt(ciphertext []byte) ([]byte, error) {
	block, err := aes.NewCipher(e.key)
	if err != nil {
		return nil, fmt.Errorf("decryption failed: cipher_creation_error=true, error=%w", err)
	}

	/* Create GCM */
	aesGCM, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("decryption failed: gcm_creation_error=true, error=%w", err)
	}

	/* Extract nonce */
	nonceSize := aesGCM.NonceSize()
	if len(ciphertext) < nonceSize {
		return nil, fmt.Errorf("decryption failed: invalid_ciphertext_length=true, length=%d", len(ciphertext))
	}

	nonce, ciphertext := ciphertext[:nonceSize], ciphertext[nonceSize:]

	/* Decrypt */
	plaintext, err := aesGCM.Open(nil, nonce, ciphertext, nil)
	if err != nil {
		return nil, fmt.Errorf("decryption failed: decryption_error=true, error=%w", err)
	}

	return plaintext, nil
}

/* EncryptString encrypts a string and returns base64-encoded result */
func (e *Encryption) EncryptString(plaintext string) (string, error) {
	ciphertext, err := e.Encrypt([]byte(plaintext))
	if err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(ciphertext), nil
}

/* DecryptString decrypts a base64-encoded string */
func (e *Encryption) DecryptString(ciphertext string) (string, error) {
	data, err := base64.StdEncoding.DecodeString(ciphertext)
	if err != nil {
		return "", fmt.Errorf("decryption failed: base64_decode_error=true, error=%w", err)
	}

	plaintext, err := e.Decrypt(data)
	if err != nil {
		return "", err
	}

	return string(plaintext), nil
}



