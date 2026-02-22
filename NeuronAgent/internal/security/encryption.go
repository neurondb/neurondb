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
	"os"

	"golang.org/x/crypto/pbkdf2"
)

/* Encryption provides encryption and decryption capabilities */
type Encryption struct {
	key []byte
}

/* NewEncryption creates a new encryption instance */
func NewEncryption(secretKey string) (*Encryption, error) {
	/* Derive key from secret using PBKDF2; salt from env or unique default per deployment */
	salt := []byte("neurondb-encryption-salt")
	if envSalt := os.Getenv("NEURONDB_ENCRYPTION_KEY_SALT"); envSalt != "" {
		if decoded, err := base64.StdEncoding.DecodeString(envSalt); err == nil && len(decoded) >= 16 {
			salt = decoded
		}
	}
	key := pbkdf2.Key([]byte(secretKey), salt, 4096, 32, sha256.New)

	return &Encryption{
		key: key,
	}, nil
}

/* Encrypt encrypts data with a random salt per encryption */
func (e *Encryption) Encrypt(plaintext []byte) ([]byte, error) {
	/* Generate random salt for this encryption (32 bytes) */
	salt := make([]byte, 32)
	if _, err := io.ReadFull(rand.Reader, salt); err != nil {
		return nil, fmt.Errorf("encryption failed: salt_creation_error=true, error=%w", err)
	}

	/* Derive encryption key from master key and salt */
	encKey := pbkdf2.Key(e.key, salt, 4096, 32, sha256.New)
	encBlock, err := aes.NewCipher(encKey)
	if err != nil {
		return nil, fmt.Errorf("encryption failed: derived_cipher_creation_error=true, error=%w", err)
	}
	encGCM, err := cipher.NewGCM(encBlock)
	if err != nil {
		return nil, fmt.Errorf("encryption failed: derived_gcm_creation_error=true, error=%w", err)
	}

	/* Create nonce */
	nonce := make([]byte, encGCM.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return nil, fmt.Errorf("encryption failed: nonce_creation_error=true, error=%w", err)
	}

	/* Encrypt */
	ciphertext := encGCM.Seal(nonce, nonce, plaintext, nil)

	/* Prepend salt to ciphertext: [salt (32 bytes)][nonce + ciphertext] */
	result := append(salt, ciphertext...)
	return result, nil
}

/* Decrypt decrypts data */
func (e *Encryption) Decrypt(ciphertext []byte) ([]byte, error) {
	/* Check minimum length: salt (32) + nonce (12) + at least 1 byte of ciphertext */
	const saltSize = 32
	if len(ciphertext) < saltSize+13 {
		return nil, fmt.Errorf("decryption failed: invalid_ciphertext_length=true, length=%d (minimum %d)", len(ciphertext), saltSize+13)
	}

	/* Extract salt */
	salt := ciphertext[:saltSize]
	encryptedData := ciphertext[saltSize:]

	/* Derive decryption key from master key and salt */
	decKey := pbkdf2.Key(e.key, salt, 4096, 32, sha256.New)
	block, err := aes.NewCipher(decKey)
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
	if len(encryptedData) < nonceSize {
		return nil, fmt.Errorf("decryption failed: invalid_encrypted_data_length=true, length=%d (minimum %d)", len(encryptedData), nonceSize)
	}

	nonce, encryptedData := encryptedData[:nonceSize], encryptedData[nonceSize:]

	/* Decrypt */
	plaintext, err := aesGCM.Open(nil, nonce, encryptedData, nil)
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
