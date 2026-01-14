/*-------------------------------------------------------------------------
 *
 * storage.go
 *    Storage backend interface for virtual file system
 *
 * Provides abstraction for different storage backends (database, S3, etc.)
 * for the virtual file system.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/storage/storage.go
 *
 *-------------------------------------------------------------------------
 */

package storage

import (
	"context"
	"fmt"
)

/* StorageBackend interface for storage implementations */
type StorageBackend interface {
	Store(ctx context.Context, key string, content []byte) error
	Retrieve(ctx context.Context, key string) ([]byte, error)
	Delete(ctx context.Context, key string) error
	Exists(ctx context.Context, key string) (bool, error)
}

/* NewStorageBackend creates a storage backend based on type */
func NewStorageBackend(backendType string, config map[string]interface{}) (StorageBackend, error) {
	switch backendType {
	case "database":
		return NewDatabaseStorage(config)
	case "s3":
		return NewS3Storage(config)
	default:
		return nil, fmt.Errorf("unsupported storage backend: %s", backendType)
	}
}
