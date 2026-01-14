/*-------------------------------------------------------------------------
 *
 * db_storage.go
 *    Database storage backend for virtual file system
 *
 * Stores file content directly in PostgreSQL database. Suitable for
 * small files (< 1MB). Uses BYTEA column for binary data.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/storage/db_storage.go
 *
 *-------------------------------------------------------------------------
 */

package storage

import (
	"context"
	"fmt"

	"github.com/neurondb/NeuronAgent/internal/db"
)

/* DatabaseStorage implements database storage backend */
type DatabaseStorage struct {
	queries *db.Queries
}

/* NewDatabaseStorage creates a new database storage backend */
func NewDatabaseStorage(config map[string]interface{}) (*DatabaseStorage, error) {
	queries, ok := config["queries"].(*db.Queries)
	if !ok {
		return nil, fmt.Errorf("database storage requires queries in config")
	}

	return &DatabaseStorage{
		queries: queries,
	}, nil
}

/* Store stores content in database */
func (d *DatabaseStorage) Store(ctx context.Context, key string, content []byte) error {
	/* Database storage uses the virtual_files table directly */
	/* This method is a no-op as storage happens in virtual_fs.go */
	return nil
}

/* Retrieve retrieves content from database */
func (d *DatabaseStorage) Retrieve(ctx context.Context, key string) ([]byte, error) {
	/* Database storage uses the virtual_files table directly */
	/* This method is a no-op as retrieval happens in virtual_fs.go */
	return nil, fmt.Errorf("database storage retrieval handled by virtual_fs")
}

/* Delete deletes content from database */
func (d *DatabaseStorage) Delete(ctx context.Context, key string) error {
	/* Database storage uses the virtual_files table directly */
	/* This method is a no-op as deletion happens in virtual_fs.go */
	return nil
}

/* Exists checks if content exists in database */
func (d *DatabaseStorage) Exists(ctx context.Context, key string) (bool, error) {
	/* Database storage uses the virtual_files table directly */
	/* This method is a no-op as existence check happens in virtual_fs.go */
	return false, nil
}
