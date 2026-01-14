/*-------------------------------------------------------------------------
 *
 * migrations.go
 *    Database operations
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/db/migrations.go
 *
 *-------------------------------------------------------------------------
 */

package db

import (
	"context"
	"fmt"
	"path/filepath"

	"github.com/jmoiron/sqlx"
)

type MigrationRunner struct {
	db            *sqlx.DB
	schemaMgr     *SchemaManager
	migrationsDir string
}

func NewMigrationRunner(db *sqlx.DB, migrationsDir string) (*MigrationRunner, error) {
	schemaMgr := NewSchemaManager(db)

	/* Get absolute path */
	absPath, err := filepath.Abs(migrationsDir)
	if err != nil {
		return nil, fmt.Errorf("failed to get absolute path: %w", err)
	}

	runner := &MigrationRunner{
		db:            db,
		schemaMgr:     schemaMgr,
		migrationsDir: absPath,
	}

	/* Load migrations */
	if err := schemaMgr.LoadMigrations(absPath); err != nil {
		return nil, fmt.Errorf("failed to load migrations: %w", err)
	}

	return runner, nil
}

/* Run runs all pending migrations */
func (mr *MigrationRunner) Run(ctx context.Context) error {
	return mr.schemaMgr.Migrate(ctx)
}

/* Status returns migration status */
func (mr *MigrationRunner) Status(ctx context.Context) (int, int, error) {
	current, err := mr.schemaMgr.GetCurrentVersion(ctx)
	if err != nil {
		return 0, 0, err
	}
	total := len(mr.schemaMgr.migrations)
	return current, total, nil
}

/* Rollback rolls back the last migration */
func (mr *MigrationRunner) Rollback(ctx context.Context) error {
	return mr.schemaMgr.Rollback(ctx)
}
