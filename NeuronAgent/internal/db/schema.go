/*-------------------------------------------------------------------------
 *
 * schema.go
 *    Database operations
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/db/schema.go
 *
 *-------------------------------------------------------------------------
 */

package db

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/jmoiron/sqlx"
)

type Migration struct {
	Version int
	Name    string
	SQL     string
}

type SchemaManager struct {
	db         *sqlx.DB
	migrations []Migration
}

func NewSchemaManager(db *sqlx.DB) *SchemaManager {
	return &SchemaManager{
		db:         db,
		migrations: []Migration{},
	}
}

/* LoadMigrations loads migrations from directory */
func (sm *SchemaManager) LoadMigrations(dir string) error {
	files, err := os.ReadDir(dir)
	if err != nil {
		return fmt.Errorf("failed to read migrations directory: %w", err)
	}

	for _, file := range files {
		if !strings.HasSuffix(file.Name(), ".sql") {
			continue
		}

		/* Parse version from filename (e.g., "001_initial_schema.sql" -> 1) */
		var version int
		var name string
		parts := strings.SplitN(strings.TrimSuffix(file.Name(), ".sql"), "_", 2)
		if len(parts) >= 1 {
			fmt.Sscanf(parts[0], "%d", &version)
		}
		if len(parts) >= 2 {
			name = parts[1]
		}

		/* Read SQL file */
		path := filepath.Join(dir, file.Name())
		sql, err := os.ReadFile(path)
		if err != nil {
			return fmt.Errorf("failed to read migration file %s: %w", file.Name(), err)
		}

		sm.migrations = append(sm.migrations, Migration{
			Version: version,
			Name:    name,
			SQL:     string(sql),
		})
	}

	/* Sort by version */
	sort.Slice(sm.migrations, func(i, j int) bool {
		return sm.migrations[i].Version < sm.migrations[j].Version
	})

	return nil
}

/* GetCurrentVersion gets the current migration version */
func (sm *SchemaManager) GetCurrentVersion(ctx context.Context) (int, error) {
	/* Check if schema_migrations table exists */
	var exists bool
	err := sm.db.GetContext(ctx, &exists, `
		SELECT EXISTS (
			SELECT FROM information_schema.tables 
			WHERE table_schema = 'neurondb_agent' 
			AND table_name = 'schema_migrations'
		)
	`)
	if err != nil || !exists {
		return 0, nil
	}

	var version int
	err = sm.db.GetContext(ctx, &version, `
		SELECT version FROM neurondb_agent.schema_migrations 
		ORDER BY version DESC LIMIT 1
	`)
	if err != nil {
		return 0, nil
	}

	return version, nil
}

/* Migrate runs all pending migrations */
func (sm *SchemaManager) Migrate(ctx context.Context) error {
	/* Create schema_migrations table if it doesn't exist */
	_, err := sm.db.ExecContext(ctx, `
		CREATE SCHEMA IF NOT EXISTS neurondb_agent;
		CREATE TABLE IF NOT EXISTS neurondb_agent.schema_migrations (
			version INT PRIMARY KEY,
			name TEXT NOT NULL,
			applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
		);
	`)
	if err != nil {
		return fmt.Errorf("failed to create schema_migrations table: %w", err)
	}

	currentVersion, err := sm.GetCurrentVersion(ctx)
	if err != nil {
		return fmt.Errorf("failed to get current version: %w", err)
	}

	/* Run pending migrations */
	for _, migration := range sm.migrations {
		if migration.Version <= currentVersion {
			continue
		}

		/* Run migration in transaction */
		tx, err := sm.db.BeginTxx(ctx, nil)
		if err != nil {
			return fmt.Errorf("failed to begin transaction: %w", err)
		}

		_, err = tx.ExecContext(ctx, migration.SQL)
		if err != nil {
			tx.Rollback()
			return fmt.Errorf("failed to run migration %d: %w", migration.Version, err)
		}

		/* Record migration */
		_, err = tx.ExecContext(ctx, `
			INSERT INTO neurondb_agent.schema_migrations (version, name)
			VALUES ($1, $2)
		`, migration.Version, migration.Name)
		if err != nil {
			tx.Rollback()
			return fmt.Errorf("failed to record migration %d: %w", migration.Version, err)
		}

		if err := tx.Commit(); err != nil {
			return fmt.Errorf("failed to commit migration %d: %w", migration.Version, err)
		}
	}

	return nil
}

/* Rollback rolls back the last migration (if supported) */
func (sm *SchemaManager) Rollback(ctx context.Context) error {
	currentVersion, err := sm.GetCurrentVersion(ctx)
	if err != nil {
		return fmt.Errorf("failed to get current version: %w", err)
	}

	if currentVersion == 0 {
		return fmt.Errorf("no migrations to rollback")
	}

	/* Find migration to rollback */
	var migrationToRollback *Migration
	for _, m := range sm.migrations {
		if m.Version == currentVersion {
			migrationToRollback = &m
			break
		}
	}

	if migrationToRollback == nil {
		return fmt.Errorf("migration version %d not found", currentVersion)
	}

	/* Note: Full rollback requires storing rollback SQL */
	/* For now, we just remove the version record */
	_, err = sm.db.ExecContext(ctx, `
		DELETE FROM neurondb_agent.schema_migrations 
		WHERE version = $1
	`, currentVersion)
	if err != nil {
		return fmt.Errorf("failed to rollback migration: %w", err)
	}

	return nil
}
