/*-------------------------------------------------------------------------
 *
 * transactions.go
 *    Database transaction management for NeuronAgent
 *
 * Provides transaction handling, retry logic, and transaction utilities
 * for safe database operations with automatic rollback on errors.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/db/transactions.go
 *
 *-------------------------------------------------------------------------
 */

package db

import (
	"context"
	"fmt"
	"time"

	"github.com/jmoiron/sqlx"
)

/* Transaction represents a database transaction */
type Transaction struct {
	tx *sqlx.Tx
}

/* BeginTransaction begins a new transaction */
func (d *DB) BeginTransaction(ctx context.Context) (*Transaction, error) {
	tx, err := d.BeginTxx(ctx, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to begin transaction: %w", err)
	}
	return &Transaction{tx: tx}, nil
}

/* Commit commits the transaction */
func (t *Transaction) Commit() error {
	return t.tx.Commit()
}

/* Rollback rolls back the transaction */
func (t *Transaction) Rollback() error {
	return t.tx.Rollback()
}

/* Exec executes a query in the transaction */
func (t *Transaction) Exec(ctx context.Context, query string, args ...interface{}) error {
	_, err := t.tx.ExecContext(ctx, query, args...)
	return err
}

/* Query executes a query and returns rows */
func (t *Transaction) Query(ctx context.Context, query string, args ...interface{}) (*sqlx.Rows, error) {
	return t.tx.QueryxContext(ctx, query, args...)
}

/* Get executes a query and scans into dest */
func (t *Transaction) Get(ctx context.Context, dest interface{}, query string, args ...interface{}) error {
	return t.tx.GetContext(ctx, dest, query, args...)
}

/* Select executes a query and scans into dest slice */
func (t *Transaction) Select(ctx context.Context, dest interface{}, query string, args ...interface{}) error {
	return t.tx.SelectContext(ctx, dest, query, args...)
}

/* RetryWithBackoff retries a function with exponential backoff */
func RetryWithBackoff(ctx context.Context, maxRetries int, fn func() error) error {
	var lastErr error
	backoff := 100 * time.Millisecond

	for i := 0; i < maxRetries; i++ {
		if err := fn(); err == nil {
			return nil
		} else {
			lastErr = err
		}

		if i < maxRetries-1 {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(backoff):
				backoff *= 2 /* Exponential backoff */
			}
		}
	}

	return fmt.Errorf("max retries exceeded: %w", lastErr)
}

/* WithTransaction executes a function within a transaction */
func (d *DB) WithTransaction(ctx context.Context, fn func(*Transaction) error) error {
	tx, err := d.BeginTransaction(ctx)
	if err != nil {
		return err
	}

	defer func() {
		if p := recover(); p != nil {
			rollbackErr := tx.Rollback()
			if rollbackErr != nil {
				err = fmt.Errorf("transaction panic occurred (%v) and rollback failed: %w", p, rollbackErr)
			} else {
				err = fmt.Errorf("transaction panic occurred: %v", p)
			}
		} else if err != nil {
			rollbackErr := tx.Rollback()
			if rollbackErr != nil {
				err = fmt.Errorf("transaction error (%w) and rollback failed: %w", err, rollbackErr)
			}
		} else {
			commitErr := tx.Commit()
			if commitErr != nil {
				err = fmt.Errorf("transaction commit failed: %w", commitErr)
			}
		}
	}()

	err = fn(tx)
	return err
}
