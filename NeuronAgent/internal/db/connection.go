/*-------------------------------------------------------------------------
 *
 * connection.go
 *    Database connection management for NeuronAgent
 *
 * Provides PostgreSQL connection pooling, retry logic, and connection
 * management with health checks.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/db/connection.go
 *
 *-------------------------------------------------------------------------
 */

package db

import (
	"context"
	"fmt"
	"math/rand"
	"strings"
	"time"

	"github.com/jmoiron/sqlx"
	_ "github.com/lib/pq"
	"github.com/neurondb/NeuronAgent/internal/metrics"
	"github.com/neurondb/NeuronAgent/internal/utils"
)

/* ConnectionInfo holds details about the database connection */
type ConnectionInfo struct {
	Host     string
	Port     int
	Database string
	User     string
}

/* DB manages PostgreSQL connections */
type DB struct {
	*sqlx.DB
	poolConfig PoolConfig
	connInfo   *ConnectionInfo
}

type PoolConfig struct {
	MaxOpenConns    int
	MaxIdleConns    int
	ConnMaxLifetime time.Duration
	ConnMaxIdleTime time.Duration
}

/* NewDB creates a new database instance */
func NewDB(connStr string, poolConfig PoolConfig) (*DB, error) {
	return NewDBWithRetry(connStr, poolConfig, 3, 2*time.Second)
}

/* NewDBWithRetry creates a new database instance with retry logic */
func NewDBWithRetry(connStr string, poolConfig PoolConfig, maxRetries int, retryDelay time.Duration) (*DB, error) {
	connInfo := parseConnectionInfo(connStr)

	var db *sqlx.DB
	var err error

	for attempt := 0; attempt < maxRetries; attempt++ {
		db, err = sqlx.Connect("postgres", connStr)
		if err == nil {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			pingErr := db.PingContext(ctx)
			cancel()
			if pingErr == nil {
				db.SetMaxOpenConns(poolConfig.MaxOpenConns)
				db.SetMaxIdleConns(poolConfig.MaxIdleConns)
				db.SetConnMaxLifetime(poolConfig.ConnMaxLifetime)
				db.SetConnMaxIdleTime(poolConfig.ConnMaxIdleTime)
				
				/* Set search_path and timezone on the connection pool */
				/* Note: This sets it for the initial connection, but we need it for all connections */
				/* We'll add it to the connection string instead */
				
				/* Test the connection works */
				_, err = db.Exec("SELECT 1")
				if err != nil {
					db.Close()
					/* Log retry attempt for connection test failure */
					metrics.WarnWithContext(context.Background(), "Database connection test failed, will retry", map[string]interface{}{
						"attempt":      attempt + 1,
						"max_retries":  maxRetries,
						"error":        err.Error(),
						"connection":   connInfo.Host,
					})
					/* Retry connection test failures */
					if attempt < maxRetries-1 {
						delay := retryDelay
						/* Add jitter: ±25% variation to prevent thundering herd */
						jitter := float64(delay) * 0.25
						jitterAmount := time.Duration(jitter * (rand.Float64()*2 - 1)) /* -0.25 to +0.25 */
						delay = delay + jitterAmount
						time.Sleep(delay)
						retryDelay *= 2
					}
					continue
				}

				metrics.InfoWithContext(context.Background(), "Database connection established", map[string]interface{}{
					"attempt":    attempt + 1,
					"connection": connInfo.Host,
					"database":   connInfo.Database,
				})

				return &DB{
					DB:         db,
					poolConfig: poolConfig,
					connInfo:   connInfo,
				}, nil
			}
			db.Close()
			err = pingErr
		}

		if attempt < maxRetries-1 {
			/* Add jitter: ±25% variation to prevent thundering herd */
			delay := retryDelay
			jitter := float64(delay) * 0.25
			jitterAmount := time.Duration(jitter * (rand.Float64()*2 - 1)) /* -0.25 to +0.25 */
			delay = delay + jitterAmount
			
			/* Log retry attempt */
			metrics.WarnWithContext(context.Background(), "Database connection failed, retrying", map[string]interface{}{
				"attempt":      attempt + 1,
				"max_retries":  maxRetries,
				"retry_delay":  delay.String(),
				"error":        err.Error(),
				"connection":   connInfo.Host,
			})
			
			time.Sleep(delay)
			retryDelay *= 2
		}
	}

	connInfoStr := utils.FormatConnectionInfo(connInfo.Host, connInfo.Port, connInfo.Database, connInfo.User)
	return nil, fmt.Errorf("failed to connect to %s after %d attempts (last error: %w)", connInfoStr, maxRetries, err)
}

/* parseConnectionInfo extracts connection information from connection string */
func parseConnectionInfo(connStr string) *ConnectionInfo {
	info := &ConnectionInfo{
		Host:     "unknown",
		Port:     5432,
		Database: "unknown",
		User:     "unknown",
	}

	parts := strings.Split(connStr, " ")
	for _, part := range parts {
		if strings.HasPrefix(part, "host=") {
			info.Host = strings.TrimPrefix(part, "host=")
		} else if strings.HasPrefix(part, "port=") {
			fmt.Sscanf(strings.TrimPrefix(part, "port="), "%d", &info.Port)
		} else if strings.HasPrefix(part, "dbname=") {
			info.Database = strings.TrimPrefix(part, "dbname=")
		} else if strings.HasPrefix(part, "user=") {
			info.User = strings.TrimPrefix(part, "user=")
		}
	}

	return info
}

/* GetConnInfoString returns a formatted string of connection details */
func (d *DB) GetConnInfoString() string {
	if d.connInfo == nil {
		return "unknown database connection"
	}
	return utils.FormatConnectionInfo(d.connInfo.Host, d.connInfo.Port, d.connInfo.Database, d.connInfo.User)
}

/* HealthCheck tests the database connection */
func (d *DB) HealthCheck(ctx context.Context) error {
	if d.DB == nil {
		return fmt.Errorf("database connection not established: %s (connection pool is nil, ensure NewDB() was called successfully)", d.GetConnInfoString())
	}

	var result int
	err := d.DB.GetContext(ctx, &result, "SELECT 1")
	if err != nil {
		return fmt.Errorf("health check failed on %s: query='SELECT 1', error=%w", d.GetConnInfoString(), err)
	}
	return nil
}

/* GetPoolStats returns connection pool statistics */
func (d *DB) GetPoolStats() (openConns, idleConns, inUse int) {
	if d.DB == nil {
		return 0, 0, 0
	}
	stats := d.DB.Stats()
	return stats.OpenConnections, stats.Idle, stats.InUse
}

/* Close closes the connection pool */
func (d *DB) Close() error {
	if d.DB == nil {
		return nil
	}
	return d.DB.Close()
}
