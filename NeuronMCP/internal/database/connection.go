/*-------------------------------------------------------------------------
 *
 * connection.go
 *    Database connection management for NeuronMCP
 *
 * Provides PostgreSQL connection pooling, retry logic, and connection
 * management with NeuronDB type registration.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/database/connection.go
 *
 *-------------------------------------------------------------------------
 */

package database

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgtype"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/neurondb/NeuronMCP/internal/config"
	"github.com/neurondb/NeuronMCP/internal/security"
	"github.com/neurondb/NeuronMCP/internal/validation"
)

/* ConnectionState represents the state of a database connection */
type ConnectionState int

const (
	StateDisconnected ConnectionState = iota
	StateConnecting
	StateConnected
	StateFailed
)

/* Database manages PostgreSQL connections */
type Database struct {
	pool      *pgxpool.Pool
	host      string
	port      int
	database  string
	user      string
	state     ConnectionState
	lastError error
	mu        sync.RWMutex /* Protects state and lastError fields */
}

/* NewDatabase creates a new database instance */
func NewDatabase() *Database {
	return &Database{
		state: StateDisconnected,
	}
}

/* Connect connects to the database using the provided configuration */
func (d *Database) Connect(cfg *config.DatabaseConfig) error {
	if cfg == nil {
		return fmt.Errorf("database configuration cannot be nil")
	}
	return d.ConnectWithRetry(cfg, 3, 2*time.Second)
}

/* ConnectWithRetry connects to the database with retry logic */
func (d *Database) ConnectWithRetry(cfg *config.DatabaseConfig, maxRetries int, retryDelay time.Duration) error {
	if cfg == nil {
		return fmt.Errorf("database configuration cannot be nil")
	}
	if maxRetries < 1 {
		maxRetries = 1
	}
	if retryDelay < 0 {
		retryDelay = 2 * time.Second
	}

	/* Use pgxpool.ParseConfig with individual fields instead of string concatenation */
	/* This prevents SQL injection and properly handles special characters in passwords */
	var poolConfig *pgxpool.Config
	var err error
	
	if cfg.ConnectionString != nil && *cfg.ConnectionString != "" {
		/* If connection string is provided, parse it */
		poolConfig, err = pgxpool.ParseConfig(*cfg.ConnectionString)
	} else {
		/* Build config from individual fields - safer than string concatenation */
		host := cfg.GetHost()
		port := cfg.GetPort()
		db := cfg.GetDatabase()
		user := cfg.GetUser()
		
		/* Create connection string without password first */
		connStr := fmt.Sprintf("host=%s port=%d user=%s dbname=%s",
			host, port, user, db)
		
		/* Add SSL mode */
		if cfg.SSL != nil {
			if sslBool, ok := cfg.SSL.(bool); ok {
				if sslBool {
					connStr += " sslmode=require"
				} else {
					connStr += " sslmode=disable"
				}
			} else if sslStr, ok := cfg.SSL.(string); ok {
				connStr += fmt.Sprintf(" sslmode=%s", sslStr)
			}
		} else {
			connStr += " sslmode=prefer"
		}
		
		/* Parse config - this handles password escaping properly */
		poolConfig, err = pgxpool.ParseConfig(connStr)
		if err == nil && cfg.Password != nil && *cfg.Password != "" {
			/* Set password directly in config - pgx handles escaping */
			poolConfig.ConnConfig.Password = *cfg.Password
		}
	}
	if err != nil {
		host := cfg.GetHost()
		port := cfg.GetPort()
		db := cfg.GetDatabase()
		user := cfg.GetUser()
		/* Sanitize error to remove any sensitive information */
		sanitizedErr := security.SanitizeError(err)
		return fmt.Errorf("failed to parse connection string for database '%s' on host '%s:%d' as user '%s': %w (connection string format may be invalid)", db, host, port, user, sanitizedErr)
	}

	/* Register NeuronDB custom types (vector, vector[], etc.) */
	poolConfig.AfterConnect = func(ctx context.Context, conn *pgx.Conn) error {
		conn.TypeMap().RegisterType(&pgtype.Type{
			Codec: &pgtype.TextCodec{},
			Name:  "vector",
			OID:   17648,
		})
		conn.TypeMap().RegisterType(&pgtype.Type{
			Codec: &pgtype.ArrayCodec{ElementType: &pgtype.Type{Name: "text", Codec: &pgtype.TextCodec{}}},
			Name:  "_vector",
			OID:   17656,
		})
		return nil
	}

	if cfg.Pool != nil {
		poolConfig.MinConns = int32(cfg.Pool.GetMin())
		poolConfig.MaxConns = int32(cfg.Pool.GetMax())
		poolConfig.MaxConnIdleTime = cfg.Pool.GetIdleTimeout()
		poolConfig.MaxConnLifetime = time.Hour
		poolConfig.HealthCheckPeriod = 30 * time.Second /* Enhanced health checks */
		/* Enable connection health checks */
		poolConfig.ConnConfig.ConnectTimeout = cfg.Pool.GetConnectionTimeout()
	} else {
		poolConfig.MinConns = 2 /* Default minimum for better performance */
		poolConfig.MaxConns = 10
		poolConfig.HealthCheckPeriod = 30 * time.Second
	}

	var host, dbName, dbUser string
	var dbPort int
	if cfg.ConnectionString != nil && *cfg.ConnectionString != "" {
		host = "unknown"
		dbName = "unknown"
		dbUser = "unknown"
		dbPort = 0
	} else {
		host = cfg.GetHost()
		dbPort = cfg.GetPort()
		dbName = cfg.GetDatabase()
		dbUser = cfg.GetUser()
	}
	d.host = host
	d.port = dbPort
	d.database = dbName
	d.user = dbUser

	var pool *pgxpool.Pool
	var lastErr error
	baseDelay := retryDelay
	
	d.mu.Lock()
	d.state = StateConnecting
	d.lastError = nil
	d.mu.Unlock()
	
	for attempt := 0; attempt < maxRetries; attempt++ {
		pool, err = pgxpool.NewWithConfig(context.Background(), poolConfig)
		if err == nil {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			pingErr := pool.Ping(ctx)
			cancel()
			
			if pingErr == nil {
				d.mu.Lock()
				d.pool = pool
				d.state = StateConnected
				d.lastError = nil
				d.mu.Unlock()
				return nil
			}
			/* Sanitize error to remove sensitive information */
			sanitizedPingErr := security.SanitizeError(pingErr)
			lastErr = fmt.Errorf("connection ping failed: database '%s' on host '%s:%d' as user '%s': %w", dbName, host, dbPort, dbUser, sanitizedPingErr)
			if pool != nil {
				pool.Close()
			}
		} else {
			/* Sanitize error to remove sensitive information */
			sanitizedErr := security.SanitizeError(err)
			lastErr = fmt.Errorf("failed to create connection pool: database '%s' on host '%s:%d' as user '%s': %w", dbName, host, dbPort, dbUser, sanitizedErr)
		}

		if attempt < maxRetries-1 {
			currentDelay := baseDelay * time.Duration(1<<uint(attempt))
			time.Sleep(currentDelay)
		}
	}

	d.mu.Lock()
	d.state = StateFailed
	/* Sanitize error before storing */
	d.lastError = security.SanitizeError(lastErr)
	d.mu.Unlock()
	/* Sanitize error in return message */
	sanitizedLastErr := security.SanitizeError(lastErr)
	return fmt.Errorf("failed to connect to database '%s' on host '%s:%d' as user '%s' after %d attempts (last error: %v)", dbName, host, dbPort, dbUser, maxRetries, sanitizedLastErr)
}

/* IsConnected checks if the database is connected */
func (d *Database) IsConnected() bool {
	if d == nil {
		return false
	}
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.pool != nil && d.state == StateConnected
}

/* GetConnectionState returns the current connection state */
func (d *Database) GetConnectionState() ConnectionState {
	if d == nil {
		return StateDisconnected
	}
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.state
}

/* GetLastError returns the last connection error */
func (d *Database) GetLastError() error {
	if d == nil {
		return nil
	}
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.lastError
}

/* HealthCheck performs a health check on the database connection */
func (d *Database) HealthCheck(ctx context.Context) error {
	if d == nil || d.pool == nil {
		return fmt.Errorf("database connection not established")
	}
	if d.state != StateConnected {
		return fmt.Errorf("database connection is not in connected state: state=%d", d.state)
	}
	err := d.pool.Ping(ctx)
	if err != nil {
		d.mu.Lock()
		d.state = StateFailed
		d.lastError = err
		d.mu.Unlock()
		return fmt.Errorf("health check failed: %w", err)
	}
	return nil
}

/* Query executes a query and returns rows with automatic reconnection */
func (d *Database) Query(ctx context.Context, query string, args ...interface{}) (pgx.Rows, error) {
	if d == nil || d.pool == nil {
		db, host, port, user := d.getDBInfo()
		return nil, fmt.Errorf("database connection not established: database '%s' on host '%s:%d' as user '%s' (connection pool is nil, ensure Connect() was called successfully)", db, host, port, user)
	}
	
	if err := d.pool.Ping(ctx); err != nil {
		db, host, port, user := d.getDBInfo()
		return nil, fmt.Errorf("database connection lost: database '%s' on host '%s:%d' as user '%s': %w (connection pool ping failed, may need to reconnect)", db, host, port, user, err)
	}
	
	rows, err := d.pool.Query(ctx, query, args...)
	if err != nil {
		db, host, port, user := d.getDBInfo()
		return nil, fmt.Errorf("query execution failed on database '%s' on host '%s:%d' as user '%s': query='%s', error=%w", db, host, port, user, query, err)
	}
	return rows, nil
}

/* QueryRow executes a query and returns a single row */
func (d *Database) QueryRow(ctx context.Context, query string, args ...interface{}) pgx.Row {
	if d == nil || d.pool == nil {
		db, host, port, user := d.getDBInfo()
		return &errorRow{err: fmt.Errorf("database connection not established: database '%s' on host '%s:%d' as user '%s' (connection pool is nil, ensure Connect() was called successfully)", db, host, port, user)}
	}
	return d.pool.QueryRow(ctx, query, args...)
}

/* getDBInfo returns database connection info for error messages */
func (d *Database) getDBInfo() (string, string, int, string) {
	if d == nil {
		return "unknown", "unknown", 0, "unknown"
	}
	return d.database, d.host, d.port, d.user
}

/* Exec executes a query without returning rows */
func (d *Database) Exec(ctx context.Context, query string, args ...interface{}) (pgconn.CommandTag, error) {
	if d == nil || d.pool == nil {
		db, host, port, user := d.getDBInfo()
		return pgconn.CommandTag{}, fmt.Errorf("database connection not established: database '%s' on host '%s:%d' as user '%s' (connection pool is nil, ensure Connect() was called successfully)", db, host, port, user)
	}
	tag, err := d.pool.Exec(ctx, query, args...)
	if err != nil {
		db, host, port, user := d.getDBInfo()
		return pgconn.CommandTag{}, fmt.Errorf("query execution failed on database '%s' on host '%s:%d' as user '%s': query='%s', error=%w", db, host, port, user, query, err)
	}
	return tag, nil
}

/* Begin starts a transaction */
func (d *Database) Begin(ctx context.Context) (pgx.Tx, error) {
	if d == nil || d.pool == nil {
		db, host, port, user := d.getDBInfo()
		return nil, fmt.Errorf("database connection not established: database '%s' on host '%s:%d' as user '%s' (connection pool is nil, ensure Connect() was called successfully)", db, host, port, user)
	}
	tx, err := d.pool.Begin(ctx)
	if err != nil {
		db, host, port, user := d.getDBInfo()
		return nil, fmt.Errorf("failed to begin transaction on database '%s' on host '%s:%d' as user '%s': %w", db, host, port, user, err)
	}
	return tx, nil
}

/* Close closes the connection pool */
/* Safe to call multiple times - uses mutex and nil checks to prevent panics */
func (d *Database) Close() {
	if d == nil {
		return
	}
	
	d.mu.Lock()
	
	/* Check if already closed */
	if d.pool == nil {
		d.state = StateDisconnected
		d.mu.Unlock()
		return
	}
	
	/* Get reference to pool and clear it */
	pool := d.pool
	d.pool = nil
	d.state = StateDisconnected
	d.lastError = nil
	
	/* Unlock before closing to avoid holding lock during I/O */
	/* pool.Close() may take time and we don't want to block other operations */
	d.mu.Unlock()
	
	/* Close the pool (this may take time) */
	pool.Close()
}

/* TestConnection tests the database connection */
func (d *Database) TestConnection(ctx context.Context) error {
	if d == nil || d.pool == nil {
		db, host, port, user := d.getDBInfo()
		return fmt.Errorf("database connection not established: database '%s' on host '%s:%d' as user '%s' (connection pool is nil, ensure Connect() was called successfully)", db, host, port, user)
	}
	err := d.pool.Ping(ctx)
	if err != nil {
		db, host, port, user := d.getDBInfo()
		return fmt.Errorf("connection test failed for database '%s' on host '%s:%d' as user '%s': %w", db, host, port, user, err)
	}
	return nil
}

/* GetPoolStats returns pool statistics */
func (d *Database) GetPoolStats() *PoolStats {
	if d == nil || d.pool == nil {
		return nil
	}
	stats := d.pool.Stat()
	if stats == nil {
		return nil
	}
	return &PoolStats{
		TotalConns:       stats.TotalConns(),
		AcquiredConns:    stats.AcquiredConns(),
		IdleConns:        stats.IdleConns(),
		ConstructingConns: stats.ConstructingConns(),
	}
}

/* PoolStats holds connection pool statistics */
type PoolStats struct {
	TotalConns      int32
	AcquiredConns   int32
	IdleConns       int32
	ConstructingConns int32
}

/* EscapeIdentifier escapes a SQL identifier for safe use */
/* Uses validation package for proper escaping */
func EscapeIdentifier(identifier string) string {
	return validation.EscapeSQLIdentifier(identifier)
}

/* errorRow is a row that always returns an error */
type errorRow struct {
	err error
}

func (r *errorRow) Scan(dest ...interface{}) error {
	return r.err
}

