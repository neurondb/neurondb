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
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgtype"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/neurondb/NeuronMCP/internal/config"
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
	pool     *pgxpool.Pool
	host     string
	port     int
	database string
	user     string
	state    ConnectionState
	lastError error
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

	var connStr string
	var err error

	if cfg.ConnectionString != nil && *cfg.ConnectionString != "" {
		connStr = *cfg.ConnectionString
	} else {
		host := cfg.GetHost()
		port := cfg.GetPort()
		db := cfg.GetDatabase()
		user := cfg.GetUser()
		password := ""
		if cfg.Password != nil {
			password = *cfg.Password
		}

		connStr = fmt.Sprintf("host=%s port=%d user=%s dbname=%s",
			host, port, user, db)
		
		if password != "" {
			connStr += fmt.Sprintf(" password=%s", password)
		}

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
	}

	poolConfig, err := pgxpool.ParseConfig(connStr)
	if err != nil {
		host := cfg.GetHost()
		port := cfg.GetPort()
		db := cfg.GetDatabase()
		user := cfg.GetUser()
		return fmt.Errorf("failed to parse connection string for database '%s' on host '%s:%d' as user '%s': %w (connection string format may be invalid)", db, host, port, user, err)
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
		poolConfig.HealthCheckPeriod = 1 * time.Minute
	} else {
		poolConfig.MinConns = 0
		poolConfig.MaxConns = 10
		poolConfig.HealthCheckPeriod = 1 * time.Minute
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
	
	d.state = StateConnecting
	d.lastError = nil
	
	for attempt := 0; attempt < maxRetries; attempt++ {
		pool, err = pgxpool.NewWithConfig(context.Background(), poolConfig)
		if err == nil {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			pingErr := pool.Ping(ctx)
			cancel()
			
			if pingErr == nil {
				d.pool = pool
				d.state = StateConnected
				d.lastError = nil
				return nil
			}
			lastErr = fmt.Errorf("connection ping failed: database '%s' on host '%s:%d' as user '%s': %w", dbName, host, dbPort, dbUser, pingErr)
			if pool != nil {
				pool.Close()
			}
		} else {
			lastErr = fmt.Errorf("failed to create connection pool: database '%s' on host '%s:%d' as user '%s': %w", dbName, host, dbPort, dbUser, err)
		}

		if attempt < maxRetries-1 {
			currentDelay := baseDelay * time.Duration(1<<uint(attempt))
			time.Sleep(currentDelay)
		}
	}

	d.state = StateFailed
	d.lastError = lastErr
	return fmt.Errorf("failed to connect to database '%s' on host '%s:%d' as user '%s' after %d attempts (last error: %v)", dbName, host, dbPort, dbUser, maxRetries, lastErr)
}

/* IsConnected checks if the database is connected */
func (d *Database) IsConnected() bool {
	return d != nil && d.pool != nil && d.state == StateConnected
}

/* GetConnectionState returns the current connection state */
func (d *Database) GetConnectionState() ConnectionState {
	if d == nil {
		return StateDisconnected
	}
	return d.state
}

/* GetLastError returns the last connection error */
func (d *Database) GetLastError() error {
	if d == nil {
		return nil
	}
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
		d.state = StateFailed
		d.lastError = err
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
func (d *Database) Close() {
	if d != nil && d.pool != nil {
		d.pool.Close()
		d.pool = nil
		d.state = StateDisconnected
		d.lastError = nil
	}
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

/* EscapeIdentifier escapes a SQL identifier */
func EscapeIdentifier(identifier string) string {
	return fmt.Sprintf(`"%s"`, identifier)
}

/* errorRow is a row that always returns an error */
type errorRow struct {
	err error
}

func (r *errorRow) Scan(dest ...interface{}) error {
	return r.err
}

