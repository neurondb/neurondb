package database

import (
	"context"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/neurondb/NeuronMCP/internal/config"
)

// Database manages PostgreSQL connections
type Database struct {
	pool *pgxpool.Pool
}

// NewDatabase creates a new database instance
func NewDatabase() *Database {
	return &Database{}
}

// Connect connects to the database using the provided configuration
func (d *Database) Connect(cfg *config.DatabaseConfig) error {
	var connStr string

	if cfg.ConnectionString != nil && *cfg.ConnectionString != "" {
		connStr = *cfg.ConnectionString
	} else {
		// Build connection string from components
		host := cfg.GetHost()
		port := cfg.GetPort()
		db := cfg.GetDatabase()
		user := cfg.GetUser()
		password := ""
		if cfg.Password != nil {
			password = *cfg.Password
		}

		connStr = fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s",
			host, port, user, password, db)

		// Add SSL if configured
		if cfg.SSL != nil {
			if sslBool, ok := cfg.SSL.(bool); ok {
				if sslBool {
					connStr += " sslmode=require"
				} else {
					connStr += " sslmode=disable"
				}
			}
		}
	}

	// Parse connection string
	poolConfig, err := pgxpool.ParseConfig(connStr)
	if err != nil {
		return fmt.Errorf("failed to parse connection string: %w", err)
	}

	// Apply pool settings
	if cfg.Pool != nil {
		poolConfig.MinConns = int32(cfg.Pool.GetMin())
		poolConfig.MaxConns = int32(cfg.Pool.GetMax())
		poolConfig.MaxConnIdleTime = cfg.Pool.GetIdleTimeout()
		poolConfig.MaxConnLifetime = time.Hour
		poolConfig.HealthCheckPeriod = 1 * time.Minute
	}

	// Create pool
	pool, err := pgxpool.NewWithConfig(context.Background(), poolConfig)
	if err != nil {
		return fmt.Errorf("failed to create connection pool: %w", err)
	}

	d.pool = pool
	return nil
}

// Query executes a query and returns rows
func (d *Database) Query(ctx context.Context, query string, args ...interface{}) (pgx.Rows, error) {
	if d.pool == nil {
		return nil, fmt.Errorf("database not connected")
	}
	return d.pool.Query(ctx, query, args...)
}

// QueryRow executes a query and returns a single row
func (d *Database) QueryRow(ctx context.Context, query string, args ...interface{}) pgx.Row {
	if d.pool == nil {
		// Return a row that will error on scan
		return &errorRow{err: fmt.Errorf("database not connected")}
	}
	return d.pool.QueryRow(ctx, query, args...)
}

// Exec executes a query without returning rows
func (d *Database) Exec(ctx context.Context, query string, args ...interface{}) (pgconn.CommandTag, error) {
	if d.pool == nil {
		return pgconn.CommandTag{}, fmt.Errorf("database not connected")
	}
	return d.pool.Exec(ctx, query, args...)
}

// Begin starts a transaction
func (d *Database) Begin(ctx context.Context) (pgx.Tx, error) {
	if d.pool == nil {
		return nil, fmt.Errorf("database not connected")
	}
	return d.pool.Begin(ctx)
}

// Close closes the connection pool
func (d *Database) Close() {
	if d.pool != nil {
		d.pool.Close()
	}
}

// TestConnection tests the database connection
func (d *Database) TestConnection(ctx context.Context) error {
	if d.pool == nil {
		return fmt.Errorf("database not connected")
	}
	return d.pool.Ping(ctx)
}

// GetPoolStats returns pool statistics
func (d *Database) GetPoolStats() *PoolStats {
	if d.pool == nil {
		return nil
	}
	stats := d.pool.Stat()
	return &PoolStats{
		TotalConns:     stats.TotalConns(),
		AcquiredConns:  stats.AcquiredConns(),
		IdleConns:      stats.IdleConns(),
		ConstructingConns: stats.ConstructingConns(),
	}
}

// PoolStats holds connection pool statistics
type PoolStats struct {
	TotalConns      int32
	AcquiredConns   int32
	IdleConns       int32
	ConstructingConns int32
}

// EscapeIdentifier escapes a SQL identifier
func EscapeIdentifier(identifier string) string {
	// Simple escaping - in production, use pgx's built-in escaping
	return fmt.Sprintf(`"%s"`, identifier)
}

// errorRow is a row that always returns an error
type errorRow struct {
	err error
}

func (r *errorRow) Scan(dest ...interface{}) error {
	return r.err
}

