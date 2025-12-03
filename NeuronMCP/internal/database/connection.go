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

// Database manages PostgreSQL connections
type Database struct {
	pool     *pgxpool.Pool
	host     string
	port     int
	database string
	user     string
}

// NewDatabase creates a new database instance
func NewDatabase() *Database {
	return &Database{}
}

// Connect connects to the database using the provided configuration
func (d *Database) Connect(cfg *config.DatabaseConfig) error {
	return d.ConnectWithRetry(cfg, 3, 2*time.Second)
}

// ConnectWithRetry connects to the database with retry logic
func (d *Database) ConnectWithRetry(cfg *config.DatabaseConfig, maxRetries int, retryDelay time.Duration) error {
	var connStr string
	var err error

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

		// Build connection string properly
		connStr = fmt.Sprintf("host=%s port=%d user=%s dbname=%s",
			host, port, user, db)
		
		if password != "" {
			connStr += fmt.Sprintf(" password=%s", password)
		}

		// Add SSL if configured
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
			// Default to prefer SSL
			connStr += " sslmode=prefer"
		}
	}

	// Parse connection string
	poolConfig, err := pgxpool.ParseConfig(connStr)
	if err != nil {
		host := cfg.GetHost()
		port := cfg.GetPort()
		db := cfg.GetDatabase()
		user := cfg.GetUser()
		return fmt.Errorf("failed to parse connection string for database '%s' on host '%s:%d' as user '%s': %w (connection string format may be invalid)", db, host, port, user, err)
	}

	// Register NeuronDB custom types (vector, vector[], etc.)
	// These OIDs are from NeuronDB extension
	// Note: We cast to text in queries for compatibility, but register types for future use
	poolConfig.AfterConnect = func(ctx context.Context, conn *pgx.Conn) error {
		// Register vector type (OID 17648) as text for scanning
		// This allows pgx to handle vector types by treating them as text
		conn.TypeMap().RegisterType(&pgtype.Type{
			Codec: &pgtype.TextCodec{},
			Name:  "vector",
			OID:   17648,
		})
		// Register vector[] type (OID 17656) as text array
		conn.TypeMap().RegisterType(&pgtype.Type{
			Codec: &pgtype.ArrayCodec{ElementType: &pgtype.Type{Name: "text", Codec: &pgtype.TextCodec{}}},
			Name:  "_vector",
			OID:   17656,
		})
		return nil
	}

	// Apply pool settings
	if cfg.Pool != nil {
		poolConfig.MinConns = int32(cfg.Pool.GetMin())
		poolConfig.MaxConns = int32(cfg.Pool.GetMax())
		poolConfig.MaxConnIdleTime = cfg.Pool.GetIdleTimeout()
		poolConfig.MaxConnLifetime = time.Hour
		poolConfig.HealthCheckPeriod = 1 * time.Minute
	} else {
		// Set defaults to prevent immediate connection attempts
		poolConfig.MinConns = 0  // Don't create connections until needed
		poolConfig.MaxConns = 10
		poolConfig.HealthCheckPeriod = 1 * time.Minute
	}

	// Store connection info for error messages
	var host, dbName, dbUser string
	var dbPort int
	if cfg.ConnectionString != nil && *cfg.ConnectionString != "" {
		// Try to extract info from connection string if possible
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

	// Retry connection
	var pool *pgxpool.Pool
	var lastErr error
	for attempt := 0; attempt < maxRetries; attempt++ {
		pool, err = pgxpool.NewWithConfig(context.Background(), poolConfig)
		if err == nil {
			// Test the connection
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			if err := pool.Ping(ctx); err == nil {
				d.pool = pool
				return nil
			}
			lastErr = fmt.Errorf("connection ping failed: database '%s' on host '%s:%d' as user '%s': %w", dbName, host, dbPort, dbUser, err)
			pool.Close()
		} else {
			lastErr = fmt.Errorf("failed to create connection pool: database '%s' on host '%s:%d' as user '%s': %w", dbName, host, dbPort, dbUser, err)
		}

		if attempt < maxRetries-1 {
			time.Sleep(retryDelay)
			retryDelay *= 2 // Exponential backoff
		}
	}

	return fmt.Errorf("failed to connect to database '%s' on host '%s:%d' as user '%s' after %d attempts (last error: %v)", dbName, host, dbPort, dbUser, maxRetries, lastErr)
}

// IsConnected checks if the database is connected
func (d *Database) IsConnected() bool {
	return d.pool != nil
}

// Query executes a query and returns rows
func (d *Database) Query(ctx context.Context, query string, args ...interface{}) (pgx.Rows, error) {
	if d.pool == nil {
		return nil, fmt.Errorf("database connection not established: database '%s' on host '%s:%d' as user '%s' (connection pool is nil, ensure Connect() was called successfully)", d.database, d.host, d.port, d.user)
	}
	rows, err := d.pool.Query(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("query execution failed on database '%s' on host '%s:%d' as user '%s': query='%s', error=%w", d.database, d.host, d.port, d.user, query, err)
	}
	return rows, nil
}

// QueryRow executes a query and returns a single row
func (d *Database) QueryRow(ctx context.Context, query string, args ...interface{}) pgx.Row {
	if d.pool == nil {
		// Return a row that will error on scan
		return &errorRow{err: fmt.Errorf("database connection not established: database '%s' on host '%s:%d' as user '%s' (connection pool is nil, ensure Connect() was called successfully)", d.database, d.host, d.port, d.user)}
	}
	return d.pool.QueryRow(ctx, query, args...)
}

// Exec executes a query without returning rows
func (d *Database) Exec(ctx context.Context, query string, args ...interface{}) (pgconn.CommandTag, error) {
	if d.pool == nil {
		return pgconn.CommandTag{}, fmt.Errorf("database connection not established: database '%s' on host '%s:%d' as user '%s' (connection pool is nil, ensure Connect() was called successfully)", d.database, d.host, d.port, d.user)
	}
	tag, err := d.pool.Exec(ctx, query, args...)
	if err != nil {
		return pgconn.CommandTag{}, fmt.Errorf("query execution failed on database '%s' on host '%s:%d' as user '%s': query='%s', error=%w", d.database, d.host, d.port, d.user, query, err)
	}
	return tag, nil
}

// Begin starts a transaction
func (d *Database) Begin(ctx context.Context) (pgx.Tx, error) {
	if d.pool == nil {
		return nil, fmt.Errorf("database connection not established: database '%s' on host '%s:%d' as user '%s' (connection pool is nil, ensure Connect() was called successfully)", d.database, d.host, d.port, d.user)
	}
	tx, err := d.pool.Begin(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to begin transaction on database '%s' on host '%s:%d' as user '%s': %w", d.database, d.host, d.port, d.user, err)
	}
	return tx, nil
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
		return fmt.Errorf("database connection not established: database '%s' on host '%s:%d' as user '%s' (connection pool is nil, ensure Connect() was called successfully)", d.database, d.host, d.port, d.user)
	}
	err := d.pool.Ping(ctx)
	if err != nil {
		return fmt.Errorf("connection test failed for database '%s' on host '%s:%d' as user '%s': %w", d.database, d.host, d.port, d.user, err)
	}
	return nil
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

