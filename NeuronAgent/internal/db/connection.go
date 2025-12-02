package db

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/jmoiron/sqlx"
	_ "github.com/lib/pq"
	"github.com/neurondb/NeuronAgent/internal/utils"
)

// ConnectionInfo holds details about the database connection
type ConnectionInfo struct {
	Host     string
	Port     int
	Database string
	User     string
}

// DB manages PostgreSQL connections
type DB struct {
	*sqlx.DB
	poolConfig PoolConfig
	connInfo   *ConnectionInfo // Stores connection details for error messages
}

type PoolConfig struct {
	MaxOpenConns    int
	MaxIdleConns    int
	ConnMaxLifetime time.Duration
	ConnMaxIdleTime time.Duration
}

// NewDB creates a new database instance
func NewDB(connStr string, poolConfig PoolConfig) (*DB, error) {
	return NewDBWithRetry(connStr, poolConfig, 3, 2*time.Second)
}

// NewDBWithRetry creates a new database instance with retry logic
func NewDBWithRetry(connStr string, poolConfig PoolConfig, maxRetries int, retryDelay time.Duration) (*DB, error) {
	// Parse connection string to extract connection info
	connInfo := parseConnectionInfo(connStr)
	
	var db *sqlx.DB
	var err error
	
	for attempt := 0; attempt < maxRetries; attempt++ {
		db, err = sqlx.Connect("postgres", connStr)
		if err == nil {
			// Test the connection
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			pingErr := db.PingContext(ctx)
			cancel()
			if pingErr == nil {
				db.SetMaxOpenConns(poolConfig.MaxOpenConns)
				db.SetMaxIdleConns(poolConfig.MaxIdleConns)
				db.SetConnMaxLifetime(poolConfig.ConnMaxLifetime)
				db.SetConnMaxIdleTime(poolConfig.ConnMaxIdleTime)
				
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
			time.Sleep(retryDelay)
			retryDelay *= 2 // Exponential backoff
		}
	}
	
	connInfoStr := utils.FormatConnectionInfo(connInfo.Host, connInfo.Port, connInfo.Database, connInfo.User)
	return nil, fmt.Errorf("failed to connect to %s after %d attempts (last error: %w)", connInfoStr, maxRetries, err)
}

// parseConnectionInfo extracts connection information from connection string
func parseConnectionInfo(connStr string) *ConnectionInfo {
	info := &ConnectionInfo{
		Host:     "unknown",
		Port:     5432,
		Database: "unknown",
		User:     "unknown",
	}
	
	// Simple parsing - in production, use proper connection string parser
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

// GetConnInfoString returns a formatted string of connection details
func (d *DB) GetConnInfoString() string {
	if d.connInfo == nil {
		return "unknown database connection"
	}
	return utils.FormatConnectionInfo(d.connInfo.Host, d.connInfo.Port, d.connInfo.Database, d.connInfo.User)
}

// HealthCheck tests the database connection
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

// Close closes the connection pool
func (d *DB) Close() error {
	if d.DB == nil {
		return nil
	}
	return d.DB.Close()
}
