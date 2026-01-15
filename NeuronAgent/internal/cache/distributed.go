/*-------------------------------------------------------------------------
 *
 * distributed.go
 *    Distributed caching layer
 *
 * Provides PostgreSQL-based distributed cache with multi-level caching
 * (L1: in-memory, L2: PostgreSQL with LISTEN/NOTIFY, L3: PostgreSQL persistent).
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/cache/distributed.go
 *
 *-------------------------------------------------------------------------
 */

package cache

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/lib/pq"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/metrics"
)

/* DistributedCache provides distributed caching */
type DistributedCache struct {
	l1Cache    CacheManagerInterface // In-memory cache interface
	pgCache    *PostgreSQLCache     // PostgreSQL cache with LISTEN/NOTIFY
	dbCache    *DBCache              // PostgreSQL persistent cache
	enabled    bool
	mu         sync.RWMutex
	listeners  map[string]chan struct{} // Key invalidation listeners
}

/* CacheManagerInterface defines cache operations */
type CacheManagerInterface interface {
	GetCachedResponse(ctx context.Context, key string) (interface{}, bool)
	CacheResponse(ctx context.Context, key string, response interface{}, ttl time.Duration) error
	DeleteResponse(key string)
	DeleteEmbedding(key string)
	DeleteToolResult(key string)
}

/* PostgreSQLCache provides PostgreSQL-based caching with LISTEN/NOTIFY */
type PostgreSQLCache struct {
	queries  *db.Queries
	listener *pq.Listener
	enabled  bool
	mu       sync.RWMutex
}

/* DBCache provides PostgreSQL-based persistent caching */
type DBCache struct {
	queries *db.Queries
}

/* NewDistributedCache creates a new distributed cache */
func NewDistributedCache(queries *db.Queries, l1Cache CacheManagerInterface) *DistributedCache {
	return &DistributedCache{
		l1Cache:   l1Cache,
		pgCache:   NewPostgreSQLCache(queries),
		dbCache:   NewDBCache(queries),
		enabled:   false,
		listeners: make(map[string]chan struct{}),
	}
}

/* Enable enables distributed caching */
func (dc *DistributedCache) Enable(ctx context.Context) error {
	dc.mu.Lock()
	defer dc.mu.Unlock()

	if dc.enabled {
		return nil
	}

	/* Enable PostgreSQL cache with LISTEN/NOTIFY */
	if err := dc.pgCache.Enable(ctx); err != nil {
		return fmt.Errorf("distributed cache enable failed: pg_cache_error=true, error=%w", err)
	}

	/* Start listening for cache invalidation notifications */
	go dc.startInvalidationListener(ctx)

	dc.enabled = true
	metrics.InfoWithContext(ctx, "Distributed cache enabled", map[string]interface{}{
		"cache_type": "postgresql_listen_notify",
	})

	return nil
}

/* startInvalidationListener listens for cache invalidation notifications */
func (dc *DistributedCache) startInvalidationListener(ctx context.Context) {
	notificationChan := dc.pgCache.GetNotificationChannel()
	for {
		select {
		case <-ctx.Done():
			return
		case notification := <-notificationChan:
			if notification != nil {
				/* Parse notification payload (key) */
				key := notification.Extra
				if key != "" {
					/* Invalidate in L1 */
					dc.l1Cache.DeleteResponse(key)
					dc.l1Cache.DeleteEmbedding(key)
					dc.l1Cache.DeleteToolResult(key)

					/* Notify local listeners */
					dc.mu.RLock()
					if listener, ok := dc.listeners[key]; ok {
						select {
						case listener <- struct{}{}:
						default:
						}
					}
					dc.mu.RUnlock()

					metrics.InfoWithContext(ctx, "Cache invalidated via NOTIFY", map[string]interface{}{
						"key": key,
					})
				}
			}
		}
	}
}

/* Get retrieves a value from cache (L1 -> L2 -> L3) */
func (dc *DistributedCache) Get(ctx context.Context, key string) (interface{}, bool) {
	if !dc.enabled {
		/* Fall back to L1 only */
		return dc.l1Cache.GetCachedResponse(ctx, key)
	}

	/* Try L1 (in-memory) */
	if value, ok := dc.l1Cache.GetCachedResponse(ctx, key); ok {
		metrics.InfoWithContext(ctx, "Cache hit L1", map[string]interface{}{
			"key": key,
		})
		return value, true
	}

	/* Try L2 (PostgreSQL with LISTEN/NOTIFY) */
	if dc.enabled && dc.pgCache.enabled {
		if value, ok := dc.pgCache.Get(ctx, key); ok {
			/* Promote to L1 */
			dc.l1Cache.CacheResponse(ctx, key, value, 5*time.Minute)
			metrics.InfoWithContext(ctx, "Cache hit L2", map[string]interface{}{
				"key": key,
			})
			return value, true
		}
	}

	/* Try L3 (PostgreSQL persistent) */
	if value, ok := dc.dbCache.Get(ctx, key); ok {
		/* Promote to L2 and L1 */
		if dc.enabled && dc.pgCache.enabled {
			dc.pgCache.Set(ctx, key, value, 10*time.Minute)
		}
		dc.l1Cache.CacheResponse(ctx, key, value, 5*time.Minute)
		metrics.InfoWithContext(ctx, "Cache hit L3", map[string]interface{}{
			"key": key,
		})
		return value, true
	}

	metrics.InfoWithContext(ctx, "Cache miss", map[string]interface{}{
		"key": key,
	})
	return nil, false
}

/* Set stores a value in all cache levels */
func (dc *DistributedCache) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
	/* Store in L1 */
	if err := dc.l1Cache.CacheResponse(ctx, key, value, ttl); err != nil {
		return err
	}

	/* Store in L2 (PostgreSQL with LISTEN/NOTIFY) */
	if dc.enabled && dc.pgCache.enabled {
		if err := dc.pgCache.Set(ctx, key, value, ttl); err != nil {
			metrics.WarnWithContext(ctx, "Failed to cache in PostgreSQL", map[string]interface{}{
				"key":   key,
				"error": err.Error(),
			})
		}
	}

	/* Store in L3 (PostgreSQL) for long-term persistence */
	if dc.enabled {
		if err := dc.dbCache.Set(ctx, key, value, ttl); err != nil {
			metrics.WarnWithContext(ctx, "Failed to cache in database", map[string]interface{}{
				"key":   key,
				"error": err.Error(),
			})
		}
	}

	return nil
}

/* Invalidate invalidates a key across all cache levels */
func (dc *DistributedCache) Invalidate(ctx context.Context, key string) error {
	/* Invalidate L1 */
	dc.l1Cache.DeleteResponse(key)
	dc.l1Cache.DeleteEmbedding(key)
	dc.l1Cache.DeleteToolResult(key)

	/* Invalidate L2 and notify other nodes */
	if dc.enabled && dc.pgCache.enabled {
		if err := dc.pgCache.Delete(ctx, key); err != nil {
			metrics.WarnWithContext(ctx, "Failed to invalidate in PostgreSQL cache", map[string]interface{}{
				"key":   key,
				"error": err.Error(),
			})
		}
		/* Send NOTIFY to invalidate other nodes */
		dc.pgCache.NotifyInvalidation(ctx, key)
	}

	/* Invalidate L3 */
	if err := dc.dbCache.Delete(ctx, key); err != nil {
		return err
	}

	return nil
}

/* NewPostgreSQLCache creates a new PostgreSQL cache with LISTEN/NOTIFY */
func NewPostgreSQLCache(queries *db.Queries) *PostgreSQLCache {
	return &PostgreSQLCache{
		queries: queries,
		enabled: false,
	}
}

/* Enable enables PostgreSQL cache with LISTEN/NOTIFY */
func (pgc *PostgreSQLCache) Enable(ctx context.Context) error {
	pgc.mu.Lock()
	defer pgc.mu.Unlock()

	if pgc.enabled {
		return nil
	}

	/* Get connection string for LISTEN/NOTIFY */
	/* Query database for connection parameters */
	connStr := ""
	if pgc.queries.DB != nil {
		var dbname, usr, host, port string
		/* Get connection parameters from database */
		err := pgc.queries.DB.QueryRowContext(ctx, `
			SELECT 
				current_database() as database,
				current_user as user,
				inet_server_addr()::text as host,
				inet_server_port()::text as port
		`).Scan(&dbname, &usr, &host, &port)
		if err == nil {
			/* Build connection string */
			if host != "" && port != "" {
				connStr = fmt.Sprintf("host=%s port=%s dbname=%s user=%s sslmode=disable", host, port, dbname, usr)
			} else {
				/* If inet_server_addr/port return NULL (local connection), use defaults */
				connStr = fmt.Sprintf("dbname=%s user=%s sslmode=disable", dbname, usr)
			}
		} else {
			/* Fallback: query just database and user */
			if err := pgc.queries.DB.QueryRowContext(ctx, "SELECT current_database(), current_user").Scan(&dbname, &usr); err == nil {
				connStr = fmt.Sprintf("dbname=%s user=%s sslmode=disable", dbname, usr)
			}
		}
	}

	if connStr == "" {
		return fmt.Errorf("postgresql cache enable failed: could_not_build_connection_string=true")
	}

	/* Create pq.Listener for LISTEN/NOTIFY */
	reportProblem := func(ev pq.ListenerEventType, err error) {
		if err != nil {
			metrics.WarnWithContext(ctx, "PostgreSQL LISTEN error", map[string]interface{}{
				"event": int(ev),
				"error": err.Error(),
			})
		}
	}

	listener := pq.NewListener(connStr, 10*time.Second, time.Minute, reportProblem)
	if err := listener.Listen("cache_invalidation"); err != nil {
		return fmt.Errorf("postgresql cache enable failed: listen_error=true, error=%w", err)
	}

	pgc.listener = listener
	pgc.enabled = true

	metrics.InfoWithContext(ctx, "PostgreSQL cache enabled with LISTEN/NOTIFY", nil)
	return nil
}

/* GetNotificationChannel returns the notification channel */
func (pgc *PostgreSQLCache) GetNotificationChannel() <-chan *pq.Notification {
	if pgc.listener != nil {
		return pgc.listener.Notify
	}
	return nil
}

/* Get retrieves from PostgreSQL cache */
func (pgc *PostgreSQLCache) Get(ctx context.Context, key string) (interface{}, bool) {
	if !pgc.enabled {
		return nil, false
	}

	/* Use same query as DBCache but with shorter TTL for L2 */
	query := `SELECT value, expires_at
		FROM neurondb_agent.cache
		WHERE key = $1 AND expires_at > NOW()`

	type CacheRow struct {
		Value     string    `db:"value"`
		ExpiresAt time.Time `db:"expires_at"`
	}

	var row CacheRow
	err := pgc.queries.DB.GetContext(ctx, &row, query, key)
	if err != nil {
		return nil, false
	}

	/* Deserialize value */
	var value interface{}
	if err := json.Unmarshal([]byte(row.Value), &value); err != nil {
		return nil, false
	}

	return value, true
}

/* Set stores in PostgreSQL cache */
func (pgc *PostgreSQLCache) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
	if !pgc.enabled {
		return nil
	}

	query := `INSERT INTO neurondb_agent.cache
		(key, value, expires_at, created_at)
		VALUES ($1, $2::jsonb, NOW() + $3, NOW())
		ON CONFLICT (key) DO UPDATE
		SET value = $2::jsonb, expires_at = NOW() + $3`

	valueJSON, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("postgresql cache set failed: json_marshal_error=true, error=%w", err)
	}

	_, err = pgc.queries.DB.ExecContext(ctx, query, key, valueJSON, ttl)
	return err
}

/* Delete deletes from PostgreSQL cache */
func (pgc *PostgreSQLCache) Delete(ctx context.Context, key string) error {
	if !pgc.enabled {
		return nil
	}

	query := `DELETE FROM neurondb_agent.cache WHERE key = $1`
	_, err := pgc.queries.DB.ExecContext(ctx, query, key)
	return err
}

/* NotifyInvalidation sends a NOTIFY to invalidate cache on other nodes */
func (pgc *PostgreSQLCache) NotifyInvalidation(ctx context.Context, key string) error {
	if !pgc.enabled {
		return nil
	}

	/* Send NOTIFY to invalidate cache on other nodes */
	query := `SELECT pg_notify('cache_invalidation', $1)`
	_, err := pgc.queries.DB.ExecContext(ctx, query, key)
	return err
}

/* Close closes the PostgreSQL cache listener */
func (pgc *PostgreSQLCache) Close() error {
	pgc.mu.Lock()
	defer pgc.mu.Unlock()

	if pgc.listener != nil {
		pgc.listener.Close()
		pgc.listener = nil
	}
	pgc.enabled = false
	return nil
}

/* NewDBCache creates a new database cache */
func NewDBCache(queries *db.Queries) *DBCache {
	return &DBCache{
		queries: queries,
	}
}

/* Get retrieves from database */
func (dbc *DBCache) Get(ctx context.Context, key string) (interface{}, bool) {
	query := `SELECT value, expires_at
		FROM neurondb_agent.cache
		WHERE key = $1 AND expires_at > NOW()`

	type CacheRow struct {
		Value     string    `db:"value"`
		ExpiresAt time.Time `db:"expires_at"`
	}

	var row CacheRow
	err := dbc.queries.DB.GetContext(ctx, &row, query, key)
	if err != nil {
		return nil, false
	}

	/* Deserialize value */
	var value interface{}
	if err := json.Unmarshal([]byte(row.Value), &value); err != nil {
		return nil, false
	}

	return value, true
}

/* Set stores in database */
func (dbc *DBCache) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
	query := `INSERT INTO neurondb_agent.cache
		(key, value, expires_at, created_at)
		VALUES ($1, $2::jsonb, NOW() + $3, NOW())
		ON CONFLICT (key) DO UPDATE
		SET value = $2::jsonb, expires_at = NOW() + $3`

	valueJSON, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("db cache set failed: json_marshal_error=true, error=%w", err)
	}

	_, err = dbc.queries.DB.ExecContext(ctx, query, key, valueJSON, ttl)
	return err
}

/* Delete deletes from database */
func (dbc *DBCache) Delete(ctx context.Context, key string) error {
	query := `DELETE FROM neurondb_agent.cache WHERE key = $1`
	_, err := dbc.queries.DB.ExecContext(ctx, query, key)
	return err
}

