/*-------------------------------------------------------------------------
 *
 * pool.go
 *    Enhanced connection pool management for NeuronMCP
 *
 * Provides dynamic pool sizing, health checks, and automatic scaling.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/database/pool.go
 *
 *-------------------------------------------------------------------------
 */

package database

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

/* defaultHealthCheckInterval is the interval between pool health checks */
const defaultHealthCheckInterval = 30 * time.Second

/* PoolManager manages connection pool with dynamic sizing */
type PoolManager struct {
	pool                *pgxpool.Pool
	mu                  sync.RWMutex
	minConns            int32
	maxConns            int32
	targetConns         int32
	healthCheckInterval time.Duration
	lastHealthCheck     time.Time
	healthStatus        string
	stopCh              chan struct{}
}

/* NewPoolManager creates a new pool manager */
func NewPoolManager(pool *pgxpool.Pool, minConns, maxConns int32) *PoolManager {
	pm := &PoolManager{
		pool:                pool,
		minConns:            minConns,
		maxConns:            maxConns,
		targetConns:         (minConns + maxConns) / 2,
		healthCheckInterval: defaultHealthCheckInterval,
		healthStatus:        "unknown",
		stopCh:              make(chan struct{}),
	}

	/* Start health check goroutine */
	go pm.healthCheckLoop()

	return pm
}

/* Close stops the health check goroutine */
func (pm *PoolManager) Close() {
	close(pm.stopCh)
}

/* healthCheckLoop periodically checks pool health */
func (pm *PoolManager) healthCheckLoop() {
	ticker := time.NewTicker(pm.healthCheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			pm.performHealthCheck()
		case <-pm.stopCh:
			return
		}
	}
}

/* performHealthCheck performs a health check on the pool */
func (pm *PoolManager) performHealthCheck() {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	if pm.pool == nil {
		pm.healthStatus = "disconnected"
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	stats := pm.pool.Stat()
	if stats == nil {
		pm.healthStatus = "unavailable"
		return
	}

	/* Check pool health */
	err := pm.pool.Ping(ctx)
	if err != nil {
		pm.healthStatus = "unhealthy"
		return
	}

	/* Adjust target connections based on utilization */
	utilization := float64(stats.AcquiredConns()) / float64(pm.maxConns)
	if utilization > 0.8 {
		/* High utilization - increase target */
		if pm.targetConns < pm.maxConns {
			pm.targetConns = min(pm.targetConns+1, pm.maxConns)
		}
	} else if utilization < 0.3 {
		/* Low utilization - decrease target */
		if pm.targetConns > pm.minConns {
			pm.targetConns = max(pm.targetConns-1, pm.minConns)
		}
	}

	pm.healthStatus = "healthy"
	pm.lastHealthCheck = time.Now()
}

/* GetHealthStatus returns pool health status */
func (pm *PoolManager) GetHealthStatus() map[string]interface{} {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	if pm.pool == nil {
		return map[string]interface{}{
			"status": "disconnected",
		}
	}

	stats := pm.pool.Stat()
	if stats == nil {
		return map[string]interface{}{
			"status": "unavailable",
		}
	}

	utilization := 0.0
	if pm.maxConns > 0 {
		utilization = float64(stats.AcquiredConns()) / float64(pm.maxConns)
	}

	return map[string]interface{}{
		"status":               pm.healthStatus,
		"total_connections":    stats.TotalConns(),
		"acquired_connections": stats.AcquiredConns(),
		"idle_connections":     stats.IdleConns(),
		"max_connections":      pm.maxConns,
		"min_connections":      pm.minConns,
		"target_connections":   pm.targetConns,
		"utilization":          utilization,
		"last_health_check":    pm.lastHealthCheck,
	}
}

/* min returns the minimum of two int32 values */
func min(a, b int32) int32 {
	if a < b {
		return a
	}
	return b
}

/* max returns the maximum of two int32 values */
func max(a, b int32) int32 {
	if a > b {
		return a
	}
	return b
}

/* StreamQueryResult represents a streaming query result */
type StreamQueryResult struct {
	Rows   <-chan map[string]interface{}
	Errors <-chan error
	Done   <-chan struct{}
}

/* StreamQuery streams query results instead of loading all into memory */
func (d *Database) StreamQuery(ctx context.Context, query string, args ...interface{}) (*StreamQueryResult, error) {
	if d == nil || d.pool == nil {
		return nil, fmt.Errorf("database connection not established")
	}

	rows, err := d.pool.Query(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("query execution failed: %w", err)
	}

	/* Create channels */
	rowChan := make(chan map[string]interface{}, 10)
	errChan := make(chan error, 1)
	doneChan := make(chan struct{})

	/* Stream rows in background */
	go func() {
		defer close(rowChan)
		defer close(errChan)
		defer close(doneChan)
		defer rows.Close()

		fieldDescriptions := rows.FieldDescriptions()
		for rows.Next() {
			/* Check context cancellation */
			select {
			case <-ctx.Done():
				errChan <- ctx.Err()
				return
			default:
			}

			/* Scan row into map */
			values := make([]interface{}, len(fieldDescriptions))
			valuePointers := make([]interface{}, len(fieldDescriptions))
			for i := range values {
				valuePointers[i] = &values[i]
			}

			if err := rows.Scan(valuePointers...); err != nil {
				errChan <- fmt.Errorf("failed to scan row: %w", err)
				return
			}

			/* Build map */
			rowMap := make(map[string]interface{})
			for i, desc := range fieldDescriptions {
				rowMap[string(desc.Name)] = values[i]
			}

			/* Send row */
			select {
			case <-ctx.Done():
				return
			case rowChan <- rowMap:
			}
		}

		/* Check for errors */
		if err := rows.Err(); err != nil {
			errChan <- err
		}
	}()

	return &StreamQueryResult{
		Rows:   rowChan,
		Errors: errChan,
		Done:   doneChan,
	}, nil
}
