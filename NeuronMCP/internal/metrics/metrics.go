/*-------------------------------------------------------------------------
 *
 * metrics.go
 *    Metrics collection for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/metrics/metrics.go
 *
 *-------------------------------------------------------------------------
 */

package metrics

import (
	"sync"
	"time"

	"github.com/neurondb/NeuronMCP/internal/database"
)

/* Metrics holds collected metrics */
type Metrics struct {
	RequestCount      int64                  `json:"requestCount"`
	ErrorCount        int64                  `json:"errorCount"`
	TotalDuration     time.Duration           `json:"totalDuration"`
	AverageDuration   time.Duration           `json:"averageDuration"`
	MethodCounts      map[string]int64        `json:"methodCounts"`
	ToolCounts        map[string]int64        `json:"toolCounts"`
	ErrorCounts       map[string]int64        `json:"errorCounts"`
	PoolStats         *PoolMetrics           `json:"poolStats,omitempty"`
}

/* PoolMetrics holds connection pool metrics */
type PoolMetrics struct {
	TotalConnections   int     `json:"totalConnections"`
	ActiveConnections  int     `json:"activeConnections"`
	IdleConnections    int     `json:"idleConnections"`
	MaxConnections     int     `json:"maxConnections"`
	Utilization        float64 `json:"utilization"`
}

/* Collector collects metrics */
type Collector struct {
	mu            sync.RWMutex
	requestCount  int64
	errorCount    int64
	totalDuration time.Duration
	methodCounts  map[string]int64
	toolCounts    map[string]int64
	errorCounts   map[string]int64
	db            *database.Database
}

/* NewCollector creates a new metrics collector */
func NewCollector() *Collector {
	return NewCollectorWithDB(nil)
}

/* NewCollectorWithDB creates a new metrics collector with database */
func NewCollectorWithDB(db *database.Database) *Collector {
	return &Collector{
		methodCounts: make(map[string]int64),
		toolCounts:   make(map[string]int64),
		errorCounts:  make(map[string]int64),
		db:           db,
	}
}

/* IncrementRequest increments request count */
func (c *Collector) IncrementRequest(method string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.requestCount++
	c.methodCounts[method]++
}

/* IncrementError increments error count */
func (c *Collector) IncrementError(method string, errorType string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.errorCount++
	if errorType != "" {
		c.errorCounts[errorType]++
	}
}

/* AddDuration adds to total duration */
func (c *Collector) AddDuration(duration time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.totalDuration += duration
}

/* IncrementTool increments tool usage count */
func (c *Collector) IncrementTool(toolName string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.toolCounts[toolName]++
}

/* GetMetrics returns current metrics */
func (c *Collector) GetMetrics() Metrics {
	c.mu.RLock()
	defer c.mu.RUnlock()

	avgDuration := time.Duration(0)
	if c.requestCount > 0 {
		avgDuration = c.totalDuration / time.Duration(c.requestCount)
	}

	metrics := Metrics{
		RequestCount:    c.requestCount,
		ErrorCount:      c.errorCount,
		TotalDuration:   c.totalDuration,
		AverageDuration: avgDuration,
		MethodCounts:    copyMap(c.methodCounts),
		ToolCounts:      copyMap(c.toolCounts),
		ErrorCounts:     copyMap(c.errorCounts),
	}

	/* Add pool stats if database is available */
	if c.db != nil {
		poolStats := c.db.GetPoolStats()
		if poolStats != nil {
			utilization := 0.0
			maxConns := int(poolStats.TotalConns)
			if maxConns > 0 {
				utilization = float64(poolStats.AcquiredConns) / float64(maxConns)
			}
			metrics.PoolStats = &PoolMetrics{
				TotalConnections:  int(poolStats.TotalConns),
				ActiveConnections: int(poolStats.AcquiredConns),
				IdleConnections:   int(poolStats.IdleConns),
				MaxConnections:    maxConns,
				Utilization:       utilization,
			}
		}
	}

	return metrics
}

/* Reset resets all metrics */
func (c *Collector) Reset() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.requestCount = 0
	c.errorCount = 0
	c.totalDuration = 0
	c.methodCounts = make(map[string]int64)
	c.toolCounts = make(map[string]int64)
	c.errorCounts = make(map[string]int64)
}

/* copyMap creates a copy of a map */
func copyMap(m map[string]int64) map[string]int64 {
	result := make(map[string]int64, len(m))
	for k, v := range m {
		result[k] = v
	}
	return result
}

