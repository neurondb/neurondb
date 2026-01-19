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
	ToolMetrics       map[string]*ToolMetrics `json:"toolMetrics,omitempty"`
	CustomMetrics     map[string]interface{}  `json:"customMetrics,omitempty"`
	PoolStats         *PoolMetrics           `json:"poolStats,omitempty"`
}

/* ToolMetrics holds per-tool metrics */
type ToolMetrics struct {
	Count           int64         `json:"count"`
	ErrorCount      int64         `json:"errorCount"`
	TotalDuration   time.Duration `json:"totalDuration"`
	AverageDuration time.Duration `json:"averageDuration"`
	MinDuration     time.Duration `json:"minDuration"`
	MaxDuration     time.Duration `json:"maxDuration"`
	LastUsed        time.Time     `json:"lastUsed"`
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
	toolMetrics   map[string]*ToolMetrics
	customMetrics map[string]interface{}
	db            *database.Database
}

/* NewCollector creates a new metrics collector */
func NewCollector() *Collector {
	return NewCollectorWithDB(nil)
}

/* NewCollectorWithDB creates a new metrics collector with database */
func NewCollectorWithDB(db *database.Database) *Collector {
	return &Collector{
		methodCounts:  make(map[string]int64),
		toolCounts:    make(map[string]int64),
		errorCounts:   make(map[string]int64),
		toolMetrics:   make(map[string]*ToolMetrics),
		customMetrics: make(map[string]interface{}),
		db:            db,
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
	
	/* Initialize tool metrics if not exists */
	if c.toolMetrics[toolName] == nil {
		c.toolMetrics[toolName] = &ToolMetrics{
			MinDuration: time.Hour, /* Initialize with large value */
		}
	}
	c.toolMetrics[toolName].Count++
	c.toolMetrics[toolName].LastUsed = time.Now()
}

/* RecordToolExecution records tool execution with duration */
func (c *Collector) RecordToolExecution(toolName string, duration time.Duration, err error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	/* Initialize tool metrics if not exists */
	if c.toolMetrics[toolName] == nil {
		c.toolMetrics[toolName] = &ToolMetrics{
			MinDuration: time.Hour, /* Initialize with large value */
		}
	}
	
	tm := c.toolMetrics[toolName]
	tm.Count++
	tm.TotalDuration += duration
	if tm.Count > 0 {
		tm.AverageDuration = tm.TotalDuration / time.Duration(tm.Count)
	}
	
	/* Update min/max duration */
	if duration < tm.MinDuration {
		tm.MinDuration = duration
	}
	if duration > tm.MaxDuration {
		tm.MaxDuration = duration
	}
	
	tm.LastUsed = time.Now()
	
	if err != nil {
		tm.ErrorCount++
		c.errorCount++
	}
	
	/* Also update tool counts for backward compatibility */
	c.toolCounts[toolName]++
}

/* IncrementToolError increments tool error count */
func (c *Collector) IncrementToolError(toolName string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	if c.toolMetrics[toolName] == nil {
		c.toolMetrics[toolName] = &ToolMetrics{
			MinDuration: time.Hour,
		}
	}
	c.toolMetrics[toolName].ErrorCount++
	c.errorCount++
}

/* SetCustomMetric sets a custom business metric */
func (c *Collector) SetCustomMetric(name string, value interface{}) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.customMetrics[name] = value
}

/* IncrementCustomMetric increments a custom counter metric */
func (c *Collector) IncrementCustomMetric(name string, delta int64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	if val, ok := c.customMetrics[name].(int64); ok {
		c.customMetrics[name] = val + delta
	} else {
		c.customMetrics[name] = delta
	}
}

/* GetMetrics returns current metrics */
func (c *Collector) GetMetrics() Metrics {
	c.mu.RLock()
	defer c.mu.RUnlock()

	avgDuration := time.Duration(0)
	if c.requestCount > 0 {
		avgDuration = c.totalDuration / time.Duration(c.requestCount)
	}

	/* Copy tool metrics */
	toolMetricsCopy := make(map[string]*ToolMetrics)
	for k, v := range c.toolMetrics {
		if v != nil {
			tmCopy := *v
			toolMetricsCopy[k] = &tmCopy
		}
	}
	
	/* Copy custom metrics */
	customMetricsCopy := make(map[string]interface{})
	for k, v := range c.customMetrics {
		customMetricsCopy[k] = v
	}

	metrics := Metrics{
		RequestCount:    c.requestCount,
		ErrorCount:      c.errorCount,
		TotalDuration:   c.totalDuration,
		AverageDuration: avgDuration,
		MethodCounts:    copyMap(c.methodCounts),
		ToolCounts:      copyMap(c.toolCounts),
		ErrorCounts:     copyMap(c.errorCounts),
		ToolMetrics:     toolMetricsCopy,
		CustomMetrics:   customMetricsCopy,
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
	c.toolMetrics = make(map[string]*ToolMetrics)
	c.customMetrics = make(map[string]interface{})
}

/* copyMap creates a copy of a map */
func copyMap(m map[string]int64) map[string]int64 {
	result := make(map[string]int64, len(m))
	for k, v := range m {
		result[k] = v
	}
	return result
}

