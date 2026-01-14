/*-------------------------------------------------------------------------
 *
 * resource_quota.go
 *    Resource quota enforcement middleware for memory/CPU limits per tool
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/middleware/resource_quota.go
 *
 *-------------------------------------------------------------------------
 */

package middleware

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/validation"
)

/* ResourceQuotaMiddleware enforces resource limits per tool execution */
type ResourceQuotaMiddleware struct {
	logger           *logging.Logger
	quota            *validation.ResourceQuota
	concurrentOps    int32
	maxConcurrent    int32
	mu               sync.RWMutex
	toolStats        map[string]*ToolResourceStats
	enableThrottling bool
}

/* ToolResourceStats tracks resource usage per tool */
type ToolResourceStats struct {
	TotalExecutions   int64
	TotalDuration     time.Duration
	AverageMemoryMB   float64
	PeakMemoryMB      int64
	LastExecutionTime time.Time
	Errors            int64
	Throttled         int64
}

/* ResourceQuotaConfig configures resource quota behavior */
type ResourceQuotaConfig struct {
	MaxMemoryMB       int   // Maximum memory per operation (MB)
	MaxVectorDim      int   // Maximum vector dimension
	MaxBatchSize      int   // Maximum batch size
	MaxConcurrent     int   // Maximum concurrent operations
	EnableThrottling  bool  // Enable adaptive throttling
}

/* NewResourceQuotaMiddleware creates a new resource quota middleware */
func NewResourceQuotaMiddleware(logger *logging.Logger, config ResourceQuotaConfig) *ResourceQuotaMiddleware {
	if config.MaxMemoryMB == 0 {
		config.MaxMemoryMB = 1024 // 1GB default
	}
	if config.MaxVectorDim == 0 {
		config.MaxVectorDim = 10000
	}
	if config.MaxBatchSize == 0 {
		config.MaxBatchSize = 10000
	}
	if config.MaxConcurrent == 0 {
		config.MaxConcurrent = 100
	}

	quota := &validation.ResourceQuota{
		MaxMemoryBytes:   int64(config.MaxMemoryMB) * 1024 * 1024,
		MaxVectorSize:    config.MaxVectorDim,
		MaxBatchSize:     config.MaxBatchSize,
		MaxCPUTimeMs:     300000, // 5 minutes
	}

	return &ResourceQuotaMiddleware{
		logger:           logger,
		quota:            quota,
		maxConcurrent:    int32(config.MaxConcurrent),
		toolStats:        make(map[string]*ToolResourceStats),
		enableThrottling: config.EnableThrottling,
	}
}

/* Execute enforces resource quotas during tool execution */
func (m *ResourceQuotaMiddleware) Execute(ctx context.Context, params map[string]interface{}, next MiddlewareFunc) (interface{}, error) {
	start := time.Now()
	
	/* Check concurrent operations limit */
	current := atomic.AddInt32(&m.concurrentOps, 1)
	if current > m.maxConcurrent {
		atomic.AddInt32(&m.concurrentOps, -1)
		m.recordThrottle(params)
		return nil, fmt.Errorf("resource quota exceeded: too many concurrent operations (%d/%d)", current, m.maxConcurrent)
	}
	defer atomic.AddInt32(&m.concurrentOps, -1)

	/* Validate vector parameters against quota */
	if err := m.validateVectorParams(params); err != nil {
		return nil, fmt.Errorf("resource quota validation failed: %w", err)
	}

	/* Validate batch size */
	if err := m.validateBatchSize(params); err != nil {
		return nil, fmt.Errorf("resource quota validation failed: %w", err)
	}

	/* Track memory before execution */
	var memBefore runtime.MemStats
	runtime.ReadMemStats(&memBefore)

	/* Execute tool */
	result, err := next(ctx, params)

	/* Track memory after execution */
	var memAfter runtime.MemStats
	runtime.ReadMemStats(&memAfter)

	duration := time.Since(start)
	memUsedMB := int64((memAfter.Alloc - memBefore.Alloc) / (1024 * 1024))

	/* Record stats */
	m.recordStats(params, duration, memUsedMB, err)

	/* Check if memory usage exceeded quota (log warning, don't fail) */
	if memUsedMB > int64(m.quota.MaxMemoryBytes/(1024*1024)) {
		m.logger.Warn("Tool execution exceeded memory quota", map[string]interface{}{
			"memory_used_mb": memUsedMB,
			"quota_mb":       m.quota.MaxMemoryBytes / (1024 * 1024),
			"duration":       duration,
			"params":         params,
		})
	}

	return result, err
}

/* validateVectorParams validates vector-related parameters against quota */
func (m *ResourceQuotaMiddleware) validateVectorParams(params map[string]interface{}) error {
	/* Check query_vector */
	if queryVector, ok := params["query_vector"].([]interface{}); ok {
		if err := validation.ValidateVectorSize(len(queryVector), *m.quota); err != nil {
			return err
		}
	}

	/* Check embeddings */
	if embeddings, ok := params["embeddings"].([]interface{}); ok {
		for i, emb := range embeddings {
			if embVec, ok := emb.([]interface{}); ok {
				if err := validation.ValidateVectorSize(len(embVec), *m.quota); err != nil {
					return fmt.Errorf("embedding[%d]: %w", i, err)
				}
			}
		}
	}

	/* Check dimension parameter */
	if dim, ok := params["dimension"].(float64); ok {
		if int(dim) > m.quota.MaxVectorSize {
			return fmt.Errorf("requested dimension %d exceeds maximum %d", int(dim), m.quota.MaxVectorSize)
		}
	}

	return nil
}

/* validateBatchSize validates batch size parameters against quota */
func (m *ResourceQuotaMiddleware) validateBatchSize(params map[string]interface{}) error {
	batchSizeParams := []string{"batch_size", "limit", "top_k", "k"}
	
	for _, param := range batchSizeParams {
		if size, ok := params[param].(float64); ok {
			if int(size) > m.quota.MaxBatchSize {
				return fmt.Errorf("%s %d exceeds maximum batch size %d", param, int(size), m.quota.MaxBatchSize)
			}
		}
	}

	return nil
}

/* recordStats records execution statistics */
func (m *ResourceQuotaMiddleware) recordStats(params map[string]interface{}, duration time.Duration, memUsedMB int64, err error) {
	toolName := "unknown"
	if name, ok := params["_tool_name"].(string); ok {
		toolName = name
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	stats, exists := m.toolStats[toolName]
	if !exists {
		stats = &ToolResourceStats{}
		m.toolStats[toolName] = stats
	}

	stats.TotalExecutions++
	stats.TotalDuration += duration
	stats.LastExecutionTime = time.Now()
	
	if memUsedMB > stats.PeakMemoryMB {
		stats.PeakMemoryMB = memUsedMB
	}
	
	/* Update rolling average */
	alpha := 0.2 // Exponential smoothing factor
	stats.AverageMemoryMB = alpha*float64(memUsedMB) + (1-alpha)*stats.AverageMemoryMB

	if err != nil {
		stats.Errors++
	}
}

/* recordThrottle records throttling event */
func (m *ResourceQuotaMiddleware) recordThrottle(params map[string]interface{}) {
	toolName := "unknown"
	if name, ok := params["_tool_name"].(string); ok {
		toolName = name
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	stats, exists := m.toolStats[toolName]
	if !exists {
		stats = &ToolResourceStats{}
		m.toolStats[toolName] = stats
	}

	stats.Throttled++
}

/* GetStats returns statistics for a specific tool */
func (m *ResourceQuotaMiddleware) GetStats(toolName string) *ToolResourceStats {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	if stats, exists := m.toolStats[toolName]; exists {
		statsCopy := *stats
		return &statsCopy
	}
	return nil
}

/* GetAllStats returns statistics for all tools */
func (m *ResourceQuotaMiddleware) GetAllStats() map[string]*ToolResourceStats {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	result := make(map[string]*ToolResourceStats)
	for name, stats := range m.toolStats {
		statsCopy := *stats
		result[name] = &statsCopy
	}
	return result
}

/* Name returns the middleware name */
func (m *ResourceQuotaMiddleware) Name() string {
	return "resource_quota"
}

