/*-------------------------------------------------------------------------
 *
 * health.go
 *    Health check handler for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/health/health.go
 *
 *-------------------------------------------------------------------------
 */

package health

import (
	"context"
	"time"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/resources"
	"github.com/neurondb/NeuronMCP/internal/tools"
)

const (
	DefaultHealthCheckTimeout = 10 * time.Second
)

/* HealthStatus represents health status */
type HealthStatus struct {
	Status      string                 `json:"status"`
	Database    DatabaseHealth         `json:"database"`
	Tools       ToolsHealth            `json:"tools"`
	Resources   ResourcesHealth        `json:"resources"`
	Pool        PoolHealth             `json:"pool,omitempty"`
	Timestamp   time.Time              `json:"timestamp"`
}

/* DatabaseHealth represents database health */
type DatabaseHealth struct {
	Status    string        `json:"status"`
	Latency   time.Duration `json:"latency,omitempty"`
	Error     string        `json:"error,omitempty"`
}

/* ToolsHealth represents tools health */
type ToolsHealth struct {
	Status      string `json:"status"`
	TotalCount  int    `json:"totalCount"`
	AvailableCount int `json:"availableCount"`
}

/* ResourcesHealth represents resources health */
type ResourcesHealth struct {
	Status      string `json:"status"`
	TotalCount  int    `json:"totalCount"`
	AvailableCount int `json:"availableCount"`
}

/* PoolHealth represents connection pool health */
type PoolHealth struct {
	Status          string `json:"status"`
	TotalConnections int   `json:"totalConnections"`
	IdleConnections  int   `json:"idleConnections"`
	ActiveConnections int `json:"activeConnections"`
	MaxConnections   int   `json:"maxConnections"`
	Utilization     float64 `json:"utilization"`
}

/* Checker performs health checks */
type Checker struct {
	db           *database.Database
	logger       *logging.Logger
	toolRegistry *tools.ToolRegistry
	resources    *resources.Manager
}

/* NewChecker creates a new health checker */
func NewChecker(db *database.Database, logger *logging.Logger) *Checker {
	return &Checker{
		db:     db,
		logger: logger,
	}
}

/* SetToolRegistry sets the tool registry for dynamic tool counting */
func (c *Checker) SetToolRegistry(registry *tools.ToolRegistry) {
	c.toolRegistry = registry
}

/* SetResources sets the resources manager for dynamic resource counting */
func (c *Checker) SetResources(manager *resources.Manager) {
	c.resources = manager
}

/* Check performs a health check */
func (c *Checker) Check(ctx context.Context) *HealthStatus {
	if c == nil {
		return &HealthStatus{
			Status:    "unknown",
			Timestamp: time.Now(),
		}
	}

	/* Add timeout context */
	healthCtx, cancel := context.WithTimeout(ctx, DefaultHealthCheckTimeout)
	defer cancel()

	status := &HealthStatus{
		Timestamp: time.Now(),
	}

	/* Check database */
	dbHealth := c.checkDatabase(healthCtx)
	status.Database = dbHealth

	/* Check tools */
	toolsHealth := c.checkTools(healthCtx)
	status.Tools = toolsHealth

	/* Check resources */
	resourcesHealth := c.checkResources(healthCtx)
	status.Resources = resourcesHealth

	/* Check connection pool */
	poolHealth := c.checkPool(healthCtx)
	status.Pool = poolHealth

	/* Overall status */
	if dbHealth.Status == "healthy" && toolsHealth.Status == "healthy" && resourcesHealth.Status == "healthy" && poolHealth.Status == "healthy" {
		status.Status = "healthy"
	} else {
		status.Status = "degraded"
	}

	return status
}

/* checkDatabase checks database health */
func (c *Checker) checkDatabase(ctx context.Context) DatabaseHealth {
	if c == nil || c.db == nil {
		return DatabaseHealth{
			Status: "unknown",
			Error:  "database instance is not initialized",
		}
	}

	start := time.Now()
	
	var result int
	err := c.db.QueryRow(ctx, "SELECT 1").Scan(&result)
	latency := time.Since(start)

	if err != nil {
		return DatabaseHealth{
			Status: "unhealthy",
			Error:  err.Error(),
		}
	}

	return DatabaseHealth{
		Status:  "healthy",
		Latency: latency,
	}
}

/* checkTools checks tools health */
func (c *Checker) checkTools(ctx context.Context) ToolsHealth {
	if c == nil {
		return ToolsHealth{
			Status:        "unknown",
			TotalCount:    0,
			AvailableCount: 0,
		}
	}

	if c.toolRegistry == nil {
		return ToolsHealth{
			Status:        "unknown",
			TotalCount:    0,
			AvailableCount: 0,
		}
	}

	/* Get actual tool count from registry */
	definitions := c.toolRegistry.GetAllDefinitions()
	totalCount := len(definitions)
	
	/* For now, assume all registered tools are available */
	/* In a full implementation, we could test each tool's availability */
	availableCount := totalCount
	status := "healthy"
	
	if totalCount == 0 {
		status = "degraded"
	}

	return ToolsHealth{
		Status:        status,
		TotalCount:    totalCount,
		AvailableCount: availableCount,
	}
}

/* checkResources checks resources health */
func (c *Checker) checkResources(ctx context.Context) ResourcesHealth {
	if c == nil {
		return ResourcesHealth{
			Status:        "unknown",
			TotalCount:    0,
			AvailableCount: 0,
		}
	}

	if c.resources == nil {
		return ResourcesHealth{
			Status:        "unknown",
			TotalCount:    0,
			AvailableCount: 0,
		}
	}

	/* Get actual resource count from manager */
	definitions := c.resources.ListResources()
	totalCount := len(definitions)
	
	/* For now, assume all registered resources are available */
	/* In a full implementation, we could test each resource's availability */
	availableCount := totalCount
	status := "healthy"
	
	if totalCount == 0 {
		status = "degraded"
	}

	return ResourcesHealth{
		Status:        status,
		TotalCount:    totalCount,
		AvailableCount: availableCount,
	}
}

/* checkPool checks connection pool health */
func (c *Checker) checkPool(ctx context.Context) PoolHealth {
	if c == nil || c.db == nil {
		return PoolHealth{
			Status: "unknown",
		}
	}

	stats := c.db.GetPoolStats()
	if stats == nil {
		return PoolHealth{
			Status: "unknown",
		}
	}

	totalConns := int(stats.TotalConns)
	activeConns := int(stats.AcquiredConns)
	idleConns := int(stats.IdleConns)

	utilization := 0.0
	if totalConns > 0 {
		utilization = float64(activeConns) / float64(totalConns)
	}

	status := "healthy"
	if utilization > 0.9 {
		status = "warning" /* Pool is 90%+ utilized */
	} else if utilization > 0.95 {
		status = "critical" /* Pool is 95%+ utilized */
	}

	return PoolHealth{
		Status:           status,
		TotalConnections: totalConns,
		IdleConnections:  idleConns,
		ActiveConnections: activeConns,
		MaxConnections:   totalConns,
		Utilization:      utilization,
	}
}

