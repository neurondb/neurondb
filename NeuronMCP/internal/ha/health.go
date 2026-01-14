/*-------------------------------------------------------------------------
 *
 * health.go
 *    Health check and failover support
 *
 * Implements HA features as specified in Phase 2.3.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/ha/health.go
 *
 *-------------------------------------------------------------------------
 */

package ha

import (
	"context"
	"fmt"
	"time"
)

/* HealthStatus represents health status */
type HealthStatus string

const (
	HealthStatusHealthy   HealthStatus = "healthy"
	HealthStatusDegraded  HealthStatus = "degraded"
	HealthStatusUnhealthy HealthStatus = "unhealthy"
)

/* HealthCheck represents a health check */
type HealthCheck struct {
	Name        string
	Status      HealthStatus
	LastCheck   time.Time
	Message     string
	Details     map[string]interface{}
}

/* HealthChecker performs health checks */
type HealthChecker struct {
	checks map[string]*HealthCheck
}

/* NewHealthChecker creates a new health checker */
func NewHealthChecker() *HealthChecker {
	return &HealthChecker{
		checks: make(map[string]*HealthCheck),
	}
}

/* RegisterCheck registers a health check */
func (h *HealthChecker) RegisterCheck(name string, checkFunc func() (HealthStatus, string, map[string]interface{})) {
	status, message, details := checkFunc()
	h.checks[name] = &HealthCheck{
		Name:      name,
		Status:    status,
		LastCheck: time.Now(),
		Message:   message,
		Details:   details,
	}
}

/* GetHealth returns overall health status */
func (h *HealthChecker) GetHealth() HealthStatus {
	hasUnhealthy := false
	hasDegraded := false

	for _, check := range h.checks {
		switch check.Status {
		case HealthStatusUnhealthy:
			hasUnhealthy = true
		case HealthStatusDegraded:
			hasDegraded = true
		}
	}

	if hasUnhealthy {
		return HealthStatusUnhealthy
	}
	if hasDegraded {
		return HealthStatusDegraded
	}
	return HealthStatusHealthy
}

/* GetAllChecks returns all health checks */
func (h *HealthChecker) GetAllChecks() map[string]*HealthCheck {
	return h.checks
}

/* LoadBalancer manages load balancing */
type LoadBalancer struct {
	instances []Instance
	algorithm string /* "round_robin", "least_connections", "weighted" */
}

/* Instance represents a server instance */
type Instance struct {
	ID       string
	Address  string
	Healthy  bool
	Weight   int
	Connections int
}

/* NewLoadBalancer creates a new load balancer */
func NewLoadBalancer(algorithm string) *LoadBalancer {
	return &LoadBalancer{
		instances: []Instance{},
		algorithm: algorithm,
	}
}

/* AddInstance adds an instance */
func (l *LoadBalancer) AddInstance(instance Instance) {
	l.instances = append(l.instances, instance)
}

/* SelectInstance selects an instance using the load balancing algorithm */
func (l *LoadBalancer) SelectInstance() *Instance {
	if len(l.instances) == 0 {
		return nil
	}

	/* Filter healthy instances */
	healthy := []Instance{}
	for _, inst := range l.instances {
		if inst.Healthy {
			healthy = append(healthy, inst)
		}
	}

	if len(healthy) == 0 {
		return nil
	}

	switch l.algorithm {
	case "round_robin":
		/* Simple round robin - in production, use atomic counter */
		return &healthy[0]
	case "least_connections":
		/* Select instance with least connections */
		minConn := healthy[0]
		for _, inst := range healthy {
			if inst.Connections < minConn.Connections {
				minConn = inst
			}
		}
		return &minConn
	case "weighted":
		/* Weighted round robin */
		totalWeight := 0
		for _, inst := range healthy {
			totalWeight += inst.Weight
		}
		/* In production, use proper weighted selection */
		return &healthy[0]
	default:
		return &healthy[0]
	}
}

/* FailoverManager manages failover */
type FailoverManager struct {
	primary   *Instance
	replicas  []Instance
	healthChecker *HealthChecker
}

/* NewFailoverManager creates a new failover manager */
func NewFailoverManager(primary *Instance, replicas []Instance) *FailoverManager {
	return &FailoverManager{
		primary:       primary,
		replicas:      replicas,
		healthChecker: NewHealthChecker(),
	}
}

/* CheckHealth checks health of primary */
func (f *FailoverManager) CheckHealth(ctx context.Context) bool {
	/* In production, perform actual health check */
	return f.primary.Healthy
}

/* Failover performs failover to a replica */
func (f *FailoverManager) Failover() (*Instance, error) {
	/* Find healthy replica */
	for _, replica := range f.replicas {
		if replica.Healthy {
			/* Promote replica to primary */
			f.primary = &replica
			return &replica, nil
		}
	}

	return nil, fmt.Errorf("no healthy replica available")
}

