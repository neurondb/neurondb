/*-------------------------------------------------------------------------
 *
 * profiler.go
 *    Performance profiling
 *
 * Provides detailed performance analysis, bottleneck identification,
 * and resource usage tracking.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/observability/profiler.go
 *
 *-------------------------------------------------------------------------
 */

package observability

import (
	"context"
	"time"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
)

/* Profiler provides performance profiling */
type Profiler struct {
	queries *db.Queries
}

/* PerformanceProfile represents a performance profile */
type PerformanceProfile struct {
	ExecutionID    uuid.UUID
	TotalTime      time.Duration
	LLMTime        time.Duration
	ToolTime       time.Duration
	MemoryTime     time.Duration
	DatabaseTime   time.Duration
	Bottlenecks    []Bottleneck
	ResourceUsage  ResourceUsage
}

/* Bottleneck represents a performance bottleneck */
type Bottleneck struct {
	Type        string
	Description string
	Duration    time.Duration
	Percentage  float64
}

/* ResourceUsage represents resource usage */
type ResourceUsage struct {
	CPUTime    time.Duration
	MemoryMB   float64
	NetworkMB  float64
	GPUTime    time.Duration
}

/* NewProfiler creates a new profiler */
func NewProfiler(queries *db.Queries) *Profiler {
	return &Profiler{
		queries: queries,
	}
}

/* ProfileExecution profiles an execution */
func (p *Profiler) ProfileExecution(ctx context.Context, executionID uuid.UUID) (*PerformanceProfile, error) {
	query := `SELECT 
		total_time, llm_time, tool_time, memory_time, database_time,
		cpu_time, memory_mb, network_mb, gpu_time
		FROM neurondb_agent.performance_profiles
		WHERE execution_id = $1`

	type ProfileRow struct {
		TotalTime    time.Duration `db:"total_time"`
		LLMTime      time.Duration `db:"llm_time"`
		ToolTime     time.Duration `db:"tool_time"`
		MemoryTime   time.Duration `db:"memory_time"`
		DatabaseTime time.Duration `db:"database_time"`
		CPUTime      time.Duration `db:"cpu_time"`
		MemoryMB     float64       `db:"memory_mb"`
		NetworkMB    float64       `db:"network_mb"`
		GPUTime      time.Duration `db:"gpu_time"`
	}

	var row ProfileRow
	err := p.queries.DB.GetContext(ctx, &row, query, executionID)
	if err != nil {
		return nil, err
	}

	profile := &PerformanceProfile{
		ExecutionID: executionID,
		TotalTime:   row.TotalTime,
		LLMTime:     row.LLMTime,
		ToolTime:    row.ToolTime,
		MemoryTime:  row.MemoryTime,
		DatabaseTime: row.DatabaseTime,
		ResourceUsage: ResourceUsage{
			CPUTime:   row.CPUTime,
			MemoryMB:  row.MemoryMB,
			NetworkMB: row.NetworkMB,
			GPUTime:   row.GPUTime,
		},
	}

	/* Identify bottlenecks */
	profile.Bottlenecks = p.identifyBottlenecks(profile)

	return profile, nil
}

/* identifyBottlenecks identifies performance bottlenecks */
func (p *Profiler) identifyBottlenecks(profile *PerformanceProfile) []Bottleneck {
	bottlenecks := make([]Bottleneck, 0)
	total := profile.TotalTime

	if total == 0 {
		return bottlenecks
	}

	/* Check LLM time */
	if profile.LLMTime > 0 {
		percentage := float64(profile.LLMTime) / float64(total) * 100
		if percentage > 50 {
			bottlenecks = append(bottlenecks, Bottleneck{
				Type:        "llm",
				Description: "LLM calls are taking significant time",
				Duration:    profile.LLMTime,
				Percentage:  percentage,
			})
		}
	}

	/* Check tool time */
	if profile.ToolTime > 0 {
		percentage := float64(profile.ToolTime) / float64(total) * 100
		if percentage > 30 {
			bottlenecks = append(bottlenecks, Bottleneck{
				Type:        "tool",
				Description: "Tool execution is taking significant time",
				Duration:    profile.ToolTime,
				Percentage:  percentage,
			})
		}
	}

	/* Check database time */
	if profile.DatabaseTime > 0 {
		percentage := float64(profile.DatabaseTime) / float64(total) * 100
		if percentage > 20 {
			bottlenecks = append(bottlenecks, Bottleneck{
				Type:        "database",
				Description: "Database operations are taking significant time",
				Duration:    profile.DatabaseTime,
				Percentage:  percentage,
			})
		}
	}

	return bottlenecks
}

/* StoreProfile stores a performance profile */
func (p *Profiler) StoreProfile(ctx context.Context, profile *PerformanceProfile) error {
	query := `INSERT INTO neurondb_agent.performance_profiles
		(execution_id, total_time, llm_time, tool_time, memory_time, database_time,
		 cpu_time, memory_mb, network_mb, gpu_time, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW())
		ON CONFLICT (execution_id) DO UPDATE
		SET total_time = $2, llm_time = $3, tool_time = $4, memory_time = $5, database_time = $6,
		    cpu_time = $7, memory_mb = $8, network_mb = $9, gpu_time = $10, updated_at = NOW()`

	_, err := p.queries.DB.ExecContext(ctx, query,
		profile.ExecutionID,
		profile.TotalTime,
		profile.LLMTime,
		profile.ToolTime,
		profile.MemoryTime,
		profile.DatabaseTime,
		profile.ResourceUsage.CPUTime,
		profile.ResourceUsage.MemoryMB,
		profile.ResourceUsage.NetworkMB,
		profile.ResourceUsage.GPUTime,
	)

	return err
}

