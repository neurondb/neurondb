/*-------------------------------------------------------------------------
 *
 * runtime_trace.go
 *    Execution tracing for observability
 *
 * Provides execution trace recording for decision tree visualization
 * and performance profiling.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/runtime_trace.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"context"
	"encoding/json"
	"time"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/metrics"
)

/* ExecutionTracer traces execution steps */
type ExecutionTracer struct {
	queries      *db.Queries
	executionID  uuid.UUID
	currentStep  *TraceStep
	steps        []*TraceStep
	startTime    time.Time
}

/* TraceStep represents a step in execution trace */
type TraceStep struct {
	ID          uuid.UUID
	StepType    string
	Description string
	InputData   map[string]interface{}
	OutputData  map[string]interface{}
	ParentID    *uuid.UUID
	Timestamp   time.Time
	Duration    time.Duration
}

/* NewExecutionTracer creates a new execution tracer */
func NewExecutionTracer(queries *db.Queries, executionID uuid.UUID) *ExecutionTracer {
	return &ExecutionTracer{
		queries:     queries,
		executionID: executionID,
		steps:       make([]*TraceStep, 0),
		startTime:   time.Now(),
	}
}

/* StartStep starts a new trace step */
func (et *ExecutionTracer) StartStep(ctx context.Context, stepType, description string, inputData map[string]interface{}) *TraceStep {
	step := &TraceStep{
		ID:          uuid.New(),
		StepType:    stepType,
		Description: description,
		InputData:   inputData,
		OutputData:  make(map[string]interface{}),
		ParentID:    nil,
		Timestamp:   time.Now(),
	}

	if et.currentStep != nil {
		step.ParentID = &et.currentStep.ID
	}

	et.currentStep = step
	et.steps = append(et.steps, step)

	return step
}

/* EndStep ends the current trace step */
func (et *ExecutionTracer) EndStep(ctx context.Context, outputData map[string]interface{}) {
	if et.currentStep == nil {
		return
	}

	et.currentStep.OutputData = outputData
	et.currentStep.Duration = time.Since(et.currentStep.Timestamp)

	/* Store step in database */
	et.storeStep(ctx, et.currentStep)

	/* Move to parent step if exists */
	if et.currentStep.ParentID != nil {
		for _, step := range et.steps {
			if step.ID == *et.currentStep.ParentID {
				et.currentStep = step
				return
			}
		}
	}

	et.currentStep = nil
}

/* storeStep stores a trace step in database */
func (et *ExecutionTracer) storeStep(ctx context.Context, step *TraceStep) {
	query := `INSERT INTO neurondb_agent.execution_trace
		(id, execution_id, step_type, description, input_data, output_data, parent_id, timestamp)
		VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7, $8)`

	inputJSON, _ := json.Marshal(step.InputData)
	outputJSON, _ := json.Marshal(step.OutputData)

	_, err := et.queries.DB.ExecContext(ctx, query,
		step.ID,
		et.executionID,
		step.StepType,
		step.Description,
		inputJSON,
		outputJSON,
		step.ParentID,
		step.Timestamp,
	)

	if err != nil {
		metrics.WarnWithContext(ctx, "Failed to store execution trace step", map[string]interface{}{
			"execution_id": et.executionID.String(),
			"step_id":      step.ID.String(),
			"error":        err.Error(),
		})
	}
}

/* StoreProfile stores performance profile */
func (et *ExecutionTracer) StoreProfile(ctx context.Context, profile *PerformanceProfileData) error {
	query := `INSERT INTO neurondb_agent.performance_profiles
		(execution_id, total_time, llm_time, tool_time, memory_time, database_time,
		 cpu_time, memory_mb, network_mb, gpu_time, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW())
		ON CONFLICT (execution_id) DO UPDATE
		SET total_time = $2, llm_time = $3, tool_time = $4, memory_time = $5, database_time = $6,
		    cpu_time = $7, memory_mb = $8, network_mb = $9, gpu_time = $10, updated_at = NOW()`

	_, err := et.queries.DB.ExecContext(ctx, query,
		et.executionID,
		profile.TotalTime,
		profile.LLMTime,
		profile.ToolTime,
		profile.MemoryTime,
		profile.DatabaseTime,
		profile.ResourceUsageData.CPUTime,
		profile.ResourceUsageData.MemoryMB,
		profile.ResourceUsageData.NetworkMB,
		profile.ResourceUsageData.GPUTime,
	)

	return err
}

/* PerformanceProfileData represents performance metrics for storage */
type PerformanceProfileData struct {
	TotalTime    time.Duration
	LLMTime      time.Duration
	ToolTime     time.Duration
	MemoryTime   time.Duration
	DatabaseTime time.Duration
	ResourceUsageData ResourceUsageData
}

/* ResourceUsageData represents resource usage for storage */
type ResourceUsageData struct {
	CPUTime   time.Duration
	MemoryMB  float64
	NetworkMB float64
	GPUTime   time.Duration
}

