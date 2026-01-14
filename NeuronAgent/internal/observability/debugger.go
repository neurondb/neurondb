/*-------------------------------------------------------------------------
 *
 * debugger.go
 *    Debugging tools and execution viewer
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/observability/debugger.go
 *
 *-------------------------------------------------------------------------
 */

package observability

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
)

/* ExecutionStep represents a step in agent execution */
type ExecutionStep struct {
	ID        string                 `json:"id"`
	StepType  string                 `json:"step_type"` // "load", "context", "prompt", "llm", "tool", "memory", "store"
	Timestamp time.Time              `json:"timestamp"`
	Duration  time.Duration          `json:"duration"`
	Input     map[string]interface{} `json:"input,omitempty"`
	Output    map[string]interface{} `json:"output,omitempty"`
	Error     *string                `json:"error,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

/* ExecutionTrace represents a complete execution trace */
type ExecutionTrace struct {
	ID          uuid.UUID              `json:"id"`
	SessionID   uuid.UUID              `json:"session_id"`
	AgentID     uuid.UUID              `json:"agent_id"`
	UserMessage string                 `json:"user_message"`
	Steps       []ExecutionStep        `json:"steps"`
	StartTime   time.Time              `json:"start_time"`
	EndTime     *time.Time             `json:"end_time,omitempty"`
	Duration    time.Duration          `json:"duration"`
	Success     bool                   `json:"success"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

/* Debugger manages execution traces for debugging */
type Debugger struct {
	traces    map[uuid.UUID]*ExecutionTrace
	mu        sync.RWMutex
	maxTraces int
}

/* NewDebugger creates a new debugger */
func NewDebugger(maxTraces int) *Debugger {
	if maxTraces <= 0 {
		maxTraces = 1000
	}
	return &Debugger{
		traces:    make(map[uuid.UUID]*ExecutionTrace),
		maxTraces: maxTraces,
	}
}

/* StartTrace starts a new execution trace */
func (d *Debugger) StartTrace(sessionID, agentID uuid.UUID, userMessage string) *ExecutionTrace {
	d.mu.Lock()
	defer d.mu.Unlock()

	traceID := uuid.New()
	trace := &ExecutionTrace{
		ID:          traceID,
		SessionID:   sessionID,
		AgentID:     agentID,
		UserMessage: userMessage,
		Steps:       make([]ExecutionStep, 0),
		StartTime:   time.Now(),
		Metadata:    make(map[string]interface{}),
	}

	d.traces[traceID] = trace

	/* Cleanup old traces if needed */
	if len(d.traces) > d.maxTraces {
		d.cleanupOldTraces()
	}

	return trace
}

/* AddStep adds a step to a trace */
func (d *Debugger) AddStep(traceID uuid.UUID, stepType string, input, output map[string]interface{}, err error, metadata map[string]interface{}) {
	d.mu.Lock()
	defer d.mu.Unlock()

	trace, exists := d.traces[traceID]
	if !exists {
		return
	}

	step := ExecutionStep{
		ID:        uuid.New().String(),
		StepType:  stepType,
		Timestamp: time.Now(),
		Input:     input,
		Output:    output,
		Metadata:  metadata,
	}

	if err != nil {
		errStr := err.Error()
		step.Error = &errStr
	}

	if len(trace.Steps) > 0 {
		lastStep := trace.Steps[len(trace.Steps)-1]
		step.Duration = step.Timestamp.Sub(lastStep.Timestamp)
	} else {
		step.Duration = step.Timestamp.Sub(trace.StartTime)
	}

	trace.Steps = append(trace.Steps, step)
}

/* EndTrace ends an execution trace */
func (d *Debugger) EndTrace(traceID uuid.UUID, success bool) {
	d.mu.Lock()
	defer d.mu.Unlock()

	trace, exists := d.traces[traceID]
	if !exists {
		return
	}

	now := time.Now()
	trace.EndTime = &now
	trace.Duration = now.Sub(trace.StartTime)
	trace.Success = success
}

/* GetTrace gets a trace by ID */
func (d *Debugger) GetTrace(traceID uuid.UUID) (*ExecutionTrace, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	trace, exists := d.traces[traceID]
	if !exists {
		return nil, fmt.Errorf("trace not found: %s", traceID.String())
	}

	return trace, nil
}

/* ListTraces lists traces with filters */
func (d *Debugger) ListTraces(agentID, sessionID *uuid.UUID, limit int) []*ExecutionTrace {
	d.mu.RLock()
	defer d.mu.RUnlock()

	results := make([]*ExecutionTrace, 0)
	for _, trace := range d.traces {
		if agentID != nil && trace.AgentID != *agentID {
			continue
		}
		if sessionID != nil && trace.SessionID != *sessionID {
			continue
		}
		results = append(results, trace)
	}

	/* Sort by start time descending */
	for i := 0; i < len(results)-1; i++ {
		for j := i + 1; j < len(results); j++ {
			if results[i].StartTime.Before(results[j].StartTime) {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	if limit > 0 && limit < len(results) {
		return results[:limit]
	}

	return results
}

/* cleanupOldTraces removes old traces */
func (d *Debugger) cleanupOldTraces() {
	/* Sort by start time and remove oldest */
	type traceWithTime struct {
		id    uuid.UUID
		start time.Time
	}

	traces := make([]traceWithTime, 0, len(d.traces))
	for id, trace := range d.traces {
		traces = append(traces, traceWithTime{id: id, start: trace.StartTime})
	}

	/* Sort by start time ascending */
	for i := 0; i < len(traces)-1; i++ {
		for j := i + 1; j < len(traces); j++ {
			if traces[i].start.After(traces[j].start) {
				traces[i], traces[j] = traces[j], traces[i]
			}
		}
	}

	/* Remove oldest 10% */
	removeCount := len(d.traces) / 10
	if removeCount == 0 {
		removeCount = 1
	}
	for i := 0; i < removeCount && i < len(traces); i++ {
		delete(d.traces, traces[i].id)
	}
}

/* ExportTrace exports a trace as JSON */
func (t *ExecutionTrace) ExportTrace() ([]byte, error) {
	return json.MarshalIndent(t, "", "  ")
}

/* GetStepByType gets steps of a specific type */
func (t *ExecutionTrace) GetStepByType(stepType string) []ExecutionStep {
	steps := make([]ExecutionStep, 0)
	for _, step := range t.Steps {
		if step.StepType == stepType {
			steps = append(steps, step)
		}
	}
	return steps
}

/* GetTotalDuration calculates total duration of steps */
func (t *ExecutionTrace) GetTotalDuration() time.Duration {
	var total time.Duration
	for _, step := range t.Steps {
		total += step.Duration
	}
	return total
}

/* ContextKey is a type for context keys */
const traceIDKey contextKey = "trace_id"

/* WithTraceID adds trace ID to context */
func WithTraceID(ctx context.Context, traceID uuid.UUID) context.Context {
	return context.WithValue(ctx, traceIDKey, traceID)
}

/* TraceIDFromContext gets trace ID from context */
func TraceIDFromContext(ctx context.Context) (uuid.UUID, bool) {
	traceID, ok := ctx.Value(traceIDKey).(uuid.UUID)
	return traceID, ok
}
