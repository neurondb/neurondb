package tools

import (
	"context"
	"fmt"
	"time"

	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/metrics"
)

// Executor handles tool execution with timeout and error handling
type Executor struct {
	registry *Registry
	timeout  time.Duration
}

// NewExecutor creates a new tool executor
func NewExecutor(registry *Registry, timeout time.Duration) *Executor {
	return &Executor{
		registry: registry,
		timeout:  timeout,
	}
}

// Execute executes a tool with timeout
func (e *Executor) Execute(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	start := time.Now()
	
	// Create context with timeout
	ctx, cancel := context.WithTimeout(ctx, e.timeout)
	defer cancel()

	// Execute tool
	result, err := e.registry.Execute(ctx, tool, args)
	duration := time.Since(start)
	
	// Record metrics
	status := "success"
	if err != nil {
		status = "error"
	}
	metrics.RecordToolExecution(tool.Name, status, duration)
	
	if err != nil {
		return "", fmt.Errorf("tool execution failed: %w", err)
	}

	return result, nil
}

// ExecuteByName executes a tool by name
func (e *Executor) ExecuteByName(ctx context.Context, toolName string, args map[string]interface{}) (string, error) {
	tool, err := e.registry.Get(toolName)
	if err != nil {
		return "", fmt.Errorf("tool not found: %w", err)
	}

	return e.Execute(ctx, tool, args)
}

