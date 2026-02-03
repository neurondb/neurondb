/*-------------------------------------------------------------------------
 *
 * executor.go
 *    Workflow step executor
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/workflow/executor.go
 *
 *-------------------------------------------------------------------------
 */

package workflow

import (
	"context"
	"fmt"
	"time"

	"github.com/neurondb/NeuronMCP/internal/logging"
)

/* ToolExecutor executes a tool within a workflow step */
type ToolExecutor interface {
	ExecuteTool(ctx context.Context, toolName string, arguments map[string]interface{}) (interface{}, error)
}

/* Executor executes workflow steps */
type Executor struct {
	manager      *Manager
	toolExecutor ToolExecutor
	logger       *logging.Logger
}

/* NewExecutor creates a new workflow executor */
func NewExecutor(manager *Manager, toolExecutor ToolExecutor, logger *logging.Logger) *Executor {
	return &Executor{
		manager:      manager,
		toolExecutor: toolExecutor,
		logger:       logger,
	}
}

/* ExecuteWorkflow executes a workflow */
func (e *Executor) ExecuteWorkflow(ctx context.Context, executionID string) error {
	if ctx == nil {
		return fmt.Errorf("context cannot be nil")
	}

	/* Get execution state */
	exec, err := e.manager.GetExecution(executionID)
	if err != nil {
		return fmt.Errorf("failed to get execution: %w", err)
	}

	/* Get workflow */
	workflow, err := e.manager.GetWorkflow(exec.WorkflowID)
	if err != nil {
		return e.manager.UpdateExecution(executionID, func(e *ExecutionState) {
			e.Status = "failed"
			errMsg := fmt.Sprintf("workflow not found: %s", exec.WorkflowID)
			e.Error = &errMsg
		})
	}

	/* Update status to running */
	err = e.manager.UpdateExecution(executionID, func(e *ExecutionState) {
		e.Status = "running"
	})
	if err != nil {
		return err
	}

	/* Execute steps in order, respecting dependencies */
	completedSteps := make(map[string]bool)

	for {
		/* Find next step to execute */
		nextStep := e.findNextStep(workflow.Steps, completedSteps, exec)
		if nextStep == nil {
			/* All steps completed */
			now := time.Now()
			return e.manager.UpdateExecution(executionID, func(e *ExecutionState) {
				e.Status = "completed"
				e.CompletedAt = &now
			})
		}

		/* Check if cancelled */
		exec, _ = e.manager.GetExecution(executionID)
		if exec.Status == "cancelled" {
			return nil
		}

		/* Execute step */
		err := e.executeStep(ctx, executionID, *nextStep, exec)
		if err != nil {
			/* Handle error based on step configuration */
			if nextStep.OnError == "stop" {
				errMsg := err.Error()
				return e.manager.UpdateExecution(executionID, func(e *ExecutionState) {
					e.Status = "failed"
					e.Error = &errMsg
				})
			} else if nextStep.OnError == "continue" {
				/* Continue to next step */
				completedSteps[nextStep.ID] = true
				continue
			}
			/* Default: stop on error */
			errMsg := err.Error()
			return e.manager.UpdateExecution(executionID, func(e *ExecutionState) {
				e.Status = "failed"
				e.Error = &errMsg
			})
		}

		completedSteps[nextStep.ID] = true
	}
}

/* findNextStep finds the next step to execute */
func (e *Executor) findNextStep(steps []Step, completed map[string]bool, exec *ExecutionState) *Step {
	for _, step := range steps {
		/* Skip if already completed */
		if completed[step.ID] {
			continue
		}

		/* Check dependencies */
		allDepsMet := true
		for _, depID := range step.DependsOn {
			if !completed[depID] {
				allDepsMet = false
				break
			}
		}

		if allDepsMet {
			return &step
		}
	}

	return nil
}

/* executeStep executes a single workflow step */
func (e *Executor) executeStep(ctx context.Context, executionID string, step Step, exec *ExecutionState) error {
	/* Update current step */
	err := e.manager.UpdateExecution(executionID, func(e *ExecutionState) {
		e.CurrentStep = step.ID
	})
	if err != nil {
		return err
	}

	/* Evaluate condition if present */
	/* Note: Condition evaluation is not yet implemented. Steps with conditions will always execute. */
	/* Future implementation will support expressions like "variable == 'value'" or "status == 'success'" */
	if step.Condition != nil && *step.Condition != "" {
		if e.logger != nil {
			e.logger.Warn("workflow step condition not yet implemented; step will execute regardless",
				map[string]interface{}{"step_id": step.ID, "condition": *step.Condition})
		}
	}

	/* Prepare arguments with variable substitution */
	arguments := e.substituteVariables(step.Arguments, exec.Variables)

	/* Set timeout if specified */
	stepCtx := ctx
	if step.Timeout != nil {
		var cancel context.CancelFunc
		stepCtx, cancel = context.WithTimeout(ctx, *step.Timeout)
		defer cancel()
	}

	/* Execute tool with retries */
	var lastErr error
	for attempt := 0; attempt <= step.Retries; attempt++ {
		if attempt > 0 {
			/* Wait before retry */
			time.Sleep(time.Duration(attempt) * time.Second)
		}

		result, err := e.toolExecutor.ExecuteTool(stepCtx, step.Tool, arguments)
		if err == nil {
			/* Store result */
			return e.manager.UpdateExecution(executionID, func(e *ExecutionState) {
				e.Results[step.ID] = result
			})
		}

		lastErr = err
	}

	return fmt.Errorf("step %s failed after %d attempts: %w", step.ID, step.Retries+1, lastErr)
}

/* substituteVariables substitutes variables in arguments */
func (e *Executor) substituteVariables(args map[string]interface{}, variables map[string]interface{}) map[string]interface{} {
	if args == nil {
		return nil
	}

	result := make(map[string]interface{})
	for k, v := range args {
		result[k] = e.substituteValue(v, variables)
	}

	return result
}

/* substituteValue substitutes variables in a value */
func (e *Executor) substituteValue(value interface{}, variables map[string]interface{}) interface{} {
	switch v := value.(type) {
	case string:
		/* Check if it's a variable reference {{var}} */
		if len(v) > 4 && v[:2] == "{{" && v[len(v)-2:] == "}}" {
			varName := v[2 : len(v)-2]
			if varValue, exists := variables[varName]; exists {
				return varValue
			}
		}
		return v
	case map[string]interface{}:
		result := make(map[string]interface{})
		for k, val := range v {
			result[k] = e.substituteValue(val, variables)
		}
		return result
	case []interface{}:
		result := make([]interface{}, len(v))
		for i, val := range v {
			result[i] = e.substituteValue(val, variables)
		}
		return result
	default:
		return v
	}
}
