/*-------------------------------------------------------------------------
 *
 * tools.go
 *    Workflow orchestration tools
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/workflow/tools.go
 *
 *-------------------------------------------------------------------------
 */

package workflow

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/pkg/mcp"
)

/* maxWorkflowDuration caps how long a background workflow can run */
const maxWorkflowDuration = 1 * time.Hour

/* WorkflowManagerTool provides workflow management */
type WorkflowManagerTool struct {
	manager *Manager
	logger  *logging.Logger
}

/* NewWorkflowManagerTool creates a workflow manager tool */
func NewWorkflowManagerTool(manager *Manager, logger *logging.Logger) *WorkflowManagerTool {
	return &WorkflowManagerTool{
		manager: manager,
		logger:  logger,
	}
}

/* CreateWorkflowTool creates a workflow */
type CreateWorkflowTool struct {
	baseTool *BaseToolWrapper
	manager  *Manager
	logger   *logging.Logger
}

/* NewCreateWorkflowTool creates a new create workflow tool */
func NewCreateWorkflowTool(manager *Manager, logger *logging.Logger) *CreateWorkflowTool {
	return &CreateWorkflowTool{
		baseTool: &BaseToolWrapper{
			name:        "workflow_create",
			description: "Create a new workflow with steps",
			inputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"id": map[string]interface{}{
						"type":        "string",
						"description": "Workflow ID",
					},
					"name": map[string]interface{}{
						"type":        "string",
						"description": "Workflow name",
					},
					"description": map[string]interface{}{
						"type":        "string",
						"description": "Workflow description",
					},
					"steps": map[string]interface{}{
						"type":        "array",
						"description": "Workflow steps",
						"items": map[string]interface{}{
							"type": "object",
							"properties": map[string]interface{}{
								"id": map[string]interface{}{
									"type": "string",
								},
								"name": map[string]interface{}{
									"type": "string",
								},
								"tool": map[string]interface{}{
									"type": "string",
								},
								"arguments": map[string]interface{}{
									"type": "object",
								},
							},
						},
					},
				},
				"required": []interface{}{"id", "name", "steps"},
			},
			version: "2.0.0",
		},
		manager: manager,
		logger:  logger,
	}
}

/* Name returns the tool name */
func (t *CreateWorkflowTool) Name() string { return t.baseTool.Name() }

/* Description returns the tool description */
func (t *CreateWorkflowTool) Description() string { return t.baseTool.Description() }

/* InputSchema returns the input schema */
func (t *CreateWorkflowTool) InputSchema() map[string]interface{} { return t.baseTool.InputSchema() }

/* Version returns the tool version */
func (t *CreateWorkflowTool) Version() string { return t.baseTool.Version() }

/* OutputSchema returns the output schema */
func (t *CreateWorkflowTool) OutputSchema() map[string]interface{} { return t.baseTool.OutputSchema() }

/* Deprecated returns whether the tool is deprecated */
func (t *CreateWorkflowTool) Deprecated() bool { return t.baseTool.Deprecated() }

/* Deprecation returns deprecation information */
func (t *CreateWorkflowTool) Deprecation() *mcp.DeprecationInfo { return t.baseTool.Deprecation() }

/* Execute creates a workflow */
func (t *CreateWorkflowTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	id, _ := params["id"].(string)
	name, _ := params["name"].(string)
	description, _ := params["description"].(string)
	stepsData, _ := params["steps"].([]interface{})

	if id == "" {
		return errorResult("workflow ID is required", "VALIDATION_ERROR", nil), nil
	}
	if name == "" {
		return errorResult("workflow name is required", "VALIDATION_ERROR", nil), nil
	}

	/* Parse steps */
	steps := make([]Step, 0, len(stepsData))
	for i, stepData := range stepsData {
		stepMap, ok := stepData.(map[string]interface{})
		if !ok {
			return errorResult(fmt.Sprintf("invalid step at index %d", i), "VALIDATION_ERROR", nil), nil
		}

		stepID, _ := stepMap["id"].(string)
		stepName, _ := stepMap["name"].(string)
		tool, _ := stepMap["tool"].(string)
		arguments, _ := stepMap["arguments"].(map[string]interface{})

		step := Step{
			ID:        stepID,
			Name:      stepName,
			Tool:      tool,
			Arguments: arguments,
		}

		steps = append(steps, step)
	}

	workflow := &Workflow{
		ID:          id,
		Name:        name,
		Description: description,
		Steps:       steps,
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}

	if err := t.manager.RegisterWorkflow(workflow); err != nil {
		return errorResult(fmt.Sprintf("failed to create workflow: %v", err), "WORKFLOW_ERROR", nil), nil
	}

	return successResult(map[string]interface{}{
		"workflow_id": id,
		"name":        name,
		"steps_count": len(steps),
	}), nil
}

/* ExecuteWorkflowTool executes a workflow */
type ExecuteWorkflowTool struct {
	baseTool *BaseToolWrapper
	manager  *Manager
	executor *Executor
	logger   *logging.Logger
}

/* NewExecuteWorkflowTool creates a new execute workflow tool */
func NewExecuteWorkflowTool(manager *Manager, executor *Executor, logger *logging.Logger) *ExecuteWorkflowTool {
	return &ExecuteWorkflowTool{
		baseTool: &BaseToolWrapper{
			name:        "workflow_execute",
			description: "Execute a workflow",
			inputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"workflow_id": map[string]interface{}{
						"type":        "string",
						"description": "Workflow ID to execute",
					},
					"variables": map[string]interface{}{
						"type":        "object",
						"description": "Workflow variables",
					},
				},
				"required": []interface{}{"workflow_id"},
			},
			version: "2.0.0",
		},
		manager:  manager,
		executor: executor,
		logger:   logger,
	}
}

/* Name returns the tool name */
func (t *ExecuteWorkflowTool) Name() string { return t.baseTool.Name() }

/* Description returns the tool description */
func (t *ExecuteWorkflowTool) Description() string { return t.baseTool.Description() }

/* InputSchema returns the input schema */
func (t *ExecuteWorkflowTool) InputSchema() map[string]interface{} { return t.baseTool.InputSchema() }

/* Version returns the tool version */
func (t *ExecuteWorkflowTool) Version() string { return t.baseTool.Version() }

/* OutputSchema returns the output schema */
func (t *ExecuteWorkflowTool) OutputSchema() map[string]interface{} { return t.baseTool.OutputSchema() }

/* Deprecated returns whether the tool is deprecated */
func (t *ExecuteWorkflowTool) Deprecated() bool { return t.baseTool.Deprecated() }

/* Deprecation returns deprecation information */
func (t *ExecuteWorkflowTool) Deprecation() *mcp.DeprecationInfo { return t.baseTool.Deprecation() }

/* Execute executes a workflow */
func (t *ExecuteWorkflowTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	workflowID, _ := params["workflow_id"].(string)
	variables, _ := params["variables"].(map[string]interface{})

	if workflowID == "" {
		return errorResult("workflow_id is required", "VALIDATION_ERROR", nil), nil
	}

	executionID, err := t.manager.CreateExecution(workflowID, variables)
	if err != nil {
		return errorResult(fmt.Sprintf("failed to create execution: %v", err), "WORKFLOW_ERROR", nil), nil
	}

	/* Execute workflow in background with max duration so it cannot run indefinitely */
	runCtx, runCancel := context.WithTimeout(context.WithoutCancel(ctx), maxWorkflowDuration)
	go func() {
		defer runCancel()
		if err := t.executor.ExecuteWorkflow(runCtx, executionID); err != nil {
			if t.logger != nil {
				t.logger.Error("Workflow execution failed", err, map[string]interface{}{
					"execution_id": executionID,
					"workflow_id":  workflowID,
				})
			}
		}
	}()

	return successResult(map[string]interface{}{
		"execution_id": executionID,
		"workflow_id":  workflowID,
		"status":       "running",
	}), nil
}

/* WorkflowStatusTool checks workflow execution status */
type WorkflowStatusTool struct {
	baseTool *BaseToolWrapper
	manager  *Manager
	logger   *logging.Logger
}

/* NewWorkflowStatusTool creates a new workflow status tool */
func NewWorkflowStatusTool(manager *Manager, logger *logging.Logger) *WorkflowStatusTool {
	return &WorkflowStatusTool{
		baseTool: &BaseToolWrapper{
			name:        "workflow_status",
			description: "Check workflow execution status",
			inputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"execution_id": map[string]interface{}{
						"type":        "string",
						"description": "Execution ID",
					},
				},
				"required": []interface{}{"execution_id"},
			},
			version: "2.0.0",
		},
		manager: manager,
		logger:  logger,
	}
}

/* Name returns the tool name */
func (t *WorkflowStatusTool) Name() string { return t.baseTool.Name() }

/* Description returns the tool description */
func (t *WorkflowStatusTool) Description() string { return t.baseTool.Description() }

/* InputSchema returns the input schema */
func (t *WorkflowStatusTool) InputSchema() map[string]interface{} { return t.baseTool.InputSchema() }

/* Version returns the tool version */
func (t *WorkflowStatusTool) Version() string { return t.baseTool.Version() }

/* OutputSchema returns the output schema */
func (t *WorkflowStatusTool) OutputSchema() map[string]interface{} { return t.baseTool.OutputSchema() }

/* Deprecated returns whether the tool is deprecated */
func (t *WorkflowStatusTool) Deprecated() bool { return t.baseTool.Deprecated() }

/* Deprecation returns deprecation information */
func (t *WorkflowStatusTool) Deprecation() *mcp.DeprecationInfo { return t.baseTool.Deprecation() }

/* Execute checks workflow status */
func (t *WorkflowStatusTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	executionID, _ := params["execution_id"].(string)

	if executionID == "" {
		return errorResult("execution_id is required", "VALIDATION_ERROR", nil), nil
	}

	exec, err := t.manager.GetExecution(executionID)
	if err != nil {
		return errorResult(fmt.Sprintf("execution not found: %v", err), "NOT_FOUND", nil), nil
	}

	/* Convert to JSON-serializable format */
	execJSON, _ := json.Marshal(exec)
	var execMap map[string]interface{}
	json.Unmarshal(execJSON, &execMap)

	return successResult(execMap), nil
}

/* ListWorkflowsTool lists all workflows */
type ListWorkflowsTool struct {
	baseTool *BaseToolWrapper
	manager  *Manager
	logger   *logging.Logger
}

/* NewListWorkflowsTool creates a new list workflows tool */
func NewListWorkflowsTool(manager *Manager, logger *logging.Logger) *ListWorkflowsTool {
	return &ListWorkflowsTool{
		baseTool: &BaseToolWrapper{
			name:        "workflow_list",
			description: "List all available workflows",
			inputSchema: map[string]interface{}{
				"type":       "object",
				"properties": map[string]interface{}{},
			},
			version: "2.0.0",
		},
		manager: manager,
		logger:  logger,
	}
}

/* Name returns the tool name */
func (t *ListWorkflowsTool) Name() string { return t.baseTool.Name() }

/* Description returns the tool description */
func (t *ListWorkflowsTool) Description() string { return t.baseTool.Description() }

/* InputSchema returns the input schema */
func (t *ListWorkflowsTool) InputSchema() map[string]interface{} { return t.baseTool.InputSchema() }

/* Version returns the tool version */
func (t *ListWorkflowsTool) Version() string { return t.baseTool.Version() }

/* OutputSchema returns the output schema */
func (t *ListWorkflowsTool) OutputSchema() map[string]interface{} { return t.baseTool.OutputSchema() }

/* Deprecated returns whether the tool is deprecated */
func (t *ListWorkflowsTool) Deprecated() bool { return t.baseTool.Deprecated() }

/* Deprecation returns deprecation information */
func (t *ListWorkflowsTool) Deprecation() *mcp.DeprecationInfo { return t.baseTool.Deprecation() }

/* Execute lists workflows */
func (t *ListWorkflowsTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	workflows := t.manager.ListWorkflows()

	workflowsList := make([]map[string]interface{}, 0, len(workflows))
	for _, wf := range workflows {
		workflowsList = append(workflowsList, map[string]interface{}{
			"id":          wf.ID,
			"name":        wf.Name,
			"description": wf.Description,
			"steps_count": len(wf.Steps),
		})
	}

	return successResult(map[string]interface{}{
		"workflows": workflowsList,
		"count":     len(workflowsList),
	}), nil
}
