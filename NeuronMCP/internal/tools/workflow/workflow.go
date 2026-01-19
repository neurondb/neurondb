/*-------------------------------------------------------------------------
 *
 * workflow.go
 *    Workflow execution engine for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/workflow/workflow.go
 *
 *-------------------------------------------------------------------------
 */

package workflow

import (
	"fmt"
	"sync"
	"time"
)

/* Workflow represents a workflow definition */
type Workflow struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Steps       []Step                 `json:"steps"`
	Variables   map[string]interface{} `json:"variables,omitempty"`
	CreatedAt   time.Time              `json:"createdAt"`
	UpdatedAt   time.Time              `json:"updatedAt"`
}

/* Step represents a workflow step */
type Step struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Tool        string                 `json:"tool"`
	Arguments   map[string]interface{} `json:"arguments,omitempty"`
	Condition   *string                `json:"condition,omitempty"` /* JavaScript expression */
	OnError     string                 `json:"onError,omitempty"` /* continue, stop, retry */
	Retries    int                    `json:"retries,omitempty"`
	Timeout    *time.Duration         `json:"timeout,omitempty"`
	DependsOn  []string               `json:"dependsOn,omitempty"` /* Step IDs this step depends on */
}

/* ExecutionState represents workflow execution state */
type ExecutionState struct {
	WorkflowID  string                 `json:"workflowId"`
	Status      string                 `json:"status"` /* pending, running, completed, failed, cancelled */
	CurrentStep string                 `json:"currentStep,omitempty"`
	Results     map[string]interface{} `json:"results,omitempty"`
	Variables   map[string]interface{} `json:"variables,omitempty"`
	StartedAt   time.Time              `json:"startedAt"`
	UpdatedAt   time.Time              `json:"updatedAt"`
	CompletedAt *time.Time             `json:"completedAt,omitempty"`
	Error       *string                `json:"error,omitempty"`
}

/* Manager manages workflows */
type Manager struct {
	mu         sync.RWMutex
	workflows  map[string]*Workflow
	executions map[string]*ExecutionState
}

/* NewManager creates a new workflow manager */
func NewManager() *Manager {
	return &Manager{
		workflows:  make(map[string]*Workflow),
		executions: make(map[string]*ExecutionState),
	}
}

/* RegisterWorkflow registers a workflow */
func (m *Manager) RegisterWorkflow(workflow *Workflow) error {
	if workflow == nil {
		return fmt.Errorf("workflow cannot be nil")
	}
	if workflow.ID == "" {
		return fmt.Errorf("workflow ID cannot be empty")
	}
	if workflow.Name == "" {
		return fmt.Errorf("workflow name cannot be empty")
	}
	if len(workflow.Steps) == 0 {
		return fmt.Errorf("workflow must have at least one step")
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	workflow.UpdatedAt = time.Now()
	if workflow.CreatedAt.IsZero() {
		workflow.CreatedAt = time.Now()
	}

	m.workflows[workflow.ID] = workflow
	return nil
}

/* GetWorkflow retrieves a workflow */
func (m *Manager) GetWorkflow(id string) (*Workflow, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	workflow, exists := m.workflows[id]
	if !exists {
		return nil, fmt.Errorf("workflow not found: %s", id)
	}

	return workflow, nil
}

/* ListWorkflows lists all workflows */
func (m *Manager) ListWorkflows() []*Workflow {
	m.mu.RLock()
	defer m.mu.RUnlock()

	workflows := make([]*Workflow, 0, len(m.workflows))
	for _, wf := range m.workflows {
		workflows = append(workflows, wf)
	}

	return workflows
}

/* GetExecution retrieves an execution state */
func (m *Manager) GetExecution(executionID string) (*ExecutionState, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	exec, exists := m.executions[executionID]
	if !exists {
		return nil, fmt.Errorf("execution not found: %s", executionID)
	}

	return exec, nil
}

/* CreateExecution creates a new execution state */
func (m *Manager) CreateExecution(workflowID string, variables map[string]interface{}) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	workflow, exists := m.workflows[workflowID]
	if !exists {
		return "", fmt.Errorf("workflow not found: %s", workflowID)
	}

	executionID := fmt.Sprintf("exec-%d", time.Now().UnixNano())
	exec := &ExecutionState{
		WorkflowID: workflowID,
		Status:     "pending",
		Results:    make(map[string]interface{}),
		Variables:  variables,
		StartedAt:  time.Now(),
		UpdatedAt:  time.Now(),
	}

	if exec.Variables == nil {
		exec.Variables = make(map[string]interface{})
	}
	/* Merge workflow variables */
	for k, v := range workflow.Variables {
		if _, exists := exec.Variables[k]; !exists {
			exec.Variables[k] = v
		}
	}

	m.executions[executionID] = exec
	return executionID, nil
}

/* UpdateExecution updates an execution state */
func (m *Manager) UpdateExecution(executionID string, updates func(*ExecutionState)) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	exec, exists := m.executions[executionID]
	if !exists {
		return fmt.Errorf("execution not found: %s", executionID)
	}

	updates(exec)
	exec.UpdatedAt = time.Now()

	return nil
}

/* CancelExecution cancels an execution */
func (m *Manager) CancelExecution(executionID string) error {
	return m.UpdateExecution(executionID, func(exec *ExecutionState) {
		if exec.Status == "running" || exec.Status == "pending" {
			exec.Status = "cancelled"
		}
	})
}
