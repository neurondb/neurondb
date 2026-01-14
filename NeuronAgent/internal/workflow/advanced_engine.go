/*-------------------------------------------------------------------------
 *
 * advanced_engine.go
 *    Advanced workflow engine enhancements
 *
 * Provides parallel execution, conditional branching, workflow versioning,
 * templates, event-driven workflows, and enhanced error handling.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/workflow/advanced_engine.go
 *
 *-------------------------------------------------------------------------
 */

package workflow

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
)

/* AdvancedWorkflowEngine extends the base workflow engine with advanced features */
type AdvancedWorkflowEngine struct {
	engine  *Engine
	queries *db.Queries
}

/* NewAdvancedWorkflowEngine creates an advanced workflow engine */
func NewAdvancedWorkflowEngine(engine *Engine, queries *db.Queries) *AdvancedWorkflowEngine {
	return &AdvancedWorkflowEngine{
		engine:  engine,
		queries: queries,
	}
}

/* ExecuteParallel executes workflow steps in parallel where possible */
func (awe *AdvancedWorkflowEngine) ExecuteParallel(ctx context.Context, executionID uuid.UUID, steps []db.WorkflowStep, stepOutputs map[uuid.UUID]map[string]interface{}, inputs map[string]interface{}) (map[uuid.UUID]map[string]interface{}, error) {
	/* Build dependency graph */
	dependencyGraph := make(map[uuid.UUID][]uuid.UUID)
	inDegree := make(map[uuid.UUID]int)
	
	for _, step := range steps {
		inDegree[step.ID] = 0
		if step.Dependencies != nil {
			for _, depIDStr := range step.Dependencies {
		depID, _ := uuid.Parse(depIDStr)
				if dependencyGraph[depID] == nil {
					dependencyGraph[depID] = make([]uuid.UUID, 0)
				}
				dependencyGraph[depID] = append(dependencyGraph[depID], step.ID)
				inDegree[step.ID]++
			}
		}
	}

	/* Execute in parallel using topological levels */
	resultOutputs := make(map[uuid.UUID]map[string]interface{})
	var mu sync.Mutex
	
	for {
		/* Find steps with no dependencies (ready to execute) */
		readySteps := make([]db.WorkflowStep, 0)
		for _, step := range steps {
			if inDegree[step.ID] == 0 && resultOutputs[step.ID] == nil {
				readySteps = append(readySteps, step)
			}
		}

		if len(readySteps) == 0 {
			break /* No more steps to execute */
		}

		/* Execute ready steps in parallel */
		var wg sync.WaitGroup
		errors := make(chan error, len(readySteps))

		for _, step := range readySteps {
			wg.Add(1)
			go func(s db.WorkflowStep) {
				defer wg.Done()

				/* Build inputs */
				stepInputs := awe.buildStepInputs(&s, inputs, resultOutputs)

				/* Execute step */
				outputs, err := awe.engine.ExecuteStep(ctx, executionID, &s, stepInputs)
				if err != nil {
					errors <- fmt.Errorf("parallel step execution failed: step_id='%s', step_name='%s', error=%w", s.ID.String(), s.StepName, err)
					return
				}

				mu.Lock()
				resultOutputs[s.ID] = outputs
				mu.Unlock()

				/* Update dependencies - mark dependent steps as ready */
				for _, dependentID := range dependencyGraph[s.ID] {
					mu.Lock()
					inDegree[dependentID]--
					mu.Unlock()
				}
			}(step)
		}

		wg.Wait()
		close(errors)

		/* Check for errors */
		for err := range errors {
			if err != nil {
				return nil, err
			}
		}
	}

	return resultOutputs, nil
}

/* ExecuteConditional executes workflow with conditional branching */
func (awe *AdvancedWorkflowEngine) ExecuteConditional(ctx context.Context, executionID uuid.UUID, step *db.WorkflowStep, inputs map[string]interface{}, condition string) (map[string]interface{}, error) {
	/* Evaluate condition */
	result, err := awe.evaluateCondition(ctx, condition, inputs)
	if err != nil {
		return nil, fmt.Errorf("condition evaluation failed: step_id='%s', condition='%s', error=%w", step.ID.String(), condition, err)
	}

	if !result {
		/* Condition false - skip step */
		return make(map[string]interface{}), nil
	}

	/* Condition true - execute step */
	return awe.engine.ExecuteStep(ctx, executionID, step, inputs)
}

/* VersionWorkflow creates a new version of a workflow */
func (awe *AdvancedWorkflowEngine) VersionWorkflow(ctx context.Context, workflowID uuid.UUID, version string, description string) (uuid.UUID, error) {
	/* Get original workflow */
	workflow, err := awe.queries.GetWorkflowByID(ctx, workflowID)
	if err != nil {
		return uuid.Nil, fmt.Errorf("workflow versioning failed: workflow_not_found=true, workflow_id='%s', error=%w", workflowID.String(), err)
	}

	/* Get workflow steps */
	steps, err := awe.queries.ListWorkflowSteps(ctx, workflowID)
	if err != nil {
		return uuid.Nil, fmt.Errorf("workflow versioning failed: steps_retrieval_failed=true, error=%w", err)
	}

	/* Create new workflow version */
	newWorkflow := &db.Workflow{
		Name:          fmt.Sprintf("%s (v%s)", workflow.Name, version),
		DAGDefinition: workflow.DAGDefinition,
		Status:        "draft",
	}

	if err := awe.queries.CreateWorkflow(ctx, newWorkflow); err != nil {
		return uuid.Nil, fmt.Errorf("workflow versioning failed: creation_failed=true, error=%w", err)
	}

	/* Copy steps */
	for _, step := range steps {
		newStep := &db.WorkflowStep{
			WorkflowID:   newWorkflow.ID,
			StepName:     step.StepName,
			StepType:     step.StepType,
			Inputs:       step.Inputs,
			Outputs:      step.Outputs,
			Dependencies: step.Dependencies,
			RetryConfig:  step.RetryConfig,
		}
		if err := awe.queries.CreateWorkflowStep(ctx, newStep); err != nil {
			return uuid.Nil, fmt.Errorf("workflow versioning failed: step_copy_failed=true, step_name='%s', error=%w", step.StepName, err)
		}
	}

	return newWorkflow.ID, nil
}

/* CreateWorkflowTemplate creates a reusable workflow template */
func (awe *AdvancedWorkflowEngine) CreateWorkflowTemplate(ctx context.Context, name string, description string, steps []db.WorkflowStep, parameters []TemplateParameter) (uuid.UUID, error) {
	/* Create template workflow */
	/* Store description and metadata in DAGDefinition */
	dagDefinition := map[string]interface{}{
		"description": description,
		"template":    true,
		"parameters":  parameters,
		"created_at":  time.Now().Format(time.RFC3339),
	}
	template := &db.Workflow{
		Name:          fmt.Sprintf("Template: %s", name),
		DAGDefinition: db.FromMap(dagDefinition),
		Status:        "template",
	}

	if err := awe.queries.CreateWorkflow(ctx, template); err != nil {
		return uuid.Nil, fmt.Errorf("template creation failed: creation_failed=true, error=%w", err)
	}

	/* Create template steps */
	for _, step := range steps {
		newStep := &db.WorkflowStep{
			WorkflowID:   template.ID,
			StepName:     step.StepName,
			StepType:     step.StepType,
			Inputs:       step.Inputs,
			Outputs:      step.Outputs,
			Dependencies: step.Dependencies,
			RetryConfig:  step.RetryConfig,
		}
		if err := awe.queries.CreateWorkflowStep(ctx, newStep); err != nil {
			return uuid.Nil, fmt.Errorf("template creation failed: step_creation_failed=true, step_name='%s', error=%w", step.StepName, err)
		}
	}

	return template.ID, nil
}

/* InstantiateTemplate creates a workflow instance from a template */
func (awe *AdvancedWorkflowEngine) InstantiateTemplate(ctx context.Context, templateID uuid.UUID, parameterValues map[string]interface{}) (uuid.UUID, error) {
	/* Get template */
	template, err := awe.queries.GetWorkflowByID(ctx, templateID)
	if err != nil {
		return uuid.Nil, fmt.Errorf("template instantiation failed: template_not_found=true, template_id='%s', error=%w", templateID.String(), err)
	}

	/* Check if it's a template */
	if template.Status != "template" {
		return uuid.Nil, fmt.Errorf("template instantiation failed: not_a_template=true, workflow_id='%s'", templateID.String())
	}

	/* Get template steps */
	steps, err := awe.queries.ListWorkflowSteps(ctx, templateID)
	if err != nil {
		return uuid.Nil, fmt.Errorf("template instantiation failed: steps_retrieval_failed=true, error=%w", err)
	}

	/* Create new workflow instance */
	/* Extract description from template's DAGDefinition if present */
	var description string
	if template.DAGDefinition != nil {
		if desc, ok := template.DAGDefinition["description"].(string); ok {
			description = desc
		}
	}
	/* Store metadata in DAGDefinition */
	dagDefinition := map[string]interface{}{
		"description": description,
		"template_id": templateID.String(),
		"parameters":  parameterValues,
		"created_at":  time.Now().Format(time.RFC3339),
	}
	instance := &db.Workflow{
		Name:          template.Name,
		DAGDefinition: db.FromMap(dagDefinition),
		Status:        "draft",
	}

	if err := awe.queries.CreateWorkflow(ctx, instance); err != nil {
		return uuid.Nil, fmt.Errorf("template instantiation failed: creation_failed=true, error=%w", err)
	}

	/* Create instance steps with parameter substitution */
	for _, step := range steps {
		/* Substitute parameters in step inputs */
		var inputs map[string]interface{}
		if step.Inputs != nil {
			inputsMap := step.Inputs.ToMap()
			inputs = awe.substituteParameters(inputsMap, parameterValues)
		}

		newStep := &db.WorkflowStep{
			WorkflowID:   instance.ID,
			StepName:     step.StepName,
			StepType:     step.StepType,
			Inputs:       db.FromMap(inputs),
			Outputs:      step.Outputs,
			Dependencies: step.Dependencies,
			RetryConfig:  step.RetryConfig,
		}
		if err := awe.queries.CreateWorkflowStep(ctx, newStep); err != nil {
			return uuid.Nil, fmt.Errorf("template instantiation failed: step_creation_failed=true, step_name='%s', error=%w", step.StepName, err)
		}
	}

	return instance.ID, nil
}

/* RegisterEventTrigger registers an event trigger for a workflow */
func (awe *AdvancedWorkflowEngine) RegisterEventTrigger(ctx context.Context, workflowID uuid.UUID, eventType string, eventFilter map[string]interface{}) error {
	query := `INSERT INTO neurondb_agent.workflow_triggers
		(workflow_id, event_type, event_filter, enabled, created_at)
		VALUES ($1, $2, $3::jsonb, $4, $5)
		ON CONFLICT (workflow_id, event_type) DO UPDATE
		SET event_filter = $3::jsonb, enabled = $4, updated_at = $5`

	_, err := awe.queries.DB.ExecContext(ctx, query, workflowID, eventType, eventFilter, true, time.Now())
	if err != nil {
		return fmt.Errorf("event trigger registration failed: workflow_id='%s', event_type='%s', error=%w", workflowID.String(), eventType, err)
	}

	return nil
}

/* TriggerWorkflowOnEvent triggers a workflow when an event occurs */
func (awe *AdvancedWorkflowEngine) TriggerWorkflowOnEvent(ctx context.Context, eventType string, eventData map[string]interface{}) error {
	/* Find workflows with matching triggers */
	query := `SELECT workflow_id, event_filter
		FROM neurondb_agent.workflow_triggers
		WHERE event_type = $1 AND enabled = true`

	type TriggerRow struct {
		WorkflowID  uuid.UUID              `db:"workflow_id"`
		EventFilter map[string]interface{} `db:"event_filter"`
	}

	var triggers []TriggerRow
	err := awe.queries.DB.SelectContext(ctx, &triggers, query, eventType)
	if err != nil {
		return fmt.Errorf("event trigger lookup failed: event_type='%s', error=%w", eventType, err)
	}

	/* Check each trigger filter */
	for _, trigger := range triggers {
		if awe.matchesEventFilter(eventData, trigger.EventFilter) {
			/* Trigger workflow */
			_, err := awe.engine.ExecuteWorkflow(ctx, trigger.WorkflowID, "event", map[string]interface{}{"event_type": eventType}, eventData)
			if err != nil {
				return fmt.Errorf("event-triggered workflow execution failed: workflow_id='%s', event_type='%s', error=%w", trigger.WorkflowID.String(), eventType, err)
			}
		}
	}

	return nil
}

/* Helper types */

type TemplateParameter struct {
	Name        string
	Type        string
	Description string
	Default     interface{}
	Required    bool
}

/* Helper methods */

func (awe *AdvancedWorkflowEngine) buildStepInputs(step *db.WorkflowStep, workflowInputs map[string]interface{}, stepOutputs map[uuid.UUID]map[string]interface{}) map[string]interface{} {
	/* This would be similar to Engine.buildStepInputs */
	/* For now, return a simplified version */
	inputs := make(map[string]interface{})
	
	/* Copy workflow inputs */
	for k, v := range workflowInputs {
		inputs[k] = v
	}

	/* Add previous step outputs */
	if step.Dependencies != nil {
		for _, depIDStr := range step.Dependencies {
		depID, _ := uuid.Parse(depIDStr)
			if outputs, exists := stepOutputs[depID]; exists {
				for k, v := range outputs {
					inputs[fmt.Sprintf("%s.%s", depID.String(), k)] = v
				}
			}
		}
	}

	return inputs
}

func (awe *AdvancedWorkflowEngine) evaluateCondition(ctx context.Context, condition string, inputs map[string]interface{}) (bool, error) {
	if condition == "" {
		return true, nil
	}

	/* Parse and evaluate condition expression */
	/* Supports: ==, !=, >, <, >=, <=, &&, ||, in */
	
	/* Trim whitespace */
	condition = strings.TrimSpace(condition)
	
	/* Handle logical operators (&&, ||) */
	if strings.Contains(condition, " && ") {
		parts := strings.Split(condition, " && ")
		for _, part := range parts {
			result, err := awe.evaluateCondition(ctx, strings.TrimSpace(part), inputs)
			if err != nil {
				return false, err
			}
			if !result {
				return false, nil
			}
		}
		return true, nil
	}
	
	if strings.Contains(condition, " || ") {
		parts := strings.Split(condition, " || ")
		for _, part := range parts {
			result, err := awe.evaluateCondition(ctx, strings.TrimSpace(part), inputs)
			if err != nil {
				return false, err
			}
			if result {
				return true, nil
			}
		}
		return false, nil
	}
	
	/* Handle comparison operators */
	if strings.Contains(condition, " == ") {
		return awe.evaluateComparison(condition, " == ", inputs, func(a, b interface{}) bool {
			return compareValues(a, b) == 0
		})
	}
	
	if strings.Contains(condition, " != ") {
		return awe.evaluateComparison(condition, " != ", inputs, func(a, b interface{}) bool {
			return compareValues(a, b) != 0
		})
	}
	
	if strings.Contains(condition, " >= ") {
		return awe.evaluateComparison(condition, " >= ", inputs, func(a, b interface{}) bool {
			return compareValues(a, b) >= 0
		})
	}
	
	if strings.Contains(condition, " <= ") {
		return awe.evaluateComparison(condition, " <= ", inputs, func(a, b interface{}) bool {
			return compareValues(a, b) <= 0
		})
	}
	
	if strings.Contains(condition, " > ") {
		return awe.evaluateComparison(condition, " > ", inputs, func(a, b interface{}) bool {
			return compareValues(a, b) > 0
		})
	}
	
	if strings.Contains(condition, " < ") {
		return awe.evaluateComparison(condition, " < ", inputs, func(a, b interface{}) bool {
			return compareValues(a, b) < 0
		})
	}
	
	/* Handle 'in' operator */
	if strings.Contains(condition, " in ") {
		return awe.evaluateInOperator(condition, inputs)
	}
	
	/* If no operator found, treat as boolean value */
	if val, ok := inputs[condition]; ok {
		if boolVal, ok := val.(bool); ok {
			return boolVal, nil
		}
	}
	
	/* Default to false if condition cannot be evaluated */
	return false, fmt.Errorf("unable to evaluate condition: '%s'", condition)
}

func (awe *AdvancedWorkflowEngine) evaluateComparison(condition, operator string, inputs map[string]interface{}, compareFunc func(interface{}, interface{}) bool) (bool, error) {
	parts := strings.Split(condition, operator)
	if len(parts) != 2 {
		return false, fmt.Errorf("invalid comparison expression: '%s'", condition)
	}
	
	left := strings.TrimSpace(parts[0])
	right := strings.TrimSpace(parts[1])
	
	/* Get left value */
	leftVal := awe.getValue(left, inputs)
	
	/* Get right value (could be literal or variable) */
	rightVal := awe.getValue(right, inputs)
	
	return compareFunc(leftVal, rightVal), nil
}

func (awe *AdvancedWorkflowEngine) evaluateInOperator(condition string, inputs map[string]interface{}) (bool, error) {
	parts := strings.Split(condition, " in ")
	if len(parts) != 2 {
		return false, fmt.Errorf("invalid 'in' expression: '%s'", condition)
	}
	
	left := strings.TrimSpace(parts[0])
	right := strings.TrimSpace(parts[1])
	
	/* Get left value */
	leftVal := awe.getValue(left, inputs)
	
	/* Parse right side as array */
	/* Remove brackets if present */
	right = strings.TrimSpace(right)
	if strings.HasPrefix(right, "[") && strings.HasSuffix(right, "]") {
		right = right[1 : len(right)-1]
	}
	
	/* Split by comma */
	rightParts := strings.Split(right, ",")
	for _, part := range rightParts {
		part = strings.TrimSpace(part)
		/* Remove quotes if present */
		if (strings.HasPrefix(part, "'") && strings.HasSuffix(part, "'")) ||
			(strings.HasPrefix(part, "\"") && strings.HasSuffix(part, "\"")) {
			part = part[1 : len(part)-1]
		}
		
		rightVal := awe.parseValue(part)
		if compareValues(leftVal, rightVal) == 0 {
			return true, nil
		}
	}
	
	return false, nil
}

func (awe *AdvancedWorkflowEngine) getValue(expr string, inputs map[string]interface{}) interface{} {
	expr = strings.TrimSpace(expr)
	
	/* Check if it's a variable reference (input.xxx or just xxx) */
	if strings.HasPrefix(expr, "input.") {
		key := strings.TrimPrefix(expr, "input.")
		if val, ok := inputs[key]; ok {
			return val
		}
	} else if val, ok := inputs[expr]; ok {
		return val
	}
	
	/* Try to parse as literal value */
	return awe.parseValue(expr)
}

func (awe *AdvancedWorkflowEngine) parseValue(expr string) interface{} {
	expr = strings.TrimSpace(expr)
	
	/* Remove quotes if present */
	if (strings.HasPrefix(expr, "'") && strings.HasSuffix(expr, "'")) ||
		(strings.HasPrefix(expr, "\"") && strings.HasSuffix(expr, "\"")) {
		return expr[1 : len(expr)-1]
	}
	
	/* Try boolean */
	if expr == "true" {
		return true
	}
	if expr == "false" {
		return false
	}
	
	/* Try integer */
	if intVal, err := strconv.ParseInt(expr, 10, 64); err == nil {
		return intVal
	}
	
	/* Try float */
	if floatVal, err := strconv.ParseFloat(expr, 64); err == nil {
		return floatVal
	}
	
	/* Return as string */
	return expr
}

func compareValues(a, b interface{}) int {
	/* Convert to comparable types */
	aFloat, aIsFloat := toFloat(a)
	bFloat, bIsFloat := toFloat(b)
	
	if aIsFloat && bIsFloat {
		if aFloat < bFloat {
			return -1
		}
		if aFloat > bFloat {
			return 1
		}
		return 0
	}
	
	/* String comparison */
	aStr := fmt.Sprintf("%v", a)
	bStr := fmt.Sprintf("%v", b)
	
	if aStr < bStr {
		return -1
	}
	if aStr > bStr {
		return 1
	}
	return 0
}

func toFloat(v interface{}) (float64, bool) {
	switch val := v.(type) {
	case int:
		return float64(val), true
	case int64:
		return float64(val), true
	case float32:
		return float64(val), true
	case float64:
		return val, true
	default:
		return 0, false
	}
}

func (awe *AdvancedWorkflowEngine) substituteParameters(config map[string]interface{}, parameterValues map[string]interface{}) map[string]interface{} {
	configJSON, _ := json.Marshal(config)
	configStr := string(configJSON)

	/* Simple parameter substitution */
	for paramName, paramValue := range parameterValues {
		placeholder := fmt.Sprintf("{{%s}}", paramName)
		valueStr := fmt.Sprintf("%v", paramValue)
		configStr = replaceAll(configStr, placeholder, valueStr)
	}

	var result map[string]interface{}
	json.Unmarshal([]byte(configStr), &result)
	return result
}

func (awe *AdvancedWorkflowEngine) matchesEventFilter(eventData map[string]interface{}, filter map[string]interface{}) bool {
	if filter == nil || len(filter) == 0 {
		return true /* No filter = match all */
	}

	/* Check if event data matches filter */
	for key, filterValue := range filter {
		eventValue, exists := eventData[key]
		if !exists {
			return false
		}
		if eventValue != filterValue {
			return false
		}
	}

	return true
}

func replaceAll(s, old, new string) string {
	result := ""
	start := 0
	for {
		idx := findSubstring(s, old, start)
		if idx == -1 {
			result += s[start:]
			break
		}
		result += s[start:idx] + new
		start = idx + len(old)
	}
	return result
}

func findSubstring(s, substr string, start int) int {
	for i := start; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return i
		}
	}
	return -1
}

