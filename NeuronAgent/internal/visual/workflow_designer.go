/*-------------------------------------------------------------------------
 *
 * workflow_designer.go
 *    Visual workflow designer for NeuronAgent
 *
 * Provides data structures and API for visual workflow design.
 * Actual UI implementation would be separate (e.g., React frontend).
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/visual/workflow_designer.go
 *
 *-------------------------------------------------------------------------
 */

package visual

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/google/uuid"
	"github.com/lib/pq"
	"github.com/neurondb/NeuronAgent/internal/db"
)

/* WorkflowDesigner provides visual workflow design capabilities */
type WorkflowDesigner struct {
	queries *db.Queries
}

/* WorkflowGraph represents a visual workflow graph */
type WorkflowGraph struct {
	Nodes []WorkflowNode `json:"nodes"`
	Edges []WorkflowEdge `json:"edges"`
}

/* WorkflowNode represents a node in the workflow graph */
type WorkflowNode struct {
	ID          string                 `json:"id"`
	Type        string                 `json:"type"`
	Label       string                 `json:"label"`
	Position    Position               `json:"position"`
	Data        map[string]interface{} `json:"data"`
	Config      map[string]interface{} `json:"config"`
}

/* WorkflowEdge represents an edge in the workflow graph */
type WorkflowEdge struct {
	ID        string `json:"id"`
	Source    string `json:"source"`
	Target    string `json:"target"`
	Label     string `json:"label,omitempty"`
	Condition string `json:"condition,omitempty"`
}

/* Position represents node position in the visual editor */
type Position struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
}

/* NewWorkflowDesigner creates a new workflow designer */
func NewWorkflowDesigner(queries *db.Queries) *WorkflowDesigner {
	return &WorkflowDesigner{
		queries: queries,
	}
}

/* SaveWorkflowGraph saves a workflow graph */
func (wd *WorkflowDesigner) SaveWorkflowGraph(ctx context.Context, workflowID uuid.UUID, graph *WorkflowGraph) error {
	/* Convert graph to workflow steps */
	steps, err := wd.graphToSteps(graph)
	if err != nil {
		return fmt.Errorf("workflow graph save failed: conversion_error=true, error=%w", err)
	}

	/* Delete existing steps */
	deleteQuery := `DELETE FROM neurondb_agent.workflow_steps WHERE workflow_id = $1`
	_, err = wd.queries.DB.ExecContext(ctx, deleteQuery, workflowID)
	if err != nil {
		return fmt.Errorf("workflow graph save failed: deletion_error=true, error=%w", err)
	}

	/* Create new steps */
	for _, step := range steps {
		if err := wd.queries.CreateWorkflowStep(ctx, step); err != nil {
			return fmt.Errorf("workflow graph save failed: step_creation_error=true, step_name='%s', error=%w", step.StepName, err)
		}
	}

	/* Update workflow with graph metadata */
	graphJSON, _ := json.Marshal(graph)
	updateQuery := `UPDATE neurondb_agent.workflows
		SET dag_definition = $1::jsonb
		WHERE id = $2`

	_, err = wd.queries.DB.ExecContext(ctx, updateQuery, graphJSON, workflowID)
	if err != nil {
		return fmt.Errorf("workflow graph save failed: update_error=true, error=%w", err)
	}

	return nil
}

/* LoadWorkflowGraph loads a workflow graph */
func (wd *WorkflowDesigner) LoadWorkflowGraph(ctx context.Context, workflowID uuid.UUID) (*WorkflowGraph, error) {
	/* Get workflow */
	workflow, err := wd.queries.GetWorkflowByID(ctx, workflowID)
	if err != nil {
		return nil, fmt.Errorf("workflow graph load failed: workflow_not_found=true, error=%w", err)
	}

	/* Parse graph from DAG definition */
	if workflow.DAGDefinition != nil {
		graphJSON, err := workflow.DAGDefinition.Value()
		if err == nil {
			var graph WorkflowGraph
			if err := json.Unmarshal(graphJSON.([]byte), &graph); err == nil {
				return &graph, nil
			}
		}
	}

	/* If no graph, convert steps to graph */
	steps, err := wd.queries.ListWorkflowSteps(ctx, workflowID)
	if err != nil {
		return nil, fmt.Errorf("workflow graph load failed: steps_retrieval_error=true, error=%w", err)
	}

	graph, err := wd.stepsToGraph(steps)
	if err != nil {
		return nil, fmt.Errorf("workflow graph load failed: conversion_error=true, error=%w", err)
	}

	return graph, nil
}

/* ValidateWorkflowGraph validates a workflow graph */
func (wd *WorkflowDesigner) ValidateWorkflowGraph(graph *WorkflowGraph) ([]string, error) {
	errors := make([]string, 0)

	/* Check for cycles */
	if wd.hasCycle(graph) {
		errors = append(errors, "workflow graph contains cycles")
	}

	/* Check for orphaned nodes */
	nodeIDs := make(map[string]bool)
	for _, node := range graph.Nodes {
		nodeIDs[node.ID] = true
	}

	for _, edge := range graph.Edges {
		if !nodeIDs[edge.Source] {
			errors = append(errors, fmt.Sprintf("edge references non-existent source node: %s", edge.Source))
		}
		if !nodeIDs[edge.Target] {
			errors = append(errors, fmt.Sprintf("edge references non-existent target node: %s", edge.Target))
		}
	}

	return errors, nil
}

/* Helper methods */

func (wd *WorkflowDesigner) graphToSteps(graph *WorkflowGraph) ([]*db.WorkflowStep, error) {
	steps := make([]*db.WorkflowStep, 0)

	/* Create step for each node */
	for _, node := range graph.Nodes {
		/* Convert config to inputs */
		inputs := node.Config
		if inputs == nil {
			inputs = make(map[string]interface{})
		}
		
		step := &db.WorkflowStep{
			StepName: node.Label,
			StepType: node.Type,
			Inputs:   inputs,
		}

		/* Add dependencies from edges */
		dependencies := make([]string, 0)
		for _, edge := range graph.Edges {
			if edge.Target == node.ID {
				/* Source is already a string ID */
				dependencies = append(dependencies, edge.Source)
			}
		}
		step.Dependencies = pq.StringArray(dependencies)

		steps = append(steps, step)
	}

	return steps, nil
}

func (wd *WorkflowDesigner) stepsToGraph(steps []db.WorkflowStep) (*WorkflowGraph, error) {
	graph := &WorkflowGraph{
		Nodes: make([]WorkflowNode, 0),
		Edges: make([]WorkflowEdge, 0),
	}

	/* Create nodes */
	for i, step := range steps {
		/* Convert inputs to config */
		config := step.Inputs
		if config == nil {
			config = make(map[string]interface{})
		}
		
		node := WorkflowNode{
			ID:       step.ID.String(),
			Type:     step.StepType,
			Label:    step.StepName,
			Position: Position{X: float64(i * 200), Y: float64(i * 100)},
			Data:     make(map[string]interface{}),
			Config:   config,
		}
		graph.Nodes = append(graph.Nodes, node)

		/* Create edges from dependencies */
		if step.Dependencies != nil {
			for _, depID := range step.Dependencies {
				edge := WorkflowEdge{
					ID:     fmt.Sprintf("%s-%s", depID, step.ID.String()),
					Source: depID,
					Target: step.ID.String(),
				}
				graph.Edges = append(graph.Edges, edge)
			}
		}
	}

	return graph, nil
}

func (wd *WorkflowDesigner) hasCycle(graph *WorkflowGraph) bool {
	/* Simple cycle detection using DFS */
	visited := make(map[string]bool)
	recStack := make(map[string]bool)

	var dfs func(nodeID string) bool
	dfs = func(nodeID string) bool {
		visited[nodeID] = true
		recStack[nodeID] = true

		/* Check neighbors */
		for _, edge := range graph.Edges {
			if edge.Source == nodeID {
				target := edge.Target
				if !visited[target] {
					if dfs(target) {
						return true
					}
				} else if recStack[target] {
					return true
				}
			}
		}

		recStack[nodeID] = false
		return false
	}

	/* Check all nodes */
	for _, node := range graph.Nodes {
		if !visited[node.ID] {
			if dfs(node.ID) {
				return true
			}
		}
	}

	return false
}

