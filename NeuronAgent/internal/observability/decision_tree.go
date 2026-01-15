/*-------------------------------------------------------------------------
 *
 * decision_tree.go
 *    Decision tree visualization
 *
 * Provides visualization of agent decision-making process,
 * tool call chains, and memory retrieval paths.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/observability/decision_tree.go
 *
 *-------------------------------------------------------------------------
 */

package observability

import (
	"context"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
)

/* DecisionTreeVisualizer provides decision tree visualization */
type DecisionTreeVisualizer struct {
	queries *db.Queries
}

/* DecisionNode represents a node in the decision tree */
type DecisionNode struct {
	ID          string
	Type        string // "llm_call", "tool_call", "memory_retrieval", "decision"
	Description string
	Input       map[string]interface{}
	Output      map[string]interface{}
	Children    []*DecisionNode
	Timestamp   string
}

/* NewDecisionTreeVisualizer creates a new decision tree visualizer */
func NewDecisionTreeVisualizer(queries *db.Queries) *DecisionTreeVisualizer {
	return &DecisionTreeVisualizer{
		queries: queries,
	}
}

/* BuildDecisionTree builds decision tree for an execution */
func (dtv *DecisionTreeVisualizer) BuildDecisionTree(ctx context.Context, executionID uuid.UUID) (*DecisionNode, error) {
	/* Query execution trace */
	query := `SELECT id, step_type, description, input_data, output_data, timestamp, parent_id
		FROM neurondb_agent.execution_trace
		WHERE execution_id = $1
		ORDER BY timestamp ASC`

	type TraceRow struct {
		ID          uuid.UUID              `db:"id"`
		StepType    string                  `db:"step_type"`
		Description string                  `db:"description"`
		InputData   map[string]interface{}  `db:"input_data"`
		OutputData  map[string]interface{}  `db:"output_data"`
		Timestamp   string                  `db:"timestamp"`
		ParentID    *uuid.UUID              `db:"parent_id"`
	}

	var rows []TraceRow
	err := dtv.queries.DB.SelectContext(ctx, &rows, query, executionID)
	if err != nil {
		return nil, err
	}

	/* Build tree structure */
	nodes := make(map[uuid.UUID]*DecisionNode)
	var root *DecisionNode

	for _, row := range rows {
		node := &DecisionNode{
			ID:          row.ID.String(),
			Type:        row.StepType,
			Description: row.Description,
			Input:       row.InputData,
			Output:      row.OutputData,
			Timestamp:   row.Timestamp,
			Children:    make([]*DecisionNode, 0),
		}

		nodes[row.ID] = node

		if row.ParentID == nil {
			root = node
		} else {
			if parent, exists := nodes[*row.ParentID]; exists {
				parent.Children = append(parent.Children, node)
			}
		}
	}

	return root, nil
}

/* GetToolCallChain gets tool call chain for an execution */
func (dtv *DecisionTreeVisualizer) GetToolCallChain(ctx context.Context, executionID uuid.UUID) ([]*DecisionNode, error) {
	query := `SELECT id, step_type, description, input_data, output_data, timestamp
		FROM neurondb_agent.execution_trace
		WHERE execution_id = $1 AND step_type = 'tool_call'
		ORDER BY timestamp ASC`

	type TraceRow struct {
		ID          uuid.UUID              `db:"id"`
		StepType    string                 `db:"step_type"`
		Description string                 `db:"description"`
		InputData   map[string]interface{} `db:"input_data"`
		OutputData  map[string]interface{} `db:"output_data"`
		Timestamp   string                 `db:"timestamp"`
	}

	var rows []TraceRow
	err := dtv.queries.DB.SelectContext(ctx, &rows, query, executionID)
	if err != nil {
		return nil, err
	}

	chain := make([]*DecisionNode, len(rows))
	for i, row := range rows {
		chain[i] = &DecisionNode{
			ID:          row.ID.String(),
			Type:        row.StepType,
			Description: row.Description,
			Input:       row.InputData,
			Output:      row.OutputData,
			Timestamp:   row.Timestamp,
		}
	}

	return chain, nil
}

/* GetMemoryRetrievalPath gets memory retrieval path for an execution */
func (dtv *DecisionTreeVisualizer) GetMemoryRetrievalPath(ctx context.Context, executionID uuid.UUID) ([]*DecisionNode, error) {
	query := `SELECT id, step_type, description, input_data, output_data, timestamp
		FROM neurondb_agent.execution_trace
		WHERE execution_id = $1 AND step_type = 'memory_retrieval'
		ORDER BY timestamp ASC`

	type TraceRow struct {
		ID          uuid.UUID              `db:"id"`
		StepType    string                 `db:"step_type"`
		Description string                 `db:"description"`
		InputData   map[string]interface{} `db:"input_data"`
		OutputData  map[string]interface{} `db:"output_data"`
		Timestamp   string                 `db:"timestamp"`
	}

	var rows []TraceRow
	err := dtv.queries.DB.SelectContext(ctx, &rows, query, executionID)
	if err != nil {
		return nil, err
	}

	path := make([]*DecisionNode, len(rows))
	for i, row := range rows {
		path[i] = &DecisionNode{
			ID:          row.ID.String(),
			Type:        row.StepType,
			Description: row.Description,
			Input:       row.InputData,
			Output:      row.OutputData,
			Timestamp:   row.Timestamp,
		}
	}

	return path, nil
}

