package tools

import (
	"context"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

// VectorGraphTool performs graph operations on vgraph type
type VectorGraphTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewVectorGraphTool creates a new vector graph tool
func NewVectorGraphTool(db *database.Database, logger *logging.Logger) *VectorGraphTool {
	return &VectorGraphTool{
		BaseTool: NewBaseTool(
			"vector_graph",
			"Perform graph operations: BFS, DFS, PageRank, community detection on vgraph type",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"operation": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"bfs", "dfs", "pagerank", "community_detection"},
						"description": "Graph operation to perform",
					},
					"graph": map[string]interface{}{
						"type":        "string",
						"description": "vgraph value as string",
					},
					"start_node": map[string]interface{}{
						"type":        "number",
						"description": "Starting node index (for BFS, DFS)",
					},
					"max_depth": map[string]interface{}{
						"type":        "number",
						"description": "Maximum depth for BFS (-1 for unlimited)",
					},
					"damping_factor": map[string]interface{}{
						"type":        "number",
						"default":     0.85,
						"description": "Damping factor for PageRank (0.0-1.0)",
					},
					"max_iterations": map[string]interface{}{
						"type":        "number",
						"default":     100,
						"description": "Maximum iterations for PageRank or community detection",
					},
					"tolerance": map[string]interface{}{
						"type":        "number",
						"default":     1e-6,
						"description": "Convergence tolerance for PageRank",
					},
				},
				"required": []interface{}{"operation", "graph"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes the graph operation
func (t *VectorGraphTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for vector_graph tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	operation, _ := params["operation"].(string)
	graph, _ := params["graph"].(string)

	if graph == "" {
		return Error("graph is required and cannot be empty", "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "graph",
			"params":    params,
		}), nil
	}

	var query string
	var queryParams []interface{}

	switch operation {
	case "bfs":
		startNode, ok := params["start_node"].(float64)
		if !ok {
			return Error("start_node is required for BFS", "VALIDATION_ERROR", nil), nil
		}
		maxDepth := -1
		if md, ok := params["max_depth"].(float64); ok {
			maxDepth = int(md)
		}
		query = "SELECT * FROM vgraph_bfs($1::vgraph, $2::int, $3::int)"
		queryParams = []interface{}{graph, int(startNode), maxDepth}
	case "dfs":
		startNode, ok := params["start_node"].(float64)
		if !ok {
			return Error("start_node is required for DFS", "VALIDATION_ERROR", nil), nil
		}
		query = "SELECT * FROM vgraph_dfs($1::vgraph, $2::int)"
		queryParams = []interface{}{graph, int(startNode)}
	case "pagerank":
		dampingFactor := 0.85
		if df, ok := params["damping_factor"].(float64); ok {
			dampingFactor = df
		}
		maxIterations := 100
		if mi, ok := params["max_iterations"].(float64); ok {
			maxIterations = int(mi)
		}
		tolerance := 1e-6
		if tol, ok := params["tolerance"].(float64); ok {
			tolerance = tol
		}
		query = "SELECT * FROM vgraph_pagerank($1::vgraph, $2::float8, $3::int, $4::float8)"
		queryParams = []interface{}{graph, dampingFactor, maxIterations, tolerance}
	case "community_detection":
		maxIterations := 10
		if mi, ok := params["max_iterations"].(float64); ok {
			maxIterations = int(mi)
		}
		query = "SELECT * FROM vgraph_community_detection($1::vgraph, $2::int)"
		queryParams = []interface{}{graph, maxIterations}
	default:
		return Error(fmt.Sprintf("Unknown operation: %s", operation), "VALIDATION_ERROR", nil), nil
	}

	results, err := t.executor.ExecuteQuery(ctx, query, queryParams)
	if err != nil {
		t.logger.Error("Vector graph operation failed", err, params)
		return Error(fmt.Sprintf("Vector graph operation failed: operation='%s', error=%v", operation, err), "EXECUTION_ERROR", map[string]interface{}{
			"operation": operation,
			"error":     err.Error(),
		}), nil
	}

	return Success(map[string]interface{}{
		"results": results,
		"count":   len(results),
	}, map[string]interface{}{
		"operation": operation,
		"count":     len(results),
	}), nil
}

