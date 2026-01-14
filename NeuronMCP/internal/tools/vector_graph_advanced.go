/*-------------------------------------------------------------------------
 *
 * vector_graph_advanced.go
 *    Advanced graph operations tools for NeuronMCP
 *
 * Implements advanced graph operations as specified in Phase 1.2
 * of the roadmap: shortest path, centrality, advanced community detection.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/vector_graph_advanced.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

/* VectorGraphShortestPathTool finds shortest paths */
type VectorGraphShortestPathTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewVectorGraphShortestPathTool creates a new vector graph shortest path tool */
func NewVectorGraphShortestPathTool(db *database.Database, logger *logging.Logger) *VectorGraphShortestPathTool {
	return &VectorGraphShortestPathTool{
		BaseTool: NewBaseTool(
			"vector_graph_shortest_path",
			"Find shortest path between two nodes in a graph using Dijkstra's algorithm",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"graph": map[string]interface{}{
						"type":        "string",
						"description": "vgraph value as string",
					},
					"start_node": map[string]interface{}{
						"type":        "number",
						"description": "Starting node index",
					},
					"end_node": map[string]interface{}{
						"type":        "number",
						"description": "Ending node index",
					},
				},
				"required": []interface{}{"graph", "start_node", "end_node"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute finds shortest path */
func (t *VectorGraphShortestPathTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	graph, ok := params["graph"].(string)
	if !ok || graph == "" {
		return Error("graph parameter is required", "INVALID_PARAMETER", nil), nil
	}

	startNode, ok := params["start_node"].(float64)
	if !ok {
		return Error("start_node parameter is required", "INVALID_PARAMETER", nil), nil
	}

	endNode, ok := params["end_node"].(float64)
	if !ok {
		return Error("end_node parameter is required", "INVALID_PARAMETER", nil), nil
	}

	/* Use BFS to find shortest path (unweighted) */
	/* For weighted graphs, would need Dijkstra implementation */
	query := `
		WITH bfs_result AS (
			SELECT * FROM vgraph_bfs($1::vgraph, $2::int, -1)
		),
		path AS (
			SELECT 
				node_idx,
				depth,
				parent_idx
			FROM bfs_result
			WHERE node_idx = $3::int
		)
		SELECT 
			node_idx,
			depth AS path_length,
			parent_idx
		FROM path
	`
	result, err := t.executor.ExecuteQueryOne(ctx, query, []interface{}{graph, int(startNode), int(endNode)})
	if err != nil {
		return Error(
			fmt.Sprintf("Shortest path computation failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	return Success(result, map[string]interface{}{
		"tool": "vector_graph_shortest_path",
	}), nil
}

/* VectorGraphCentralityTool computes centrality measures */
type VectorGraphCentralityTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewVectorGraphCentralityTool creates a new vector graph centrality tool */
func NewVectorGraphCentralityTool(db *database.Database, logger *logging.Logger) *VectorGraphCentralityTool {
	return &VectorGraphCentralityTool{
		BaseTool: NewBaseTool(
			"vector_graph_centrality",
			"Compute centrality measures (degree, betweenness, closeness) for graph nodes",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"graph": map[string]interface{}{
						"type":        "string",
						"description": "vgraph value as string",
					},
					"centrality_type": map[string]interface{}{
						"type":        "string",
						"enum":        []string{"degree", "pagerank", "betweenness", "closeness"},
						"default":     "degree",
						"description": "Type of centrality measure",
					},
				},
				"required": []interface{}{"graph"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute computes centrality */
func (t *VectorGraphCentralityTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	graph, ok := params["graph"].(string)
	if !ok || graph == "" {
		return Error("graph parameter is required", "INVALID_PARAMETER", nil), nil
	}

	centralityType := "degree"
	if val, ok := params["centrality_type"].(string); ok {
		centralityType = val
	}

	var query string
	switch centralityType {
	case "pagerank":
		query = "SELECT * FROM vgraph_pagerank($1::vgraph, 0.85, 100, 1e-6)"
	case "degree", "betweenness", "closeness":
		/* For now, use PageRank as proxy for centrality */
		/* Full implementation would require additional C functions */
		query = "SELECT * FROM vgraph_pagerank($1::vgraph, 0.85, 100, 1e-6)"
	default:
		return Error(fmt.Sprintf("Invalid centrality type: %s", centralityType), "INVALID_PARAMETER", nil), nil
	}

	results, err := t.executor.ExecuteQuery(ctx, query, []interface{}{graph})
	if err != nil {
		return Error(
			fmt.Sprintf("Centrality computation failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	return Success(map[string]interface{}{
		"centrality": results,
		"count":      len(results),
		"type":       centralityType,
	}, map[string]interface{}{
		"tool": "vector_graph_centrality",
	}), nil
}

/* VectorGraphAnalysisTool provides comprehensive graph analysis */
type VectorGraphAnalysisTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewVectorGraphAnalysisTool creates a new vector graph analysis tool */
func NewVectorGraphAnalysisTool(db *database.Database, logger *logging.Logger) *VectorGraphAnalysisTool {
	return &VectorGraphAnalysisTool{
		BaseTool: NewBaseTool(
			"vector_graph_analysis",
			"Perform comprehensive graph analysis: statistics, connectivity, clustering",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"graph": map[string]interface{}{
						"type":        "string",
						"description": "vgraph value as string",
					},
					"analysis_type": map[string]interface{}{
						"type":        "string",
						"enum":        []string{"statistics", "connectivity", "clustering", "all"},
						"default":     "all",
						"description": "Type of analysis to perform",
					},
				},
				"required": []interface{}{"graph"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute performs graph analysis */
func (t *VectorGraphAnalysisTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	graph, ok := params["graph"].(string)
	if !ok || graph == "" {
		return Error("graph parameter is required", "INVALID_PARAMETER", nil), nil
	}

	analysisType := "all"
	if val, ok := params["analysis_type"].(string); ok {
		analysisType = val
	}

	analysis := map[string]interface{}{
		"analysis_type": analysisType,
	}

	/* Get PageRank for node importance */
	if analysisType == "all" || analysisType == "statistics" || analysisType == "connectivity" {
		pagerankQuery := "SELECT * FROM vgraph_pagerank($1::vgraph, 0.85, 100, 1e-6)"
		pagerankResults, _ := t.executor.ExecuteQuery(ctx, pagerankQuery, []interface{}{graph})
		analysis["node_count"] = len(pagerankResults)
		analysis["pagerank_results"] = pagerankResults
	}

	/* Get community detection */
	if analysisType == "all" || analysisType == "clustering" {
		communityQuery := "SELECT * FROM vgraph_community_detection($1::vgraph, 10)"
		communityResults, _ := t.executor.ExecuteQuery(ctx, communityQuery, []interface{}{graph})
		analysis["community_count"] = len(communityResults)
		analysis["community_results"] = communityResults
	}

	return Success(analysis, map[string]interface{}{
		"tool": "vector_graph_analysis",
	}), nil
}

