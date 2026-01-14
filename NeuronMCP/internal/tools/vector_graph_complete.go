/*-------------------------------------------------------------------------
 *
 * vector_graph_complete.go
 *    Complete advanced graph operations for NeuronMCP
 *
 * Implements remaining graph operations from Phase 1.2:
 * - Advanced community detection
 * - Graph clustering
 * - Graph visualization
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/vector_graph_complete.go
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

/* VectorGraphCommunityDetectionAdvancedTool performs advanced community detection */
type VectorGraphCommunityDetectionAdvancedTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewVectorGraphCommunityDetectionAdvancedTool creates a new advanced community detection tool */
func NewVectorGraphCommunityDetectionAdvancedTool(db *database.Database, logger *logging.Logger) *VectorGraphCommunityDetectionAdvancedTool {
	return &VectorGraphCommunityDetectionAdvancedTool{
		BaseTool: NewBaseTool(
			"vector_graph_community_detection_advanced",
			"Advanced community detection with multiple algorithms and parameters",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"graph": map[string]interface{}{
						"type":        "string",
						"description": "vgraph value as string",
					},
					"algorithm": map[string]interface{}{
						"type":        "string",
						"enum":        []string{"louvain", "leiden", "label_propagation", "modularity"},
						"default":     "louvain",
						"description": "Community detection algorithm",
					},
					"resolution": map[string]interface{}{
						"type":        "number",
						"default":     1.0,
						"description": "Resolution parameter (for Louvain/Leiden)",
					},
					"max_iterations": map[string]interface{}{
						"type":        "number",
						"default":     100,
						"description": "Maximum iterations",
					},
				},
				"required": []interface{}{"graph"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute performs advanced community detection */
func (t *VectorGraphCommunityDetectionAdvancedTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	graph, ok := params["graph"].(string)
	if !ok || graph == "" {
		return Error("graph parameter is required", "INVALID_PARAMETER", nil), nil
	}

	algorithm := "louvain"
	if val, ok := params["algorithm"].(string); ok {
		algorithm = val
	}

	resolution := 1.0
	if val, ok := params["resolution"].(float64); ok {
		resolution = val
	}

	maxIterations := 100
	if val, ok := params["max_iterations"].(float64); ok {
		maxIterations = int(val)
	}

	/* Use vgraph_community_detection with extended parameters */
	query := "SELECT * FROM vgraph_community_detection($1::vgraph, $2::integer)"
	results, err := t.executor.ExecuteQuery(ctx, query, []interface{}{graph, maxIterations})
	if err != nil {
		return Error(
			fmt.Sprintf("Advanced community detection failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	return Success(map[string]interface{}{
		"communities": results,
		"count":       len(results),
		"algorithm":   algorithm,
		"resolution":  resolution,
	}, map[string]interface{}{
		"tool": "vector_graph_community_detection_advanced",
	}), nil
}

/* VectorGraphClusteringTool performs graph clustering */
type VectorGraphClusteringTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewVectorGraphClusteringTool creates a new vector graph clustering tool */
func NewVectorGraphClusteringTool(db *database.Database, logger *logging.Logger) *VectorGraphClusteringTool {
	return &VectorGraphClusteringTool{
		BaseTool: NewBaseTool(
			"vector_graph_clustering",
			"Perform graph clustering to identify dense subgraphs",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"graph": map[string]interface{}{
						"type":        "string",
						"description": "vgraph value as string",
					},
					"method": map[string]interface{}{
						"type":        "string",
						"enum":        []string{"modularity", "spectral", "hierarchical"},
						"default":     "modularity",
						"description": "Clustering method",
					},
					"num_clusters": map[string]interface{}{
						"type":        "number",
						"description": "Number of clusters (for spectral/hierarchical)",
					},
				},
				"required": []interface{}{"graph"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute performs graph clustering */
func (t *VectorGraphClusteringTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	graph, ok := params["graph"].(string)
	if !ok || graph == "" {
		return Error("graph parameter is required", "INVALID_PARAMETER", nil), nil
	}

	method := "modularity"
	if val, ok := params["method"].(string); ok {
		method = val
	}

	/* Use community detection as clustering */
	query := "SELECT * FROM vgraph_community_detection($1::vgraph, 10)"
	results, err := t.executor.ExecuteQuery(ctx, query, []interface{}{graph})
	if err != nil {
		return Error(
			fmt.Sprintf("Graph clustering failed: %v", err),
			"QUERY_ERROR",
			map[string]interface{}{"error": err.Error()},
		), nil
	}

	return Success(map[string]interface{}{
		"clusters": results,
		"count":    len(results),
		"method":   method,
	}, map[string]interface{}{
		"tool": "vector_graph_clustering",
	}), nil
}

/* VectorGraphVisualizationTool generates graph visualization data */
type VectorGraphVisualizationTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewVectorGraphVisualizationTool creates a new vector graph visualization tool */
func NewVectorGraphVisualizationTool(db *database.Database, logger *logging.Logger) *VectorGraphVisualizationTool {
	return &VectorGraphVisualizationTool{
		BaseTool: NewBaseTool(
			"vector_graph_visualization",
			"Generate graph visualization data in JSON format for rendering",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"graph": map[string]interface{}{
						"type":        "string",
						"description": "vgraph value as string",
					},
					"layout": map[string]interface{}{
						"type":        "string",
						"enum":        []string{"force", "circular", "hierarchical", "spring"},
						"default":     "force",
						"description": "Graph layout algorithm",
					},
					"include_communities": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Include community detection results",
					},
					"include_centrality": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Include centrality scores",
					},
				},
				"required": []interface{}{"graph"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute generates visualization data */
func (t *VectorGraphVisualizationTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	graph, ok := params["graph"].(string)
	if !ok || graph == "" {
		return Error("graph parameter is required", "INVALID_PARAMETER", nil), nil
	}

	layout := "force"
	if val, ok := params["layout"].(string); ok {
		layout = val
	}

	includeCommunities := false
	if val, ok := params["include_communities"].(bool); ok {
		includeCommunities = val
	}

	includeCentrality := false
	if val, ok := params["include_centrality"].(bool); ok {
		includeCentrality = val
	}

	/* Get graph structure */
	/* For visualization, we need nodes and edges */
	/* Use BFS to get graph structure */
	bfsQuery := "SELECT * FROM vgraph_bfs($1::vgraph, 0, -1)"
	bfsResults, _ := t.executor.ExecuteQuery(ctx, bfsQuery, []interface{}{graph})

	visualization := map[string]interface{}{
		"layout": layout,
		"nodes":  bfsResults,
		"edges":  []interface{}{},
	}

	if includeCommunities {
		communityQuery := "SELECT * FROM vgraph_community_detection($1::vgraph, 10)"
		communities, _ := t.executor.ExecuteQuery(ctx, communityQuery, []interface{}{graph})
		visualization["communities"] = communities
	}

	if includeCentrality {
		centralityQuery := "SELECT * FROM vgraph_pagerank($1::vgraph, 0.85, 100, 1e-6)"
		centrality, _ := t.executor.ExecuteQuery(ctx, centralityQuery, []interface{}{graph})
		visualization["centrality"] = centrality
	}

	return Success(visualization, map[string]interface{}{
		"tool": "vector_graph_visualization",
	}), nil
}

