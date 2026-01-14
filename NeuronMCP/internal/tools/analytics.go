/*-------------------------------------------------------------------------
 *
 * analytics.go
 *    Tool implementation for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/analytics.go
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

/* ClusterDataTool performs clustering on data */
type ClusterDataTool struct {
	*BaseTool
	executor     *QueryExecutor
	logger       *logging.Logger
	configHelper *database.ConfigHelper
}

/* NewClusterDataTool creates a new cluster data tool */
func NewClusterDataTool(db *database.Database, logger *logging.Logger) *ClusterDataTool {
	return &ClusterDataTool{
		BaseTool: NewBaseTool(
			"postgresql_cluster_data",
			"Perform clustering on vector data using K-means, GMM, DBSCAN, or hierarchical clustering",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"algorithm": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"kmeans", "gmm", "dbscan", "hierarchical", "minibatch_kmeans"},
						"default":     "kmeans",
						"description": "Clustering algorithm to use",
					},
					"table": map[string]interface{}{
						"type":        "string",
						"description": "Table name containing vectors",
					},
					"vector_column": map[string]interface{}{
						"type":        "string",
						"description": "Vector column name",
					},
					"k": map[string]interface{}{
						"type":        "number",
						"minimum":     2,
						"description": "Number of clusters (for kmeans, gmm, hierarchical)",
					},
					"eps": map[string]interface{}{
						"type":        "number",
						"default":     0.5,
						"description": "Maximum distance for DBSCAN",
					},
					"min_samples": map[string]interface{}{
						"type":        "number",
						"default":     5,
						"description": "Minimum samples for DBSCAN",
					},
					"max_iter": map[string]interface{}{
						"type":        "number",
						"default":     100,
						"description": "Maximum iterations",
					},
				},
				"required": []interface{}{"table", "vector_column"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the clustering */
func (t *ClusterDataTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for neurondb_cluster_data tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	algorithm := "kmeans"
	if a, ok := params["algorithm"].(string); ok {
		algorithm = a
	}

	table, _ := params["table"].(string)
	vectorColumn, _ := params["vector_column"].(string)

	if table == "" {
		return Error(fmt.Sprintf("table parameter is required and cannot be empty for neurondb_cluster_data tool with algorithm '%s'", algorithm), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "table",
			"algorithm": algorithm,
			"params":    params,
		}), nil
	}

	if vectorColumn == "" {
		return Error(fmt.Sprintf("vector_column parameter is required and cannot be empty for neurondb_cluster_data tool: algorithm='%s', table='%s'", algorithm, table), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "vector_column",
			"algorithm": algorithm,
			"table":     table,
			"params":    params,
		}), nil
	}

	var query string
	var queryParams []interface{}

	switch algorithm {
	case "kmeans":
		k := 3
		if kVal, ok := params["k"].(float64); ok {
			k = int(kVal)
		}
		maxIter := 100
		if m, ok := params["max_iter"].(float64); ok {
			maxIter = int(m)
		}
		query = `SELECT cluster_kmeans($1, $2, $3, $4) AS clusters`
		queryParams = []interface{}{table, vectorColumn, k, maxIter}

	case "gmm":
		k := 3
		if kVal, ok := params["k"].(float64); ok {
			k = int(kVal)
		}
		maxIter := 100
		if m, ok := params["max_iter"].(float64); ok {
			maxIter = int(m)
		}
		query = `SELECT cluster_gmm($1, $2, $3, $4) AS clusters`
		queryParams = []interface{}{table, vectorColumn, k, maxIter}

	case "dbscan":
		eps := 0.5
		if e, ok := params["eps"].(float64); ok {
			eps = e
		}
		minSamples := 5
		if m, ok := params["min_samples"].(float64); ok {
			minSamples = int(m)
		}
		query = `SELECT cluster_dbscan($1, $2, $3, $4) AS clusters`
		queryParams = []interface{}{table, vectorColumn, eps, minSamples}

	case "hierarchical":
		k := 3
		if kVal, ok := params["k"].(float64); ok {
			k = int(kVal)
		}
		query = `SELECT cluster_hierarchical($1, $2, $3, 'ward') AS clusters`
		queryParams = []interface{}{table, vectorColumn, k}

	case "minibatch_kmeans":
		k := 3
		if kVal, ok := params["k"].(float64); ok {
			k = int(kVal)
		}
		maxIter := 100
		if m, ok := params["max_iter"].(float64); ok {
			maxIter = int(m)
		}
		batchSize := 100
		if b, ok := params["batch_size"].(float64); ok {
			batchSize = int(b)
		}
		query = `SELECT cluster_minibatch_kmeans($1, $2, $3, $4, $5) AS clusters`
		queryParams = []interface{}{table, vectorColumn, k, maxIter, batchSize}

	default:
		validAlgorithms := []string{"kmeans", "gmm", "dbscan", "hierarchical", "minibatch_kmeans"}
		return Error(fmt.Sprintf("Unsupported clustering algorithm '%s' for neurondb_cluster_data tool: table='%s', vector_column='%s'. Valid algorithms are: %v", algorithm, table, vectorColumn, validAlgorithms), "VALIDATION_ERROR", map[string]interface{}{
			"algorithm":       algorithm,
			"table":           table,
			"vector_column":   vectorColumn,
			"valid_algorithms": validAlgorithms,
			"params":          params,
		}), nil
	}

	result, err := t.executor.ExecuteQueryOne(ctx, query, queryParams)
	if err != nil {
		t.logger.Error("Clustering failed", err, params)
		return Error(fmt.Sprintf("Clustering execution failed: algorithm='%s', table='%s', vector_column='%s', query_params_count=%d, error=%v", algorithm, table, vectorColumn, len(queryParams), err), "CLUSTERING_ERROR", map[string]interface{}{
			"algorithm":        algorithm,
			"table":            table,
			"vector_column":    vectorColumn,
			"query_params_count": len(queryParams),
			"error":            err.Error(),
		}), nil
	}

	return Success(result, map[string]interface{}{
		"algorithm": algorithm,
	}), nil
}

/* DetectOutliersTool detects outliers in data */
type DetectOutliersTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewDetectOutliersTool creates a new detect outliers tool */
func NewDetectOutliersTool(db *database.Database, logger *logging.Logger) *DetectOutliersTool {
	return &DetectOutliersTool{
		BaseTool: NewBaseTool(
			"postgresql_detect_outliers",
			"Detect outliers in vector data using Z-score method",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table": map[string]interface{}{
						"type":        "string",
						"description": "Table name containing vectors",
					},
					"vector_column": map[string]interface{}{
						"type":        "string",
						"description": "Vector column name",
					},
					"threshold": map[string]interface{}{
						"type":        "number",
						"default":     3.0,
						"description": "Z-score threshold",
					},
				},
				"required": []interface{}{"table", "vector_column"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the outlier detection */
func (t *DetectOutliersTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for neurondb_detect_outliers tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	table, _ := params["table"].(string)
	vectorColumn, _ := params["vector_column"].(string)
	threshold := 3.0
	if t, ok := params["threshold"].(float64); ok {
		threshold = t
	}

	if table == "" {
		return Error("table parameter is required and cannot be empty for neurondb_detect_outliers tool", "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "table",
			"params":    params,
		}), nil
	}

	if vectorColumn == "" {
		return Error(fmt.Sprintf("vector_column parameter is required and cannot be empty for neurondb_detect_outliers tool: table='%s'", table), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "vector_column",
			"table":     table,
			"params":    params,
		}), nil
	}

	if threshold <= 0 {
		return Error(fmt.Sprintf("threshold must be greater than 0 for neurondb_detect_outliers tool: table='%s', vector_column='%s', received threshold=%g", table, vectorColumn, threshold), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":     "threshold",
			"table":         table,
			"vector_column": vectorColumn,
			"threshold":     threshold,
			"params":        params,
		}), nil
	}

  /* Use NeuronDB's outlier detection function: detect_outliers_zscore(table, column, threshold, method) */
	query := `SELECT detect_outliers_zscore($1, $2, $3, 'zscore') AS outliers`
	result, err := t.executor.ExecuteQueryOne(ctx, query, []interface{}{table, vectorColumn, threshold})
	if err != nil {
		t.logger.Error("Outlier detection failed", err, params)
		return Error(fmt.Sprintf("Outlier detection execution failed: table='%s', vector_column='%s', threshold=%g, method='zscore', error=%v", table, vectorColumn, threshold, err), "OUTLIER_ERROR", map[string]interface{}{
			"table":          table,
			"vector_column": vectorColumn,
			"threshold":     threshold,
			"method":         "zscore",
			"error":          err.Error(),
		}), nil
	}

	return Success(result, map[string]interface{}{
		"method":    "zscore",
		"threshold": threshold,
	}), nil
}

/* ReduceDimensionalityTool reduces dimensionality using PCA */
type ReduceDimensionalityTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewReduceDimensionalityTool creates a new reduce dimensionality tool */
func NewReduceDimensionalityTool(db *database.Database, logger *logging.Logger) *ReduceDimensionalityTool {
	return &ReduceDimensionalityTool{
		BaseTool: NewBaseTool(
			"postgresql_reduce_dimensionality",
			"Reduce dimensionality using PCA",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table": map[string]interface{}{
						"type":        "string",
						"description": "Table name containing vectors",
					},
					"vector_column": map[string]interface{}{
						"type":        "string",
						"description": "Vector column name",
					},
					"n_components": map[string]interface{}{
						"type":        "number",
						"minimum":     1,
						"description": "Number of components to reduce to",
					},
				},
				"required": []interface{}{"table", "vector_column", "n_components"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the dimensionality reduction */
func (t *ReduceDimensionalityTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for neurondb_reduce_dimensionality tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	table, _ := params["table"].(string)
	vectorColumn, _ := params["vector_column"].(string)
	nComponents := 2
	if n, ok := params["n_components"].(float64); ok {
		nComponents = int(n)
	}

	if table == "" {
		return Error("table parameter is required and cannot be empty for neurondb_reduce_dimensionality tool", "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "table",
			"params":    params,
		}), nil
	}

	if vectorColumn == "" {
		return Error(fmt.Sprintf("vector_column parameter is required and cannot be empty for neurondb_reduce_dimensionality tool: table='%s'", table), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "vector_column",
			"table":     table,
			"params":    params,
		}), nil
	}

	if nComponents < 1 {
		return Error(fmt.Sprintf("n_components must be at least 1 for neurondb_reduce_dimensionality tool: table='%s', vector_column='%s', received n_components=%d", table, vectorColumn, nComponents), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":     "n_components",
			"table":         table,
			"vector_column": vectorColumn,
			"n_components": nComponents,
			"params":        params,
		}), nil
	}

  /* PCA in NeuronDB is typically done through training a model */
  /* Use neurondb.train with 'pca' algorithm or use dimensionality reduction functions */
  /* For now, we'll use a direct approach - PCA might need to be implemented differently */
  /* Check if there's a compute_pca function or use train with pca algorithm */
	query := `SELECT neurondb.train('default', 'pca', $1, NULL, ARRAY[$2], jsonb_build_object('n_components', $3::integer)) AS pca_model_id`
	result, err := t.executor.ExecuteQueryOne(ctx, query, []interface{}{table, vectorColumn, nComponents})
	if err != nil {
		t.logger.Error("Dimensionality reduction failed", err, params)
		return Error(fmt.Sprintf("Dimensionality reduction execution failed: table='%s', vector_column='%s', n_components=%d, method='pca', error=%v", table, vectorColumn, nComponents, err), "PCA_ERROR", map[string]interface{}{
			"table":          table,
			"vector_column": vectorColumn,
			"n_components":  nComponents,
			"method":         "pca",
			"error":          err.Error(),
		}), nil
	}

	return Success(result, map[string]interface{}{
		"n_components": nComponents,
		"method":       "pca",
	}), nil
}

