package tools

import (
	"context"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

// VectorArithmeticTool performs vector arithmetic operations
type VectorArithmeticTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewVectorArithmeticTool creates a new vector arithmetic tool
func NewVectorArithmeticTool(db *database.Database, logger *logging.Logger) *VectorArithmeticTool {
	return &VectorArithmeticTool{
		BaseTool: NewBaseTool(
			"vector_arithmetic",
			"Perform vector arithmetic operations: add, subtract, multiply, normalize, concat, norm",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"operation": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"add", "subtract", "multiply", "normalize", "concat", "norm", "dims"},
						"description": "Arithmetic operation to perform",
					},
					"vector1": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "number"},
						"description": "First vector (required for add, subtract, concat, norm, dims)",
					},
					"vector2": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "number"},
						"description": "Second vector (required for add, subtract, concat)",
					},
					"scalar": map[string]interface{}{
						"type":        "number",
						"description": "Scalar value (required for multiply)",
					},
				},
				"required": []interface{}{"operation", "vector1"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes the vector arithmetic operation
func (t *VectorArithmeticTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for vector_arithmetic tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	operation, _ := params["operation"].(string)
	vector1, _ := params["vector1"].([]interface{})

	if len(vector1) == 0 {
		return Error("vector1 cannot be empty", "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "vector1",
			"params":    params,
		}), nil
	}

	vec1Str := formatVectorFromInterface(vector1)

	var query string
	var queryParams []interface{}

	switch operation {
	case "add", "subtract", "concat":
		vector2, ok := params["vector2"].([]interface{})
		if !ok || len(vector2) == 0 {
			return Error(fmt.Sprintf("vector2 is required for %s operation", operation), "VALIDATION_ERROR", map[string]interface{}{
				"operation": operation,
				"params":    params,
			}), nil
		}
		vec2Str := formatVectorFromInterface(vector2)
		op := "+"
		if operation == "subtract" {
			op = "-"
		} else if operation == "concat" {
			query = fmt.Sprintf("SELECT vector_concat($1::vector, $2::vector) AS result")
			queryParams = []interface{}{vec1Str, vec2Str}
		} else {
			query = fmt.Sprintf("SELECT ($1::vector %s $2::vector) AS result", op)
			queryParams = []interface{}{vec1Str, vec2Str}
		}
	case "multiply":
		scalar, ok := params["scalar"].(float64)
		if !ok {
			return Error("scalar is required for multiply operation", "VALIDATION_ERROR", map[string]interface{}{
				"operation": operation,
				"params":    params,
			}), nil
		}
		query = fmt.Sprintf("SELECT ($1::vector * $2::float8) AS result")
		queryParams = []interface{}{vec1Str, scalar}
	case "normalize":
		query = fmt.Sprintf("SELECT vector_normalize($1::vector) AS result")
		queryParams = []interface{}{vec1Str}
	case "norm":
		query = fmt.Sprintf("SELECT vector_norm($1::vector) AS result")
		queryParams = []interface{}{vec1Str}
	case "dims":
		query = fmt.Sprintf("SELECT vector_dims($1::vector) AS result")
		queryParams = []interface{}{vec1Str}
	default:
		return Error(fmt.Sprintf("Unknown operation: %s", operation), "VALIDATION_ERROR", map[string]interface{}{
			"operation": operation,
			"params":    params,
		}), nil
	}

	result, err := t.executor.ExecuteQueryOne(ctx, query, queryParams)
	if err != nil {
		t.logger.Error("Vector arithmetic operation failed", err, map[string]interface{}{
			"operation": operation,
			"params":    params,
		})
		return Error(fmt.Sprintf("Vector arithmetic operation failed: operation='%s', error=%v", operation, err), "EXECUTION_ERROR", map[string]interface{}{
			"operation": operation,
			"error":     err.Error(),
		}), nil
	}

	return Success(result, map[string]interface{}{
		"operation": operation,
	}), nil
}

// VectorDistanceTool computes distance between two vectors using various metrics
type VectorDistanceTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewVectorDistanceTool creates a new vector distance tool
func NewVectorDistanceTool(db *database.Database, logger *logging.Logger) *VectorDistanceTool {
	return &VectorDistanceTool{
		BaseTool: NewBaseTool(
			"vector_distance",
			"Compute distance between two vectors using L1, L2, Hamming, Chebyshev, Minkowski, Jaccard, Dice, or Mahalanobis metrics",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"vector1": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "number"},
						"description": "First vector",
					},
					"vector2": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "number"},
						"description": "Second vector",
					},
					"metric": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"l1", "l2", "hamming", "chebyshev", "minkowski", "jaccard", "dice", "mahalanobis", "squared_l2"},
						"default":     "l2",
						"description": "Distance metric to use",
					},
					"p_value": map[string]interface{}{
						"type":        "number",
						"default":     3.0,
						"description": "P value for Minkowski distance",
					},
					"covariance": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "number"},
						"description": "Inverse covariance matrix for Mahalanobis distance",
					},
				},
				"required": []interface{}{"vector1", "vector2"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes the vector distance computation
func (t *VectorDistanceTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for vector_distance tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	vector1, _ := params["vector1"].([]interface{})
	vector2, _ := params["vector2"].([]interface{})
	metric := "l2"
	if m, ok := params["metric"].(string); ok {
		metric = m
	}

	if len(vector1) == 0 || len(vector2) == 0 {
		return Error("vectors cannot be empty", "VALIDATION_ERROR", map[string]interface{}{
			"params": params,
		}), nil
	}

	vec1Str := formatVectorFromInterface(vector1)
	vec2Str := formatVectorFromInterface(vector2)

	var query string
	var queryParams []interface{}

	switch metric {
	case "l1":
		query = "SELECT vector_l1_distance($1::vector, $2::vector) AS distance"
		queryParams = []interface{}{vec1Str, vec2Str}
	case "l2":
		query = "SELECT vector_l2_distance($1::vector, $2::vector) AS distance"
		queryParams = []interface{}{vec1Str, vec2Str}
	case "squared_l2":
		query = "SELECT vector_squared_l2_distance($1::vector, $2::vector) AS distance"
		queryParams = []interface{}{vec1Str, vec2Str}
	case "hamming":
		query = "SELECT vector_hamming_distance($1::vector, $2::vector) AS distance"
		queryParams = []interface{}{vec1Str, vec2Str}
	case "chebyshev":
		query = "SELECT vector_chebyshev_distance($1::vector, $2::vector) AS distance"
		queryParams = []interface{}{vec1Str, vec2Str}
	case "minkowski":
		pValue := 3.0
		if p, ok := params["p_value"].(float64); ok {
			pValue = p
		}
		query = "SELECT vector_minkowski_distance($1::vector, $2::vector, $3::float8) AS distance"
		queryParams = []interface{}{vec1Str, vec2Str, pValue}
	case "jaccard":
		query = "SELECT vector_jaccard_distance($1::vector, $2::vector) AS distance"
		queryParams = []interface{}{vec1Str, vec2Str}
	case "dice":
		query = "SELECT vector_dice_distance($1::vector, $2::vector) AS distance"
		queryParams = []interface{}{vec1Str, vec2Str}
	case "mahalanobis":
		covariance, ok := params["covariance"].([]interface{})
		if !ok || len(covariance) == 0 {
			return Error("covariance is required for Mahalanobis distance", "VALIDATION_ERROR", map[string]interface{}{
				"metric": metric,
				"params": params,
			}), nil
		}
		covStr := formatVectorFromInterface(covariance)
		query = "SELECT vector_mahalanobis_distance($1::vector, $2::vector, $3::vector) AS distance"
		queryParams = []interface{}{vec1Str, vec2Str, covStr}
	default:
		// Use unified distance function
		pValue := 3.0
		if p, ok := params["p_value"].(float64); ok {
			pValue = p
		}
		query = "SELECT neurondb.distance($1::vector, $2::vector, $3::text, $4::float8) AS distance"
		queryParams = []interface{}{vec1Str, vec2Str, metric, pValue}
	}

	result, err := t.executor.ExecuteQueryOne(ctx, query, queryParams)
	if err != nil {
		t.logger.Error("Vector distance computation failed", err, map[string]interface{}{
			"metric": metric,
			"params": params,
		})
		return Error(fmt.Sprintf("Vector distance computation failed: metric='%s', error=%v", metric, err), "EXECUTION_ERROR", map[string]interface{}{
			"metric": metric,
			"error":  err.Error(),
		}), nil
	}

	return Success(result, map[string]interface{}{
		"metric": metric,
	}), nil
}

// VectorSimilarityUnifiedTool computes similarity using unified function
type VectorSimilarityUnifiedTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewVectorSimilarityUnifiedTool creates a new unified similarity tool
func NewVectorSimilarityUnifiedTool(db *database.Database, logger *logging.Logger) *VectorSimilarityUnifiedTool {
	return &VectorSimilarityUnifiedTool{
		BaseTool: NewBaseTool(
			"vector_similarity_unified",
			"Compute similarity between two vectors using unified similarity function (cosine, inner_product, l2)",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"vector1": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "number"},
						"description": "First vector",
					},
					"vector2": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "number"},
						"description": "Second vector",
					},
					"metric": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"cosine", "inner_product", "l2"},
						"default":     "cosine",
						"description": "Similarity metric to use",
					},
				},
				"required": []interface{}{"vector1", "vector2"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes the similarity computation
func (t *VectorSimilarityUnifiedTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for vector_similarity_unified tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	vector1, _ := params["vector1"].([]interface{})
	vector2, _ := params["vector2"].([]interface{})
	metric := "cosine"
	if m, ok := params["metric"].(string); ok {
		metric = m
	}

	if len(vector1) == 0 || len(vector2) == 0 {
		return Error("vectors cannot be empty", "VALIDATION_ERROR", map[string]interface{}{
			"params": params,
		}), nil
	}

	vec1Str := formatVectorFromInterface(vector1)
	vec2Str := formatVectorFromInterface(vector2)

	query := "SELECT neurondb.similarity($1::vector, $2::vector, $3::text) AS similarity"
	queryParams := []interface{}{vec1Str, vec2Str, metric}

	result, err := t.executor.ExecuteQueryOne(ctx, query, queryParams)
	if err != nil {
		t.logger.Error("Vector similarity computation failed", err, map[string]interface{}{
			"metric": metric,
			"params": params,
		})
		return Error(fmt.Sprintf("Vector similarity computation failed: metric='%s', error=%v", metric, err), "EXECUTION_ERROR", map[string]interface{}{
			"metric": metric,
			"error":  err.Error(),
		}), nil
	}

	return Success(result, map[string]interface{}{
		"metric": metric,
	}), nil
}

