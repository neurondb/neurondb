package tools

import (
	"context"
	"encoding/base64"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

// VecmapOperationsTool performs operations on vecmap (sparse vector) type
type VecmapOperationsTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewVecmapOperationsTool creates a new vecmap operations tool
func NewVecmapOperationsTool(db *database.Database, logger *logging.Logger) *VecmapOperationsTool {
	return &VecmapOperationsTool{
		BaseTool: NewBaseTool(
			"vecmap_operations",
			"Perform operations on vecmap (sparse vector) type: distances, arithmetic, norm",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"operation": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"l2_distance", "cosine_distance", "inner_product", "l1_distance", "add", "subtract", "multiply_scalar", "norm"},
						"description": "Vecmap operation",
					},
					"vecmap1": map[string]interface{}{
						"type":        "string",
						"description": "First vecmap (base64-encoded bytea)",
					},
					"vecmap2": map[string]interface{}{
						"type":        "string",
						"description": "Second vecmap (base64-encoded bytea, for distance/arithmetic)",
					},
					"scalar": map[string]interface{}{
						"type":        "number",
						"description": "Scalar value (for multiply_scalar)",
					},
				},
				"required": []interface{}{"operation", "vecmap1"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes the vecmap operation
func (t *VecmapOperationsTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for vecmap_operations tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	operation, _ := params["operation"].(string)
	vecmap1, _ := params["vecmap1"].(string)

	if vecmap1 == "" {
		return Error("vecmap1 is required and cannot be empty", "VALIDATION_ERROR", nil), nil
	}

	// Decode base64 vecmap data
	vecmap1Bytes, err := base64.StdEncoding.DecodeString(vecmap1)
	if err != nil {
		return Error(fmt.Sprintf("Invalid base64 vecmap1 data: %v", err), "VALIDATION_ERROR", nil), nil
	}

	var query string
	var queryParams []interface{}

	switch operation {
	case "l2_distance", "cosine_distance", "inner_product", "l1_distance", "add", "subtract":
		vecmap2, ok := params["vecmap2"].(string)
		if !ok || vecmap2 == "" {
			return Error("vecmap2 is required for this operation", "VALIDATION_ERROR", nil), nil
		}
		vecmap2Bytes, err := base64.StdEncoding.DecodeString(vecmap2)
		if err != nil {
			return Error(fmt.Sprintf("Invalid base64 vecmap2 data: %v", err), "VALIDATION_ERROR", nil), nil
		}

		switch operation {
		case "l2_distance":
			query = "SELECT vecmap_l2_distance($1::bytea, $2::bytea) AS result"
		case "cosine_distance":
			query = "SELECT vecmap_cosine_distance($1::bytea, $2::bytea) AS result"
		case "inner_product":
			query = "SELECT vecmap_inner_product($1::bytea, $2::bytea) AS result"
		case "l1_distance":
			query = "SELECT vecmap_l1_distance($1::bytea, $2::bytea) AS result"
		case "add":
			query = "SELECT encode(vecmap_add($1::bytea, $2::bytea), 'base64') AS result"
		case "subtract":
			query = "SELECT encode(vecmap_sub($1::bytea, $2::bytea), 'base64') AS result"
		}
		queryParams = []interface{}{vecmap1Bytes, vecmap2Bytes}
	case "multiply_scalar":
		scalar, ok := params["scalar"].(float64)
		if !ok {
			return Error("scalar is required for multiply_scalar", "VALIDATION_ERROR", nil), nil
		}
		query = "SELECT encode(vecmap_mul_scalar($1::bytea, $2::real), 'base64') AS result"
		queryParams = []interface{}{vecmap1Bytes, float32(scalar)}
	case "norm":
		query = "SELECT vecmap_norm($1::bytea) AS result"
		queryParams = []interface{}{vecmap1Bytes}
	default:
		return Error(fmt.Sprintf("Unknown operation: %s", operation), "VALIDATION_ERROR", nil), nil
	}

	result, err := t.executor.ExecuteQueryOne(ctx, query, queryParams)
	if err != nil {
		t.logger.Error("Vecmap operation failed", err, params)
		return Error(fmt.Sprintf("Vecmap operation failed: operation='%s', error=%v", operation, err), "EXECUTION_ERROR", map[string]interface{}{
			"operation": operation,
			"error":     err.Error(),
		}), nil
	}

	return Success(result, map[string]interface{}{
		"operation": operation,
	}), nil
}

