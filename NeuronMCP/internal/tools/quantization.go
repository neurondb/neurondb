package tools

import (
	"context"
	"encoding/base64"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

// VectorQuantizationTool performs vector quantization operations
type VectorQuantizationTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewVectorQuantizationTool creates a new vector quantization tool
func NewVectorQuantizationTool(db *database.Database, logger *logging.Logger) *VectorQuantizationTool {
	return &VectorQuantizationTool{
		BaseTool: NewBaseTool(
			"vector_quantize",
			"Quantize or dequantize vectors using int8, fp16, binary, uint8, ternary, or int4 formats",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"operation": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"to_int8", "from_int8", "to_fp16", "from_fp16", "to_binary", "from_binary", "to_uint8", "from_uint8", "to_ternary", "from_ternary", "to_int4", "from_int4", "to_halfvec", "from_halfvec", "to_sparsevec", "from_sparsevec", "to_bit", "from_bit"},
						"description": "Quantization operation",
					},
					"vector": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "number"},
						"description": "Input vector (for quantization operations)",
					},
					"data": map[string]interface{}{
						"type":        "string",
						"description": "Base64-encoded quantized data (for dequantization operations)",
					},
				},
				"required": []interface{}{"operation"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes the quantization operation
func (t *VectorQuantizationTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for vector_quantize tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	operation, _ := params["operation"].(string)

	var query string
	var queryParams []interface{}

	switch operation {
	case "to_int8":
		vector, ok := params["vector"].([]interface{})
		if !ok || len(vector) == 0 {
			return Error("vector is required for to_int8 operation", "VALIDATION_ERROR", map[string]interface{}{
				"operation": operation,
			}), nil
		}
		vecStr := formatVectorFromInterface(vector)
		query = "SELECT encode(vector_to_int8($1::vector), 'base64') AS quantized_data"
		queryParams = []interface{}{vecStr}
	case "from_int8":
		data, ok := params["data"].(string)
		if !ok || data == "" {
			return Error("data is required for from_int8 operation", "VALIDATION_ERROR", map[string]interface{}{
				"operation": operation,
			}), nil
		}
		decoded, err := base64.StdEncoding.DecodeString(data)
		if err != nil {
			return Error(fmt.Sprintf("Invalid base64 data: %v", err), "VALIDATION_ERROR", map[string]interface{}{
				"operation": operation,
			}), nil
		}
		query = "SELECT vector_to_array(int8_to_vector($1::bytea)) AS vector"
		queryParams = []interface{}{decoded}
	case "to_fp16":
		vector, ok := params["vector"].([]interface{})
		if !ok || len(vector) == 0 {
			return Error("vector is required for to_fp16 operation", "VALIDATION_ERROR", map[string]interface{}{
				"operation": operation,
			}), nil
		}
		vecStr := formatVectorFromInterface(vector)
		query = "SELECT encode(vector_to_fp16($1::vector), 'base64') AS quantized_data"
		queryParams = []interface{}{vecStr}
	case "from_fp16":
		data, ok := params["data"].(string)
		if !ok || data == "" {
			return Error("data is required for from_fp16 operation", "VALIDATION_ERROR", map[string]interface{}{
				"operation": operation,
			}), nil
		}
		decoded, err := base64.StdEncoding.DecodeString(data)
		if err != nil {
			return Error(fmt.Sprintf("Invalid base64 data: %v", err), "VALIDATION_ERROR", map[string]interface{}{
				"operation": operation,
			}), nil
		}
		query = "SELECT vector_to_array(fp16_to_vector($1::bytea)) AS vector"
		queryParams = []interface{}{decoded}
	case "to_binary":
		vector, ok := params["vector"].([]interface{})
		if !ok || len(vector) == 0 {
			return Error("vector is required for to_binary operation", "VALIDATION_ERROR", map[string]interface{}{
				"operation": operation,
			}), nil
		}
		vecStr := formatVectorFromInterface(vector)
		query = "SELECT encode(vector_to_binary($1::vector), 'base64') AS quantized_data"
		queryParams = []interface{}{vecStr}
	case "from_binary":
		data, ok := params["data"].(string)
		if !ok || data == "" {
			return Error("data is required for from_binary operation", "VALIDATION_ERROR", map[string]interface{}{
				"operation": operation,
			}), nil
		}
		decoded, err := base64.StdEncoding.DecodeString(data)
		if err != nil {
			return Error(fmt.Sprintf("Invalid base64 data: %v", err), "VALIDATION_ERROR", map[string]interface{}{
				"operation": operation,
			}), nil
		}
		query = "SELECT vector_to_array(binary_to_vector($1::bytea)) AS vector"
		queryParams = []interface{}{decoded}
	case "to_uint8":
		vector, ok := params["vector"].([]interface{})
		if !ok || len(vector) == 0 {
			return Error("vector is required for to_uint8 operation", "VALIDATION_ERROR", map[string]interface{}{
				"operation": operation,
			}), nil
		}
		vecStr := formatVectorFromInterface(vector)
		query = "SELECT encode(vector_to_uint8($1::vector), 'base64') AS quantized_data"
		queryParams = []interface{}{vecStr}
	case "from_uint8":
		data, ok := params["data"].(string)
		if !ok || data == "" {
			return Error("data is required for from_uint8 operation", "VALIDATION_ERROR", map[string]interface{}{
				"operation": operation,
			}), nil
		}
		decoded, err := base64.StdEncoding.DecodeString(data)
		if err != nil {
			return Error(fmt.Sprintf("Invalid base64 data: %v", err), "VALIDATION_ERROR", map[string]interface{}{
				"operation": operation,
			}), nil
		}
		query = "SELECT vector_to_array(uint8_to_vector($1::bytea)) AS vector"
		queryParams = []interface{}{decoded}
	case "to_ternary":
		vector, ok := params["vector"].([]interface{})
		if !ok || len(vector) == 0 {
			return Error("vector is required for to_ternary operation", "VALIDATION_ERROR", map[string]interface{}{
				"operation": operation,
			}), nil
		}
		vecStr := formatVectorFromInterface(vector)
		query = "SELECT encode(vector_to_ternary($1::vector), 'base64') AS quantized_data"
		queryParams = []interface{}{vecStr}
	case "from_ternary":
		data, ok := params["data"].(string)
		if !ok || data == "" {
			return Error("data is required for from_ternary operation", "VALIDATION_ERROR", map[string]interface{}{
				"operation": operation,
			}), nil
		}
		decoded, err := base64.StdEncoding.DecodeString(data)
		if err != nil {
			return Error(fmt.Sprintf("Invalid base64 data: %v", err), "VALIDATION_ERROR", map[string]interface{}{
				"operation": operation,
			}), nil
		}
		query = "SELECT vector_to_array(ternary_to_vector($1::bytea)) AS vector"
		queryParams = []interface{}{decoded}
	case "to_int4":
		vector, ok := params["vector"].([]interface{})
		if !ok || len(vector) == 0 {
			return Error("vector is required for to_int4 operation", "VALIDATION_ERROR", map[string]interface{}{
				"operation": operation,
			}), nil
		}
		vecStr := formatVectorFromInterface(vector)
		query = "SELECT encode(vector_to_int4($1::vector), 'base64') AS quantized_data"
		queryParams = []interface{}{vecStr}
	case "from_int4":
		data, ok := params["data"].(string)
		if !ok || data == "" {
			return Error("data is required for from_int4 operation", "VALIDATION_ERROR", map[string]interface{}{
				"operation": operation,
			}), nil
		}
		decoded, err := base64.StdEncoding.DecodeString(data)
		if err != nil {
			return Error(fmt.Sprintf("Invalid base64 data: %v", err), "VALIDATION_ERROR", map[string]interface{}{
				"operation": operation,
			}), nil
		}
		query = "SELECT vector_to_array(int4_to_vector($1::bytea)) AS vector"
		queryParams = []interface{}{decoded}
	case "to_halfvec":
		vector, ok := params["vector"].([]interface{})
		if !ok || len(vector) == 0 {
			return Error("vector is required for to_halfvec operation", "VALIDATION_ERROR", map[string]interface{}{
				"operation": operation,
			}), nil
		}
		vecStr := formatVectorFromInterface(vector)
		query = "SELECT vector_to_array(vector_to_halfvec($1::vector)::vector) AS vector"
		queryParams = []interface{}{vecStr}
	case "from_halfvec":
		vector, ok := params["vector"].([]interface{})
		if !ok || len(vector) == 0 {
			return Error("vector is required for from_halfvec operation", "VALIDATION_ERROR", map[string]interface{}{
				"operation": operation,
			}), nil
		}
		vecStr := formatVectorFromInterface(vector)
		query = "SELECT vector_to_array(halfvec_to_vector($1::halfvec)) AS vector"
		queryParams = []interface{}{vecStr}
	case "to_sparsevec":
		vector, ok := params["vector"].([]interface{})
		if !ok || len(vector) == 0 {
			return Error("vector is required for to_sparsevec operation", "VALIDATION_ERROR", map[string]interface{}{
				"operation": operation,
			}), nil
		}
		vecStr := formatVectorFromInterface(vector)
		query = "SELECT vector_to_array(vector_to_sparsevec($1::vector)::vector) AS vector"
		queryParams = []interface{}{vecStr}
	case "from_sparsevec":
		vector, ok := params["vector"].([]interface{})
		if !ok || len(vector) == 0 {
			return Error("vector is required for from_sparsevec operation", "VALIDATION_ERROR", map[string]interface{}{
				"operation": operation,
			}), nil
		}
		vecStr := formatVectorFromInterface(vector)
		query = "SELECT vector_to_array(sparsevec_to_vector($1::sparsevec)) AS vector"
		queryParams = []interface{}{vecStr}
	case "to_bit", "from_bit":
		return Error("bit operations not yet implemented", "NOT_IMPLEMENTED", map[string]interface{}{
			"operation": operation,
		}), nil
	default:
		return Error(fmt.Sprintf("Unknown operation: %s", operation), "VALIDATION_ERROR", map[string]interface{}{
			"operation": operation,
		}), nil
	}

	result, err := t.executor.ExecuteQueryOne(ctx, query, queryParams)
	if err != nil {
		t.logger.Error("Vector quantization operation failed", err, map[string]interface{}{
			"operation": operation,
			"params":    params,
		})
		return Error(fmt.Sprintf("Vector quantization operation failed: operation='%s', error=%v", operation, err), "EXECUTION_ERROR", map[string]interface{}{
			"operation": operation,
			"error":     err.Error(),
		}), nil
	}

	return Success(result, map[string]interface{}{
		"operation": operation,
	}), nil
}

// QuantizationAnalysisTool analyzes quantization options
type QuantizationAnalysisTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewQuantizationAnalysisTool creates a new quantization analysis tool
func NewQuantizationAnalysisTool(db *database.Database, logger *logging.Logger) *QuantizationAnalysisTool {
	return &QuantizationAnalysisTool{
		BaseTool: NewBaseTool(
			"quantization_analyze",
			"Analyze quantization options for a vector (int8, fp16, binary, uint8, ternary, int4) or compare distances",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"operation": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"analyze_int8", "analyze_fp16", "analyze_binary", "analyze_uint8", "analyze_ternary", "analyze_int4", "compare_distances"},
						"description": "Analysis operation",
					},
					"vector": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "number"},
						"description": "Input vector",
					},
					"vector1": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "number"},
						"description": "First vector (for compare_distances)",
					},
					"vector2": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "number"},
						"description": "Second vector (for compare_distances)",
					},
					"metric": map[string]interface{}{
						"type":        "string",
						"description": "Distance metric for compare_distances",
					},
				},
				"required": []interface{}{"operation"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes the quantization analysis
func (t *QuantizationAnalysisTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for quantization_analyze tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	operation, _ := params["operation"].(string)

	var query string
	var queryParams []interface{}

	switch operation {
	case "analyze_int8":
		vector, ok := params["vector"].([]interface{})
		if !ok || len(vector) == 0 {
			return Error("vector is required for analyze_int8", "VALIDATION_ERROR", nil), nil
		}
		vecStr := formatVectorFromInterface(vector)
		query = "SELECT quantize_analyze_int8($1::vector) AS analysis"
		queryParams = []interface{}{vecStr}
	case "analyze_fp16":
		vector, ok := params["vector"].([]interface{})
		if !ok || len(vector) == 0 {
			return Error("vector is required for analyze_fp16", "VALIDATION_ERROR", nil), nil
		}
		vecStr := formatVectorFromInterface(vector)
		query = "SELECT quantize_analyze_fp16($1::vector) AS analysis"
		queryParams = []interface{}{vecStr}
	case "analyze_binary":
		vector, ok := params["vector"].([]interface{})
		if !ok || len(vector) == 0 {
			return Error("vector is required for analyze_binary", "VALIDATION_ERROR", nil), nil
		}
		vecStr := formatVectorFromInterface(vector)
		query = "SELECT quantize_analyze_binary($1::vector) AS analysis"
		queryParams = []interface{}{vecStr}
	case "analyze_uint8":
		vector, ok := params["vector"].([]interface{})
		if !ok || len(vector) == 0 {
			return Error("vector is required for analyze_uint8", "VALIDATION_ERROR", nil), nil
		}
		vecStr := formatVectorFromInterface(vector)
		query = "SELECT quantize_analyze_uint8($1::vector) AS analysis"
		queryParams = []interface{}{vecStr}
	case "analyze_ternary":
		vector, ok := params["vector"].([]interface{})
		if !ok || len(vector) == 0 {
			return Error("vector is required for analyze_ternary", "VALIDATION_ERROR", nil), nil
		}
		vecStr := formatVectorFromInterface(vector)
		query = "SELECT quantize_analyze_ternary($1::vector) AS analysis"
		queryParams = []interface{}{vecStr}
	case "analyze_int4":
		vector, ok := params["vector"].([]interface{})
		if !ok || len(vector) == 0 {
			return Error("vector is required for analyze_int4", "VALIDATION_ERROR", nil), nil
		}
		vecStr := formatVectorFromInterface(vector)
		query = "SELECT quantize_analyze_int4($1::vector) AS analysis"
		queryParams = []interface{}{vecStr}
	case "compare_distances":
		vector1, ok1 := params["vector1"].([]interface{})
		vector2, ok2 := params["vector2"].([]interface{})
		metric, ok3 := params["metric"].(string)
		if !ok1 || !ok2 || !ok3 || len(vector1) == 0 || len(vector2) == 0 || metric == "" {
			return Error("vector1, vector2, and metric are required for compare_distances", "VALIDATION_ERROR", nil), nil
		}
		vec1Str := formatVectorFromInterface(vector1)
		vec2Str := formatVectorFromInterface(vector2)
		query = "SELECT quantize_compare_distances($1::vector, $2::vector, $3::text) AS comparison"
		queryParams = []interface{}{vec1Str, vec2Str, metric}
	default:
		return Error(fmt.Sprintf("Unknown operation: %s", operation), "VALIDATION_ERROR", nil), nil
	}

	result, err := t.executor.ExecuteQueryOne(ctx, query, queryParams)
	if err != nil {
		t.logger.Error("Quantization analysis failed", err, map[string]interface{}{
			"operation": operation,
		})
		return Error(fmt.Sprintf("Quantization analysis failed: operation='%s', error=%v", operation, err), "EXECUTION_ERROR", map[string]interface{}{
			"operation": operation,
			"error":    err.Error(),
		}), nil
	}

	return Success(result, map[string]interface{}{
		"operation": operation,
	}), nil
}

