package tools

import (
	"context"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

// ONNXTool manages ONNX models
type ONNXTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewONNXTool creates a new ONNX tool
func NewONNXTool(db *database.Database, logger *logging.Logger) *ONNXTool {
	return &ONNXTool{
		BaseTool: NewBaseTool(
			"onnx_model",
			"Manage ONNX models: import, export, info, predict",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"operation": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"info", "import", "export", "predict"},
						"description": "ONNX operation",
					},
					"model_id": map[string]interface{}{
						"type":        "number",
						"description": "Model ID (for export, predict)",
					},
					"onnx_path": map[string]interface{}{
						"type":        "string",
						"description": "ONNX file path (for import)",
					},
					"features": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "number"},
						"description": "Feature vector (for predict)",
					},
				},
				"required": []interface{}{"operation"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes ONNX operation
func (t *ONNXTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for onnx_model tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	operation, _ := params["operation"].(string)

	var query string
	var queryParams []interface{}

	switch operation {
	case "info":
		query = "SELECT * FROM neurondb_onnx_info()"
		queryParams = []interface{}{}
	case "import":
		onnxPath, _ := params["onnx_path"].(string)
		if onnxPath == "" {
			return Error("onnx_path is required for import", "VALIDATION_ERROR", nil), nil
		}
		query = "SELECT import_onnx_model($1::text) AS model_id"
		queryParams = []interface{}{onnxPath}
	case "export":
		modelID, ok := params["model_id"].(float64)
		if !ok {
			return Error("model_id is required for export", "VALIDATION_ERROR", nil), nil
		}
		query = "SELECT export_model_to_onnx($1::int) AS onnx_data"
		queryParams = []interface{}{int(modelID)}
	case "predict":
		modelID, ok := params["model_id"].(float64)
		features, ok2 := params["features"].([]interface{})
		if !ok || !ok2 {
			return Error("model_id and features are required for predict", "VALIDATION_ERROR", nil), nil
		}
		vecStr := formatVectorFromInterface(features)
		query = "SELECT predict_onnx_model($1::int, $2::vector) AS prediction"
		queryParams = []interface{}{int(modelID), vecStr}
	default:
		return Error(fmt.Sprintf("Unknown operation: %s", operation), "VALIDATION_ERROR", nil), nil
	}

	result, err := t.executor.ExecuteQueryOne(ctx, query, queryParams)
	if err != nil {
		t.logger.Error("ONNX operation failed", err, params)
		return Error(fmt.Sprintf("ONNX operation failed: operation='%s', error=%v", operation, err), "EXECUTION_ERROR", map[string]interface{}{
			"operation": operation,
			"error":    err.Error(),
		}), nil
	}

	return Success(result, map[string]interface{}{
		"operation": operation,
	}), nil
}

