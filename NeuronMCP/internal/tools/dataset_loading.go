package tools

import (
	"context"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

// DatasetLoadingTool loads HuggingFace datasets
type DatasetLoadingTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewDatasetLoadingTool creates a new dataset loading tool
func NewDatasetLoadingTool(db *database.Database, logger *logging.Logger) *DatasetLoadingTool {
	return &DatasetLoadingTool{
		BaseTool: NewBaseTool(
			"load_dataset",
			"Load HuggingFace dataset into database",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"dataset_name": map[string]interface{}{
						"type":        "string",
						"description": "HuggingFace dataset name",
					},
					"split": map[string]interface{}{
						"type":        "string",
						"default":     "train",
						"description": "Dataset split (train, test, validation)",
					},
					"config": map[string]interface{}{
						"type":        "string",
						"description": "Dataset configuration name (optional)",
					},
					"streaming": map[string]interface{}{
						"type":        "boolean",
						"default":     false,
						"description": "Enable streaming mode",
					},
					"cache_dir": map[string]interface{}{
						"type":        "string",
						"description": "Cache directory path (optional)",
					},
				},
				"required": []interface{}{"dataset_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes the dataset loading
func (t *DatasetLoadingTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for load_dataset tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	datasetName, _ := params["dataset_name"].(string)
	split := "train"
	if s, ok := params["split"].(string); ok && s != "" {
		split = s
	}
	config, _ := params["config"].(string)
	streaming := false
	if s, ok := params["streaming"].(bool); ok {
		streaming = s
	}
	cacheDir, _ := params["cache_dir"].(string)

	if datasetName == "" {
		return Error("dataset_name is required and cannot be empty", "VALIDATION_ERROR", nil), nil
	}

	// Build query with optional parameters
	var query string
	var queryParams []interface{}

	if config != "" && cacheDir != "" {
		query = "SELECT * FROM neurondb.load_dataset($1::text, $2::text, $3::text, $4::boolean, $5::text)"
		queryParams = []interface{}{datasetName, split, config, streaming, cacheDir}
	} else if config != "" {
		query = "SELECT * FROM neurondb.load_dataset($1::text, $2::text, $3::text, $4::boolean, NULL)"
		queryParams = []interface{}{datasetName, split, config, streaming}
	} else if cacheDir != "" {
		query = "SELECT * FROM neurondb.load_dataset($1::text, $2::text, NULL, $3::boolean, $4::text)"
		queryParams = []interface{}{datasetName, split, streaming, cacheDir}
	} else {
		query = "SELECT * FROM neurondb.load_dataset($1::text, $2::text, NULL, $3::boolean, NULL)"
		queryParams = []interface{}{datasetName, split, streaming}
	}

	results, err := t.executor.ExecuteQuery(ctx, query, queryParams)
	if err != nil {
		t.logger.Error("Dataset loading failed", err, params)
		return Error(fmt.Sprintf("Dataset loading failed: dataset_name='%s', error=%v", datasetName, err), "EXECUTION_ERROR", map[string]interface{}{
			"dataset_name": datasetName,
			"error":        err.Error(),
		}), nil
	}

	return Success(map[string]interface{}{
		"rows":  results,
		"count":  len(results),
		"dataset": datasetName,
		"split":   split,
	}, map[string]interface{}{
		"dataset": datasetName,
		"split":   split,
		"count":   len(results),
	}), nil
}

