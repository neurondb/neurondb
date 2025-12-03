package tools

import (
	"context"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

// TuneHNSWIndexTool automatically tunes HNSW index parameters
type TuneHNSWIndexTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

// NewTuneHNSWIndexTool creates a new TuneHNSWIndexTool
func NewTuneHNSWIndexTool(db *database.Database, logger *logging.Logger) *TuneHNSWIndexTool {
	return &TuneHNSWIndexTool{
		BaseTool: NewBaseTool(
			"tune_hnsw_index",
			"Automatically optimize HNSW index parameters (m, ef_construction) based on dataset characteristics",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table": map[string]interface{}{
						"type":        "string",
						"description": "The name of the table containing the vector column",
					},
					"vector_column": map[string]interface{}{
						"type":        "string",
						"description": "The name of the vector column to tune",
					},
				},
				"required": []interface{}{"table", "vector_column"},
			},
		),
		db:     db,
		logger: logger,
	}
}

// Execute tunes the HNSW index
func (t *TuneHNSWIndexTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for tune_hnsw_index tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	table, _ := params["table"].(string)
	vectorColumn, _ := params["vector_column"].(string)

	if table == "" {
		return Error("table parameter is required and cannot be empty for tune_hnsw_index tool", "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "table",
			"params":    params,
		}), nil
	}

	if vectorColumn == "" {
		return Error(fmt.Sprintf("vector_column parameter is required and cannot be empty for tune_hnsw_index tool: table='%s'", table), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "vector_column",
			"table":     table,
			"params":    params,
		}), nil
	}

	// Use NeuronDB's index_tune_hnsw function: index_tune_hnsw(table, vector_col)
	query := `SELECT index_tune_hnsw($1, $2) AS tuning_result`
	executor := NewQueryExecutor(t.db)
	result, err := executor.ExecuteQueryOne(ctx, query, []interface{}{table, vectorColumn})
	if err != nil {
		t.logger.Error("HNSW index tuning failed", err, params)
		return Error(fmt.Sprintf("HNSW index tuning execution failed: table='%s', vector_column='%s', error=%v", table, vectorColumn, err), "TUNING_ERROR", map[string]interface{}{
			"table":          table,
			"vector_column": vectorColumn,
			"error":          err.Error(),
		}), nil
	}

	return Success(result, map[string]interface{}{
		"table":          table,
		"vector_column": vectorColumn,
		"index_type":    "hnsw",
	}), nil
}

// TuneIVFIndexTool automatically tunes IVF index parameters
type TuneIVFIndexTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

// NewTuneIVFIndexTool creates a new TuneIVFIndexTool
func NewTuneIVFIndexTool(db *database.Database, logger *logging.Logger) *TuneIVFIndexTool {
	return &TuneIVFIndexTool{
		BaseTool: NewBaseTool(
			"tune_ivf_index",
			"Automatically optimize IVF index parameters (num_lists, probes) based on dataset characteristics",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table": map[string]interface{}{
						"type":        "string",
						"description": "The name of the table containing the vector column",
					},
					"vector_column": map[string]interface{}{
						"type":        "string",
						"description": "The name of the vector column to tune",
					},
				},
				"required": []interface{}{"table", "vector_column"},
			},
		),
		db:     db,
		logger: logger,
	}
}

// Execute tunes the IVF index
func (t *TuneIVFIndexTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for tune_ivf_index tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	table, _ := params["table"].(string)
	vectorColumn, _ := params["vector_column"].(string)

	if table == "" {
		return Error("table parameter is required and cannot be empty for tune_ivf_index tool", "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "table",
			"params":    params,
		}), nil
	}

	if vectorColumn == "" {
		return Error(fmt.Sprintf("vector_column parameter is required and cannot be empty for tune_ivf_index tool: table='%s'", table), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "vector_column",
			"table":     table,
			"params":    params,
		}), nil
	}

	// Use NeuronDB's index_tune_ivf function: index_tune_ivf(table, vector_col)
	query := `SELECT index_tune_ivf($1, $2) AS tuning_result`
	executor := NewQueryExecutor(t.db)
	result, err := executor.ExecuteQueryOne(ctx, query, []interface{}{table, vectorColumn})
	if err != nil {
		t.logger.Error("IVF index tuning failed", err, params)
		return Error(fmt.Sprintf("IVF index tuning execution failed: table='%s', vector_column='%s', error=%v", table, vectorColumn, err), "TUNING_ERROR", map[string]interface{}{
			"table":          table,
			"vector_column": vectorColumn,
			"error":          err.Error(),
		}), nil
	}

	return Success(result, map[string]interface{}{
		"table":          table,
		"vector_column": vectorColumn,
		"index_type":    "ivf",
	}), nil
}


