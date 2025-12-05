package tools

import (
	"context"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

// TimeSeriesTool performs time series analysis
type TimeSeriesTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewTimeSeriesTool creates a new time series tool
func NewTimeSeriesTool(db *database.Database, logger *logging.Logger) *TimeSeriesTool {
	return &TimeSeriesTool{
		BaseTool: NewBaseTool(
			"timeseries_analysis",
			"Perform time series analysis: ARIMA, forecasting",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"operation": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"arima", "forecast", "seasonal_decompose"},
						"description": "Time series operation",
					},
					"table": map[string]interface{}{
						"type":        "string",
						"description": "Table name with time series data",
					},
					"value_column": map[string]interface{}{
						"type":        "string",
						"description": "Value column name",
					},
					"time_column": map[string]interface{}{
						"type":        "string",
						"description": "Time/timestamp column name",
					},
					"p": map[string]interface{}{
						"type":        "number",
						"description": "ARIMA p parameter",
					},
					"d": map[string]interface{}{
						"type":        "number",
						"description": "ARIMA d parameter",
					},
					"q": map[string]interface{}{
						"type":        "number",
						"description": "ARIMA q parameter",
					},
					"forecast_periods": map[string]interface{}{
						"type":        "number",
						"description": "Number of periods to forecast",
					},
				},
				"required": []interface{}{"operation", "table", "value_column", "time_column"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes time series analysis
func (t *TimeSeriesTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for timeseries_analysis tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	operation, _ := params["operation"].(string)
	table, _ := params["table"].(string)
	valueColumn, _ := params["value_column"].(string)
	timeColumn, _ := params["time_column"].(string)

	if table == "" || valueColumn == "" || timeColumn == "" {
		return Error("table, value_column, and time_column are required", "VALIDATION_ERROR", nil), nil
	}

	var query string
	var queryParams []interface{}

	switch operation {
	case "arima":
		p, _ := params["p"].(float64)
		d, _ := params["d"].(float64)
		q, _ := params["q"].(float64)
		query = "SELECT * FROM train_arima($1::text, $2::text, $3::text, $4::int, $5::int, $6::int)"
		queryParams = []interface{}{table, valueColumn, timeColumn, int(p), int(d), int(q)}
	case "forecast":
		forecastPeriods, _ := params["forecast_periods"].(float64)
		if forecastPeriods <= 0 {
			return Error("forecast_periods must be greater than 0", "VALIDATION_ERROR", nil), nil
		}
		query = "SELECT * FROM forecast_time_series($1::text, $2::text, $3::text, $4::int)"
		queryParams = []interface{}{table, valueColumn, timeColumn, int(forecastPeriods)}
	case "seasonal_decompose":
		query = "SELECT * FROM seasonal_decompose($1::text, $2::text, $3::text)"
		queryParams = []interface{}{table, valueColumn, timeColumn}
	default:
		return Error(fmt.Sprintf("Unknown operation: %s", operation), "VALIDATION_ERROR", nil), nil
	}

	results, err := t.executor.ExecuteQuery(ctx, query, queryParams)
	if err != nil {
		t.logger.Error("Time series analysis failed", err, params)
		return Error(fmt.Sprintf("Time series analysis failed: operation='%s', error=%v", operation, err), "EXECUTION_ERROR", map[string]interface{}{
			"operation": operation,
			"error":    err.Error(),
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

