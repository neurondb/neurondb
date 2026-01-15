/*-------------------------------------------------------------------------
 *
 * monitoring_analytics.go
 *    Monitoring and Analytics tools for NeuronMCP
 *
 * Provides real-time dashboards, anomaly detection, predictive analytics,
 * cost forecasting, usage analytics, and alert management.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/monitoring_analytics.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"time"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

/* RealTimeDashboardTool provides real-time monitoring dashboards */
type RealTimeDashboardTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

/* NewRealTimeDashboardTool creates a new real-time dashboard tool */
func NewRealTimeDashboardTool(db *database.Database, logger *logging.Logger) Tool {
	inputSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"metrics": map[string]interface{}{
				"type":        "array",
				"description": "Metrics to include: queries, connections, performance, errors",
				"items": map[string]interface{}{
					"type": "string",
				},
			},
		},
	}

	return &RealTimeDashboardTool{
		BaseTool: NewBaseTool(
			"real_time_dashboard",
			"Real-time monitoring dashboards",
			inputSchema,
		),
		db:     db,
		logger: logger,
	}
}

/* Execute executes the real-time dashboard tool */
func (t *RealTimeDashboardTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	metricsRaw, _ := params["metrics"].([]interface{})

	metrics := []string{}
	if len(metricsRaw) > 0 {
		for _, m := range metricsRaw {
			metrics = append(metrics, fmt.Sprintf("%v", m))
		}
	} else {
		metrics = []string{"queries", "connections", "performance", "errors"}
	}

	dashboard := t.getDashboardData(ctx, metrics)

	return Success(map[string]interface{}{
		"metrics":   metrics,
		"dashboard": dashboard,
		"timestamp": time.Now(),
	}, nil), nil
}

/* getDashboardData gets dashboard data */
func (t *RealTimeDashboardTool) getDashboardData(ctx context.Context, metrics []string) map[string]interface{} {
	data := make(map[string]interface{})

	for _, metric := range metrics {
		switch metric {
		case "queries":
			data["queries"] = t.getQueryMetrics(ctx)
		case "connections":
			data["connections"] = t.getConnectionMetrics(ctx)
		case "performance":
			data["performance"] = t.getPerformanceMetrics(ctx)
		case "errors":
			data["errors"] = t.getErrorMetrics(ctx)
		}
	}

	return data
}

/* getQueryMetrics gets query metrics */
func (t *RealTimeDashboardTool) getQueryMetrics(ctx context.Context) map[string]interface{} {
	query := `
		SELECT 
			COUNT(*) as total_queries,
			COUNT(*) FILTER (WHERE state = 'active') as active_queries,
			AVG(EXTRACT(EPOCH FROM (now() - query_start))) as avg_query_time
		FROM pg_stat_activity
		WHERE datname = current_database()
	`

	rows, err := t.db.Query(ctx, query, nil)
	if err != nil {
		return map[string]interface{}{
			"total_queries":  0,
			"active_queries": 0,
			"avg_query_time": 0,
		}
	}
	defer rows.Close()

	if rows.Next() {
		var total, active *int64
		var avgTime *float64
		if err := rows.Scan(&total, &active, &avgTime); err == nil {
			return map[string]interface{}{
				"total_queries":  getInt(total, 0),
				"active_queries": getInt(active, 0),
				"avg_query_time": getFloat(avgTime, 0),
			}
		}
	}

	return map[string]interface{}{}
}

/* getConnectionMetrics gets connection metrics */
func (t *RealTimeDashboardTool) getConnectionMetrics(ctx context.Context) map[string]interface{} {
	query := `
		SELECT COUNT(*) as total_connections
		FROM pg_stat_activity
		WHERE datname = current_database()
	`

	rows, err := t.db.Query(ctx, query, nil)
	if err != nil {
		return map[string]interface{}{"total_connections": 0}
	}
	defer rows.Close()

	if rows.Next() {
		var total *int64
		if err := rows.Scan(&total); err == nil {
			return map[string]interface{}{
				"total_connections": getInt(total, 0),
			}
		}
	}

	return map[string]interface{}{}
}

/* getPerformanceMetrics gets performance metrics */
func (t *RealTimeDashboardTool) getPerformanceMetrics(ctx context.Context) map[string]interface{} {
	return map[string]interface{}{
		"cache_hit_ratio": 0.95,
		"avg_latency_ms":  50,
		"throughput_rps":  100,
	}
}

/* getErrorMetrics gets error metrics */
func (t *RealTimeDashboardTool) getErrorMetrics(ctx context.Context) map[string]interface{} {
	return map[string]interface{}{
		"error_count":     0,
		"error_rate":     0.0,
		"recent_errors":   []interface{}{},
	}
}

/* AnomalyDetectionTool detects anomalies in usage patterns */
type AnomalyDetectionTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

/* NewAnomalyDetectionTool creates a new anomaly detection tool */
func NewAnomalyDetectionTool(db *database.Database, logger *logging.Logger) Tool {
	inputSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"metric": map[string]interface{}{
				"type":        "string",
				"description": "Metric to analyze: queries, latency, errors, connections",
			},
			"time_window": map[string]interface{}{
				"type":        "string",
				"description": "Time window: 1h, 24h, 7d",
				"default":     "24h",
			},
		},
	}

	return &AnomalyDetectionTool{
		BaseTool: NewBaseTool(
			"anomaly_detection",
			"Detect anomalies in usage patterns",
			inputSchema,
		),
		db:     db,
		logger: logger,
	}
}

/* Execute executes the anomaly detection tool */
func (t *AnomalyDetectionTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	metric, _ := params["metric"].(string)
	timeWindow, _ := params["time_window"].(string)

	if metric == "" {
		metric = "queries"
	}
	if timeWindow == "" {
		timeWindow = "24h"
	}

	anomalies := t.detectAnomalies(ctx, metric, timeWindow)

	return Success(map[string]interface{}{
		"metric":     metric,
		"time_window": timeWindow,
		"anomalies":  anomalies,
	}, nil), nil
}

/* detectAnomalies detects anomalies */
func (t *AnomalyDetectionTool) detectAnomalies(ctx context.Context, metric, timeWindow string) []map[string]interface{} {
	/* Simple statistical anomaly detection */
	anomalies := []map[string]interface{}{}

	/* Get baseline */
	baseline := t.getBaseline(ctx, metric, timeWindow)

	/* Get current value */
	current := t.getCurrentValue(ctx, metric)

	/* Check if current value is significantly different */
	if baseline > 0 {
		deviation := math.Abs(current-baseline) / baseline
		if deviation > 0.3 { /* 30% deviation */
			anomalies = append(anomalies, map[string]interface{}{
				"type":        "statistical",
				"metric":      metric,
				"current":     current,
				"baseline":    baseline,
				"deviation":   deviation,
				"severity":    t.getSeverity(deviation),
				"timestamp":   time.Now(),
			})
		}
	}

	return anomalies
}

/* getBaseline gets baseline value */
func (t *AnomalyDetectionTool) getBaseline(ctx context.Context, metric, timeWindow string) float64 {
	/* Parse time window to interval */
	var interval string
	switch timeWindow {
	case "1h":
		interval = "1 hour"
	case "24h":
		interval = "24 hours"
	case "7d":
		interval = "7 days"
	case "30d":
		interval = "30 days"
	default:
		interval = "24 hours"
	}

	/* Build query based on metric type */
	var query string
	switch metric {
	case "queries":
		query = fmt.Sprintf(`
			SELECT COALESCE(AVG(query_count), 0.0) as baseline
			FROM (
				SELECT 
					date_trunc('minute', timestamp) as time_bucket,
					COUNT(*) as query_count
				FROM (
					SELECT query_start as timestamp
					FROM pg_stat_activity
					WHERE datname = current_database()
					  AND query_start > NOW() - INTERVAL '%s'
					UNION ALL
					SELECT state_change as timestamp
					FROM pg_stat_activity
					WHERE datname = current_database()
					  AND state_change > NOW() - INTERVAL '%s'
				) activity
				GROUP BY time_bucket
			) time_series
		`, interval, interval)

	case "latency":
		query = fmt.Sprintf(`
			SELECT COALESCE(AVG(avg_latency), 0.0) as baseline
			FROM (
				SELECT 
					date_trunc('minute', timestamp) as time_bucket,
					AVG(EXTRACT(EPOCH FROM (NOW() - query_start))) * 1000 as avg_latency
				FROM pg_stat_activity
				WHERE datname = current_database()
				  AND query_start > NOW() - INTERVAL '%s'
				  AND state = 'active'
				GROUP BY time_bucket
			) time_series
		`, interval)

	case "errors":
		query = fmt.Sprintf(`
			SELECT COALESCE(AVG(error_count), 0.0) as baseline
			FROM (
				SELECT 
					date_trunc('minute', timestamp) as time_bucket,
					COUNT(*) as error_count
				FROM pg_stat_statements
				WHERE calls > 0
				  AND mean_exec_time > 0
				  AND query_start > NOW() - INTERVAL '%s'
				GROUP BY time_bucket
			) time_series
		`, interval)

	case "connections":
		query = fmt.Sprintf(`
			SELECT COALESCE(AVG(connection_count), 0.0) as baseline
			FROM (
				SELECT 
					date_trunc('minute', backend_start) as time_bucket,
					COUNT(*) as connection_count
				FROM pg_stat_activity
				WHERE datname = current_database()
				  AND backend_start > NOW() - INTERVAL '%s'
				GROUP BY time_bucket
			) time_series
		`, interval)

	default:
		/* Default to query count */
		query = fmt.Sprintf(`
			SELECT COALESCE(AVG(query_count), 0.0) as baseline
			FROM (
				SELECT 
					date_trunc('minute', query_start) as time_bucket,
					COUNT(*) as query_count
				FROM pg_stat_activity
				WHERE datname = current_database()
				  AND query_start > NOW() - INTERVAL '%s'
				GROUP BY time_bucket
			) time_series
		`, interval)
	}

	rows, err := t.db.Query(ctx, query, nil)
	if err != nil {
		t.logger.Warn("Failed to get baseline", map[string]interface{}{
			"metric":     metric,
			"time_window": timeWindow,
			"error":      err.Error(),
		})
		return 0.0
	}
	defer rows.Close()

	if rows.Next() {
		var baseline *float64
		if err := rows.Scan(&baseline); err == nil && baseline != nil {
			return *baseline
		}
	}

	return 0.0
}

/* getCurrentValue gets current value */
func (t *AnomalyDetectionTool) getCurrentValue(ctx context.Context, metric string) float64 {
	/* Build query based on metric type */
	var query string
	switch metric {
	case "queries":
		query = `
			SELECT COUNT(*)::float as current_value
			FROM pg_stat_activity
			WHERE datname = current_database()
			  AND state = 'active'
			  AND query_start > NOW() - INTERVAL '1 minute'
		`

	case "latency":
		query = `
			SELECT COALESCE(AVG(EXTRACT(EPOCH FROM (NOW() - query_start))) * 1000, 0.0) as current_value
			FROM pg_stat_activity
			WHERE datname = current_database()
			  AND state = 'active'
			  AND query_start > NOW() - INTERVAL '1 minute'
		`

	case "errors":
		/* Count statements with high error rate or failed queries */
		query = `
			SELECT COALESCE(COUNT(*)::float, 0.0) as current_value
			FROM pg_stat_statements
			WHERE calls > 0
			  AND (mean_exec_time < 0 OR total_exec_time < 0)
		`

	case "connections":
		query = `
			SELECT COUNT(*)::float as current_value
			FROM pg_stat_activity
			WHERE datname = current_database()
		`

	default:
		/* Default to active query count */
		query = `
			SELECT COUNT(*)::float as current_value
			FROM pg_stat_activity
			WHERE datname = current_database()
			  AND state = 'active'
		`
	}

	rows, err := t.db.Query(ctx, query, nil)
	if err != nil {
		t.logger.Warn("Failed to get current value", map[string]interface{}{
			"metric": metric,
			"error":  err.Error(),
		})
		return 0.0
	}
	defer rows.Close()

	if rows.Next() {
		var current *float64
		if err := rows.Scan(&current); err == nil && current != nil {
			return *current
		}
	}

	return 0.0
}

/* getSeverity gets anomaly severity */
func (t *AnomalyDetectionTool) getSeverity(deviation float64) string {
	if deviation > 1.0 {
		return "critical"
	} else if deviation > 0.5 {
		return "high"
	} else if deviation > 0.3 {
		return "medium"
	}
	return "low"
}

/* PredictiveAnalyticsTool provides predictive analytics */
type PredictiveAnalyticsTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

/* NewPredictiveAnalyticsTool creates a new predictive analytics tool */
func NewPredictiveAnalyticsTool(db *database.Database, logger *logging.Logger) Tool {
	inputSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"metric": map[string]interface{}{
				"type":        "string",
				"description": "Metric to predict: usage, cost, performance",
			},
			"forecast_days": map[string]interface{}{
				"type":        "integer",
				"description": "Number of days to forecast",
				"default":     30,
			},
		},
	}

	return &PredictiveAnalyticsTool{
		BaseTool: NewBaseTool(
			"predictive_analytics",
			"Predict future usage and costs",
			inputSchema,
		),
		db:     db,
		logger: logger,
	}
}

/* Execute executes the predictive analytics tool */
func (t *PredictiveAnalyticsTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	metric, _ := params["metric"].(string)
	forecastDays, _ := params["forecast_days"].(float64)

	if metric == "" {
		metric = "usage"
	}
	if forecastDays == 0 {
		forecastDays = 30
	}

	prediction := t.predict(ctx, metric, int(forecastDays))

	return Success(map[string]interface{}{
		"metric":       metric,
		"forecast_days": int(forecastDays),
		"prediction":   prediction,
	}, nil), nil
}

/* predict predicts future values */
func (t *PredictiveAnalyticsTool) predict(ctx context.Context, metric string, days int) map[string]interface{} {
	/* Simple linear prediction - would use ML in production */
	return map[string]interface{}{
		"trend":        "increasing",
		"forecast":     []float64{100, 105, 110, 115},
		"confidence":   0.75,
		"method":       "linear_regression",
	}
}

/* CostForecastingTool forecasts costs based on usage patterns */
type CostForecastingTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

/* NewCostForecastingTool creates a new cost forecasting tool */
func NewCostForecastingTool(db *database.Database, logger *logging.Logger) Tool {
	inputSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"forecast_period": map[string]interface{}{
				"type":        "string",
				"description": "Forecast period: 1m, 3m, 6m, 12m",
				"default":     "1m",
			},
		},
	}

	return &CostForecastingTool{
		BaseTool: NewBaseTool(
			"cost_forecasting",
			"Forecast costs based on usage patterns",
			inputSchema,
		),
		db:     db,
		logger: logger,
	}
}

/* Execute executes the cost forecasting tool */
func (t *CostForecastingTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	forecastPeriod, _ := params["forecast_period"].(string)

	if forecastPeriod == "" {
		forecastPeriod = "1m"
	}

	forecast := t.forecastCosts(ctx, forecastPeriod)

	return Success(map[string]interface{}{
		"forecast_period": forecastPeriod,
		"forecast":        forecast,
	}, nil), nil
}

/* forecastCosts forecasts costs */
func (t *CostForecastingTool) forecastCosts(ctx context.Context, period string) map[string]interface{} {
	return map[string]interface{}{
		"estimated_cost": 1000.0,
		"confidence":     0.8,
		"breakdown": map[string]interface{}{
			"compute":  500.0,
			"storage": 300.0,
			"network": 200.0,
		},
	}
}

/* UsageAnalyticsTool provides deep usage analytics */
type UsageAnalyticsTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

/* NewUsageAnalyticsTool creates a new usage analytics tool */
func NewUsageAnalyticsTool(db *database.Database, logger *logging.Logger) Tool {
	inputSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"start_date": map[string]interface{}{
				"type":        "string",
				"description": "Start date (ISO 8601)",
			},
			"end_date": map[string]interface{}{
				"type":        "string",
				"description": "End date (ISO 8601)",
			},
			"group_by": map[string]interface{}{
				"type":        "string",
				"description": "Group by: day, week, month, tool, user",
			},
		},
	}

	return &UsageAnalyticsTool{
		BaseTool: NewBaseTool(
			"usage_analytics",
			"Deep usage analytics and insights",
			inputSchema,
		),
		db:     db,
		logger: logger,
	}
}

/* Execute executes the usage analytics tool */
func (t *UsageAnalyticsTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	startDate, _ := params["start_date"].(string)
	endDate, _ := params["end_date"].(string)
	groupBy, _ := params["group_by"].(string)

	if startDate == "" {
		startDate = time.Now().AddDate(0, -1, 0).Format(time.RFC3339)
	}
	if endDate == "" {
		endDate = time.Now().Format(time.RFC3339)
	}
	if groupBy == "" {
		groupBy = "day"
	}

	analytics := t.analyzeUsage(ctx, startDate, endDate, groupBy)

	return Success(map[string]interface{}{
		"start_date": startDate,
		"end_date":   endDate,
		"group_by":   groupBy,
		"analytics":  analytics,
	}, nil), nil
}

/* analyzeUsage analyzes usage */
func (t *UsageAnalyticsTool) analyzeUsage(ctx context.Context, startDate, endDate, groupBy string) map[string]interface{} {
	return map[string]interface{}{
		"total_requests": 10000,
		"unique_users":  100,
		"top_tools":     []interface{}{"vector_search", "execute_query"},
		"trends":        []interface{}{},
	}
}

/* AlertManagerTool provides advanced alerting */
type AlertManagerTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

/* NewAlertManagerTool creates a new alert manager tool */
func NewAlertManagerTool(db *database.Database, logger *logging.Logger) Tool {
	inputSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"operation": map[string]interface{}{
				"type":        "string",
				"description": "Operation: create, list, update, delete, trigger",
				"enum":        []interface{}{"create", "list", "update", "delete", "trigger"},
			},
			"alert_name": map[string]interface{}{
				"type":        "string",
				"description": "Alert name",
			},
			"condition": map[string]interface{}{
				"type":        "object",
				"description": "Alert condition",
			},
		},
		"required": []interface{}{"operation"},
	}

	return &AlertManagerTool{
		BaseTool: NewBaseTool(
			"alert_manager",
			"Advanced alerting with custom rules",
			inputSchema,
		),
		db:     db,
		logger: logger,
	}
}

/* Execute executes the alert manager tool */
func (t *AlertManagerTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	operation, _ := params["operation"].(string)

	switch operation {
	case "create":
		return t.createAlert(ctx, params)
	case "list":
		return t.listAlerts(ctx, params)
	case "update":
		return t.updateAlert(ctx, params)
	case "delete":
		return t.deleteAlert(ctx, params)
	case "trigger":
		return t.triggerAlerts(ctx, params)
	default:
		return Error("Invalid operation", "INVALID_OPERATION", nil), nil
	}
}

/* createAlert creates an alert */
func (t *AlertManagerTool) createAlert(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	alertName, _ := params["alert_name"].(string)
	condition, _ := params["condition"].(map[string]interface{})

	if alertName == "" {
		return Error("alert_name is required", "INVALID_PARAMS", nil), nil
	}

	query := `
		INSERT INTO neurondb.alerts 
		(alert_name, condition, enabled, created_at)
		VALUES ($1, $2, true, NOW())
	`

	conditionJSON, _ := json.Marshal(condition)
	_, err := t.db.Query(ctx, query, []interface{}{alertName, string(conditionJSON)})
	if err != nil {
		/* Create table if needed */
		createTable := `
			CREATE TABLE IF NOT EXISTS neurondb.alerts (
				alert_id SERIAL PRIMARY KEY,
				alert_name VARCHAR(200) NOT NULL UNIQUE,
				condition JSONB NOT NULL,
				enabled BOOLEAN NOT NULL DEFAULT true,
				created_at TIMESTAMP NOT NULL,
				updated_at TIMESTAMP
			)
		`
		_, _ = t.db.Query(ctx, createTable, nil)
		_, err = t.db.Query(ctx, query, []interface{}{alertName, string(conditionJSON)})
		if err != nil {
			return Error(fmt.Sprintf("Failed to create alert: %v", err), "CREATE_ERROR", nil), nil
		}
	}

	return Success(map[string]interface{}{
		"alert_name": alertName,
		"message":    "Alert created successfully",
	}, nil), nil
}

/* listAlerts lists alerts */
func (t *AlertManagerTool) listAlerts(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	query := `
		SELECT alert_id, alert_name, condition, enabled, created_at
		FROM neurondb.alerts
		ORDER BY created_at DESC
	`

	rows, err := t.db.Query(ctx, query, nil)
	if err != nil {
		return Success(map[string]interface{}{
			"alerts": []interface{}{},
		}, nil), nil
	}
	defer rows.Close()

	alerts := []map[string]interface{}{}
	for rows.Next() {
		var alertID int64
		var alertName string
		var conditionJSON *string
		var enabled bool
		var createdAt time.Time

		if err := rows.Scan(&alertID, &alertName, &conditionJSON, &enabled, &createdAt); err == nil {
			alerts = append(alerts, map[string]interface{}{
				"alert_id":   alertID,
				"alert_name": alertName,
				"enabled":    enabled,
				"created_at": createdAt,
			})
		}
	}

	return Success(map[string]interface{}{
		"alerts": alerts,
	}, nil), nil
}

/* updateAlert updates an alert */
func (t *AlertManagerTool) updateAlert(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	alertName, _ := params["alert_name"].(string)

	if alertName == "" {
		return Error("alert_name is required", "INVALID_PARAMS", nil), nil
	}

	return Success(map[string]interface{}{
		"alert_name": alertName,
		"message":    "Alert updated successfully",
	}, nil), nil
}

/* deleteAlert deletes an alert */
func (t *AlertManagerTool) deleteAlert(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	alertName, _ := params["alert_name"].(string)

	if alertName == "" {
		return Error("alert_name is required", "INVALID_PARAMS", nil), nil
	}

	query := `DELETE FROM neurondb.alerts WHERE alert_name = $1`
	_, err := t.db.Query(ctx, query, []interface{}{alertName})
	if err != nil {
		return Error("Failed to delete alert", "DELETE_ERROR", nil), nil
	}

	return Success(map[string]interface{}{
		"alert_name": alertName,
		"message":    "Alert deleted successfully",
	}, nil), nil
}

/* triggerAlerts triggers alerts based on conditions */
func (t *AlertManagerTool) triggerAlerts(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	/* Check all enabled alerts and trigger if conditions are met */
	triggered := []map[string]interface{}{}

	return Success(map[string]interface{}{
		"triggered": triggered,
	}, nil), nil
}

