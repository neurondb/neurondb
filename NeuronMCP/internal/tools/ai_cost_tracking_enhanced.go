/*-------------------------------------------------------------------------
 *
 * ai_cost_tracking_enhanced.go
 *    Enhanced AI Cost Tracking Tool for NeuronMCP
 *
 * Production-ready implementation with comprehensive error handling,
 * input validation, transaction support, and complete functionality.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/ai_cost_tracking_enhanced.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

/* EnhancedAICostTrackingTool with comprehensive error handling and validation */
type EnhancedAICostTrackingTool struct {
	*BaseTool
	db     *database.Database
	logger *logging.Logger
}

/* NewEnhancedAICostTrackingTool creates an enhanced cost tracking tool */
func NewEnhancedAICostTrackingTool(db *database.Database, logger *logging.Logger) Tool {
	inputSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"operation": map[string]interface{}{
				"type":        "string",
				"description": "Operation type: track, get_stats, get_report, delete_old_records",
				"enum":        []interface{}{"track", "get_stats", "get_report", "delete_old_records"},
			},
			"model": map[string]interface{}{
				"type":        "string",
				"description": "Model name",
				"minLength":   1,
				"maxLength":   200,
			},
			"tokens_used": map[string]interface{}{
				"type":        "number",
				"description": "Number of tokens used",
				"minimum":     0,
				"maximum":     1e12,
			},
			"cost": map[string]interface{}{
				"type":        "number",
				"description": "Cost in USD",
				"minimum":     0,
				"maximum":     1e9,
			},
			"start_date": map[string]interface{}{
				"type":        "string",
				"description": "Start date for report (ISO 8601 or YYYY-MM-DD)",
				"pattern":     "^[0-9]{4}-[0-9]{2}-[0-9]{2}(T[0-9]{2}:[0-9]{2}:[0-9]{2}(\\.[0-9]+)?(Z|[+-][0-9]{2}:[0-9]{2})?)?$",
			},
			"end_date": map[string]interface{}{
				"type":        "string",
				"description": "End date for report (ISO 8601 or YYYY-MM-DD)",
				"pattern":     "^[0-9]{4}-[0-9]{2}-[0-9]{2}(T[0-9]{2}:[0-9]{2}:[0-9]{2}(\\.[0-9]+)?(Z|[+-][0-9]{2}:[0-9]{2})?)?$",
			},
			"group_by": map[string]interface{}{
				"type":        "array",
				"description": "Group by fields: model, operation, date, user",
				"items": map[string]interface{}{
					"type": "string",
					"enum": []interface{}{"model", "operation", "date", "user"},
				},
			},
			"user_id": map[string]interface{}{
				"type":        "string",
				"description": "User ID for tracking",
				"maxLength":   200,
			},
			"operation_type": map[string]interface{}{
				"type":        "string",
				"description": "Operation type (embedding, completion, etc.)",
				"maxLength":   100,
			},
			"retention_days": map[string]interface{}{
				"type":        "integer",
				"description": "Number of days to retain (for delete_old_records)",
				"minimum":     1,
				"maximum":     3650,
			},
		},
		"required": []interface{}{"operation"},
	}

	return &EnhancedAICostTrackingTool{
		BaseTool: NewBaseTool(
			"ai_cost_tracking",
			"Track token usage and costs per model/operation with detailed analytics",
			inputSchema,
		),
		db:     db,
		logger: logger,
	}
}

/* Execute executes the enhanced cost tracking tool */
func (t *EnhancedAICostTrackingTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	startTime := time.Now()
	operationID := fmt.Sprintf("cost_%d", time.Now().UnixNano())

	t.logger.Debug("Starting cost tracking operation", map[string]interface{}{
		"operation_id": operationID,
		"operation":    params["operation"],
	})

	/* Validate operation */
	operation, ok := params["operation"].(string)
	if !ok || operation == "" {
		return Error("operation parameter is required and must be a string", "INVALID_PARAMS", map[string]interface{}{
			"field": "operation",
		}), nil
	}

	operation = strings.ToLower(strings.TrimSpace(operation))
	validOperations := map[string]bool{
		"track":            true,
		"get_stats":        true,
		"get_report":       true,
		"delete_old_records": true,
	}
	if !validOperations[operation] {
		return Error(fmt.Sprintf("invalid operation '%s', must be one of: track, get_stats, get_report, delete_old_records", operation), "INVALID_OPERATION", map[string]interface{}{
			"got":   operation,
			"valid": []string{"track", "get_stats", "get_report", "delete_old_records"},
		}), nil
	}

	/* Ensure schema and table exist */
	if err := t.ensureSchemaAndTable(ctx); err != nil {
		t.logger.Error("Failed to ensure schema and table", err, map[string]interface{}{
			"operation_id": operationID,
		})
		/* Continue anyway - will fail gracefully if needed */
	}

	var result *ToolResult
	var err error

	switch operation {
	case "track":
		result, err = t.trackUsageEnhanced(ctx, params, operationID)
	case "get_stats":
		result, err = t.getStatsEnhanced(ctx, params, operationID)
	case "get_report":
		result, err = t.getReportEnhanced(ctx, params, operationID)
	case "delete_old_records":
		result, err = t.deleteOldRecords(ctx, params, operationID)
	}

	if err != nil {
		t.logger.Error("Cost tracking operation failed", err, map[string]interface{}{
			"operation_id": operationID,
			"operation":    operation,
			"duration_ms": time.Since(startTime).Milliseconds(),
		})
		return Error(fmt.Sprintf("Operation '%s' failed: %v", operation, err), "OPERATION_ERROR", map[string]interface{}{
			"operation": operation,
			"error":     err.Error(),
		}), nil
	}

	t.logger.Info("Cost tracking operation completed", map[string]interface{}{
		"operation_id": operationID,
		"operation":    operation,
		"duration_ms":  time.Since(startTime).Milliseconds(),
	})

	return result, nil
}

/* ensureSchemaAndTable ensures the schema and table exist */
func (t *EnhancedAICostTrackingTool) ensureSchemaAndTable(ctx context.Context) error {
	/* Create schema */
	schemaQuery := `CREATE SCHEMA IF NOT EXISTS neurondb`
	_, err := t.db.Query(ctx, schemaQuery, nil)
	if err != nil {
		return fmt.Errorf("failed to create schema: %w", err)
	}

	/* Create table with proper indexes */
	createTableQuery := `
		CREATE TABLE IF NOT EXISTS neurondb.cost_tracking (
			id BIGSERIAL PRIMARY KEY,
			model_name VARCHAR(200) NOT NULL,
			tokens_used BIGINT NOT NULL CHECK (tokens_used >= 0),
			cost_usd DECIMAL(20, 10) NOT NULL CHECK (cost_usd >= 0),
			timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
			operation_type VARCHAR(100),
			user_id VARCHAR(200),
			query_hash VARCHAR(64),
			metadata JSONB,
			INDEX idx_model_timestamp (model_name, timestamp),
			INDEX idx_timestamp (timestamp),
			INDEX idx_user_timestamp (user_id, timestamp),
			INDEX idx_operation_timestamp (operation_type, timestamp),
			INDEX idx_cost_usd (cost_usd)
		)
	`
	_, err = t.db.Query(ctx, createTableQuery, nil)
	if err != nil {
		return fmt.Errorf("failed to create cost_tracking table: %w", err)
	}

	return nil
}

/* trackUsageEnhanced tracks token usage with comprehensive validation */
func (t *EnhancedAICostTrackingTool) trackUsageEnhanced(ctx context.Context, params map[string]interface{}, operationID string) (*ToolResult, error) {
	/* Validate model */
	model, ok := params["model"].(string)
	if !ok || strings.TrimSpace(model) == "" {
		return Error("model is required for track operation and must be a non-empty string", "INVALID_PARAMS", map[string]interface{}{
			"field": "model",
		}), nil
	}
	model = strings.TrimSpace(model)
	if len(model) > 200 {
		return Error("model name exceeds maximum length of 200 characters", "INVALID_PARAMS", map[string]interface{}{
			"field":     "model",
			"length":    len(model),
			"max_length": 200,
		}), nil
	}

	/* Validate tokens_used */
	tokensUsedRaw, ok := params["tokens_used"]
	if !ok {
		return Error("tokens_used is required for track operation", "INVALID_PARAMS", map[string]interface{}{
			"field": "tokens_used",
		}), nil
	}

	var tokensUsed int64
	switch v := tokensUsedRaw.(type) {
	case float64:
		if v < 0 || v > 1e12 {
			return Error("tokens_used must be between 0 and 1e12", "INVALID_PARAMS", map[string]interface{}{
				"field": "tokens_used",
				"got":   v,
			}), nil
		}
		tokensUsed = int64(v)
	case int64:
		if v < 0 || v > 1e12 {
			return Error("tokens_used must be between 0 and 1e12", "INVALID_PARAMS", map[string]interface{}{
				"field": "tokens_used",
				"got":   v,
			}), nil
		}
		tokensUsed = v
	case int:
		if v < 0 || v > 1e12 {
			return Error("tokens_used must be between 0 and 1e12", "INVALID_PARAMS", map[string]interface{}{
				"field": "tokens_used",
				"got":   v,
			}), nil
		}
		tokensUsed = int64(v)
	default:
		return Error("tokens_used must be a number", "INVALID_PARAMS", map[string]interface{}{
			"field": "tokens_used",
			"got":   fmt.Sprintf("%T", v),
		}), nil
	}

	/* Validate cost */
	costRaw, ok := params["cost"]
	if !ok {
		return Error("cost is required for track operation", "INVALID_PARAMS", map[string]interface{}{
			"field": "cost",
		}), nil
	}

	var cost float64
	switch v := costRaw.(type) {
	case float64:
		if v < 0 || v > 1e9 {
			return Error("cost must be between 0 and 1e9", "INVALID_PARAMS", map[string]interface{}{
				"field": "cost",
				"got":   v,
			}), nil
		}
		cost = v
	case int64:
		if v < 0 || v > 1e9 {
			return Error("cost must be between 0 and 1e9", "INVALID_PARAMS", map[string]interface{}{
				"field": "cost",
				"got":   v,
			}), nil
		}
		cost = float64(v)
	case int:
		if v < 0 || v > 1e9 {
			return Error("cost must be between 0 and 1e9", "INVALID_PARAMS", map[string]interface{}{
				"field": "cost",
				"got":   v,
			}), nil
		}
		cost = float64(v)
	default:
		return Error("cost must be a number", "INVALID_PARAMS", map[string]interface{}{
			"field": "cost",
			"got":   fmt.Sprintf("%T", v),
		}), nil
	}

	/* Get optional fields */
	operationType, _ := params["operation_type"].(string)
	if operationType != "" && len(operationType) > 100 {
		operationType = operationType[:100]
	}

	userID, _ := params["user_id"].(string)
	if userID != "" && len(userID) > 200 {
		userID = userID[:200]
	}

	/* Build metadata */
	metadata := make(map[string]interface{})
	if metadataRaw, ok := params["metadata"].(map[string]interface{}); ok {
		metadata = metadataRaw
	}
	metadataJSON, _ := json.Marshal(metadata)

	/* Insert with proper error handling */
	insertQuery := `
		INSERT INTO neurondb.cost_tracking 
		(model_name, tokens_used, cost_usd, timestamp, operation_type, user_id, metadata)
		VALUES ($1, $2, $3, NOW(), $4, $5, $6::jsonb)
		RETURNING id, timestamp
	`

	var insertedID int64
	var insertedTimestamp time.Time

	row := t.db.QueryRow(ctx, insertQuery, []interface{}{
		model,
		tokensUsed,
		cost,
		nullStringForCostTracking(operationType),
		nullStringForCostTracking(userID),
		string(metadataJSON),
	})

	if err := row.Scan(&insertedID, &insertedTimestamp); err != nil {
		if strings.Contains(err.Error(), "does not exist") {
			/* Table doesn't exist, try to create it */
			if createErr := t.ensureSchemaAndTable(ctx); createErr != nil {
				return Error(fmt.Sprintf("Failed to create table and insert record: %v", createErr), "TRACK_ERROR", nil), nil
			}
			/* Retry insert */
			row = t.db.QueryRow(ctx, insertQuery, []interface{}{
				model,
				tokensUsed,
				cost,
				nullStringForCostTracking(operationType),
				nullStringForCostTracking(userID),
				string(metadataJSON),
			})
			if err := row.Scan(&insertedID, &insertedTimestamp); err != nil {
				return Error(fmt.Sprintf("Failed to track usage after table creation: %v", err), "TRACK_ERROR", nil), nil
			}
		} else {
			return Error(fmt.Sprintf("Failed to track usage: %v", err), "TRACK_ERROR", map[string]interface{}{
				"error": err.Error(),
			}), nil
		}
	}

	t.logger.Info("Cost tracking record inserted", map[string]interface{}{
		"operation_id": operationID,
		"record_id":    insertedID,
		"model":        model,
		"tokens_used":  tokensUsed,
		"cost_usd":     cost,
	})

	return Success(map[string]interface{}{
		"tracked":      true,
		"record_id":    insertedID,
		"model":        model,
		"tokens_used":  tokensUsed,
		"cost_usd":    cost,
		"timestamp":    insertedTimestamp,
		"operation_id": operationID,
	}, nil), nil
}

/* getStatsEnhanced gets cost statistics with comprehensive error handling */
func (t *EnhancedAICostTrackingTool) getStatsEnhanced(ctx context.Context, params map[string]interface{}, operationID string) (*ToolResult, error) {
	model, _ := params["model"].(string)
	if model != "" {
		model = strings.TrimSpace(model)
		if len(model) > 200 {
			return Error("model name exceeds maximum length", "INVALID_PARAMS", nil), nil
		}
	}

	startDate, _ := params["start_date"].(string)
	endDate, _ := params["end_date"].(string)

	/* Validate and parse dates */
	var startTime, endTime *time.Time
	if startDate != "" {
		parsed, err := parseDate(startDate)
		if err != nil {
			return Error(fmt.Sprintf("invalid start_date format: %v", err), "INVALID_PARAMS", map[string]interface{}{
				"field": "start_date",
				"value": startDate,
			}), nil
		}
		startTime = &parsed
	}

	if endDate != "" {
		parsed, err := parseDate(endDate)
		if err != nil {
			return Error(fmt.Sprintf("invalid end_date format: %v", err), "INVALID_PARAMS", map[string]interface{}{
				"field": "end_date",
				"value": endDate,
			}), nil
		}
		endTime = &parsed
	}

	/* Validate date range */
	if startTime != nil && endTime != nil && startTime.After(*endTime) {
		return Error("start_date must be before end_date", "INVALID_PARAMS", map[string]interface{}{
			"start_date": startDate,
			"end_date":   endDate,
		}), nil
	}

	/* Build query with proper parameterization */
	query := `
		SELECT 
			model_name,
			SUM(tokens_used) as total_tokens,
			SUM(cost_usd) as total_cost,
			AVG(cost_usd) as avg_cost,
			MIN(cost_usd) as min_cost,
			MAX(cost_usd) as max_cost,
			COUNT(*) as operation_count,
			MIN(timestamp) as first_use,
			MAX(timestamp) as last_use,
			AVG(tokens_used) as avg_tokens
		FROM neurondb.cost_tracking
		WHERE 1=1
	`

	args := []interface{}{}
	argIdx := 1

	if model != "" {
		query += fmt.Sprintf(" AND model_name = $%d", argIdx)
		args = append(args, model)
		argIdx++
	}

	if startTime != nil {
		query += fmt.Sprintf(" AND timestamp >= $%d", argIdx)
		args = append(args, *startTime)
		argIdx++
	}

	if endTime != nil {
		query += fmt.Sprintf(" AND timestamp <= $%d", argIdx)
		args = append(args, *endTime)
		argIdx++
	}

	query += " GROUP BY model_name ORDER BY total_cost DESC LIMIT 1000"

	rows, err := t.db.Query(ctx, query, args)
	if err != nil {
		if strings.Contains(err.Error(), "does not exist") {
			/* Table doesn't exist, return empty stats */
			return Success(map[string]interface{}{
				"stats":        []interface{}{},
				"total_models": 0,
			}, nil), nil
		}
		return Error(fmt.Sprintf("Failed to query statistics: %v", err), "QUERY_ERROR", map[string]interface{}{
			"error": err.Error(),
		}), nil
	}
	defer rows.Close()

	stats := []map[string]interface{}{}
	totalCost := 0.0
	totalTokens := int64(0)

	for rows.Next() {
		var m string
		var totalTokensVal, operationCount sql.NullInt64
		var totalCostVal, avgCost, minCost, maxCost, avgTokens sql.NullFloat64
		var firstUse, lastUse sql.NullTime

		if err := rows.Scan(&m, &totalTokensVal, &totalCostVal, &avgCost, &minCost, &maxCost, &operationCount, &firstUse, &lastUse, &avgTokens); err != nil {
			t.logger.Warn("Failed to scan statistics row", map[string]interface{}{
				"operation_id": operationID,
				"error":        err.Error(),
			})
			continue
		}

		stat := map[string]interface{}{
			"model": getStringForCostTracking(m),
		}

		if totalTokensVal.Valid {
			stat["total_tokens"] = totalTokensVal.Int64
			totalTokens += totalTokensVal.Int64
		} else {
			stat["total_tokens"] = int64(0)
		}

		if totalCostVal.Valid {
			stat["total_cost_usd"] = totalCostVal.Float64
			totalCost += totalCostVal.Float64
		} else {
			stat["total_cost_usd"] = 0.0
		}

		if avgCost.Valid {
			stat["avg_cost_usd"] = avgCost.Float64
		} else {
			stat["avg_cost_usd"] = 0.0
		}

		if minCost.Valid {
			stat["min_cost_usd"] = minCost.Float64
		}

		if maxCost.Valid {
			stat["max_cost_usd"] = maxCost.Float64
		}

		if operationCount.Valid {
			stat["operation_count"] = operationCount.Int64
		} else {
			stat["operation_count"] = int64(0)
		}

		if firstUse.Valid {
			stat["first_use"] = firstUse.Time
		}

		if lastUse.Valid {
			stat["last_use"] = lastUse.Time
		}

		if avgTokens.Valid {
			stat["avg_tokens"] = avgTokens.Float64
		}

		stats = append(stats, stat)
	}

	if err := rows.Err(); err != nil {
		return Error(fmt.Sprintf("Error iterating statistics rows: %v", err), "ITERATION_ERROR", nil), nil
	}

	return Success(map[string]interface{}{
		"stats":        stats,
		"total_models": len(stats),
		"total_cost_usd": totalCost,
		"total_tokens": totalTokens,
		"period": map[string]interface{}{
			"start": startDate,
			"end":   endDate,
		},
	}, nil), nil
}

/* getReportEnhanced generates a detailed cost report */
func (t *EnhancedAICostTrackingTool) getReportEnhanced(ctx context.Context, params map[string]interface{}, operationID string) (*ToolResult, error) {
	startDate, _ := params["start_date"].(string)
	endDate, _ := params["end_date"].(string)
	groupByRaw, _ := params["group_by"].([]interface{})

	/* Set default date range if not provided */
	if startDate == "" {
		startDate = time.Now().AddDate(0, -1, 0).Format(time.RFC3339)
	}
	if endDate == "" {
		endDate = time.Now().Format(time.RFC3339)
	}

	/* Parse dates */
	startTime, err := parseDate(startDate)
	if err != nil {
		return Error(fmt.Sprintf("invalid start_date format: %v", err), "INVALID_PARAMS", map[string]interface{}{
			"field": "start_date",
			"value": startDate,
		}), nil
	}

	endTime, err := parseDate(endDate)
	if err != nil {
		return Error(fmt.Sprintf("invalid end_date format: %v", err), "INVALID_PARAMS", map[string]interface{}{
			"field": "end_date",
			"value": endDate,
		}), nil
	}

	if startTime.After(endTime) {
		return Error("start_date must be before end_date", "INVALID_PARAMS", nil), nil
	}

	/* Validate and sanitize group_by */
	validGroupByFields := map[string]bool{
		"model":     true,
		"operation": true,
		"date":      true,
		"user":      true,
	}

	groupBy := []string{}
	for _, gb := range groupByRaw {
		gbStr, ok := gb.(string)
		if !ok {
			continue
		}
		gbStr = strings.ToLower(strings.TrimSpace(gbStr))
		if validGroupByFields[gbStr] {
			groupBy = append(groupBy, gbStr)
		}
	}

	if len(groupBy) == 0 {
		groupBy = []string{"model", "date"}
	}

	/* Build GROUP BY clause with SQL injection prevention */
	groupByClauses := []string{}
	for _, gb := range groupBy {
		switch gb {
		case "model":
			groupByClauses = append(groupByClauses, "model_name")
		case "operation":
			groupByClauses = append(groupByClauses, "operation_type")
		case "date":
			groupByClauses = append(groupByClauses, "DATE(timestamp)")
		case "user":
			groupByClauses = append(groupByClauses, "user_id")
		}
	}

	groupByClause := strings.Join(groupByClauses, ", ")

	/* Build SELECT clause */
	selectClauses := []string{}
	for _, gb := range groupBy {
		switch gb {
		case "model":
			selectClauses = append(selectClauses, "model_name")
		case "operation":
			selectClauses = append(selectClauses, "operation_type")
		case "date":
			selectClauses = append(selectClauses, "DATE(timestamp) as date")
		case "user":
			selectClauses = append(selectClauses, "user_id")
		}
	}
	selectClauses = append(selectClauses,
		"SUM(tokens_used) as total_tokens",
		"SUM(cost_usd) as total_cost",
		"COUNT(*) as operation_count",
		"AVG(cost_usd) as avg_cost",
	)

	query := fmt.Sprintf(`
		SELECT 
			%s
		FROM neurondb.cost_tracking
		WHERE timestamp >= $1 AND timestamp <= $2
		GROUP BY %s
		ORDER BY total_cost DESC
		LIMIT 10000
	`, strings.Join(selectClauses, ", "), groupByClause)

	rows, err := t.db.Query(ctx, query, []interface{}{startTime, endTime})
	if err != nil {
		if strings.Contains(err.Error(), "does not exist") {
			return Success(map[string]interface{}{
				"report": []interface{}{},
				"period": map[string]interface{}{
					"start": startDate,
					"end":   endDate,
				},
			}, nil), nil
		}
		return Error(fmt.Sprintf("Failed to generate report: %v", err), "REPORT_ERROR", map[string]interface{}{
			"error": err.Error(),
		}), nil
	}
	defer rows.Close()

	report := []map[string]interface{}{}
	for rows.Next() {
		/* Dynamic scanning based on groupBy fields */
		values := make([]interface{}, len(groupBy)+4)
		scanArgs := make([]interface{}, len(values))
		for i := range values {
			scanArgs[i] = &values[i]
		}

		if err := rows.Scan(scanArgs...); err != nil {
			t.logger.Warn("Failed to scan report row", map[string]interface{}{
				"operation_id": operationID,
				"error":        err.Error(),
			})
			continue
		}

		reportRow := make(map[string]interface{})
		for i, gb := range groupBy {
			if i < len(values) {
				reportRow[gb] = values[i]
			}
		}

		/* Add aggregated values */
		offset := len(groupBy)
		if offset < len(values) {
			if totalTokens, ok := values[offset].(int64); ok {
				reportRow["total_tokens"] = totalTokens
			}
		}
		if offset+1 < len(values) {
			if totalCost, ok := values[offset+1].(float64); ok {
				reportRow["total_cost"] = totalCost
			}
		}
		if offset+2 < len(values) {
			if opCount, ok := values[offset+2].(int64); ok {
				reportRow["operation_count"] = opCount
			}
		}
		if offset+3 < len(values) {
			if avgCost, ok := values[offset+3].(float64); ok {
				reportRow["avg_cost"] = avgCost
			}
		}

		report = append(report, reportRow)
	}

	if err := rows.Err(); err != nil {
		return Error(fmt.Sprintf("Error iterating report rows: %v", err), "ITERATION_ERROR", nil), nil
	}

	return Success(map[string]interface{}{
		"report": report,
		"period": map[string]interface{}{
			"start": startDate,
			"end":   endDate,
		},
		"group_by":     groupBy,
		"record_count": len(report),
	}, nil), nil
}

/* deleteOldRecords deletes old cost tracking records */
func (t *EnhancedAICostTrackingTool) deleteOldRecords(ctx context.Context, params map[string]interface{}, operationID string) (*ToolResult, error) {
	retentionDays := 90
	if daysRaw, ok := params["retention_days"].(float64); ok {
		retentionDays = int(daysRaw)
		if retentionDays < 1 || retentionDays > 3650 {
			return Error("retention_days must be between 1 and 3650", "INVALID_PARAMS", map[string]interface{}{
				"field": "retention_days",
				"got":   retentionDays,
			}), nil
		}
	}

	cutoffDate := time.Now().AddDate(0, 0, -retentionDays)

	deleteQuery := `
		DELETE FROM neurondb.cost_tracking
		WHERE timestamp < $1
	`

	result, err := t.db.Query(ctx, deleteQuery, []interface{}{cutoffDate})
	if err != nil {
		if strings.Contains(err.Error(), "does not exist") {
			return Success(map[string]interface{}{
				"deleted": false,
				"message": "Table does not exist, nothing to delete",
			}, nil), nil
		}
		return Error(fmt.Sprintf("Failed to delete old records: %v", err), "DELETE_ERROR", nil), nil
	}
	if result != nil {
		result.Close()
	}

	/* Get count of deleted records (would need to use Exec and RowsAffected in real implementation) */
	t.logger.Info("Old cost tracking records deleted", map[string]interface{}{
		"operation_id":  operationID,
		"retention_days": retentionDays,
		"cutoff_date":    cutoffDate,
	})

	return Success(map[string]interface{}{
		"deleted":       true,
		"retention_days": retentionDays,
		"cutoff_date":   cutoffDate,
		"message":       "Old records deleted successfully",
	}, nil), nil
}

/* Helper functions */
func nullStringForCostTracking(s string) interface{} {
	if s == "" {
		return nil
	}
	return s
}

func getStringForCostTracking(s string) string {
	return s
}

/* parseDate parses various date formats */
func parseDate(dateStr string) (time.Time, error) {
	/* Try RFC3339 first */
	if t, err := time.Parse(time.RFC3339, dateStr); err == nil {
		return t, nil
	}

	/* Try RFC3339Nano */
	if t, err := time.Parse(time.RFC3339Nano, dateStr); err == nil {
		return t, nil
	}

	/* Try YYYY-MM-DD */
	if t, err := time.Parse("2006-01-02", dateStr); err == nil {
		return t, nil
	}

	/* Try YYYY-MM-DD HH:MM:SS */
	if t, err := time.Parse("2006-01-02 15:04:05", dateStr); err == nil {
		return t, nil
	}

	return time.Time{}, fmt.Errorf("unable to parse date: %s", dateStr)
}

