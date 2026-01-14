/*-------------------------------------------------------------------------
 *
 * sql_tool.go
 *    Tool implementation for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/tools/sql_tool.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/auth"
	"github.com/neurondb/NeuronAgent/internal/db"
)

type SQLTool struct {
	db              *db.DB
	queries         *db.Queries
	auditLogger     *auth.AuditLogger
	dataPermChecker *auth.DataPermissionChecker
}

func NewSQLTool(queries *db.Queries) *SQLTool {
	/* DB will be set by the registry during initialization */
	return &SQLTool{
		db:              nil,
		queries:         queries,
		auditLogger:     auth.NewAuditLogger(queries),
		dataPermChecker: auth.NewDataPermissionChecker(queries),
	}
}

func (t *SQLTool) Execute(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	query, ok := args["query"].(string)
	if !ok {
		argKeys := make([]string, 0, len(args))
		for k := range args {
			argKeys = append(argKeys, k)
		}
		return "", fmt.Errorf("SQL tool execution failed: tool_name='%s', handler_type='sql', args_count=%d, arg_keys=[%v], validation_error='query parameter is required and must be a string'",
			tool.Name, len(args), argKeys)
	}

	/* Security: Only allow SELECT, EXPLAIN, and schema introspection queries */
	queryUpper := strings.TrimSpace(strings.ToUpper(query))
	queryType := "UNKNOWN"
	if strings.HasPrefix(queryUpper, "SELECT") {
		queryType = "SELECT"
	} else if strings.HasPrefix(queryUpper, "EXPLAIN") {
		queryType = "EXPLAIN"
	} else if strings.HasPrefix(queryUpper, "SHOW") {
		queryType = "SHOW"
	} else if strings.HasPrefix(queryUpper, "DESCRIBE") {
		queryType = "DESCRIBE"
	} else if strings.HasPrefix(queryUpper, "\\d") {
		queryType = "DESCRIBE"
	}

	if queryType == "UNKNOWN" {
		queryPreview := query
		if len(queryPreview) > 100 {
			queryPreview = queryPreview[:100] + "..."
		}
		return "", fmt.Errorf("SQL tool execution failed: tool_name='%s', handler_type='sql', query_type='%s', query_preview='%s', query_length=%d, validation_error='only SELECT, EXPLAIN, SHOW, and DESCRIBE queries are allowed'",
			tool.Name, queryType, queryPreview, len(query))
	}

	/* Check for dangerous keywords - comprehensive list to prevent bypasses */
	/* Include variations and subqueries that could be used to bypass restrictions */
	dangerous := []string{
		"DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE",
		"INTO", "RETURNING", "EXEC", "EXECUTE", "CALL", "COPY", "GRANT", "REVOKE",
		"BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT", "RELEASE",
	}
	var foundKeywords []string
	for _, keyword := range dangerous {
		/* Use word boundary matching to avoid false positives in strings/comments */
		/* This is more accurate than simple Contains() but still not perfect */
		/* For production, a proper SQL parser would be better */
		pattern := "\\b" + keyword + "\\b"
		matched, err := regexp.MatchString(pattern, queryUpper)
		if err != nil {
			/* If regex compilation fails, fall back to simple string contains check */
			/* This should rarely happen with our simple patterns, but handle gracefully */
			if strings.Contains(queryUpper, " "+keyword+" ") {
				foundKeywords = append(foundKeywords, keyword)
			}
		} else if matched {
			foundKeywords = append(foundKeywords, keyword)
		}
	}
	if len(foundKeywords) > 0 {
		queryPreview := query
		if len(queryPreview) > 100 {
			queryPreview = queryPreview[:100] + "..."
		}
		return "", fmt.Errorf("SQL tool execution failed: tool_name='%s', handler_type='sql', query_type='%s', query_preview='%s', query_length=%d, forbidden_keywords=[%v], validation_error='query contains forbidden keywords'",
			tool.Name, queryType, queryPreview, len(query), foundKeywords)
	}

	/* Execute query (read-only) */
	if t.db == nil {
		return "", fmt.Errorf("SQL tool execution failed: tool_name='%s', handler_type='sql', query_type='%s', query_length=%d, database_connection='not_initialized'",
			tool.Name, queryType, len(query))
	}

	connInfo := "unknown"
	if t.db != nil {
		connInfo = t.db.GetConnInfoString()
	}

	/* Apply data permissions: row filters and check permissions */
	agentID, agentOK := GetAgentIDFromContext(ctx)
	if !agentOK {
		/* Agent ID not available in context - use zero UUID */
		agentID = uuid.Nil
	}
	sessionID, sessionOK := GetSessionIDFromContext(ctx)
	if !sessionOK {
		/* Session ID not available in context - use zero UUID */
		sessionID = uuid.Nil
	}

	/* Try to extract schema and table name from query (simple case) */
	schemaName, tableName := extractTableFromQuery(query)

	/* Get principal ID from context if available */
	/* For now, we'll skip row filter application for complex queries */
	/* In production, use a SQL parser to extract table names */
	var rowFilter string
	var columnMask map[string]string

	if schemaName != "" || tableName != "" {
		/* Extract principal from context */
		principal := GetPrincipalFromContext(ctx)
		if principal != nil {
			/* Get row filter for this principal and table */
			rowFilterStr, err := t.dataPermChecker.GetRowFilterForTable(ctx, principal.ID, schemaName, tableName)
			if err == nil && rowFilterStr != "" {
				rowFilter = rowFilterStr
			}

			/* Get column mask for this principal and table */
			columnMaskMap, err := t.dataPermChecker.GetColumnMaskForTable(ctx, principal.ID, schemaName, tableName)
			if err == nil && columnMaskMap != nil {
				columnMask = columnMaskMap
			}
		}
	}

	/* Apply row filter if available */
	finalQuery := query
	if rowFilter != "" {
		finalQuery = t.dataPermChecker.ApplyRowFilter(query, rowFilter)
	}

	rows, err := t.db.QueryContext(ctx, finalQuery)
	if err != nil {
		queryPreview := query
		if len(queryPreview) > 200 {
			queryPreview = queryPreview[:200] + "..."
		}
		return "", fmt.Errorf("SQL tool query execution failed: tool_name='%s', handler_type='sql', query_type='%s', query_preview='%s', query_length=%d, database='%s', error=%w",
			tool.Name, queryType, queryPreview, len(query), connInfo, err)
	}
	defer rows.Close()

	/* Convert results to JSON */
	var results []map[string]interface{}
	columns, err := rows.Columns()
	if err != nil {
		queryPreview := query
		if len(queryPreview) > 200 {
			queryPreview = queryPreview[:200] + "..."
		}
		return "", fmt.Errorf("SQL tool column retrieval failed: tool_name='%s', handler_type='sql', query_type='%s', query_preview='%s', query_length=%d, database='%s', error=%w",
			tool.Name, queryType, queryPreview, len(query), connInfo, err)
	}

	rowCount := 0
	for rows.Next() {
		values := make([]interface{}, len(columns))
		valuePtrs := make([]interface{}, len(columns))
		for i := range values {
			valuePtrs[i] = &values[i]
		}

		if err := rows.Scan(valuePtrs...); err != nil {
			queryPreview := query
			if len(queryPreview) > 200 {
				queryPreview = queryPreview[:200] + "..."
			}
			return "", fmt.Errorf("SQL tool row scan failed: tool_name='%s', handler_type='sql', query_type='%s', query_preview='%s', query_length=%d, row_count=%d, column_count=%d, database='%s', error=%w",
				tool.Name, queryType, queryPreview, len(query), rowCount, len(columns), connInfo, err)
		}

		row := make(map[string]interface{})
		for i, col := range columns {
			row[col] = values[i]
		}
		results = append(results, row)
		rowCount++
	}

	if err := rows.Err(); err != nil {
		queryPreview := query
		if len(queryPreview) > 200 {
			queryPreview = queryPreview[:200] + "..."
		}
		return "", fmt.Errorf("SQL tool row iteration failed: tool_name='%s', handler_type='sql', query_type='%s', query_preview='%s', query_length=%d, row_count=%d, column_count=%d, database='%s', error=%w",
			tool.Name, queryType, queryPreview, len(query), rowCount, len(columns), connInfo, err)
	}

	jsonResult, err := json.Marshal(results)
	if err != nil {
		return "", fmt.Errorf("SQL tool result marshaling failed: tool_name='%s', handler_type='sql', query_type='%s', row_count=%d, column_count=%d, error=%w",
			tool.Name, queryType, rowCount, len(columns), err)
	}

	/* Apply column masking if available */
	if columnMask != nil && len(columnMask) > 0 {
		results = t.dataPermChecker.MaskColumns(results, columnMask)
		jsonResult, err = json.Marshal(results)
		if err != nil {
			return "", fmt.Errorf("SQL tool result marshaling after masking failed: tool_name='%s', handler_type='sql', query_type='%s', row_count=%d, column_count=%d, error=%w",
				tool.Name, queryType, rowCount, len(columns), err)
		}
	}

	resultStr := string(jsonResult)

	/* Audit log SQL statement - use already extracted IDs from above */

	outputs := map[string]interface{}{
		"row_count":     rowCount,
		"column_count":  len(columns),
		"success":       true,
		"result_length": len(resultStr),
	}

	/* Log audit trail (async, don't block on errors) */
	go func() {
		bgCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		var agentIDPtr, sessionIDPtr *uuid.UUID
		if agentID != uuid.Nil {
			agentIDPtr = &agentID
		}
		if sessionID != uuid.Nil {
			sessionIDPtr = &sessionID
		}

		_ = t.auditLogger.LogSQLStatement(bgCtx, nil, nil, agentIDPtr, sessionIDPtr, query, args, outputs)
	}()

	return resultStr, nil
}

func (t *SQLTool) Validate(args map[string]interface{}, schema map[string]interface{}) error {
	return ValidateArgs(args, schema)
}

/* extractTableFromQuery attempts to extract schema and table name from a simple SELECT query */
/* This is a basic implementation - for complex queries, use a proper SQL parser */
func extractTableFromQuery(query string) (schemaName, tableName string) {
	queryUpper := strings.ToUpper(strings.TrimSpace(query))

	/* Look for FROM clause */
	fromIndex := strings.Index(queryUpper, " FROM ")
	if fromIndex < 0 {
		return "", ""
	}

	/* Extract table name after FROM */
	fromPart := query[fromIndex+6:]

	/* Remove WHERE, ORDER BY, GROUP BY, LIMIT, OFFSET */
	for _, keyword := range []string{" WHERE ", " ORDER BY ", " GROUP BY ", " LIMIT ", " OFFSET "} {
		if idx := strings.Index(strings.ToUpper(fromPart), keyword); idx > 0 {
			fromPart = fromPart[:idx]
		}
	}

	fromPart = strings.TrimSpace(fromPart)
	if fromPart == "" {
		return "", ""
	}

	/* Check for schema.table format */
	parts := strings.Split(fromPart, ".")
	if len(parts) == 2 {
		return strings.TrimSpace(parts[0]), strings.TrimSpace(parts[1])
	} else if len(parts) == 1 {
		return "", strings.TrimSpace(parts[0])
	}

	return "", ""
}
