package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/neurondb/NeuronAgent/internal/db"
)

type SQLTool struct {
	db *db.DB
}

func NewSQLTool(queries *db.Queries) *SQLTool {
	// DB will be set by the registry during initialization
	return &SQLTool{db: nil}
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

	// Security: Only allow SELECT, EXPLAIN, and schema introspection queries
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

	// Check for dangerous keywords
	dangerous := []string{"DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE"}
	var foundKeywords []string
	for _, keyword := range dangerous {
		if strings.Contains(queryUpper, keyword) {
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

	// Execute query (read-only)
	if t.db == nil {
		return "", fmt.Errorf("SQL tool execution failed: tool_name='%s', handler_type='sql', query_type='%s', query_length=%d, database_connection='not_initialized'",
			tool.Name, queryType, len(query))
	}
	
	connInfo := "unknown"
	if t.db != nil {
		connInfo = t.db.GetConnInfoString()
	}
	
	rows, err := t.db.QueryContext(ctx, query)
	if err != nil {
		queryPreview := query
		if len(queryPreview) > 200 {
			queryPreview = queryPreview[:200] + "..."
		}
		return "", fmt.Errorf("SQL tool query execution failed: tool_name='%s', handler_type='sql', query_type='%s', query_preview='%s', query_length=%d, database='%s', error=%w",
			tool.Name, queryType, queryPreview, len(query), connInfo, err)
	}
	defer rows.Close()

	// Convert results to JSON
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

	return string(jsonResult), nil
}

func (t *SQLTool) Validate(args map[string]interface{}, schema map[string]interface{}) error {
	return ValidateArgs(args, schema)
}

