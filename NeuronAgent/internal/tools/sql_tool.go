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
		return "", fmt.Errorf("query parameter is required and must be a string")
	}

	// Security: Only allow SELECT, EXPLAIN, and schema introspection queries
	queryUpper := strings.TrimSpace(strings.ToUpper(query))
	if !strings.HasPrefix(queryUpper, "SELECT") &&
		!strings.HasPrefix(queryUpper, "EXPLAIN") &&
		!strings.HasPrefix(queryUpper, "SHOW") &&
		!strings.HasPrefix(queryUpper, "DESCRIBE") &&
		!strings.HasPrefix(queryUpper, "\\d") {
		return "", fmt.Errorf("only SELECT, EXPLAIN, SHOW, and DESCRIBE queries are allowed")
	}

	// Check for dangerous keywords
	dangerous := []string{"DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE"}
	for _, keyword := range dangerous {
		if strings.Contains(queryUpper, keyword) {
			return "", fmt.Errorf("query contains forbidden keyword: %s", keyword)
		}
	}

	// Execute query (read-only)
	if t.db == nil {
		return "", fmt.Errorf("database connection not initialized")
	}
	rows, err := t.db.QueryContext(ctx, query)
	if err != nil {
		return "", fmt.Errorf("query execution failed: %w", err)
	}
	defer rows.Close()

	// Convert results to JSON
	var results []map[string]interface{}
	columns, err := rows.Columns()
	if err != nil {
		return "", fmt.Errorf("failed to get columns: %w", err)
	}

	for rows.Next() {
		values := make([]interface{}, len(columns))
		valuePtrs := make([]interface{}, len(columns))
		for i := range values {
			valuePtrs[i] = &values[i]
		}

		if err := rows.Scan(valuePtrs...); err != nil {
			return "", fmt.Errorf("failed to scan row: %w", err)
		}

		row := make(map[string]interface{})
		for i, col := range columns {
			row[col] = values[i]
		}
		results = append(results, row)
	}

	if err := rows.Err(); err != nil {
		return "", fmt.Errorf("row iteration error: %w", err)
	}

	jsonResult, err := json.Marshal(results)
	if err != nil {
		return "", fmt.Errorf("failed to marshal results: %w", err)
	}

	return string(jsonResult), nil
}

func (t *SQLTool) Validate(args map[string]interface{}, schema map[string]interface{}) error {
	return ValidateArgs(args, schema)
}

