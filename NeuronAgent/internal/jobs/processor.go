package jobs

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os/exec"
	"strings"
	"time"

	"github.com/neurondb/NeuronAgent/internal/db"
)

type Processor struct {
	httpClient *http.Client
	db         *db.DB
}

func NewProcessor(database *db.DB) *Processor {
	return &Processor{
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
		db: database,
	}
}

func (p *Processor) Process(ctx context.Context, job *db.Job) (map[string]interface{}, error) {
	switch job.Type {
	case "http_call":
		return p.processHTTPCall(ctx, job)
	case "sql_task":
		return p.processSQLTask(ctx, job)
	case "shell_task":
		return p.processShellTask(ctx, job)
	default:
		return nil, fmt.Errorf("unknown job type: %s", job.Type)
	}
}

func (p *Processor) processHTTPCall(ctx context.Context, job *db.Job) (map[string]interface{}, error) {
	url, ok := job.Payload["url"].(string)
	if !ok || url == "" {
		return nil, fmt.Errorf("url is required")
	}

	method, _ := job.Payload["method"].(string)
	if method == "" {
		method = "GET"
	}

	var body io.Reader
	if bodyStr, ok := job.Payload["body"].(string); ok && bodyStr != "" {
		body = strings.NewReader(bodyStr)
	}

	req, err := http.NewRequestWithContext(ctx, method, url, body)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Add headers
	if headers, ok := job.Payload["headers"].(map[string]interface{}); ok {
		for k, v := range headers {
			if str, ok := v.(string); ok {
				req.Header.Set(k, str)
			}
		}
	}

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	// Read response body (limit to 1MB)
	bodyBytes, err := io.ReadAll(io.LimitReader(resp.Body, 1024*1024))
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	result := map[string]interface{}{
		"status_code": resp.StatusCode,
		"headers":     resp.Header,
		"body":        string(bodyBytes),
	}

	return result, nil
}

func (p *Processor) processSQLTask(ctx context.Context, job *db.Job) (map[string]interface{}, error) {
	if p.db == nil {
		return nil, fmt.Errorf("database connection not available")
	}

	query, ok := job.Payload["query"].(string)
	if !ok || query == "" {
		return nil, fmt.Errorf("query is required")
	}

	// Security: Only allow SELECT queries
	queryUpper := strings.TrimSpace(strings.ToUpper(query))
	if !strings.HasPrefix(queryUpper, "SELECT") {
		return nil, fmt.Errorf("only SELECT queries are allowed in background jobs")
	}

	// Check for dangerous keywords
	dangerous := []string{"DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE", "EXEC", "EXECUTE"}
	for _, keyword := range dangerous {
		if strings.Contains(queryUpper, keyword) {
			return nil, fmt.Errorf("query contains forbidden keyword: %s", keyword)
		}
	}

	// Execute query
	rows, err := p.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("query execution failed: %w", err)
	}
	defer rows.Close()

	// Convert results to JSON
	var results []map[string]interface{}
	columns, err := rows.Columns()
	if err != nil {
		return nil, fmt.Errorf("failed to get columns: %w", err)
	}

	for rows.Next() {
		values := make([]interface{}, len(columns))
		valuePtrs := make([]interface{}, len(columns))
		for i := range values {
			valuePtrs[i] = &values[i]
		}

		if err := rows.Scan(valuePtrs...); err != nil {
			return nil, fmt.Errorf("failed to scan row: %w", err)
		}

		row := make(map[string]interface{})
		for i, col := range columns {
			// Handle different types
			val := values[i]
			if val != nil {
				switch v := val.(type) {
				case []byte:
					row[col] = string(v)
				case time.Time:
					row[col] = v.Format(time.RFC3339)
				default:
					row[col] = val
				}
			} else {
				row[col] = nil
			}
		}
		results = append(results, row)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("row iteration error: %w", err)
	}

	return map[string]interface{}{
		"row_count": len(results),
		"rows":      results,
	}, nil
}

func (p *Processor) processShellTask(ctx context.Context, job *db.Job) (map[string]interface{}, error) {
	command, ok := job.Payload["command"].(string)
	if !ok || command == "" {
		return nil, fmt.Errorf("command is required")
	}

	// Security: Only allow whitelisted commands
	allowedCommands := []string{"ls", "pwd", "cat", "grep", "find", "head", "tail", "wc", "sort", "uniq", "echo", "date", "whoami"}
	
	parts := strings.Fields(command)
	if len(parts) == 0 {
		return nil, fmt.Errorf("empty command")
	}

	cmdName := parts[0]
	allowed := false
	for _, allowedCmd := range allowedCommands {
		if cmdName == allowedCmd {
			allowed = true
			break
		}
	}

	if !allowed {
		return nil, fmt.Errorf("command not allowed: %s", cmdName)
	}

	// Create context with timeout
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	// Execute command
	cmd := exec.CommandContext(ctx, cmdName, parts[1:]...)
	output, err := cmd.CombinedOutput()
	
	result := map[string]interface{}{
		"command":  command,
		"output":   string(output),
		"exit_code": 0,
	}

	if err != nil {
		if exitError, ok := err.(*exec.ExitError); ok {
			result["exit_code"] = exitError.ExitCode()
		}
		result["error"] = err.Error()
		return result, nil // Return result with error info, don't fail the job
	}

	return result, nil
}

