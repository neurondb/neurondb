package tools

import (
	"context"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

// TopicDiscoveryTool performs topic modeling and discovery
type TopicDiscoveryTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewTopicDiscoveryTool creates a new topic discovery tool
func NewTopicDiscoveryTool(db *database.Database, logger *logging.Logger) *TopicDiscoveryTool {
	return &TopicDiscoveryTool{
		BaseTool: NewBaseTool(
			"topic_discovery",
			"Perform topic modeling and discovery on text data",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table": map[string]interface{}{
						"type":        "string",
						"description": "Table name with text data",
					},
					"text_column": map[string]interface{}{
						"type":        "string",
						"description": "Text column name",
					},
					"num_topics": map[string]interface{}{
						"type":        "number",
						"default":     10,
						"minimum":     2,
						"maximum":     100,
						"description": "Number of topics to discover",
					},
					"algorithm": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"lda", "nmf", "bertopic"},
						"default":     "lda",
						"description": "Topic modeling algorithm",
					},
				},
				"required": []interface{}{"table", "text_column"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes topic discovery
func (t *TopicDiscoveryTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for topic_discovery tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	table, _ := params["table"].(string)
	textColumn, _ := params["text_column"].(string)
	numTopics := 10
	if n, ok := params["num_topics"].(float64); ok {
		numTopics = int(n)
	}
	algorithm := "lda"
	if a, ok := params["algorithm"].(string); ok {
		algorithm = a
	}

	if table == "" || textColumn == "" {
		return Error("table and text_column are required", "VALIDATION_ERROR", nil), nil
	}

	// Use NeuronDB topic discovery function
	query := "SELECT * FROM discover_topics($1::text, $2::text, $3::int, $4::text)"
	queryParams := []interface{}{table, textColumn, numTopics, algorithm}

	results, err := t.executor.ExecuteQuery(ctx, query, queryParams)
	if err != nil {
		t.logger.Error("Topic discovery failed", err, params)
		return Error(fmt.Sprintf("Topic discovery failed: error=%v", err), "EXECUTION_ERROR", map[string]interface{}{
			"error": err.Error(),
		}), nil
	}

	return Success(map[string]interface{}{
		"topics": results,
		"count":  len(results),
	}, map[string]interface{}{
		"num_topics": numTopics,
		"algorithm":  algorithm,
		"count":      len(results),
	}), nil
}

