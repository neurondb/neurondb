package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/neurondb/NeuronMCP/internal/config"
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
			"Load HuggingFace dataset into database using Python scripts",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"dataset_name": map[string]interface{}{
						"type":        "string",
						"description": "HuggingFace dataset name (e.g., 'sentence-transformers/embedding-training-data', 'msmarco', 'wikipedia')",
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
					"limit": map[string]interface{}{
						"type":        "number",
						"default":     1000,
						"description": "Maximum number of rows to load",
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
	limit := 1000
	if l, ok := params["limit"].(float64); ok {
		limit = int(l)
	}

	if datasetName == "" {
		return Error("dataset_name is required and cannot be empty", "VALIDATION_ERROR", nil), nil
	}

	// For all datasets, use inline Python code (works in Docker)
	// This avoids needing external script files
	return t.loadGenericDataset(ctx, datasetName, split, limit)
}

// findDatasetScript finds the dataset loading Python script
func (t *DatasetLoadingTool) findDatasetScript() string {
	possiblePaths := []string{
		"NeuronDB/dataset/gen_dataset_enhanced.py",
		"../NeuronDB/dataset/gen_dataset_enhanced.py",
		"../../NeuronDB/dataset/gen_dataset_enhanced.py",
		"/Users/ibrarahmed/pgelephant/pge/neurondb/NeuronDB/dataset/gen_dataset_enhanced.py",
	}

	// Try relative to current working directory
	cwd, _ := os.Getwd()
	for dir := cwd; dir != "/"; dir = filepath.Dir(dir) {
		testPath := filepath.Join(dir, "NeuronDB", "dataset", "gen_dataset_enhanced.py")
		if _, err := os.Stat(testPath); err == nil {
			return testPath
		}
	}

	// Try predefined paths
	for _, path := range possiblePaths {
		if absPath, err := filepath.Abs(path); err == nil {
			if _, err := os.Stat(absPath); err == nil {
				return absPath
			}
		}
	}

	return ""
}

// buildScriptArgs builds command line arguments for the dataset script
func (t *DatasetLoadingTool) buildScriptArgs(datasetName string, limit int) []string {
	scriptPath := t.findDatasetScript()
	if scriptPath == "" {
		return nil
	}

	datasetLower := strings.ToLower(datasetName)
	var args []string

	if strings.Contains(datasetLower, "msmarco") || strings.Contains(datasetLower, "ms-marco") {
		args = []string{scriptPath, "--load-msmarco", "--dbname", "neurondb", "--limit", fmt.Sprintf("%d", limit)}
	} else if strings.Contains(datasetLower, "wikipedia") {
		args = []string{scriptPath, "--load-wikipedia", "--dbname", "neurondb", "--limit", fmt.Sprintf("%d", limit)}
	} else if strings.Contains(datasetLower, "hotpot") {
		args = []string{scriptPath, "--load-hotpotqa", "--dbname", "neurondb", "--limit", fmt.Sprintf("%d", limit)}
	} else if strings.Contains(datasetLower, "sift") {
		args = []string{scriptPath, "--load-sift", "--dbname", "neurondb", "--limit", fmt.Sprintf("%d", limit)}
	} else if strings.Contains(datasetLower, "deep") {
		args = []string{scriptPath, "--load-deep1b", "--dbname", "neurondb", "--limit", fmt.Sprintf("%d", limit)}
	}

	return args
}

// scriptExists checks if the script file exists
func (t *DatasetLoadingTool) scriptExists(scriptPath string) bool {
	_, err := os.Stat(scriptPath)
	return err == nil
}

// loadGenericDataset loads a generic HuggingFace dataset using inline Python
func (t *DatasetLoadingTool) loadGenericDataset(ctx context.Context, datasetName, split string, limit int) (*ToolResult, error) {
	// Create Python code to load the dataset
	pythonCode := fmt.Sprintf(`
import os
import sys
import json

try:
    from datasets import load_dataset
    import psycopg2
    from psycopg2 import sql
    
    # Set cache directory to /tmp to avoid permission issues
    os.environ['HF_HOME'] = '/tmp/hf_cache'
    os.environ['HF_DATASETS_CACHE'] = '/tmp/hf_cache/datasets'
    os.makedirs('/tmp/hf_cache', exist_ok=True)
    os.makedirs('/tmp/hf_cache/datasets', exist_ok=True)
    
    # Database connection
    conn = psycopg2.connect(
        host=os.getenv('PGHOST', 'localhost'),
        port=int(os.getenv('PGPORT', '5432')),
        user=os.getenv('PGUSER', 'postgres'),
        password=os.getenv('PGPASSWORD', ''),
        database=os.getenv('PGDATABASE', 'postgres')
    )
    
    # Load dataset - handle datasets with inconsistent structure
    dataset_name = '%s'
    split_name = '%s'
    limit_val = %[3]d
    
    try:
        dataset = load_dataset(dataset_name, split=split_name, streaming=True)
    except Exception as load_err:
        # Try without streaming if streaming fails
        try:
            dataset = load_dataset(dataset_name, split=split_name, streaming=False)
            # Convert to iterable
            dataset = iter(dataset)
        except Exception as e2:
            print(json.dumps({"error": f"Failed to load dataset: {str(load_err)}. Also tried non-streaming: {str(e2)}", "status": "error"}))
            sys.exit(1)
    
    # Create schema and table
    schema_name = 'datasets'
    table_name = dataset_name.replace('/', '_').replace('-', '_')
    
    with conn.cursor() as cur:
        # Use psycopg2's quote_ident for safe identifier quoting
        from psycopg2.extensions import quote_ident
        schema_quoted = quote_ident(schema_name, cur)
        table_quoted = quote_ident(table_name, cur)
        cur.execute("CREATE SCHEMA IF NOT EXISTS " + schema_quoted)
        create_table_query = """
            CREATE TABLE IF NOT EXISTS {}.{} (
                id SERIAL PRIMARY KEY,
                data JSONB
            )
        """.format(schema_quoted, table_quoted)
        cur.execute(create_table_query)
        conn.commit()
        
        # Insert data - iterate streaming dataset correctly
        # Handle datasets with inconsistent structure
        inserted = 0
        dataset_iter = iter(dataset)
        errors = 0
        max_errors = 10
        limit = limit_val
        
        while inserted < limit and errors < max_errors:
            try:
                example = next(dataset_iter)
                # Convert to dict - datasets library returns dict-like objects
                # Handle both dict and Example objects from datasets library
                if isinstance(example, dict):
                    example_dict = example
                elif hasattr(example, 'keys') and callable(getattr(example, 'keys', None)):
                    example_dict = {k: example[k] for k in example.keys()}
                elif hasattr(example, '__dict__'):
                    example_dict = dict(example)
                else:
                    # Fallback: try to convert to dict
                    try:
                        example_dict = dict(example)
                    except:
                        example_dict = {'raw': str(example)}
                
                # Try to serialize and insert
                try:
                    json_str = json.dumps(example_dict, default=str, ensure_ascii=False)
                    # Use psycopg2's quote_ident for safe identifier quoting
                    from psycopg2.extensions import quote_ident
                    schema_quoted = quote_ident(schema_name, cur)
                    table_quoted = quote_ident(table_name, cur)
                    insert_query = "INSERT INTO {}.{} (data) VALUES (%%s::jsonb)".format(schema_quoted, table_quoted)
                    cur.execute(insert_query, (json_str,))
                    inserted += 1
                    errors = 0  # Reset error count on success
                    if inserted %% 100 == 0:
                        conn.commit()
                except Exception as insert_err:
                    errors += 1
                    if errors <= 3:  # Log first few errors for debugging
                        # Use string concatenation to avoid format string issues
                        err_msg = "Insert error (item " + str(inserted) + "): " + str(insert_err)
                        print(err_msg, file=sys.stderr)
                    if errors >= max_errors:
                        error_msg = "Too many insert errors (" + str(errors) + "), stopping"
                        print(json.dumps({"error": error_msg, "status": "error"}))
                        break
                    continue  # Skip this item
            except StopIteration:
                break
            except Exception as e:
                errors += 1
                if errors <= 3:  # Log first few errors
                    err_msg = "Iteration error: " + str(e)
                    print(err_msg, file=sys.stderr)
                if errors >= max_errors:
                    error_msg = "Too many iteration errors (" + str(errors) + "), stopping"
                    print(json.dumps({"error": error_msg, "status": "error"}))
                    break
                continue
        conn.commit()
        
    print(json.dumps({"rows_loaded": inserted, "table": f"{schema_name}.{table_name}", "status": "success"}))
    conn.close()
    
except ImportError as e:
    print(json.dumps({"error": f"datasets library not available: {e}", "status": "error"}))
    sys.exit(1)
except Exception as e:
    print(json.dumps({"error": str(e), "status": "error"}))
    sys.exit(1)
`, datasetName, split, limit)

	// Set up environment
	cfgMgr := config.NewConfigManager()
	cfgMgr.Load("")
	dbCfg := cfgMgr.GetDatabaseConfig()

	env := os.Environ()
	env = append(env, fmt.Sprintf("PGHOST=%s", dbCfg.GetHost()))
	env = append(env, fmt.Sprintf("PGPORT=%d", dbCfg.GetPort()))
	env = append(env, fmt.Sprintf("PGUSER=%s", dbCfg.GetUser()))
	env = append(env, fmt.Sprintf("PGDATABASE=%s", dbCfg.GetDatabase()))
	env = append(env, "HF_HOME=/tmp/hf_cache")
	env = append(env, "HF_DATASETS_CACHE=/tmp/hf_cache/datasets")
	env = append(env, "HOME=/tmp")
	if pwd := dbCfg.Password; pwd != nil && *pwd != "" {
		env = append(env, fmt.Sprintf("PGPASSWORD=%s", *pwd))
	}

	// Execute Python
	cmd := exec.CommandContext(ctx, "python3", "-c", pythonCode)
	cmd.Env = env
	// CombinedOutput() captures both stdout and stderr

	output, err := cmd.CombinedOutput()
	if err != nil {
		t.logger.Error("Generic dataset loading failed", err, map[string]interface{}{
			"dataset_name": datasetName,
			"output":       string(output),
		})
		return Error(
			fmt.Sprintf("Failed to load dataset '%s': %v. Output: %s", datasetName, err, string(output)),
			"EXECUTION_ERROR",
			map[string]interface{}{
				"dataset_name": datasetName,
				"error":        err.Error(),
				"output":       string(output),
			},
		), nil
	}

	// Parse JSON output
	outputStr := strings.TrimSpace(string(output))
	rowsLoaded := 0
	tableName := ""

	// Try to extract JSON from output
	if strings.Contains(outputStr, "{") {
		jsonStart := strings.Index(outputStr, "{")
		jsonEnd := strings.LastIndex(outputStr, "}") + 1
		if jsonEnd > jsonStart {
			jsonStr := outputStr[jsonStart:jsonEnd]
			var result map[string]interface{}
			if err := json.Unmarshal([]byte(jsonStr), &result); err == nil {
				if r, ok := result["rows_loaded"].(float64); ok {
					rowsLoaded = int(r)
				}
				if t, ok := result["table"].(string); ok {
					tableName = t
				}
			}
		}
	}

	return Success(map[string]interface{}{
		"dataset":     datasetName,
		"split":       split,
		"rows_loaded": rowsLoaded,
		"table":       tableName,
		"status":      "completed",
		"message":     fmt.Sprintf("Dataset '%s' loaded successfully into %s", datasetName, tableName),
	}, map[string]interface{}{
		"dataset": datasetName,
		"split":   split,
		"method":  "inline_python",
	}), nil
}

// extractRowsLoaded extracts the number of rows loaded from script output
func (t *DatasetLoadingTool) extractRowsLoaded(output string) int {
	// Look for patterns like "loaded: 1000", "inserted: 500", etc.
	lines := strings.Split(output, "\n")
	for _, line := range lines {
		lower := strings.ToLower(line)
		if strings.Contains(lower, "loaded") || strings.Contains(lower, "inserted") {
			// Try to extract number
			words := strings.Fields(line)
			for i, word := range words {
				if (strings.Contains(word, "loaded") || strings.Contains(word, "inserted")) && i+1 < len(words) {
					// Next word might be the number
					var num int
					if _, err := fmt.Sscanf(words[i+1], "%d", &num); err == nil {
						return num
					}
				}
			}
		}
	}
	return 0
}
