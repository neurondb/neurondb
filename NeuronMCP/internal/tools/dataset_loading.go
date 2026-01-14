/*-------------------------------------------------------------------------
 *
 * dataset_loading.go
 *    Tool implementation for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/dataset_loading.go
 *
 *-------------------------------------------------------------------------
 */

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
	"github.com/neurondb/NeuronMCP/internal/validation"
)

/* DatasetLoadingTool loads HuggingFace datasets */
type DatasetLoadingTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

/* NewDatasetLoadingTool creates a new dataset loading tool */
func NewDatasetLoadingTool(db *database.Database, logger *logging.Logger) *DatasetLoadingTool {
	return &DatasetLoadingTool{
		BaseTool: NewBaseTool(
			"postgresql_load_dataset",
			"Load datasets from multiple sources (HuggingFace, URLs, GitHub, S3, local files) into PostgreSQL with automatic schema detection, embedding generation, and index creation",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"source_type": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"huggingface", "url", "github", "s3", "local"},
						"default":     "huggingface",
						"description": "Data source type: 'huggingface' for HuggingFace datasets, 'url' for direct URLs, 'github' for GitHub repos, 's3' for S3 buckets, 'local' for local files",
					},
					"source_path": map[string]interface{}{
						"type":        "string",
						"description": "Dataset identifier: HuggingFace dataset name (e.g., 'sentence-transformers/embedding-training-data'), URL, GitHub repo path, S3 path (s3://bucket/key), or local file path",
					},
					"dataset_name": map[string]interface{}{
						"type":        "string",
						"description": "Deprecated: Use source_path instead. HuggingFace dataset name for backward compatibility",
					},
					"split": map[string]interface{}{
						"type":        "string",
						"default":     "train",
						"description": "Dataset split for HuggingFace datasets (train, test, validation)",
					},
					"config": map[string]interface{}{
						"type":        "string",
						"description": "Dataset configuration name for HuggingFace datasets (optional)",
					},
					"auto_embed": map[string]interface{}{
						"type":        "boolean",
						"default":     true,
						"description": "Automatically detect text columns and generate vector embeddings using NeuronDB",
					},
					"embedding_model": map[string]interface{}{
						"type":        "string",
						"default":     "default",
						"description": "Embedding model name to use for generating embeddings (default: 'default')",
					},
					"text_columns": map[string]interface{}{
						"type":        "array",
						"items":        map[string]interface{}{"type": "string"},
						"description": "Optional list of column names to embed. If not provided, text columns are auto-detected",
					},
					"table_name": map[string]interface{}{
						"type":        "string",
						"description": "Custom table name (optional, auto-generated from dataset name if not provided)",
					},
					"schema_name": map[string]interface{}{
						"type":        "string",
						"default":     "datasets",
						"description": "PostgreSQL schema name (default: 'datasets')",
					},
					"batch_size": map[string]interface{}{
						"type":        "number",
						"default":     1000,
						"description": "Batch size for loading data (default: 1000)",
					},
					"create_indexes": map[string]interface{}{
						"type":        "boolean",
						"default":     true,
						"description": "Automatically create indexes: HNSW for vector columns, GIN for full-text, B-tree for numeric (default: true)",
					},
					"format": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"csv", "json", "jsonl", "parquet", "auto"},
						"default":     "auto",
						"description": "File format hint for URL/local sources: 'csv', 'json', 'jsonl', 'parquet', or 'auto' for auto-detection",
					},
					"limit": map[string]interface{}{
						"type":        "number",
						"default":     10000,
						"description": "Maximum number of rows to load (default: 10000, 0 for unlimited)",
					},
					"streaming": map[string]interface{}{
						"type":        "boolean",
						"default":     true,
						"description": "Enable streaming mode for large datasets (default: true)",
					},
					"cache_dir": map[string]interface{}{
						"type":        "string",
						"description": "Cache directory path for downloads (optional, defaults to /tmp/hf_cache)",
					},
				},
				"required": []interface{}{"source_path"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

/* Execute executes the dataset loading */
func (t *DatasetLoadingTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for neurondb_load_dataset tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	/* Get source path (preferred) or dataset_name (backward compatibility) */
	sourcePath, _ := params["source_path"].(string)
	datasetName, _ := params["dataset_name"].(string)
	/* Support both source_path and dataset_name for backward compatibility */
	if sourcePath == "" && datasetName != "" {
		sourcePath = datasetName
	}
	
	/* Validate source_path is required */
	if err := validation.ValidateRequired(sourcePath, "source_path"); err != nil {
		return Error(fmt.Sprintf("Invalid source_path parameter: %v", err), "VALIDATION_ERROR", map[string]interface{}{
			"error":  err.Error(),
			"params": params,
		}), nil
	}
	
	/* Validate source_path length */
	if err := validation.ValidateMaxLength(sourcePath, "source_path", 2048); err != nil {
		return Error(fmt.Sprintf("Invalid source_path parameter: %v", err), "VALIDATION_ERROR", map[string]interface{}{
			"error":  err.Error(),
			"params": params,
		}), nil
	}

	/* Get source type */
	sourceType := "huggingface"
	if st, ok := params["source_type"].(string); ok && st != "" {
		sourceType = st
	}

	/* Get optional parameters */
	split := "train"
	if s, ok := params["split"].(string); ok && s != "" {
		split = s
	}
	limit := 10000
	if l, ok := params["limit"].(float64); ok {
		limit = int(l)
	}
	batchSize := 1000
	if bs, ok := params["batch_size"].(float64); ok {
		batchSize = int(bs)
	}
	autoEmbed := true
	if ae, ok := params["auto_embed"].(bool); ok {
		autoEmbed = ae
	}
	embeddingModel := "default"
	if em, ok := params["embedding_model"].(string); ok && em != "" {
		embeddingModel = em
	}
	schemaName := "datasets"
	if sn, ok := params["schema_name"].(string); ok && sn != "" {
		schemaName = sn
	}
	tableName := ""
	if tn, ok := params["table_name"].(string); ok && tn != "" {
		tableName = tn
	}
	createIndexes := true
	if ci, ok := params["create_indexes"].(bool); ok {
		createIndexes = ci
	}
	format := "auto"
	if f, ok := params["format"].(string); ok && f != "" {
		format = f
	}
	streaming := true
	if s, ok := params["streaming"].(bool); ok {
		streaming = s
	}

	/* Get text columns if specified */
	var textColumns []string
	if tc, ok := params["text_columns"].([]interface{}); ok {
		textColumns = make([]string, 0, len(tc))
		for _, col := range tc {
			if colStr, ok := col.(string); ok {
				textColumns = append(textColumns, colStr)
			}
		}
	}

	return t.loadDataset(ctx, sourceType, sourcePath, split, limit, batchSize, autoEmbed,
		embeddingModel, schemaName, tableName, createIndexes, format, streaming, textColumns)
}


/* findDatasetLoaderScript finds the dataset loader Python script */
func (t *DatasetLoadingTool) findDatasetLoaderScript() string {
	/* Try to find the dataset_loader.py script */
	possiblePaths := []string{
		"internal/tools/dataset_loader.py",
		"NeuronMCP/internal/tools/dataset_loader.py",
		"../internal/tools/dataset_loader.py",
		"../../internal/tools/dataset_loader.py",
	}

	/* Try relative to current working directory */
	cwd, _ := os.Getwd()
	for dir := cwd; dir != "/"; dir = filepath.Dir(dir) {
		testPath := filepath.Join(dir, "NeuronMCP", "internal", "tools", "dataset_loader.py")
		if _, err := os.Stat(testPath); err == nil {
			return testPath
		}
		testPath = filepath.Join(dir, "internal", "tools", "dataset_loader.py")
		if _, err := os.Stat(testPath); err == nil {
			return testPath
		}
	}

	/* Try predefined paths */
	for _, path := range possiblePaths {
		if absPath, err := filepath.Abs(path); err == nil {
			if _, err := os.Stat(absPath); err == nil {
				return absPath
			}
		}
	}

	return ""
}

/* loadDataset loads dataset using the comprehensive Python loader */
func (t *DatasetLoadingTool) loadDataset(ctx context.Context, sourceType, sourcePath, split string,
	limit, batchSize int, autoEmbed bool, embeddingModel, schemaName, tableName string,
	createIndexes bool, format string, streaming bool, textColumns []string) (*ToolResult, error) {
	/* Find the Python loader script */
	scriptPath := t.findDatasetLoaderScript()
	if scriptPath == "" {
		/* Log warning about script not found */
		t.logger.Warn("Dataset loader script not found, using fallback method", map[string]interface{}{
			"source_type": sourceType,
			"source_path": sourcePath,
			"hint":        "Please ensure dataset_loader.py is available in NeuronMCP/internal/tools/",
		})
		/* Fallback: try to use inline Python code if script not found */
		return t.loadGenericDatasetFallback(ctx, sourceType, sourcePath, split, limit)
	}

	/* Log that we found the script */
	t.logger.Info("Using dataset loader script", map[string]interface{}{
		"script_path": scriptPath,
		"source_type": sourceType,
		"source_path": sourcePath,
	})

	/* Build command arguments */
	args := []string{scriptPath}
	args = append(args, "--source-type", sourceType)
	args = append(args, "--source-path", sourcePath)
	
	if sourceType == "huggingface" {
		args = append(args, "--split", split)
	}
	if limit > 0 {
		args = append(args, "--limit", fmt.Sprintf("%d", limit))
	}
	args = append(args, "--batch-size", fmt.Sprintf("%d", batchSize))
	args = append(args, "--schema-name", schemaName)
	if tableName != "" {
		args = append(args, "--table-name", tableName)
	}
	if autoEmbed {
		args = append(args, "--auto-embed")
	} else {
		args = append(args, "--no-auto-embed")
	}
	args = append(args, "--embedding-model", embeddingModel)
	if createIndexes {
		args = append(args, "--create-indexes")
	} else {
		args = append(args, "--no-create-indexes")
	}
	if format != "auto" {
		args = append(args, "--format", format)
	}
	if streaming {
		args = append(args, "--streaming")
	}
	if len(textColumns) > 0 {
		args = append(args, "--text-columns")
		args = append(args, textColumns...)
	}

	/* Set up environment */
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

	/* Execute Python script */
	cmd := exec.CommandContext(ctx, "python3", args...)
	cmd.Env = env

	output, err := cmd.CombinedOutput()
	if err != nil {
		t.logger.Error("Dataset loading failed", err, map[string]interface{}{
			"source_type": sourceType,
			"source_path": sourcePath,
			"output":      string(output),
		})
		return Error(
			fmt.Sprintf("Failed to load dataset from %s '%s': %v. Output: %s", sourceType, sourcePath, err, string(output)),
			"EXECUTION_ERROR",
			map[string]interface{}{
				"source_type": sourceType,
				"source_path": sourcePath,
				"error":       err.Error(),
				"output":      string(output),
			},
		), nil
	}

	/* Parse JSON output - look for the final result */
	outputStr := strings.TrimSpace(string(output))
	lines := strings.Split(outputStr, "\n")
	
	var finalResult map[string]interface{}
	var rowsLoaded int
	var rowsEmbedded int
	var resultTable string
	var textColumnsResult []interface{}
	var embeddingColumnsResult []interface{}
	var indexesCreated int

	/* Find the last JSON object (final result) */
	for i := len(lines) - 1; i >= 0; i-- {
		line := strings.TrimSpace(lines[i])
		if strings.HasPrefix(line, "{") && strings.HasSuffix(line, "}") {
			var result map[string]interface{}
			if err := json.Unmarshal([]byte(line), &result); err == nil {
				if status, ok := result["status"].(string); ok && status == "success" {
					finalResult = result
					break
				}
			}
		}
	}

	if finalResult != nil {
		if r, ok := finalResult["rows_loaded"].(float64); ok {
			rowsLoaded = int(r)
		}
		if r, ok := finalResult["rows_embedded"].(float64); ok {
			rowsEmbedded = int(r)
		}
		if t, ok := finalResult["table"].(string); ok {
			resultTable = t
		}
		if tc, ok := finalResult["text_columns"].([]interface{}); ok {
			textColumnsResult = tc
		}
		if ec, ok := finalResult["embedding_columns"].([]interface{}); ok {
			embeddingColumnsResult = ec
		}
		if ic, ok := finalResult["indexes_created"].(float64); ok {
			indexesCreated = int(ic)
		}
	} else {
		/* Fallback: try to extract from any JSON in output */
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
						resultTable = t
					}
				}
			}
		}
	}

	return Success(map[string]interface{}{
		"source_type":       sourceType,
		"source_path":       sourcePath,
		"rows_loaded":       rowsLoaded,
		"rows_embedded":     rowsEmbedded,
		"table":             resultTable,
		"text_columns":      textColumnsResult,
		"embedding_columns": embeddingColumnsResult,
		"indexes_created":   indexesCreated,
		"status":            "completed",
		"message":           fmt.Sprintf("Dataset from %s '%s' loaded successfully into %s", sourceType, sourcePath, resultTable),
	}, map[string]interface{}{
		"source_type": sourceType,
		"source_path": sourcePath,
		"method":      "comprehensive_loader",
		"auto_embed":  autoEmbed,
	}), nil
}

/* loadGenericDatasetFallback loads dataset using inline Python (fallback if script not found) */
func (t *DatasetLoadingTool) loadGenericDatasetFallback(ctx context.Context, sourceType, sourcePath, split string, limit int) (*ToolResult, error) {
	/* This is a simplified fallback for backward compatibility */
	/* Only supports HuggingFace for now */
	if sourceType != "huggingface" {
		return Error(
			fmt.Sprintf("Comprehensive loader script not found. Fallback only supports HuggingFace, but got: %s", sourceType),
			"SCRIPT_NOT_FOUND",
			map[string]interface{}{
				"source_type": sourceType,
				"hint":        "Please ensure dataset_loader.py is available in NeuronMCP/internal/tools/",
			},
		), nil
	}

	/* Use the old inline Python approach for HuggingFace */
	pythonCode := fmt.Sprintf(`
import os
import sys
import json

try:
    from datasets import load_dataset
    import psycopg2
    from psycopg2 import sql
    
    os.environ['HF_HOME'] = '/tmp/hf_cache'
    os.environ['HF_DATASETS_CACHE'] = '/tmp/hf_cache/datasets'
    os.makedirs('/tmp/hf_cache', exist_ok=True)
    os.makedirs('/tmp/hf_cache/datasets', exist_ok=True)
    
    conn = psycopg2.connect(
        host=os.getenv('PGHOST', 'localhost'),
        port=int(os.getenv('PGPORT', '5432')),
        user=os.getenv('PGUSER', 'postgres'),
        password=os.getenv('PGPASSWORD', ''),
        database=os.getenv('PGDATABASE', 'postgres')
    )
    
    dataset_name = '%s'
    split_name = '%s'
    limit_val = %[3]d
    
    try:
        dataset = load_dataset(dataset_name, split=split_name, streaming=True)
    except Exception as load_err:
        try:
            dataset = load_dataset(dataset_name, split=split_name, streaming=False)
            dataset = iter(dataset)
        except Exception as e2:
            print(json.dumps({"error": f"Failed to load dataset: {str(load_err)}. Also tried non-streaming: {str(e2)}", "status": "error"}))
            sys.exit(1)
    
    schema_name = 'datasets'
    table_name = dataset_name.replace('/', '_').replace('-', '_')
    
    with conn.cursor() as cur:
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
        
        inserted = 0
        dataset_iter = iter(dataset)
        errors = 0
        max_errors = 10
        limit = limit_val
        
        while inserted < limit and errors < max_errors:
            try:
                example = next(dataset_iter)
                if isinstance(example, dict):
                    example_dict = example
                elif hasattr(example, 'keys') and callable(getattr(example, 'keys', None)):
                    example_dict = {k: example[k] for k in example.keys()}
                else:
                    try:
                        example_dict = dict(example)
                    except:
                        example_dict = {'raw': str(example)}
                
                try:
                    json_str = json.dumps(example_dict, default=str, ensure_ascii=False)
                    from psycopg2.extensions import quote_ident
                    schema_quoted = quote_ident(schema_name, cur)
                    table_quoted = quote_ident(table_name, cur)
                    insert_query = "INSERT INTO {}.{} (data) VALUES (%%s::jsonb)".format(schema_quoted, table_quoted)
                    cur.execute(insert_query, (json_str,))
                    inserted += 1
                    errors = 0
                    if inserted %% 100 == 0:
                        conn.commit()
                except Exception as insert_err:
                    errors += 1
                    if errors >= max_errors:
                        error_msg = "Too many insert errors (" + str(errors) + "), stopping"
                        print(json.dumps({"error": error_msg, "status": "error"}))
                        break
                    continue
            except StopIteration:
                break
            except Exception as e:
                errors += 1
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
`, sourcePath, split, limit)

	/* Set up environment */
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

	cmd := exec.CommandContext(ctx, "python3", "-c", pythonCode)
	cmd.Env = env

	output, err := cmd.CombinedOutput()
	if err != nil {
		t.logger.Error("Fallback dataset loading failed", err, map[string]interface{}{
			"source_path": sourcePath,
			"output":      string(output),
		})
		return Error(
			fmt.Sprintf("Failed to load dataset '%s': %v. Output: %s", sourcePath, err, string(output)),
			"EXECUTION_ERROR",
			map[string]interface{}{
				"source_path": sourcePath,
				"error":       err.Error(),
				"output":      string(output),
			},
		), nil
	}

	/* Parse JSON output */
	outputStr := strings.TrimSpace(string(output))
	rowsLoaded := 0
	tableName := ""

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
		"source_type":   sourceType,
		"source_path":   sourcePath,
		"rows_loaded":   rowsLoaded,
		"table":         tableName,
		"status":        "completed",
		"message":       fmt.Sprintf("Dataset '%s' loaded successfully into %s (using fallback method)", sourcePath, tableName),
		"method":        "fallback",
	}, map[string]interface{}{
		"source_type": sourceType,
		"source_path": sourcePath,
		"method":      "fallback",
	}), nil
}





