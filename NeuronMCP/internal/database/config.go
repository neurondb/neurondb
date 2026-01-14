/*-------------------------------------------------------------------------
 *
 * config.go
 *    Unified Configuration Helper for NeuronMCP
 *
 * Provides unified interface for retrieving all NeuronMCP configurations
 * from the database, including LLM models, indexes, workers, ML defaults,
 * tools, and system settings.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/database/config.go
 *
 *-------------------------------------------------------------------------
 */

package database

import (
	"context"
	"encoding/json"
	"fmt"
)

/* ConfigHelper provides unified configuration access */
type ConfigHelper struct {
	db *Database
}

/* NewConfigHelper creates a new configuration helper */
func NewConfigHelper(db *Database) *ConfigHelper {
	return &ConfigHelper{db: db}
}

/* LLMConfig represents LLM model configuration */
type LLMConfig struct {
	ModelID          int                    `json:"model_id"`
	ModelName        string                 `json:"model_name"`
	Provider         string                 `json:"provider"`
	ModelType        string                 `json:"model_type"`
	ContextWindow    *int                   `json:"context_window,omitempty"`
	EmbeddingDim     *int                   `json:"embedding_dimension,omitempty"`
	HasAPIKey        bool                   `json:"has_api_key"`
	APIKey           string                 `json:"api_key,omitempty"` // Only populated when explicitly requested
	BaseURL          *string                `json:"base_url,omitempty"`
	DefaultParams    map[string]interface{} `json:"default_params,omitempty"`
	RequestHeaders   map[string]interface{} `json:"request_headers,omitempty"`
	TimeoutMS        *int                   `json:"timeout_ms,omitempty"`
	RetryConfig      map[string]interface{} `json:"retry_config,omitempty"`
	RateLimitConfig  map[string]interface{} `json:"rate_limit_config,omitempty"`
}

/* IndexConfig represents vector index configuration */
type IndexConfig struct {
	TableName         string  `json:"table_name"`
	VectorColumn      string  `json:"vector_column"`
	IndexType         string  `json:"index_type"`
	HNSWM             *int    `json:"hnsw_m,omitempty"`
	HNSWEFConstruction *int   `json:"hnsw_ef_construction,omitempty"`
	HNSWEFSearch      *int    `json:"hnsw_ef_search,omitempty"`
	IVFLists          *int    `json:"ivf_lists,omitempty"`
	IVFProbes         *int    `json:"ivf_probes,omitempty"`
	DistanceMetric    string  `json:"distance_metric"`
}

/* WorkerConfig represents worker configuration */
type WorkerConfig struct {
	WorkerName string                 `json:"worker_name"`
	Enabled    bool                   `json:"enabled"`
	NaptimeMS  int                    `json:"naptime_ms"`
	Config     map[string]interface{} `json:"config"`
}

/* MLDefaults represents ML algorithm defaults */
type MLDefaults struct {
	Algorithm          string                 `json:"algorithm"`
	Hyperparameters    map[string]interface{} `json:"hyperparameters"`
	TrainingSettings   map[string]interface{} `json:"training_settings"`
	UseGPU             bool                   `json:"use_gpu"`
	GPUDevice          int                    `json:"gpu_device"`
	BatchSize          *int                   `json:"batch_size,omitempty"`
	MaxIterations      *int                   `json:"max_iterations,omitempty"`
}

/* ToolConfig represents tool-specific configuration */
type ToolConfig struct {
	ToolName        string                 `json:"tool_name"`
	DefaultParams   map[string]interface{} `json:"default_params"`
	DefaultLimit    *int                   `json:"default_limit,omitempty"`
	DefaultTimeoutMS int                   `json:"default_timeout_ms"`
	Enabled         bool                   `json:"enabled"`
}

/* SystemConfig represents system-wide configuration */
type SystemConfig struct {
	Features      map[string]interface{} `json:"features"`
	DefaultTimeoutMS int                  `json:"default_timeout_ms"`
	RateLimiting  map[string]interface{} `json:"rate_limiting"`
	Caching       map[string]interface{} `json:"caching"`
}

/* GetLLMConfig retrieves complete LLM model configuration */
func (ch *ConfigHelper) GetLLMConfig(ctx context.Context, modelName string) (*LLMConfig, error) {
	if ch.db == nil {
		return nil, fmt.Errorf("database not initialized")
	}
	
	query := `SELECT neurondb_get_model_config($1)`
	
	var resultJSON []byte
	err := ch.db.QueryRow(ctx, query, modelName).Scan(&resultJSON)
	if err != nil {
		return nil, fmt.Errorf("failed to get LLM config for %s: %w", modelName, err)
	}
	
	var config LLMConfig
	if err := json.Unmarshal(resultJSON, &config); err != nil {
		return nil, fmt.Errorf("failed to unmarshal LLM config: %w", err)
	}
	
	return &config, nil
}

/* ResolveModelKey securely retrieves API key for a model */
func (ch *ConfigHelper) ResolveModelKey(ctx context.Context, modelName string) (string, error) {
	query := `SELECT neurondb_resolve_model_key($1)`
	
	var apiKey *string
	err := ch.db.QueryRow(ctx, query, modelName).Scan(&apiKey)
	if err != nil {
		return "", fmt.Errorf("failed to resolve model key for %s: %w", modelName, err)
	}
	
	if apiKey == nil {
		return "", fmt.Errorf("no API key configured for model %s", modelName)
	}
	
	return *apiKey, nil
}

/* GetDefaultModel gets default model for operation type */
func (ch *ConfigHelper) GetDefaultModel(ctx context.Context, modelType string) (string, error) {
	query := `SELECT neurondb_get_model_for_operation($1, NULL)`
	
	var modelName *string
	err := ch.db.QueryRow(ctx, query, modelType).Scan(&modelName)
	if err != nil {
		return "", fmt.Errorf("failed to get default model for type %s: %w", modelType, err)
	}
	
	if modelName == nil {
		return "", fmt.Errorf("no default model found for type %s", modelType)
	}
	
	return *modelName, nil
}

/* GetModelForOperation gets model for operation with optional preferred model */
func (ch *ConfigHelper) GetModelForOperation(ctx context.Context, operationType, preferredModel string) (string, error) {
	query := `SELECT neurondb_get_model_for_operation($1, $2)`
	
	var modelName *string
	err := ch.db.QueryRow(ctx, query, operationType, preferredModel).Scan(&modelName)
	if err != nil {
		return "", fmt.Errorf("failed to get model for operation %s: %w", operationType, err)
	}
	
	if modelName == nil {
		return "", fmt.Errorf("no model found for operation %s", operationType)
	}
	
	return *modelName, nil
}

/* LogModelUsage logs model usage metrics */
func (ch *ConfigHelper) LogModelUsage(ctx context.Context, modelName, operationType string, tokensInput, tokensOutput, latencyMS *int, success bool, errorMsg *string) error {
	query := `SELECT neurondb_log_model_usage($1, $2, $3, $4, $5, $6, $7)`
	
	_, err := ch.db.Exec(ctx, query, modelName, operationType, tokensInput, tokensOutput, latencyMS, success, errorMsg)
	if err != nil {
		return fmt.Errorf("failed to log model usage: %w", err)
	}
	
	return nil
}

/* GetIndexConfig retrieves index configuration for table/column */
func (ch *ConfigHelper) GetIndexConfig(ctx context.Context, tableName, vectorColumn string) (*IndexConfig, error) {
	query := `SELECT neurondb_get_index_config($1, $2)`
	
	var resultJSON []byte
	err := ch.db.QueryRow(ctx, query, tableName, vectorColumn).Scan(&resultJSON)
	if err != nil {
		/* Return nil if no config found (not an error) */
		return nil, nil
	}
	
	var config IndexConfig
	if err := json.Unmarshal(resultJSON, &config); err != nil {
		return nil, fmt.Errorf("failed to unmarshal index config: %w", err)
	}
	
	config.TableName = tableName
	config.VectorColumn = vectorColumn
	
	return &config, nil
}

/* GetIndexTemplate retrieves index template by name */
func (ch *ConfigHelper) GetIndexTemplate(ctx context.Context, templateName string) (map[string]interface{}, error) {
	query := `
		SELECT config_json
		FROM neurondb.index_templates
		WHERE template_name = $1
	`
	
	var configJSON []byte
	err := ch.db.QueryRow(ctx, query, templateName).Scan(&configJSON)
	if err != nil {
		return nil, fmt.Errorf("failed to get index template %s: %w", templateName, err)
	}
	
	var config map[string]interface{}
	if err := json.Unmarshal(configJSON, &config); err != nil {
		return nil, fmt.Errorf("failed to unmarshal index template: %w", err)
	}
	
	return config, nil
}

/* GetWorkerConfig retrieves worker configuration */
func (ch *ConfigHelper) GetWorkerConfig(ctx context.Context, workerName string) (*WorkerConfig, error) {
	query := `SELECT neurondb_get_worker_config($1)`
	
	var resultJSON []byte
	err := ch.db.QueryRow(ctx, query, workerName).Scan(&resultJSON)
	if err != nil {
		return nil, fmt.Errorf("failed to get worker config for %s: %w", workerName, err)
	}
	
	var config WorkerConfig
	if err := json.Unmarshal(resultJSON, &config); err != nil {
		return nil, fmt.Errorf("failed to unmarshal worker config: %w", err)
	}
	
	return &config, nil
}

/* GetMLDefaults retrieves ML algorithm defaults */
func (ch *ConfigHelper) GetMLDefaults(ctx context.Context, algorithm string) (*MLDefaults, error) {
	query := `SELECT neurondb_get_ml_defaults($1)`
	
	var resultJSON []byte
	err := ch.db.QueryRow(ctx, query, algorithm).Scan(&resultJSON)
	if err != nil {
		return nil, fmt.Errorf("failed to get ML defaults for %s: %w", algorithm, err)
	}
	
	var defaults MLDefaults
	if err := json.Unmarshal(resultJSON, &defaults); err != nil {
		return nil, fmt.Errorf("failed to unmarshal ML defaults: %w", err)
	}
	
	return &defaults, nil
}

/* GetMLTemplate retrieves ML model template */
func (ch *ConfigHelper) GetMLTemplate(ctx context.Context, templateName string) (map[string]interface{}, error) {
	query := `
		SELECT template_config
		FROM neurondb.ml_model_templates
		WHERE template_name = $1
	`
	
	var templateJSON []byte
	err := ch.db.QueryRow(ctx, query, templateName).Scan(&templateJSON)
	if err != nil {
		return nil, fmt.Errorf("failed to get ML template %s: %w", templateName, err)
	}
	
	var template map[string]interface{}
	if err := json.Unmarshal(templateJSON, &template); err != nil {
		return nil, fmt.Errorf("failed to unmarshal ML template: %w", err)
	}
	
	return template, nil
}

/* GetToolConfig retrieves tool-specific configuration */
func (ch *ConfigHelper) GetToolConfig(ctx context.Context, toolName string) (*ToolConfig, error) {
	query := `SELECT neurondb_get_tool_config($1)`
	
	var resultJSON []byte
	err := ch.db.QueryRow(ctx, query, toolName).Scan(&resultJSON)
	if err != nil {
		/* Return default config if not found */
		return &ToolConfig{
			ToolName:        toolName,
			DefaultParams:   make(map[string]interface{}),
			DefaultTimeoutMS: 30000,
			Enabled:         true,
		}, nil
	}
	
	var config ToolConfig
	if err := json.Unmarshal(resultJSON, &config); err != nil {
		return nil, fmt.Errorf("failed to unmarshal tool config: %w", err)
	}
	
	return &config, nil
}

/* GetSystemConfig retrieves system-wide configuration */
func (ch *ConfigHelper) GetSystemConfig(ctx context.Context) (*SystemConfig, error) {
	query := `SELECT neurondb_get_system_config()`
	
	var resultJSON []byte
	err := ch.db.QueryRow(ctx, query).Scan(&resultJSON)
	if err != nil {
		return nil, fmt.Errorf("failed to get system config: %w", err)
	}
	
	var config SystemConfig
	if err := json.Unmarshal(resultJSON, &config); err != nil {
		return nil, fmt.Errorf("failed to unmarshal system config: %w", err)
	}
	
	return &config, nil
}

/* GetAllDefaults retrieves all configurations in a unified view */
func (ch *ConfigHelper) GetAllDefaults(ctx context.Context) (map[string]interface{}, error) {
	query := `SELECT neurondb_get_all_configs()`
	
	var resultJSON []byte
	err := ch.db.QueryRow(ctx, query).Scan(&resultJSON)
	if err != nil {
		return nil, fmt.Errorf("failed to get all configs: %w", err)
	}
	
	var configs map[string]interface{}
	if err := json.Unmarshal(resultJSON, &configs); err != nil {
		return nil, fmt.Errorf("failed to unmarshal all configs: %w", err)
	}
	
	return configs, nil
}

