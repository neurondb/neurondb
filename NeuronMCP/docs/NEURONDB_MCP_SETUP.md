# NeuronMCP Complete Configuration Schema Setup

## Overview

The NeuronMCP Configuration Schema provides a comprehensive, production-grade database schema that sets up **everything** needed for NeuronMCP to work seamlessly with NeuronDB. This includes LLM models with encrypted API keys, vector index configurations, worker settings, ML model defaults, tool configurations, and system-wide settings.

## Architecture

The schema follows database best practices with proper normalization, security, and extensibility:

- **13 Normalized Tables**: Organized by concern (LLM models, indexes, workers, ML, tools, system)
- **30+ Management Functions**: Complete CRUD operations for all configurations
- **Pre-populated Defaults**: 50+ LLM models, index templates, worker configs, ML defaults
- **Security**: Encrypted API keys using pgcrypto
- **Backward Compatible**: Falls back to GUC settings if database config not found

## Quick Start

### 1. Run Setup Script

```bash
cd NeuronMCP
./scripts/neuronmcp-setup.sh
```

Or with custom database connection:

```bash
DB_HOST=localhost DB_PORT=5432 DB_NAME=neurondb DB_USER=postgres ./scripts/neuronmcp-setup.sh
```

### 2. Set API Keys

```sql
-- Set API key for a model
SELECT neurondb_set_model_key('text-embedding-3-small', 'sk-your-api-key-here');

-- Set API key with expiration
SELECT neurondb_set_model_key('gpt-4', 'sk-your-key', NOW() + INTERVAL '90 days');
```

### 3. Verify Setup

```sql
-- View all active models
SELECT * FROM neurondb.v_llm_models_active;

-- View models ready for use (have API keys)
SELECT * FROM neurondb.v_llm_models_ready;

-- Get all configurations
SELECT neurondb_get_all_configs();
```

## Schema Structure

### Part 1: LLM Models & Providers (5 tables)

#### 1. `neurondb.llm_providers`
Master table for LLM providers (OpenAI, Anthropic, HuggingFace, etc.)

**Columns:**
- `provider_id` (SERIAL PRIMARY KEY)
- `provider_name` (TEXT UNIQUE) - 'openai', 'anthropic', 'huggingface', 'local', 'openai-compatible'
- `display_name` (TEXT)
- `default_base_url` (TEXT)
- `auth_method` (TEXT) - 'api_key', 'bearer', 'oauth', 'none'
- `default_timeout_ms` (INTEGER) - Default: 30000
- `rate_limit_rpm` (INTEGER) - Requests per minute
- `rate_limit_tpm` (INTEGER) - Tokens per minute
- `supports_streaming` (BOOLEAN)
- `supports_embeddings` (BOOLEAN)
- `supports_chat` (BOOLEAN)
- `supports_completion` (BOOLEAN)
- `metadata` (JSONB)
- `status` (TEXT) - 'active', 'deprecated', 'disabled'

#### 2. `neurondb.llm_models`
Catalog of all available LLM models

**Columns:**
- `model_id` (SERIAL PRIMARY KEY)
- `provider_id` (INTEGER) - References llm_providers
- `model_name` (TEXT) - e.g., 'text-embedding-3-small', 'gpt-4'
- `model_alias` (TEXT) - Short alias
- `model_type` (TEXT) - 'embedding', 'chat', 'completion', 'rerank', 'multimodal'
- `context_window` (INTEGER) - Max tokens/context length
- `embedding_dimension` (INTEGER) - For embedding models
- `max_output_tokens` (INTEGER)
- `supports_streaming` (BOOLEAN)
- `supports_function_calling` (BOOLEAN)
- `cost_per_1k_tokens_input` (NUMERIC)
- `cost_per_1k_tokens_output` (NUMERIC)
- `description` (TEXT)
- `documentation_url` (TEXT)
- `status` (TEXT) - 'available', 'disabled', 'deprecated', 'beta'
- `is_default` (BOOLEAN) - Default model for this type/provider

#### 3. `neurondb.llm_model_keys`
Secure storage for encrypted API keys

**Columns:**
- `key_id` (SERIAL PRIMARY KEY)
- `model_id` (INTEGER UNIQUE) - References llm_models
- `api_key_encrypted` (BYTEA) - Encrypted using pgcrypto
- `encryption_salt` (BYTEA)
- `key_type` (TEXT) - 'api_key', 'bearer_token', 'oauth_token'
- `expires_at` (TIMESTAMPTZ)
- `last_used_at` (TIMESTAMPTZ)
- `access_count` (INTEGER)
- `created_by` (TEXT)

#### 4. `neurondb.llm_model_configs`
Model-specific configurations

**Columns:**
- `config_id` (SERIAL PRIMARY KEY)
- `model_id` (INTEGER) - References llm_models
- `config_name` (TEXT) - Default: 'default'
- `base_url` (TEXT) - Override provider default
- `endpoint_path` (TEXT)
- `default_params` (JSONB) - temperature, top_p, etc.
- `request_headers` (JSONB)
- `timeout_ms` (INTEGER)
- `retry_config` (JSONB)
- `rate_limit_config` (JSONB)
- `is_active` (BOOLEAN)

#### 5. `neurondb.llm_model_usage`
Usage tracking and analytics

**Columns:**
- `usage_id` (BIGSERIAL PRIMARY KEY)
- `model_id` (INTEGER) - References llm_models
- `operation_type` (TEXT) - 'embedding', 'chat', 'completion', 'rerank'
- `tokens_input` (INTEGER)
- `tokens_output` (INTEGER)
- `cost` (NUMERIC)
- `latency_ms` (INTEGER)
- `success` (BOOLEAN)
- `error_message` (TEXT)
- `user_context` (TEXT)

### Part 2: Vector Index Configurations (2 tables)

#### 6. `neurondb.index_configs`
Default index configurations for vector columns

**Columns:**
- `config_id` (SERIAL PRIMARY KEY)
- `table_name` (TEXT)
- `vector_column` (TEXT)
- `index_type` (TEXT) - 'hnsw', 'ivf', 'flat'
- `hnsw_m` (INTEGER) - Default: 16
- `hnsw_ef_construction` (INTEGER) - Default: 200
- `hnsw_ef_search` (INTEGER) - Default: 64
- `ivf_lists` (INTEGER) - Default: 100
- `ivf_probes` (INTEGER) - Default: 10
- `distance_metric` (TEXT) - 'l2', 'cosine', 'inner_product'
- `is_default` (BOOLEAN)

#### 7. `neurondb.index_templates`
Reusable index templates

**Columns:**
- `template_id` (SERIAL PRIMARY KEY)
- `template_name` (TEXT UNIQUE)
- `description` (TEXT)
- `index_type` (TEXT)
- `config_json` (JSONB) - Full index configuration
- `is_default` (BOOLEAN)

### Part 3: Worker Configurations (2 tables)

#### 8. `neurondb.worker_configs`
Background worker settings

**Columns:**
- `config_id` (SERIAL PRIMARY KEY)
- `worker_name` (TEXT UNIQUE) - 'neuranq', 'neuranmon', 'neurandefrag'
- `display_name` (TEXT)
- `enabled` (BOOLEAN)
- `naptime_ms` (INTEGER) - Sleep time between iterations
- `config_json` (JSONB) - Worker-specific configuration
- `is_default` (BOOLEAN)

#### 9. `neurondb.worker_schedules`
Worker scheduling and maintenance windows

**Columns:**
- `schedule_id` (SERIAL PRIMARY KEY)
- `worker_name` (TEXT) - References worker_configs
- `schedule_name` (TEXT)
- `schedule_type` (TEXT) - 'interval', 'cron', 'maintenance_window'
- `schedule_config` (JSONB)
- `is_active` (BOOLEAN)

### Part 4: ML Model Defaults (2 tables)

#### 10. `neurondb.ml_default_configs`
Default ML training configurations per algorithm

**Columns:**
- `config_id` (SERIAL PRIMARY KEY)
- `algorithm` (TEXT UNIQUE) - 'linear_regression', 'kmeans', 'svm', etc.
- `default_hyperparameters` (JSONB)
- `default_training_settings` (JSONB)
- `use_gpu` (BOOLEAN)
- `gpu_device` (INTEGER)
- `batch_size` (INTEGER)
- `max_iterations` (INTEGER)
- `is_default` (BOOLEAN)

#### 11. `neurondb.ml_model_templates`
Pre-configured ML model templates

**Columns:**
- `template_id` (SERIAL PRIMARY KEY)
- `template_name` (TEXT UNIQUE)
- `description` (TEXT)
- `algorithm` (TEXT)
- `template_config` (JSONB)
- `is_default` (BOOLEAN)

### Part 5: Tool Configurations (1 table)

#### 12. `neurondb.tool_configs`
NeuronMCP tool-specific default settings

**Columns:**
- `config_id` (SERIAL PRIMARY KEY)
- `tool_name` (TEXT UNIQUE) - 'vector_search', 'generate_embedding', 'rag', etc.
- `display_name` (TEXT)
- `default_params` (JSONB)
- `default_limit` (INTEGER)
- `default_timeout_ms` (INTEGER) - Default: 30000
- `enabled` (BOOLEAN)
- `is_default` (BOOLEAN)

### Part 6: System Configuration (1 table)

#### 13. `neurondb.system_configs`
System-wide NeuronMCP settings

**Columns:**
- `config_id` (SERIAL PRIMARY KEY)
- `config_key` (TEXT UNIQUE)
- `config_value` (JSONB)
- `description` (TEXT)
- `is_default` (BOOLEAN)

## Management Functions

### LLM Model Key Management

```sql
-- Set API key for a model
SELECT neurondb_set_model_key('text-embedding-3-small', 'sk-your-key');

-- Get decrypted API key (for internal use)
SELECT neurondb_get_model_key('text-embedding-3-small');

-- Remove API key
SELECT neurondb_remove_model_key('text-embedding-3-small');

-- Rotate API key
SELECT neurondb_rotate_model_key('text-embedding-3-small', 'sk-new-key');
```

### LLM Model Management

```sql
-- List all models
SELECT * FROM neurondb_list_models();

-- List models by provider
SELECT * FROM neurondb_list_models('openai', NULL, NULL);

-- List models by type
SELECT * FROM neurondb_list_models(NULL, 'embedding', NULL);

-- Get complete model configuration
SELECT neurondb_get_model_config('text-embedding-3-small');

-- Enable a model
SELECT neurondb_enable_model('text-embedding-3-small');

-- Disable a model
SELECT neurondb_disable_model('text-embedding-3-small');

-- Set default model for type
SELECT neurondb_set_default_model('text-embedding-3-small', 'embedding');

-- Smart model selection
SELECT neurondb_get_model_for_operation('embedding', NULL);
SELECT neurondb_get_model_for_operation('embedding', 'text-embedding-ada-002');
```

### Model Configuration

```sql
-- Set model configuration
SELECT neurondb_set_model_config('gpt-4', 'default', '{"temperature": 0.7, "top_p": 0.9}'::jsonb);

-- Get model configuration
SELECT neurondb_get_model_config('gpt-4', 'default');

-- Reset to defaults
SELECT neurondb_reset_model_config('gpt-4');
```

### Provider Management

```sql
-- Add custom provider
SELECT neurondb_add_provider('custom-provider', '{
  "display_name": "Custom Provider",
  "default_base_url": "https://api.custom.com",
  "auth_method": "api_key"
}'::jsonb);

-- List all providers
SELECT * FROM neurondb_list_providers();
```

### Usage & Analytics

```sql
-- Log model usage
SELECT neurondb_log_model_usage(
  'text-embedding-3-small',
  'embedding',
  100,  -- tokens_input
  NULL, -- tokens_output
  150,  -- latency_ms
  true, -- success
  NULL  -- error_message
);

-- Get model statistics
SELECT neurondb_get_model_stats('text-embedding-3-small', 30); -- last 30 days

-- Get cost summary
SELECT neurondb_get_cost_summary(30); -- last 30 days
```

### Vector Index Management

```sql
-- Get index configuration
SELECT neurondb_get_index_config('documents', 'embedding');

-- Set index configuration
SELECT neurondb_set_index_config('documents', 'embedding', '{
  "index_type": "hnsw",
  "hnsw_m": 16,
  "hnsw_ef_construction": 200,
  "hnsw_ef_search": 64,
  "distance_metric": "cosine"
}'::jsonb);

-- Apply index template
SELECT neurondb_apply_index_template('hnsw-balanced', 'documents', 'embedding');

-- List index templates
SELECT * FROM neurondb_list_index_templates();
```

### Worker Management

```sql
-- Get worker configuration
SELECT neurondb_get_worker_config('neuranq');

-- Set worker configuration
SELECT neurondb_set_worker_config('neuranq', '{
  "queue_depth": 20000,
  "batch_size": 200,
  "timeout": 60000
}'::jsonb);

-- Enable worker
SELECT neurondb_enable_worker('neuranq');

-- Disable worker
SELECT neurondb_disable_worker('neuranq');
```

### ML Defaults Management

```sql
-- Get ML defaults for algorithm
SELECT neurondb_get_ml_defaults('kmeans');

-- Set ML defaults
SELECT neurondb_set_ml_defaults('kmeans', '{
  "hyperparameters": {"n_clusters": 8, "max_iter": 300},
  "use_gpu": false
}'::jsonb);

-- Apply ML template
SELECT neurondb_apply_ml_template('kmeans-fast', 'my-project');
```

### Tool Configuration

```sql
-- Get tool configuration
SELECT neurondb_get_tool_config('vector_search');

-- Set tool configuration
SELECT neurondb_set_tool_config('vector_search', '{
  "default_params": {"distance_metric": "cosine"},
  "default_limit": 20,
  "default_timeout_ms": 60000
}'::jsonb);

-- Reset tool configuration
SELECT neurondb_reset_tool_config('vector_search');
```

### System Configuration

```sql
-- Get system configuration
SELECT neurondb_get_system_config();

-- Set system configuration
SELECT neurondb_set_system_config('{
  "features": {"vector": true, "ml": true, "analytics": true},
  "default_timeout_ms": 60000,
  "rate_limiting": {"enabled": true, "requests_per_minute": 100}
}'::jsonb);

-- Get all configurations (unified view)
SELECT neurondb_get_all_configs();
```

## Pre-populated Data

### Providers (5 providers)
- **OpenAI** - API key auth, supports embeddings/chat/completion
- **Anthropic** - API key auth, supports chat/completion
- **HuggingFace** - API key auth, supports embeddings
- **Local** - No auth, supports embeddings
- **OpenAI-Compatible** - For custom endpoints

### Models (50+ models)
- **OpenAI**: 6 embedding models, 8 chat models
- **Anthropic**: 6 Claude models
- **HuggingFace**: 10 sentence-transformers models
- **Local**: 5 local embedding models
- **Reranking**: 5 reranking models

All models have NULL keys initially - configure them using `neurondb_set_model_key()`.

### Index Templates (6 templates)
- `hnsw-fast` - Fast HNSW index
- `hnsw-balanced` - Balanced HNSW (default)
- `hnsw-precise` - Precise HNSW
- `ivf-fast` - Fast IVF index
- `ivf-balanced` - Balanced IVF (default)
- `ivf-precise` - Precise IVF

### Worker Configurations (3 workers)
- `neuranq` - Queue Executor
- `neuranmon` - Auto-Tuner
- `neurandefrag` - Index Maintenance

### ML Defaults (11 algorithms)
- linear_regression, logistic_regression, random_forest, svm, kmeans, gmm, xgboost, naive_bayes, ridge, lasso, knn

### Tool Configurations (7 tools)
- vector_search, generate_embedding, batch_embedding, rag, analytics, ml_training, ml_prediction

## Security Best Practices

### Encryption Key Management

The encryption key for API keys is retrieved from PostgreSQL GUC setting `neurondb.encryption_key`. Set it securely:

```sql
-- Set encryption key (do this before setting any API keys)
ALTER SYSTEM SET neurondb.encryption_key = 'your-secure-encryption-key-here';
SELECT pg_reload_conf();
```

**Important**: Change the default encryption key in production! The default key is only for development.

### Access Control

1. **Separate Roles**: Create separate roles for key management vs. key usage
   ```sql
   CREATE ROLE neurondb_key_manager;
   CREATE ROLE neurondb_key_user;
   
   GRANT EXECUTE ON FUNCTION neurondb_set_model_key TO neurondb_key_manager;
   GRANT EXECUTE ON FUNCTION neurondb_get_model_key TO neurondb_key_user;
   ```

2. **Function Security**: All key management functions use `SECURITY DEFINER` with proper checks

3. **Audit Trail**: All key access is logged with timestamp and user

### Best Practices

- Never log API keys in application logs
- Rotate keys regularly using `neurondb_rotate_model_key()`
- Set expiration dates for temporary keys
- Monitor usage to detect anomalies
- Use environment variables for encryption keys in production

## Integration with NeuronMCP

NeuronMCP tools automatically use configurations from the database:

1. **Model Selection**: Tools automatically select default models or use preferred models
2. **Key Resolution**: API keys are automatically retrieved when models are used
3. **Default Parameters**: Tool defaults are applied automatically
4. **Usage Logging**: Operations are automatically logged for analytics
5. **Fallback**: If database config not found, falls back to GUC settings

### Example: Embedding Generation

When `generate_embedding` tool is called:
1. If model not specified, gets default embedding model from database
2. Resolves API key from database (if available)
3. Uses model configuration (timeout, retry, etc.) from database
4. Falls back to GUC settings if database config not found
5. Logs usage metrics after successful operation

## Troubleshooting

### Setup Issues

**Problem**: Setup script fails to connect
- **Solution**: Check database connection parameters (host, port, user, password)
- Verify PostgreSQL is running and accessible
- Check firewall rules

**Problem**: NeuronDB extension not found
- **Solution**: Install NeuronDB extension first: `CREATE EXTENSION neurondb;`

**Problem**: pgcrypto extension not available
- **Solution**: Install pgcrypto: `CREATE EXTENSION pgcrypto;`

### Configuration Issues

**Problem**: API key not working
- **Solution**: Verify key is set: `SELECT * FROM neurondb.v_llm_models_ready WHERE model_name = 'your-model';`
- Check key expiration: `SELECT expires_at FROM neurondb.llm_model_keys mk JOIN neurondb.llm_models m ON mk.model_id = m.model_id WHERE m.model_name = 'your-model';`
- Verify encryption key is set correctly

**Problem**: Model not found
- **Solution**: Check available models: `SELECT * FROM neurondb.v_llm_models_active;`
- Verify model name spelling
- Check model status: `SELECT status FROM neurondb.llm_models WHERE model_name = 'your-model';`

**Problem**: Default model not working
- **Solution**: Set a default model: `SELECT neurondb_set_default_model('model-name', 'embedding');`
- Verify model has API key configured
- Check model is enabled: `SELECT neurondb_enable_model('model-name');`

### Performance Issues

**Problem**: Slow configuration lookups
- **Solution**: Indexes are automatically created, but verify they exist
- Consider caching configurations in application layer
- Monitor query performance with `EXPLAIN ANALYZE`

## Migration Guide

If upgrading from a previous version:

1. **Backup**: Always backup your database before migration
2. **Run Schema**: Execute `sql/001_initial_schema.sql`
3. **Run Functions**: Execute `sql/002_functions.sql`
4. **Verify**: Run verification queries from setup script
5. **Migrate Data**: If you have existing configurations, migrate them to new schema

## Examples

### Complete Setup Example

```sql
-- 1. Setup schema (run setup script)
-- ./scripts/neuronmcp-setup.sh

-- 2. Set encryption key (IMPORTANT: change in production!)
ALTER SYSTEM SET neurondb.encryption_key = 'your-secure-key-here';
SELECT pg_reload_conf();

-- 3. Set API keys for models
SELECT neurondb_set_model_key('text-embedding-3-small', 'sk-openai-key');
SELECT neurondb_set_model_key('gpt-4', 'sk-openai-key');
SELECT neurondb_set_model_key('claude-3.5-sonnet', 'sk-anthropic-key');

-- 4. Set default models
SELECT neurondb_set_default_model('text-embedding-3-small', 'embedding');
SELECT neurondb_set_default_model('gpt-4', 'chat');

-- 5. Configure index defaults
SELECT neurondb_set_index_config('documents', 'embedding', '{
  "index_type": "hnsw",
  "hnsw_m": 16,
  "hnsw_ef_construction": 200,
  "distance_metric": "cosine"
}'::jsonb);

-- 6. Verify setup
SELECT * FROM neurondb.v_llm_models_ready;
SELECT neurondb_get_all_configs();
```

### Using in NeuronMCP Tools

Once configured, NeuronMCP tools automatically use the database configurations:

```python
# Example: generate_embedding tool automatically uses:
# - Default embedding model from database
# - API key from database
# - Model configuration from database
# - Tool defaults from database

# No additional configuration needed in tool calls!
```

## Additional Resources

- **Schema SQL**: `sql/001_initial_schema.sql`
- **Functions SQL**: `sql/002_functions.sql`
- **Setup Script**: `scripts/neuronmcp-setup.sh`
- **Go Integration**: `internal/database/config.go`

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review function error messages
3. Check database logs
4. Verify NeuronDB extension is properly installed



