/*-------------------------------------------------------------------------
 *
 * neuron-mcp.sql
 *    Complete NeuronMCP Database Setup Script
 *
 * This script sets up everything needed for NeuronMCP:
 * - Database schema (tables, indexes, views, triggers)
 * - Management functions
 * - Pre-populated default data
 *
 * This script is idempotent and can be run multiple times safely.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/neuron-mcp.sql
 *
 *-------------------------------------------------------------------------
 *
 * PREREQUISITES
 * =============
 *
 * - PostgreSQL 16 or later
 * - NeuronDB extension installed
 * - Database user with CREATE privileges
 *
 * USAGE
 * =====
 *
 * To run this setup script on a database:
 *
 *   psql -d your_database -f neuron-mcp.sql
 *
 * Or from within psql:
 *
 *   \i neuron-mcp.sql
 *
 *-------------------------------------------------------------------------
 */

-- ============================================================================
-- SECTION 1: EXTENSIONS
-- ============================================================================

-- Ensure required extensions are available
-- Note: The neurondb extension will create the neurondb schema automatically
-- Do NOT create the schema manually here, as it must be owned by the extension
CREATE EXTENSION IF NOT EXISTS neurondb;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ============================================================================
-- SECTION 2: LLM MODELS & PROVIDERS (5 tables)
-- ============================================================================

-- 1. LLM Providers Table
CREATE TABLE IF NOT EXISTS neurondb.llm_providers (
    provider_id SERIAL PRIMARY KEY,
    provider_name TEXT NOT NULL UNIQUE,  -- 'openai', 'anthropic', 'huggingface', 'local', 'openai-compatible'
    display_name TEXT NOT NULL,
    default_base_url TEXT,
    auth_method TEXT NOT NULL DEFAULT 'api_key' CHECK (auth_method IN ('api_key', 'bearer', 'oauth', 'none')),
    default_timeout_ms INTEGER DEFAULT 30000,
    rate_limit_rpm INTEGER,  -- Requests per minute
    rate_limit_tpm INTEGER, -- Tokens per minute
    supports_streaming BOOLEAN DEFAULT false,
    supports_embeddings BOOLEAN DEFAULT false,
    supports_chat BOOLEAN DEFAULT false,
    supports_completion BOOLEAN DEFAULT false,
    metadata JSONB DEFAULT '{}',
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'deprecated', 'disabled')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
COMMENT ON TABLE neurondb.llm_providers IS 'Master table for LLM providers (OpenAI, Anthropic, HuggingFace, etc.)';

-- 2. LLM Models Table
CREATE TABLE IF NOT EXISTS neurondb.llm_models (
    model_id SERIAL PRIMARY KEY,
    provider_id INTEGER NOT NULL REFERENCES neurondb.llm_providers(provider_id) ON DELETE RESTRICT,
    model_name TEXT NOT NULL,  -- e.g., 'text-embedding-3-small', 'gpt-4'
    model_alias TEXT,  -- Short alias for convenience
    model_type TEXT NOT NULL CHECK (model_type IN ('embedding', 'chat', 'completion', 'rerank', 'multimodal')),
    context_window INTEGER,  -- Max tokens/context length
    embedding_dimension INTEGER,  -- For embedding models
    max_output_tokens INTEGER,
    supports_streaming BOOLEAN DEFAULT false,
    supports_function_calling BOOLEAN DEFAULT false,
    cost_per_1k_tokens_input NUMERIC(10,6),
    cost_per_1k_tokens_output NUMERIC(10,6),
    description TEXT,
    documentation_url TEXT,
    status TEXT DEFAULT 'available' CHECK (status IN ('available', 'disabled', 'deprecated', 'beta')),
    is_default BOOLEAN DEFAULT false,  -- Default model for this type/provider
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_used_at TIMESTAMPTZ,
    UNIQUE(provider_id, model_name)
);
COMMENT ON TABLE neurondb.llm_models IS 'Catalog of all available LLM models';

-- 3. LLM Model Keys Table (Secure Storage)
CREATE TABLE IF NOT EXISTS neurondb.llm_model_keys (
    key_id SERIAL PRIMARY KEY,
    model_id INTEGER NOT NULL UNIQUE REFERENCES neurondb.llm_models(model_id) ON DELETE CASCADE,
    api_key_encrypted BYTEA NOT NULL,  -- Encrypted using pgcrypto
    encryption_salt BYTEA NOT NULL,
    key_type TEXT DEFAULT 'api_key' CHECK (key_type IN ('api_key', 'bearer_token', 'oauth_token')),
    expires_at TIMESTAMPTZ,
    last_used_at TIMESTAMPTZ,
    access_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_by TEXT DEFAULT CURRENT_USER
);
COMMENT ON TABLE neurondb.llm_model_keys IS 'Secure storage for encrypted API keys';

-- 4. LLM Model Configurations Table
CREATE TABLE IF NOT EXISTS neurondb.llm_model_configs (
    config_id SERIAL PRIMARY KEY,
    model_id INTEGER NOT NULL REFERENCES neurondb.llm_models(model_id) ON DELETE CASCADE,
    config_name TEXT DEFAULT 'default',
    base_url TEXT,  -- Override provider default
    endpoint_path TEXT,  -- API endpoint path
    default_params JSONB DEFAULT '{}',  -- temperature, top_p, etc.
    request_headers JSONB DEFAULT '{}',  -- Custom headers
    timeout_ms INTEGER,
    retry_config JSONB DEFAULT '{"max_retries": 3, "backoff_ms": 1000}',
    rate_limit_config JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(model_id, config_name)
);
COMMENT ON TABLE neurondb.llm_model_configs IS 'Model-specific configurations';

-- 5. LLM Model Usage Tracking Table
CREATE TABLE IF NOT EXISTS neurondb.llm_model_usage (
    usage_id BIGSERIAL PRIMARY KEY,
    model_id INTEGER NOT NULL REFERENCES neurondb.llm_models(model_id) ON DELETE SET NULL,
    operation_type TEXT NOT NULL CHECK (operation_type IN ('embedding', 'chat', 'completion', 'rerank')),
    tokens_input INTEGER,
    tokens_output INTEGER,
    cost NUMERIC(10,6),
    latency_ms INTEGER,
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    user_context TEXT,  -- For multi-tenant scenarios
    created_at TIMESTAMPTZ DEFAULT NOW()
);
COMMENT ON TABLE neurondb.llm_model_usage IS 'Usage tracking and analytics for LLM models';

-- ============================================================================
-- SECTION 3: VECTOR INDEX CONFIGURATIONS (2 tables)
-- ============================================================================

-- 6. Index Configurations Table
CREATE TABLE IF NOT EXISTS neurondb.index_configs (
    config_id SERIAL PRIMARY KEY,
    table_name TEXT,
    vector_column TEXT,
    index_type TEXT NOT NULL CHECK (index_type IN ('hnsw', 'ivf', 'flat')),
    hnsw_m INTEGER DEFAULT 16,  -- HNSW: number of connections
    hnsw_ef_construction INTEGER DEFAULT 200,  -- HNSW: construction parameter
    hnsw_ef_search INTEGER DEFAULT 64,  -- HNSW: search parameter
    ivf_lists INTEGER DEFAULT 100,  -- IVF: number of lists
    ivf_probes INTEGER DEFAULT 10,  -- IVF: number of probes
    distance_metric TEXT DEFAULT 'l2' CHECK (distance_metric IN ('l2', 'cosine', 'inner_product')),
    is_default BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(table_name, vector_column) WHERE table_name IS NOT NULL AND vector_column IS NOT NULL
);
COMMENT ON TABLE neurondb.index_configs IS 'Default index configurations for vector columns';

-- 7. Index Templates Table
CREATE TABLE IF NOT EXISTS neurondb.index_templates (
    template_id SERIAL PRIMARY KEY,
    template_name TEXT NOT NULL UNIQUE,
    description TEXT,
    index_type TEXT NOT NULL CHECK (index_type IN ('hnsw', 'ivf', 'flat')),
    config_json JSONB NOT NULL,  -- Full index configuration
    is_default BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
COMMENT ON TABLE neurondb.index_templates IS 'Reusable index templates for common configurations';

-- ============================================================================
-- SECTION 4: WORKER CONFIGURATIONS (2 tables)
-- ============================================================================

-- 8. Worker Configurations Table
CREATE TABLE IF NOT EXISTS neurondb.worker_configs (
    config_id SERIAL PRIMARY KEY,
    worker_name TEXT NOT NULL UNIQUE,  -- 'neuranq', 'neuranmon', 'neurandefrag'
    display_name TEXT NOT NULL,
    enabled BOOLEAN DEFAULT true,
    naptime_ms INTEGER DEFAULT 1000,  -- Sleep time between iterations
    config_json JSONB DEFAULT '{}',  -- Worker-specific configuration
    is_default BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
COMMENT ON TABLE neurondb.worker_configs IS 'Background worker settings and configurations';

-- 9. Worker Schedules Table
CREATE TABLE IF NOT EXISTS neurondb.worker_schedules (
    schedule_id SERIAL PRIMARY KEY,
    worker_name TEXT NOT NULL REFERENCES neurondb.worker_configs(worker_name) ON DELETE CASCADE,
    schedule_name TEXT NOT NULL,
    schedule_type TEXT NOT NULL CHECK (schedule_type IN ('interval', 'cron', 'maintenance_window')),
    schedule_config JSONB NOT NULL,  -- Schedule-specific configuration
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(worker_name, schedule_name)
);
COMMENT ON TABLE neurondb.worker_schedules IS 'Worker scheduling and maintenance windows';

-- ============================================================================
-- SECTION 5: ML MODEL DEFAULTS (2 tables)
-- ============================================================================

-- 10. ML Default Configurations Table
CREATE TABLE IF NOT EXISTS neurondb.ml_default_configs (
    config_id SERIAL PRIMARY KEY,
    algorithm TEXT NOT NULL UNIQUE,  -- 'linear_regression', 'kmeans', 'svm', etc.
    default_hyperparameters JSONB DEFAULT '{}',
    default_training_settings JSONB DEFAULT '{}',
    use_gpu BOOLEAN DEFAULT false,
    gpu_device INTEGER DEFAULT 0,
    batch_size INTEGER,
    max_iterations INTEGER,
    is_default BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
COMMENT ON TABLE neurondb.ml_default_configs IS 'Default ML training configurations per algorithm';

-- 11. ML Model Templates Table
CREATE TABLE IF NOT EXISTS neurondb.ml_model_templates (
    template_id SERIAL PRIMARY KEY,
    template_name TEXT NOT NULL UNIQUE,
    description TEXT,
    algorithm TEXT NOT NULL,
    template_config JSONB NOT NULL,  -- Complete template configuration
    is_default BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
COMMENT ON TABLE neurondb.ml_model_templates IS 'Pre-configured ML model templates for quick start';

-- ============================================================================
-- SECTION 6: TOOL CONFIGURATIONS (1 table)
-- ============================================================================

-- 12. Tool Configurations Table
CREATE TABLE IF NOT EXISTS neurondb.tool_configs (
    config_id SERIAL PRIMARY KEY,
    tool_name TEXT NOT NULL UNIQUE,  -- 'vector_search', 'generate_embedding', 'rag', etc.
    display_name TEXT NOT NULL,
    default_params JSONB DEFAULT '{}',  -- Tool-specific default parameters
    default_limit INTEGER,  -- For search/query tools
    default_timeout_ms INTEGER DEFAULT 30000,
    enabled BOOLEAN DEFAULT true,
    is_default BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
COMMENT ON TABLE neurondb.tool_configs IS 'NeuronMCP tool-specific default settings';

-- ============================================================================
-- SECTION 7: SYSTEM CONFIGURATION (1 table)
-- ============================================================================

-- 13. System Configuration Table
CREATE TABLE IF NOT EXISTS neurondb.system_configs (
    config_id SERIAL PRIMARY KEY,
    config_key TEXT NOT NULL UNIQUE,
    config_value JSONB NOT NULL,
    description TEXT,
    is_default BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
COMMENT ON TABLE neurondb.system_configs IS 'System-wide NeuronMCP settings and feature flags';

-- ============================================================================
-- SECTION 8: PROMPTS (1 table)
-- ============================================================================

-- 14. Prompts Table
CREATE TABLE IF NOT EXISTS neurondb.prompts (
    prompt_id SERIAL PRIMARY KEY,
    prompt_name TEXT NOT NULL UNIQUE,
    description TEXT,
    template TEXT NOT NULL,  -- Prompt template with variables
    variables JSONB DEFAULT '[]',  -- Array of variable definitions
    category TEXT,  -- Optional category for organization
    tags TEXT[],  -- Tags for searchability
    is_default BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_by TEXT DEFAULT CURRENT_USER
);
COMMENT ON TABLE neurondb.prompts IS 'MCP prompt templates with variable support';

-- ============================================================================
-- SECTION 9: INDEXES FOR PERFORMANCE
-- ============================================================================

-- LLM Provider indexes
CREATE INDEX IF NOT EXISTS idx_llm_providers_name ON neurondb.llm_providers(provider_name);
CREATE INDEX IF NOT EXISTS idx_llm_providers_status ON neurondb.llm_providers(status);

-- LLM Model indexes
CREATE INDEX IF NOT EXISTS idx_llm_models_provider ON neurondb.llm_models(provider_id);
CREATE INDEX IF NOT EXISTS idx_llm_models_name ON neurondb.llm_models(model_name);
CREATE INDEX IF NOT EXISTS idx_llm_models_type ON neurondb.llm_models(model_type);
CREATE INDEX IF NOT EXISTS idx_llm_models_status ON neurondb.llm_models(status);
CREATE INDEX IF NOT EXISTS idx_llm_models_default ON neurondb.llm_models(model_type, is_default) WHERE is_default = true;

-- LLM Key indexes
CREATE INDEX IF NOT EXISTS idx_llm_model_keys_model ON neurondb.llm_model_keys(model_id);
CREATE INDEX IF NOT EXISTS idx_llm_model_keys_last_used ON neurondb.llm_model_keys(last_used_at);

-- LLM Config indexes
CREATE INDEX IF NOT EXISTS idx_llm_model_configs_model ON neurondb.llm_model_configs(model_id);
CREATE INDEX IF NOT EXISTS idx_llm_model_configs_active ON neurondb.llm_model_configs(model_id, is_active) WHERE is_active = true;

-- LLM Usage indexes
CREATE INDEX IF NOT EXISTS idx_llm_model_usage_model ON neurondb.llm_model_usage(model_id, created_at);
CREATE INDEX IF NOT EXISTS idx_llm_model_usage_created ON neurondb.llm_model_usage(created_at);
CREATE INDEX IF NOT EXISTS idx_llm_model_usage_type ON neurondb.llm_model_usage(operation_type, created_at);

-- Index Config indexes
CREATE INDEX IF NOT EXISTS idx_index_configs_table ON neurondb.index_configs(table_name, vector_column) WHERE table_name IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_index_templates_name ON neurondb.index_templates(template_name);

-- Worker indexes
CREATE INDEX IF NOT EXISTS idx_worker_configs_name ON neurondb.worker_configs(worker_name);
CREATE INDEX IF NOT EXISTS idx_worker_schedules_worker ON neurondb.worker_schedules(worker_name);

-- ML indexes
CREATE INDEX IF NOT EXISTS idx_ml_default_configs_algorithm ON neurondb.ml_default_configs(algorithm);
CREATE INDEX IF NOT EXISTS idx_ml_model_templates_name ON neurondb.ml_model_templates(template_name);

-- Tool indexes
CREATE INDEX IF NOT EXISTS idx_tool_configs_name ON neurondb.tool_configs(tool_name);

-- System indexes
CREATE INDEX IF NOT EXISTS idx_system_configs_key ON neurondb.system_configs(config_key);

-- Prompts indexes
CREATE INDEX IF NOT EXISTS idx_prompts_name ON neurondb.prompts(prompt_name);
CREATE INDEX IF NOT EXISTS idx_prompts_category ON neurondb.prompts(category) WHERE category IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_prompts_tags ON neurondb.prompts USING GIN(tags);

-- ============================================================================
-- SECTION 10: TRIGGERS FOR AUTOMATIC TIMESTAMP UPDATES
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION neurondb.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply triggers to all tables with updated_at
CREATE TRIGGER trigger_llm_providers_updated_at
    BEFORE UPDATE ON neurondb.llm_providers
    FOR EACH ROW EXECUTE FUNCTION neurondb.update_updated_at();

CREATE TRIGGER trigger_llm_models_updated_at
    BEFORE UPDATE ON neurondb.llm_models
    FOR EACH ROW EXECUTE FUNCTION neurondb.update_updated_at();

CREATE TRIGGER trigger_llm_model_keys_updated_at
    BEFORE UPDATE ON neurondb.llm_model_keys
    FOR EACH ROW EXECUTE FUNCTION neurondb.update_updated_at();

CREATE TRIGGER trigger_llm_model_configs_updated_at
    BEFORE UPDATE ON neurondb.llm_model_configs
    FOR EACH ROW EXECUTE FUNCTION neurondb.update_updated_at();

CREATE TRIGGER trigger_index_configs_updated_at
    BEFORE UPDATE ON neurondb.index_configs
    FOR EACH ROW EXECUTE FUNCTION neurondb.update_updated_at();

CREATE TRIGGER trigger_index_templates_updated_at
    BEFORE UPDATE ON neurondb.index_templates
    FOR EACH ROW EXECUTE FUNCTION neurondb.update_updated_at();

CREATE TRIGGER trigger_worker_configs_updated_at
    BEFORE UPDATE ON neurondb.worker_configs
    FOR EACH ROW EXECUTE FUNCTION neurondb.update_updated_at();

CREATE TRIGGER trigger_worker_schedules_updated_at
    BEFORE UPDATE ON neurondb.worker_schedules
    FOR EACH ROW EXECUTE FUNCTION neurondb.update_updated_at();

CREATE TRIGGER trigger_ml_default_configs_updated_at
    BEFORE UPDATE ON neurondb.ml_default_configs
    FOR EACH ROW EXECUTE FUNCTION neurondb.update_updated_at();

CREATE TRIGGER trigger_ml_model_templates_updated_at
    BEFORE UPDATE ON neurondb.ml_model_templates
    FOR EACH ROW EXECUTE FUNCTION neurondb.update_updated_at();

CREATE TRIGGER trigger_tool_configs_updated_at
    BEFORE UPDATE ON neurondb.tool_configs
    FOR EACH ROW EXECUTE FUNCTION neurondb.update_updated_at();

CREATE TRIGGER trigger_system_configs_updated_at
    BEFORE UPDATE ON neurondb.system_configs
    FOR EACH ROW EXECUTE FUNCTION neurondb.update_updated_at();

CREATE TRIGGER trigger_prompts_updated_at
    BEFORE UPDATE ON neurondb.prompts
    FOR EACH ROW EXECUTE FUNCTION neurondb.update_updated_at();

-- ============================================================================
-- SECTION 11: CONVENIENCE VIEWS
-- ============================================================================

-- Active LLM models view with provider info
CREATE OR REPLACE VIEW neurondb.v_llm_models_active AS
SELECT 
    m.model_id,
    m.model_name,
    m.model_alias,
    p.provider_name,
    p.display_name AS provider_display_name,
    m.model_type,
    m.context_window,
    m.embedding_dimension,
    m.status,
    m.is_default,
    CASE WHEN mk.key_id IS NOT NULL THEN true ELSE false END AS has_api_key,
    mc.config_name,
    mc.base_url,
    mc.default_params,
    m.created_at,
    m.last_used_at
FROM neurondb.llm_models m
JOIN neurondb.llm_providers p ON m.provider_id = p.provider_id
LEFT JOIN neurondb.llm_model_keys mk ON m.model_id = mk.model_id
LEFT JOIN neurondb.llm_model_configs mc ON m.model_id = mc.model_id AND mc.is_active = true
WHERE m.status = 'available' AND p.status = 'active';

COMMENT ON VIEW neurondb.v_llm_models_active IS 'Active LLM models with provider information and key status';

-- Models ready for use (have keys and config)
CREATE OR REPLACE VIEW neurondb.v_llm_models_ready AS
SELECT * FROM neurondb.v_llm_models_active
WHERE has_api_key = true;

COMMENT ON VIEW neurondb.v_llm_models_ready IS 'LLM models ready for use (have API keys configured)';

-- ============================================================================
-- SECTION 12: PRE-POPULATE DEFAULT DATA
-- ============================================================================

-- Insert LLM Providers
INSERT INTO neurondb.llm_providers (provider_name, display_name, default_base_url, auth_method, supports_embeddings, supports_chat, supports_completion, supports_streaming)
VALUES
    ('openai', 'OpenAI', 'https://api.openai.com/v1', 'api_key', true, true, true, true),
    ('anthropic', 'Anthropic', 'https://api.anthropic.com', 'api_key', false, true, true, true),
    ('huggingface', 'HuggingFace', 'https://api-inference.huggingface.co', 'api_key', true, false, false, false),
    ('local', 'Local Models', NULL, 'none', true, false, false, false),
    ('openai-compatible', 'OpenAI-Compatible', NULL, 'api_key', true, true, true, true)
ON CONFLICT (provider_name) DO NOTHING;

-- Insert LLM Models (50+ models)
DO $$
DECLARE
    v_openai_id INTEGER;
    v_anthropic_id INTEGER;
    v_huggingface_id INTEGER;
    v_local_id INTEGER;
BEGIN
    SELECT provider_id INTO v_openai_id FROM neurondb.llm_providers WHERE provider_name = 'openai';
    SELECT provider_id INTO v_anthropic_id FROM neurondb.llm_providers WHERE provider_name = 'anthropic';
    SELECT provider_id INTO v_huggingface_id FROM neurondb.llm_providers WHERE provider_name = 'huggingface';
    SELECT provider_id INTO v_local_id FROM neurondb.llm_providers WHERE provider_name = 'local';

    -- OpenAI Embedding Models
    INSERT INTO neurondb.llm_models (provider_id, model_name, model_type, embedding_dimension, is_default, description)
    VALUES
        (v_openai_id, 'text-embedding-ada-002', 'embedding', 1536, true, 'OpenAI Ada embedding model'),
        (v_openai_id, 'text-embedding-3-small', 'embedding', 1536, false, 'OpenAI small embedding model'),
        (v_openai_id, 'text-embedding-3-large', 'embedding', 3072, false, 'OpenAI large embedding model'),
        (v_openai_id, 'text-embedding-3-small-512', 'embedding', 512, false, 'OpenAI small embedding model (512 dim)'),
        (v_openai_id, 'text-embedding-3-large-256', 'embedding', 256, false, 'OpenAI large embedding model (256 dim)'),
        (v_openai_id, 'text-embedding-3-large-1024', 'embedding', 1024, false, 'OpenAI large embedding model (1024 dim)')
    ON CONFLICT (provider_id, model_name) DO NOTHING;

    -- OpenAI Chat Models
    INSERT INTO neurondb.llm_models (provider_id, model_name, model_type, context_window, supports_streaming, supports_function_calling, is_default, description)
    VALUES
        (v_openai_id, 'gpt-4', 'chat', 8192, true, true, false, 'GPT-4 model'),
        (v_openai_id, 'gpt-4-turbo', 'chat', 128000, true, true, true, 'GPT-4 Turbo model'),
        (v_openai_id, 'gpt-4-turbo-preview', 'chat', 128000, true, true, false, 'GPT-4 Turbo preview'),
        (v_openai_id, 'gpt-3.5-turbo', 'chat', 16385, true, true, false, 'GPT-3.5 Turbo model'),
        (v_openai_id, 'gpt-3.5-turbo-16k', 'chat', 16385, true, true, false, 'GPT-3.5 Turbo 16k context'),
        (v_openai_id, 'gpt-4o', 'chat', 128000, true, true, false, 'GPT-4o model'),
        (v_openai_id, 'gpt-4o-mini', 'chat', 128000, true, true, false, 'GPT-4o mini model'),
        (v_openai_id, 'gpt-4-32k', 'chat', 32768, true, true, false, 'GPT-4 with 32k context')
    ON CONFLICT (provider_id, model_name) DO NOTHING;

    -- Anthropic Models
    INSERT INTO neurondb.llm_models (provider_id, model_name, model_type, context_window, supports_streaming, is_default, description)
    VALUES
        (v_anthropic_id, 'claude-3-opus', 'chat', 200000, true, false, 'Claude 3 Opus model'),
        (v_anthropic_id, 'claude-3-sonnet', 'chat', 200000, true, false, 'Claude 3 Sonnet model'),
        (v_anthropic_id, 'claude-3-haiku', 'chat', 200000, true, false, 'Claude 3 Haiku model'),
        (v_anthropic_id, 'claude-3.5-sonnet', 'chat', 200000, true, true, 'Claude 3.5 Sonnet model'),
        (v_anthropic_id, 'claude-3.5-haiku', 'chat', 200000, true, false, 'Claude 3.5 Haiku model'),
        (v_anthropic_id, 'claude-3-opus-20240229', 'chat', 200000, true, false, 'Claude 3 Opus (versioned)')
    ON CONFLICT (provider_id, model_name) DO NOTHING;

    -- HuggingFace Models
    INSERT INTO neurondb.llm_models (provider_id, model_name, model_type, embedding_dimension, description)
    VALUES
        (v_huggingface_id, 'sentence-transformers/all-MiniLM-L6-v2', 'embedding', 384, 'MiniLM L6 v2 embedding model'),
        (v_huggingface_id, 'sentence-transformers/all-mpnet-base-v2', 'embedding', 768, 'MPNet base v2 embedding model'),
        (v_huggingface_id, 'sentence-transformers/all-MiniLM-L12-v2', 'embedding', 384, 'MiniLM L12 v2 embedding model'),
        (v_huggingface_id, 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', 'embedding', 384, 'Multilingual MiniLM L12 v2'),
        (v_huggingface_id, 'sentence-transformers/distiluse-base-multilingual-cased', 'embedding', 512, 'DistilUSE multilingual model'),
        (v_huggingface_id, 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1', 'embedding', 384, 'Multi-QA MiniLM model'),
        (v_huggingface_id, 'sentence-transformers/all-distilroberta-v1', 'embedding', 768, 'DistilRoBERTa model'),
        (v_huggingface_id, 'sentence-transformers/paraphrase-albert-small-v2', 'embedding', 768, 'Paraphrase ALBERT model'),
        (v_huggingface_id, 'sentence-transformers/nli-mpnet-base-v2', 'embedding', 768, 'NLI MPNet model'),
        (v_huggingface_id, 'sentence-transformers/ms-marco-MiniLM-L-6-v2', 'embedding', 384, 'MS MARCO MiniLM model')
    ON CONFLICT (provider_id, model_name) DO NOTHING;

    -- Local Models
    INSERT INTO neurondb.llm_models (provider_id, model_name, model_type, embedding_dimension, is_default, description)
    VALUES
        (v_local_id, 'default', 'embedding', 384, true, 'Generic local embedding model'),
        (v_local_id, 'local-embedding-small', 'embedding', 384, false, 'Small local embedding model'),
        (v_local_id, 'local-embedding-base', 'embedding', 768, false, 'Base local embedding model'),
        (v_local_id, 'local-embedding-large', 'embedding', 1024, false, 'Large local embedding model'),
        (v_local_id, 'local-embedding-multilingual', 'embedding', 512, false, 'Multilingual local embedding model')
    ON CONFLICT (provider_id, model_name) DO NOTHING;

    -- Reranking Models
    INSERT INTO neurondb.llm_models (provider_id, model_name, model_type, description)
    VALUES
        (v_openai_id, 'text-search-ada-doc-001', 'rerank', 'OpenAI text search document model'),
        (v_openai_id, 'text-search-ada-query-001', 'rerank', 'OpenAI text search query model'),
        (v_huggingface_id, 'cohere/rerank-english-v3.0', 'rerank', 'Cohere reranking model'),
        (v_huggingface_id, 'cross-encoder/ms-marco-MiniLM-L-6-v2', 'rerank', 'Cross-encoder reranking model'),
        (v_huggingface_id, 'BAAI/bge-reranker-base', 'rerank', 'BAAI reranker model')
    ON CONFLICT (provider_id, model_name) DO NOTHING;
END $$;

-- Insert Index Templates
INSERT INTO neurondb.index_templates (template_name, description, index_type, config_json, is_default)
VALUES
    ('hnsw-fast', 'Fast HNSW index for quick searches', 'hnsw', '{"m": 16, "ef_construction": 100, "ef_search": 32}', false),
    ('hnsw-balanced', 'Balanced HNSW index (default)', 'hnsw', '{"m": 16, "ef_construction": 200, "ef_search": 64}', true),
    ('hnsw-precise', 'Precise HNSW index for high recall', 'hnsw', '{"m": 32, "ef_construction": 400, "ef_search": 128}', false),
    ('ivf-fast', 'Fast IVF index', 'ivf', '{"lists": 100, "probes": 10}', false),
    ('ivf-balanced', 'Balanced IVF index', 'ivf', '{"lists": 256, "probes": 32}', true),
    ('ivf-precise', 'Precise IVF index', 'ivf', '{"lists": 512, "probes": 64}', false)
ON CONFLICT (template_name) DO NOTHING;

-- Insert Worker Configurations
INSERT INTO neurondb.worker_configs (worker_name, display_name, enabled, naptime_ms, config_json, is_default)
VALUES
    ('neuranq', 'Queue Executor', true, 1000, '{"queue_depth": 10000, "batch_size": 100, "timeout": 30000, "max_retries": 3}', true),
    ('neuranmon', 'Auto-Tuner', true, 60000, '{"sample_size": 1000, "target_latency": 100.0, "target_recall": 0.95}', true),
    ('neurandefrag', 'Index Maintenance', true, 300000, '{"compact_threshold": 1000, "fragmentation_threshold": 0.3, "maintenance_window": "02:00-04:00"}', true)
ON CONFLICT (worker_name) DO NOTHING;

-- Insert ML Default Configurations
INSERT INTO neurondb.ml_default_configs (algorithm, default_hyperparameters, use_gpu, is_default)
VALUES
    ('linear_regression', '{}', false, true),
    ('logistic_regression', '{}', false, true),
    ('random_forest', '{"n_estimators": 100, "max_depth": 10}', false, true),
    ('svm', '{"C": 1.0, "kernel": "rbf"}', false, true),
    ('kmeans', '{"n_clusters": 8, "max_iter": 300}', false, true),
    ('gmm', '{"n_components": 8}', false, true),
    ('xgboost', '{"n_estimators": 100, "max_depth": 6}', true, true),
    ('naive_bayes', '{}', false, true),
    ('ridge', '{"alpha": 1.0}', false, true),
    ('lasso', '{"alpha": 1.0}', false, true),
    ('knn', '{"n_neighbors": 5}', false, true)
ON CONFLICT (algorithm) DO NOTHING;

-- Insert Tool Configurations
INSERT INTO neurondb.tool_configs (tool_name, display_name, default_params, default_limit, default_timeout_ms, enabled, is_default)
VALUES
    ('vector_search', 'Vector Search', '{"distance_metric": "l2"}', 10, 30000, true, true),
    ('generate_embedding', 'Generate Embedding', '{}', NULL, 30000, true, true),
    ('batch_embedding', 'Batch Embedding', '{}', NULL, 60000, true, true),
    ('rag', 'RAG Operations', '{"top_k": 5}', 5, 30000, true, true),
    ('analytics', 'Analytics Tools', '{}', NULL, 30000, true, true),
    ('ml_training', 'ML Training', '{}', NULL, 3600000, true, true),
    ('ml_prediction', 'ML Prediction', '{}', NULL, 30000, true, true)
ON CONFLICT (tool_name) DO NOTHING;

-- Insert System Configuration
INSERT INTO neurondb.system_configs (config_key, config_value, description, is_default)
VALUES
    ('features', '{"vector": true, "ml": true, "analytics": true, "rag": true}', 'Feature flags', true),
    ('default_timeout_ms', '30000', 'Default timeout for operations', true),
    ('rate_limiting', '{"enabled": false, "requests_per_minute": 60}', 'Rate limiting configuration', true),
    ('caching', '{"enabled": true, "ttl_seconds": 3600}', 'Caching policy', true)
ON CONFLICT (config_key) DO NOTHING;

-- Insert default prompts
INSERT INTO neurondb.prompts (prompt_name, description, template, variables, category, tags, is_default)
VALUES
    ('rag-query', 'RAG query prompt template', 'Context:\n{{context}}\n\nQuestion: {{question}}\n\nAnswer:', '[{"name": "context", "description": "Retrieved context", "required": true}, {"name": "question", "description": "User question", "required": true}]', 'rag', ARRAY['rag', 'query', 'qa'], true),
    ('summarization', 'Text summarization prompt', 'Summarize the following text:\n\n{{text}}', '[{"name": "text", "description": "Text to summarize", "required": true}]', 'text', ARRAY['summarization', 'text'], false),
    ('code-explanation', 'Code explanation prompt', 'Explain the following code:\n\n```{{language}}\n{{code}}\n```', '[{"name": "language", "description": "Programming language", "required": true}, {"name": "code", "description": "Code to explain", "required": true}]', 'code', ARRAY['code', 'explanation'], false)
ON CONFLICT (prompt_name) DO NOTHING;

-- ============================================================================
-- SECTION 13: MANAGEMENT FUNCTIONS
-- ============================================================================

-- ============================================================================
-- HELPER FUNCTIONS FOR ENCRYPTION
-- ============================================================================

-- Get encryption key from GUC or environment
CREATE OR REPLACE FUNCTION neurondb.get_encryption_key()
RETURNS TEXT AS $$
DECLARE
    v_key TEXT;
BEGIN
    BEGIN
        v_key := current_setting('neurondb.encryption_key', true);
    EXCEPTION WHEN OTHERS THEN
        v_key := NULL;
    END;
    
    IF v_key IS NULL OR v_key = '' THEN
        v_key := 'neurondb_default_encryption_key_change_in_production';
    END IF;
    
    RETURN v_key;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================================
-- LLM MODEL KEY MANAGEMENT FUNCTIONS
-- ============================================================================

-- Set/update API key for a model
CREATE OR REPLACE FUNCTION neurondb_set_model_key(
    p_model_name TEXT,
    p_api_key TEXT,
    p_expires_at TIMESTAMPTZ DEFAULT NULL
)
RETURNS BOOLEAN AS $$
DECLARE
    v_model_id INTEGER;
    v_salt BYTEA;
    v_encrypted_key BYTEA;
    v_encryption_key TEXT;
BEGIN
    SELECT model_id INTO v_model_id
    FROM neurondb.llm_models
    WHERE model_name = p_model_name;
    
    IF v_model_id IS NULL THEN
        RAISE EXCEPTION 'Model % not found', p_model_name;
    END IF;
    
    v_salt := gen_random_bytes(32);
    v_encryption_key := neurondb.get_encryption_key();
    v_encrypted_key := pgp_sym_encrypt(p_api_key, v_encryption_key);
    
    INSERT INTO neurondb.llm_model_keys (model_id, api_key_encrypted, encryption_salt, expires_at)
    VALUES (v_model_id, v_encrypted_key, v_salt, p_expires_at)
    ON CONFLICT (model_id) DO UPDATE
    SET api_key_encrypted = v_encrypted_key,
        encryption_salt = v_salt,
        expires_at = p_expires_at,
        updated_at = NOW();
    
    RETURN true;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Get decrypted API key (for internal use only)
CREATE OR REPLACE FUNCTION neurondb_get_model_key(p_model_name TEXT)
RETURNS TEXT AS $$
DECLARE
    v_model_id INTEGER;
    v_encrypted_key BYTEA;
    v_decrypted_key TEXT;
    v_encryption_key TEXT;
BEGIN
    SELECT model_id INTO v_model_id
    FROM neurondb.llm_models
    WHERE model_name = p_model_name;
    
    IF v_model_id IS NULL THEN
        RAISE EXCEPTION 'Model % not found', p_model_name;
    END IF;
    
    SELECT api_key_encrypted INTO v_encrypted_key
    FROM neurondb.llm_model_keys
    WHERE model_id = v_model_id;
    
    IF v_encrypted_key IS NULL THEN
        RETURN NULL;
    END IF;
    
    IF EXISTS (
        SELECT 1 FROM neurondb.llm_model_keys
        WHERE model_id = v_model_id
        AND expires_at IS NOT NULL
        AND expires_at < NOW()
    ) THEN
        RAISE EXCEPTION 'API key for model % has expired', p_model_name;
    END IF;
    
    v_encryption_key := neurondb.get_encryption_key();
    v_decrypted_key := pgp_sym_decrypt(v_encrypted_key, v_encryption_key);
    
    UPDATE neurondb.llm_model_keys
    SET last_used_at = NOW(),
        access_count = access_count + 1
    WHERE model_id = v_model_id;
    
    RETURN v_decrypted_key;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Remove API key from model
CREATE OR REPLACE FUNCTION neurondb_remove_model_key(p_model_name TEXT)
RETURNS BOOLEAN AS $$
DECLARE
    v_model_id INTEGER;
BEGIN
    SELECT model_id INTO v_model_id
    FROM neurondb.llm_models
    WHERE model_name = p_model_name;
    
    IF v_model_id IS NULL THEN
        RAISE EXCEPTION 'Model % not found', p_model_name;
    END IF;
    
    DELETE FROM neurondb.llm_model_keys
    WHERE model_id = v_model_id;
    
    RETURN true;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Rotate API key securely
CREATE OR REPLACE FUNCTION neurondb_rotate_model_key(
    p_model_name TEXT,
    p_new_key TEXT
)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN neurondb_set_model_key(p_model_name, p_new_key);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Internal function for NeuronMCP integration (resolves key)
CREATE OR REPLACE FUNCTION neurondb_resolve_model_key(p_model_name TEXT)
RETURNS TEXT AS $$
BEGIN
    RETURN neurondb_get_model_key(p_model_name);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================================
-- LLM MODEL MANAGEMENT FUNCTIONS
-- ============================================================================

-- List models with optional filters
CREATE OR REPLACE FUNCTION neurondb_list_models(
    p_provider_name TEXT DEFAULT NULL,
    p_model_type TEXT DEFAULT NULL,
    p_status TEXT DEFAULT NULL
)
RETURNS TABLE (
    model_id INTEGER,
    model_name TEXT,
    provider_name TEXT,
    model_type TEXT,
    status TEXT,
    has_api_key BOOLEAN,
    is_default BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        m.model_id,
        m.model_name,
        p.provider_name,
        m.model_type,
        m.status::TEXT,
        CASE WHEN mk.key_id IS NOT NULL THEN true ELSE false END AS has_api_key,
        m.is_default
    FROM neurondb.llm_models m
    JOIN neurondb.llm_providers p ON m.provider_id = p.provider_id
    LEFT JOIN neurondb.llm_model_keys mk ON m.model_id = mk.model_id
    WHERE (p_provider_name IS NULL OR p.provider_name = p.provider_name)
      AND (p_model_type IS NULL OR m.model_type = p_model_type)
      AND (p_status IS NULL OR m.status = p_status)
    ORDER BY m.model_name;
END;
$$ LANGUAGE plpgsql;

-- Get complete model configuration
CREATE OR REPLACE FUNCTION neurondb_get_model_config(p_model_name TEXT)
RETURNS JSONB AS $$
DECLARE
    v_result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'model_id', m.model_id,
        'model_name', m.model_name,
        'provider', p.provider_name,
        'model_type', m.model_type,
        'context_window', m.context_window,
        'embedding_dimension', m.embedding_dimension,
        'has_api_key', CASE WHEN mk.key_id IS NOT NULL THEN true ELSE false END,
        'config', mc.default_params,
        'base_url', mc.base_url
    ) INTO v_result
    FROM neurondb.llm_models m
    JOIN neurondb.llm_providers p ON m.provider_id = p.provider_id
    LEFT JOIN neurondb.llm_model_keys mk ON m.model_id = mk.model_id
    LEFT JOIN neurondb.llm_model_configs mc ON m.model_id = mc.model_id AND mc.is_active = true
    WHERE m.model_name = p_model_name;
    
    RETURN v_result;
END;
$$ LANGUAGE plpgsql;

-- Enable a model
CREATE OR REPLACE FUNCTION neurondb_enable_model(p_model_name TEXT)
RETURNS BOOLEAN AS $$
BEGIN
    UPDATE neurondb.llm_models
    SET status = 'available'
    WHERE model_name = p_model_name;
    
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Model % not found', p_model_name;
    END IF;
    
    RETURN true;
END;
$$ LANGUAGE plpgsql;

-- Disable a model
CREATE OR REPLACE FUNCTION neurondb_disable_model(p_model_name TEXT)
RETURNS BOOLEAN AS $$
BEGIN
    UPDATE neurondb.llm_models
    SET status = 'disabled'
    WHERE model_name = p_model_name;
    
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Model % not found', p_model_name;
    END IF;
    
    RETURN true;
END;
$$ LANGUAGE plpgsql;

-- Set default model for type
CREATE OR REPLACE FUNCTION neurondb_set_default_model(
    p_model_name TEXT,
    p_model_type TEXT
)
RETURNS BOOLEAN AS $$
BEGIN
    UPDATE neurondb.llm_models
    SET is_default = false
    WHERE model_type = p_model_type;
    
    UPDATE neurondb.llm_models
    SET is_default = true
    WHERE model_name = p_model_name AND model_type = p_model_type;
    
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Model % not found or type mismatch', p_model_name;
    END IF;
    
    RETURN true;
END;
$$ LANGUAGE plpgsql;

-- Smart model selection
CREATE OR REPLACE FUNCTION neurondb_get_model_for_operation(
    p_operation_type TEXT,
    p_preferred_model TEXT DEFAULT NULL
)
RETURNS TEXT AS $$
DECLARE
    v_model_name TEXT;
BEGIN
    IF p_preferred_model IS NOT NULL THEN
        SELECT model_name INTO v_model_name
        FROM neurondb.llm_models m
        JOIN neurondb.llm_providers p ON m.provider_id = p.provider_id
        WHERE m.model_name = p_preferred_model
          AND m.status = 'available'
          AND p.status = 'active'
          AND EXISTS (SELECT 1 FROM neurondb.llm_model_keys WHERE model_id = m.model_id);
        
        IF v_model_name IS NOT NULL THEN
            RETURN v_model_name;
        END IF;
    END IF;
    
    SELECT model_name INTO v_model_name
    FROM neurondb.llm_models m
    JOIN neurondb.llm_providers p ON m.provider_id = p.provider_id
    WHERE m.model_type = p_operation_type
      AND m.is_default = true
      AND m.status = 'available'
      AND p.status = 'active'
      AND EXISTS (SELECT 1 FROM neurondb.llm_model_keys WHERE model_id = m.model_id)
    LIMIT 1;
    
    RETURN v_model_name;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- LLM MODEL CONFIGURATION FUNCTIONS
-- ============================================================================

-- Set model configuration
CREATE OR REPLACE FUNCTION neurondb_set_model_config(
    p_model_name TEXT,
    p_config_name TEXT DEFAULT 'default',
    p_config_json JSONB DEFAULT '{}'
)
RETURNS BOOLEAN AS $$
DECLARE
    v_model_id INTEGER;
BEGIN
    SELECT model_id INTO v_model_id
    FROM neurondb.llm_models
    WHERE model_name = p_model_name;
    
    IF v_model_id IS NULL THEN
        RAISE EXCEPTION 'Model % not found', p_model_name;
    END IF;
    
    INSERT INTO neurondb.llm_model_configs (model_id, config_name, default_params)
    VALUES (v_model_id, p_config_name, p_config_json)
    ON CONFLICT (model_id, config_name) DO UPDATE
    SET default_params = p_config_json,
        updated_at = NOW();
    
    RETURN true;
END;
$$ LANGUAGE plpgsql;

-- Get model configuration
CREATE OR REPLACE FUNCTION neurondb_get_model_config(
    p_model_name TEXT,
    p_config_name TEXT DEFAULT 'default'
)
RETURNS JSONB AS $$
DECLARE
    v_result JSONB;
BEGIN
    SELECT mc.default_params INTO v_result
    FROM neurondb.llm_models m
    JOIN neurondb.llm_model_configs mc ON m.model_id = mc.model_id
    WHERE m.model_name = p_model_name
      AND mc.config_name = p_config_name
      AND mc.is_active = true;
    
    RETURN COALESCE(v_result, '{}'::jsonb);
END;
$$ LANGUAGE plpgsql;

-- Reset model configuration to defaults
CREATE OR REPLACE FUNCTION neurondb_reset_model_config(p_model_name TEXT)
RETURNS BOOLEAN AS $$
DECLARE
    v_model_id INTEGER;
BEGIN
    SELECT model_id INTO v_model_id
    FROM neurondb.llm_models
    WHERE model_name = p_model_name;
    
    IF v_model_id IS NULL THEN
        RAISE EXCEPTION 'Model % not found', p_model_name;
    END IF;
    
    UPDATE neurondb.llm_model_configs
    SET default_params = '{}',
        updated_at = NOW()
    WHERE model_id = v_model_id;
    
    RETURN true;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- PROVIDER MANAGEMENT FUNCTIONS
-- ============================================================================

-- Add custom provider
CREATE OR REPLACE FUNCTION neurondb_add_provider(
    p_provider_name TEXT,
    p_config_json JSONB
)
RETURNS INTEGER AS $$
DECLARE
    v_provider_id INTEGER;
BEGIN
    INSERT INTO neurondb.llm_providers (
        provider_name,
        display_name,
        default_base_url,
        auth_method,
        metadata
    )
    VALUES (
        p_provider_name,
        COALESCE(p_config_json->>'display_name', p_provider_name),
        p_config_json->>'default_base_url',
        COALESCE(p_config_json->>'auth_method', 'api_key'),
        p_config_json
    )
    RETURNING provider_id INTO v_provider_id;
    
    RETURN v_provider_id;
END;
$$ LANGUAGE plpgsql;

-- List all providers
CREATE OR REPLACE FUNCTION neurondb_list_providers()
RETURNS TABLE (
    provider_id INTEGER,
    provider_name TEXT,
    display_name TEXT,
    status TEXT,
    supports_embeddings BOOLEAN,
    supports_chat BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        p.provider_id,
        p.provider_name,
        p.display_name,
        p.status::TEXT,
        p.supports_embeddings,
        p.supports_chat
    FROM neurondb.llm_providers p
    ORDER BY p.provider_name;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- USAGE & ANALYTICS FUNCTIONS
-- ============================================================================

-- Log model usage
CREATE OR REPLACE FUNCTION neurondb_log_model_usage(
    p_model_name TEXT,
    p_operation_type TEXT,
    p_tokens_input INTEGER DEFAULT NULL,
    p_tokens_output INTEGER DEFAULT NULL,
    p_latency_ms INTEGER DEFAULT NULL,
    p_success BOOLEAN DEFAULT true,
    p_error_message TEXT DEFAULT NULL
)
RETURNS BIGINT AS $$
DECLARE
    v_model_id INTEGER;
    v_usage_id BIGINT;
    v_cost NUMERIC(10,6) := 0;
BEGIN
    SELECT model_id INTO v_model_id
    FROM neurondb.llm_models
    WHERE model_name = p_model_name;
    
    IF v_model_id IS NULL THEN
        RAISE EXCEPTION 'Model % not found', p_model_name;
    END IF;
    
    IF p_tokens_input IS NOT NULL OR p_tokens_output IS NOT NULL THEN
        SELECT 
            COALESCE((p_tokens_input::NUMERIC / 1000.0) * m.cost_per_1k_tokens_input, 0) +
            COALESCE((p_tokens_output::NUMERIC / 1000.0) * m.cost_per_1k_tokens_output, 0)
        INTO v_cost
        FROM neurondb.llm_models m
        WHERE m.model_id = v_model_id;
    END IF;
    
    INSERT INTO neurondb.llm_model_usage (
        model_id,
        operation_type,
        tokens_input,
        tokens_output,
        cost,
        latency_ms,
        success,
        error_message
    )
    VALUES (
        v_model_id,
        p_operation_type,
        p_tokens_input,
        p_tokens_output,
        v_cost,
        p_latency_ms,
        p_success,
        p_error_message
    )
    RETURNING usage_id INTO v_usage_id;
    
    UPDATE neurondb.llm_models
    SET last_used_at = NOW()
    WHERE model_id = v_model_id;
    
    RETURN v_usage_id;
END;
$$ LANGUAGE plpgsql;

-- Get model statistics
CREATE OR REPLACE FUNCTION neurondb_get_model_stats(
    p_model_name TEXT,
    p_days INTEGER DEFAULT 30
)
RETURNS JSONB AS $$
DECLARE
    v_model_id INTEGER;
    v_result JSONB;
BEGIN
    SELECT model_id INTO v_model_id
    FROM neurondb.llm_models
    WHERE model_name = p_model_name;
    
    IF v_model_id IS NULL THEN
        RAISE EXCEPTION 'Model % not found', p_model_name;
    END IF;
    
    SELECT jsonb_build_object(
        'total_requests', COUNT(*),
        'successful_requests', COUNT(*) FILTER (WHERE success = true),
        'failed_requests', COUNT(*) FILTER (WHERE success = false),
        'total_tokens_input', SUM(tokens_input),
        'total_tokens_output', SUM(tokens_output),
        'total_cost', SUM(cost),
        'avg_latency_ms', AVG(latency_ms),
        'operations', jsonb_object_agg(operation_type, COUNT(*))
    ) INTO v_result
    FROM neurondb.llm_model_usage
    WHERE model_id = v_model_id
      AND created_at >= NOW() - (p_days || ' days')::INTERVAL;
    
    RETURN COALESCE(v_result, '{}'::jsonb);
END;
$$ LANGUAGE plpgsql;

-- Get cost summary
CREATE OR REPLACE FUNCTION neurondb_get_cost_summary(p_days INTEGER DEFAULT 30)
RETURNS JSONB AS $$
DECLARE
    v_result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'total_cost', SUM(cost),
        'by_model', (
            SELECT jsonb_object_agg(m.model_name, SUM(u.cost))
            FROM neurondb.llm_model_usage u
            JOIN neurondb.llm_models m ON u.model_id = m.model_id
            WHERE u.created_at >= NOW() - (p_days || ' days')::INTERVAL
            GROUP BY m.model_name
        ),
        'by_operation', (
            SELECT jsonb_object_agg(operation_type, SUM(cost))
            FROM neurondb.llm_model_usage
            WHERE created_at >= NOW() - (p_days || ' days')::INTERVAL
            GROUP BY operation_type
        )
    ) INTO v_result
    FROM neurondb.llm_model_usage
    WHERE created_at >= NOW() - (p_days || ' days')::INTERVAL;
    
    RETURN COALESCE(v_result, '{}'::jsonb);
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- VECTOR INDEX MANAGEMENT FUNCTIONS
-- ============================================================================

-- Get index configuration
CREATE OR REPLACE FUNCTION neurondb_get_index_config(
    p_table_name TEXT,
    p_vector_column TEXT
)
RETURNS JSONB AS $$
DECLARE
    v_result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'index_type', index_type,
        'hnsw_m', hnsw_m,
        'hnsw_ef_construction', hnsw_ef_construction,
        'hnsw_ef_search', hnsw_ef_search,
        'ivf_lists', ivf_lists,
        'ivf_probes', ivf_probes,
        'distance_metric', distance_metric
    ) INTO v_result
    FROM neurondb.index_configs
    WHERE table_name = p_table_name
      AND vector_column = p_vector_column;
    
    RETURN COALESCE(v_result, '{}'::jsonb);
END;
$$ LANGUAGE plpgsql;

-- Set index configuration
CREATE OR REPLACE FUNCTION neurondb_set_index_config(
    p_table_name TEXT,
    p_vector_column TEXT,
    p_config_json JSONB
)
RETURNS BOOLEAN AS $$
BEGIN
    INSERT INTO neurondb.index_configs (
        table_name,
        vector_column,
        index_type,
        hnsw_m,
        hnsw_ef_construction,
        hnsw_ef_search,
        ivf_lists,
        ivf_probes,
        distance_metric
    )
    VALUES (
        p_table_name,
        p_vector_column,
        COALESCE(p_config_json->>'index_type', 'hnsw'),
        (p_config_json->>'hnsw_m')::INTEGER,
        (p_config_json->>'hnsw_ef_construction')::INTEGER,
        (p_config_json->>'hnsw_ef_search')::INTEGER,
        (p_config_json->>'ivf_lists')::INTEGER,
        (p_config_json->>'ivf_probes')::INTEGER,
        COALESCE(p_config_json->>'distance_metric', 'l2')
    )
    ON CONFLICT (table_name, vector_column) DO UPDATE
    SET index_type = EXCLUDED.index_type,
        hnsw_m = EXCLUDED.hnsw_m,
        hnsw_ef_construction = EXCLUDED.hnsw_ef_construction,
        hnsw_ef_search = EXCLUDED.hnsw_ef_search,
        ivf_lists = EXCLUDED.ivf_lists,
        ivf_probes = EXCLUDED.ivf_probes,
        distance_metric = EXCLUDED.distance_metric,
        updated_at = NOW();
    
    RETURN true;
END;
$$ LANGUAGE plpgsql;

-- Apply index template
CREATE OR REPLACE FUNCTION neurondb_apply_index_template(
    p_template_name TEXT,
    p_table_name TEXT,
    p_vector_column TEXT
)
RETURNS BOOLEAN AS $$
DECLARE
    v_config_json JSONB;
BEGIN
    SELECT config_json INTO v_config_json
    FROM neurondb.index_templates
    WHERE template_name = p_template_name;
    
    IF v_config_json IS NULL THEN
        RAISE EXCEPTION 'Template % not found', p_template_name;
    END IF;
    
    RETURN neurondb_set_index_config(p_table_name, p_vector_column, v_config_json);
END;
$$ LANGUAGE plpgsql;

-- List index templates
CREATE OR REPLACE FUNCTION neurondb_list_index_templates()
RETURNS TABLE (
    template_name TEXT,
    description TEXT,
    index_type TEXT,
    is_default BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        t.template_name,
        t.description,
        t.index_type,
        t.is_default
    FROM neurondb.index_templates t
    ORDER BY t.template_name;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- WORKER MANAGEMENT FUNCTIONS
-- ============================================================================

-- Get worker configuration
CREATE OR REPLACE FUNCTION neurondb_get_worker_config(p_worker_name TEXT)
RETURNS JSONB AS $$
DECLARE
    v_result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'worker_name', worker_name,
        'enabled', enabled,
        'naptime_ms', naptime_ms,
        'config', config_json
    ) INTO v_result
    FROM neurondb.worker_configs
    WHERE worker_name = p_worker_name;
    
    RETURN COALESCE(v_result, '{}'::jsonb);
END;
$$ LANGUAGE plpgsql;

-- Set worker configuration
CREATE OR REPLACE FUNCTION neurondb_set_worker_config(
    p_worker_name TEXT,
    p_config_json JSONB
)
RETURNS BOOLEAN AS $$
BEGIN
    INSERT INTO neurondb.worker_configs (worker_name, display_name, config_json)
    VALUES (
        p_worker_name,
        COALESCE(p_config_json->>'display_name', p_worker_name),
        p_config_json
    )
    ON CONFLICT (worker_name) DO UPDATE
    SET config_json = EXCLUDED.config_json,
        updated_at = NOW();
    
    RETURN true;
END;
$$ LANGUAGE plpgsql;

-- Enable worker
CREATE OR REPLACE FUNCTION neurondb_enable_worker(p_worker_name TEXT)
RETURNS BOOLEAN AS $$
BEGIN
    UPDATE neurondb.worker_configs
    SET enabled = true
    WHERE worker_name = p_worker_name;
    
    RETURN FOUND;
END;
$$ LANGUAGE plpgsql;

-- Disable worker
CREATE OR REPLACE FUNCTION neurondb_disable_worker(p_worker_name TEXT)
RETURNS BOOLEAN AS $$
BEGIN
    UPDATE neurondb.worker_configs
    SET enabled = false
    WHERE worker_name = p_worker_name;
    
    RETURN FOUND;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- ML DEFAULTS MANAGEMENT FUNCTIONS
-- ============================================================================

-- Get ML defaults for algorithm
CREATE OR REPLACE FUNCTION neurondb_get_ml_defaults(p_algorithm TEXT)
RETURNS JSONB AS $$
DECLARE
    v_result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'algorithm', algorithm,
        'hyperparameters', default_hyperparameters,
        'training_settings', default_training_settings,
        'use_gpu', use_gpu,
        'gpu_device', gpu_device,
        'batch_size', batch_size,
        'max_iterations', max_iterations
    ) INTO v_result
    FROM neurondb.ml_default_configs
    WHERE algorithm = p_algorithm;
    
    RETURN COALESCE(v_result, '{}'::jsonb);
END;
$$ LANGUAGE plpgsql;

-- Set ML defaults
CREATE OR REPLACE FUNCTION neurondb_set_ml_defaults(
    p_algorithm TEXT,
    p_config_json JSONB
)
RETURNS BOOLEAN AS $$
BEGIN
    INSERT INTO neurondb.ml_default_configs (
        algorithm,
        default_hyperparameters,
        default_training_settings,
        use_gpu,
        gpu_device,
        batch_size,
        max_iterations
    )
    VALUES (
        p_algorithm,
        COALESCE(p_config_json->'hyperparameters', '{}'::jsonb),
        COALESCE(p_config_json->'training_settings', '{}'::jsonb),
        COALESCE((p_config_json->>'use_gpu')::BOOLEAN, false),
        COALESCE((p_config_json->>'gpu_device')::INTEGER, 0),
        (p_config_json->>'batch_size')::INTEGER,
        (p_config_json->>'max_iterations')::INTEGER
    )
    ON CONFLICT (algorithm) DO UPDATE
    SET default_hyperparameters = EXCLUDED.default_hyperparameters,
        default_training_settings = EXCLUDED.default_training_settings,
        use_gpu = EXCLUDED.use_gpu,
        gpu_device = EXCLUDED.gpu_device,
        batch_size = EXCLUDED.batch_size,
        max_iterations = EXCLUDED.max_iterations,
        updated_at = NOW();
    
    RETURN true;
END;
$$ LANGUAGE plpgsql;

-- Apply ML template
CREATE OR REPLACE FUNCTION neurondb_apply_ml_template(
    p_template_name TEXT,
    p_project_name TEXT
)
RETURNS JSONB AS $$
DECLARE
    v_template_config JSONB;
BEGIN
    SELECT template_config INTO v_template_config
    FROM neurondb.ml_model_templates
    WHERE template_name = p_template_name;
    
    IF v_template_config IS NULL THEN
        RAISE EXCEPTION 'Template % not found', p_template_name;
    END IF;
    
    RETURN v_template_config;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- TOOL CONFIGURATION FUNCTIONS
-- ============================================================================

-- Get tool configuration
CREATE OR REPLACE FUNCTION neurondb_get_tool_config(p_tool_name TEXT)
RETURNS JSONB AS $$
DECLARE
    v_result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'tool_name', tool_name,
        'default_params', default_params,
        'default_limit', default_limit,
        'default_timeout_ms', default_timeout_ms,
        'enabled', enabled
    ) INTO v_result
    FROM neurondb.tool_configs
    WHERE tool_name = p_tool_name;
    
    RETURN COALESCE(v_result, '{}'::jsonb);
END;
$$ LANGUAGE plpgsql;

-- Set tool configuration
CREATE OR REPLACE FUNCTION neurondb_set_tool_config(
    p_tool_name TEXT,
    p_config_json JSONB
)
RETURNS BOOLEAN AS $$
BEGIN
    INSERT INTO neurondb.tool_configs (
        tool_name,
        display_name,
        default_params,
        default_limit,
        default_timeout_ms,
        enabled
    )
    VALUES (
        p_tool_name,
        COALESCE(p_config_json->>'display_name', p_tool_name),
        COALESCE(p_config_json->'default_params', '{}'::jsonb),
        (p_config_json->>'default_limit')::INTEGER,
        COALESCE((p_config_json->>'default_timeout_ms')::INTEGER, 30000),
        COALESCE((p_config_json->>'enabled')::BOOLEAN, true)
    )
    ON CONFLICT (tool_name) DO UPDATE
    SET default_params = EXCLUDED.default_params,
        default_limit = EXCLUDED.default_limit,
        default_timeout_ms = EXCLUDED.default_timeout_ms,
        enabled = EXCLUDED.enabled,
        updated_at = NOW();
    
    RETURN true;
END;
$$ LANGUAGE plpgsql;

-- Reset tool configuration
CREATE OR REPLACE FUNCTION neurondb_reset_tool_config(p_tool_name TEXT)
RETURNS BOOLEAN AS $$
BEGIN
    UPDATE neurondb.tool_configs
    SET default_params = '{}',
        updated_at = NOW()
    WHERE tool_name = p_tool_name;
    
    RETURN FOUND;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- SYSTEM CONFIGURATION FUNCTIONS
-- ============================================================================

-- Get system configuration
CREATE OR REPLACE FUNCTION neurondb_get_system_config()
RETURNS JSONB AS $$
DECLARE
    v_result JSONB := '{}'::jsonb;
    v_row RECORD;
BEGIN
    FOR v_row IN SELECT config_key, config_value FROM neurondb.system_configs
    LOOP
        v_result := v_result || jsonb_build_object(v_row.config_key, v_row.config_value);
    END LOOP;
    
    RETURN v_result;
END;
$$ LANGUAGE plpgsql;

-- Set system configuration
CREATE OR REPLACE FUNCTION neurondb_set_system_config(p_config_json JSONB)
RETURNS BOOLEAN AS $$
DECLARE
    v_key TEXT;
    v_value JSONB;
BEGIN
    FOR v_key, v_value IN SELECT * FROM jsonb_each(p_config_json)
    LOOP
        INSERT INTO neurondb.system_configs (config_key, config_value)
        VALUES (v_key, v_value)
        ON CONFLICT (config_key) DO UPDATE
        SET config_value = EXCLUDED.config_value,
            updated_at = NOW();
    END LOOP;
    
    RETURN true;
END;
$$ LANGUAGE plpgsql;

-- Get all configurations (unified view)
CREATE OR REPLACE FUNCTION neurondb_get_all_configs()
RETURNS JSONB AS $$
DECLARE
    v_result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'llm_models', (
            SELECT jsonb_agg(jsonb_build_object(
                'model_name', model_name,
                'provider', provider_name,
                'type', model_type,
                'has_key', CASE WHEN key_id IS NOT NULL THEN true ELSE false END
            ))
            FROM neurondb.v_llm_models_active
        ),
        'index_templates', (
            SELECT jsonb_agg(jsonb_build_object(
                'template_name', template_name,
                'index_type', index_type
            ))
            FROM neurondb.index_templates
        ),
        'workers', (
            SELECT jsonb_agg(jsonb_build_object(
                'worker_name', worker_name,
                'enabled', enabled
            ))
            FROM neurondb.worker_configs
        ),
        'ml_defaults', (
            SELECT jsonb_agg(algorithm)
            FROM neurondb.ml_default_configs
        ),
        'tools', (
            SELECT jsonb_agg(jsonb_build_object(
                'tool_name', tool_name,
                'enabled', enabled
            ))
            FROM neurondb.tool_configs
        ),
        'system', neurondb_get_system_config()
    ) INTO v_result;
    
    RETURN COALESCE(v_result, '{}'::jsonb);
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- COMPLETION MESSAGE
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE 'NeuronMCP Configuration Schema setup completed successfully!';
    RAISE NOTICE 'Created 14 tables, indexes, views, triggers, and pre-populated default data.';
    RAISE NOTICE 'Next steps:';
    RAISE NOTICE '1. Set API keys using: SELECT neurondb_set_model_key(''model_name'', ''api_key'');';
    RAISE NOTICE '2. Verify setup: SELECT * FROM neurondb.v_llm_models_active;';
    RAISE NOTICE '3. Check ready models: SELECT * FROM neurondb.v_llm_models_ready;';
    RAISE NOTICE '4. List prompts: SELECT * FROM neurondb.prompts;';
END $$;
