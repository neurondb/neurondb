-- -------------------------------------------------------------------------
-- NeuronDB LLM Model Storage and Management Schema (3.1.0)
-- Standard LLM model registry, providers, prompts, conversations,
-- guardrails, training tracking, evaluation, and cost management.
-- Requires: CREATE EXTENSION plpython3u for PL/Python functions.
-- -------------------------------------------------------------------------

-- =========================================================================
-- Module 1: LLM Model Registry (Standard Model Storage)
-- =========================================================================

CREATE TABLE IF NOT EXISTS neurondb.llm_models (
    model_id BIGSERIAL PRIMARY KEY,
    name text UNIQUE NOT NULL,
    family text,
    architecture jsonb NOT NULL DEFAULT '{}',
    base_model text,
    model_type text NOT NULL DEFAULT 'causal_lm',
    task text,
    license text,
    description text,
    tags text[] DEFAULT '{}',
    parameter_count bigint,
    context_length integer,
    languages text[] DEFAULT '{}',
    format text,
    quantization text,
    quantization_config jsonb DEFAULT '{}',
    tensor_type text DEFAULT 'float16',
    size_bytes bigint,
    storage_mode text DEFAULT 'filesystem',
    filesystem_path text,
    source_url text,
    sha256 text,
    is_active boolean DEFAULT true,
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now(),
    created_by text DEFAULT current_user,
    metadata jsonb DEFAULT '{}'
);
COMMENT ON TABLE neurondb.llm_models IS 'LLM model registry (HuggingFace-style metadata and storage)';

CREATE TABLE IF NOT EXISTS neurondb.llm_model_files (
    file_id BIGSERIAL PRIMARY KEY,
    model_id bigint NOT NULL REFERENCES neurondb.llm_models(model_id) ON DELETE CASCADE,
    filename text NOT NULL,
    file_type text NOT NULL,
    file_format text,
    shard_index integer,
    shard_count integer,
    size_bytes bigint,
    sha256 text,
    storage_mode text NOT NULL DEFAULT 'filesystem',
    filesystem_path text,
    file_data bytea,
    large_object_oid oid,
    created_at timestamptz DEFAULT now(),
    UNIQUE(model_id, filename)
);
COMMENT ON TABLE neurondb.llm_model_files IS 'Individual weight/config/tokenizer files per model';

CREATE TABLE IF NOT EXISTS neurondb.llm_providers (
    provider_id BIGSERIAL PRIMARY KEY,
    name text UNIQUE NOT NULL,
    provider_type text NOT NULL,
    api_base text NOT NULL,
    api_key_encrypted text,
    default_model text,
    rate_limit_rpm integer,
    rate_limit_tpm integer,
    timeout_ms integer DEFAULT 30000,
    retry_config jsonb DEFAULT '{"max_retries": 3, "backoff_factor": 2}',
    headers jsonb DEFAULT '{}',
    capabilities text[] DEFAULT '{}',
    is_active boolean DEFAULT true,
    priority integer DEFAULT 100,
    created_at timestamptz DEFAULT now()
);
COMMENT ON TABLE neurondb.llm_providers IS 'Multi-provider registry (OpenAI, HuggingFace, Ollama, vLLM, etc.)';

CREATE TABLE IF NOT EXISTS neurondb.llm_model_deployments (
    deployment_id BIGSERIAL PRIMARY KEY,
    model_id bigint NOT NULL REFERENCES neurondb.llm_models(model_id) ON DELETE CASCADE,
    provider_id bigint NOT NULL REFERENCES neurondb.llm_providers(provider_id) ON DELETE CASCADE,
    endpoint_model_name text,
    gpu_memory_gb float,
    tensor_parallel_size integer DEFAULT 1,
    max_batch_size integer,
    is_active boolean DEFAULT true,
    health_check_url text,
    last_health_check timestamptz,
    health_status text DEFAULT 'unknown',
    created_at timestamptz DEFAULT now()
);
COMMENT ON TABLE neurondb.llm_model_deployments IS 'Map models to providers (same model can be served by multiple providers)';

-- Training runs (needed for llm_model_versions and llm_adapters)
CREATE TABLE IF NOT EXISTS neurondb.llm_training_runs (
    run_id BIGSERIAL PRIMARY KEY,
    model_id bigint REFERENCES neurondb.llm_models(model_id) ON DELETE SET NULL,
    run_name text,
    run_type text NOT NULL,
    base_model_name text,
    dataset_name text,
    dataset_size bigint,
    config jsonb NOT NULL DEFAULT '{}',
    deepspeed_config jsonb DEFAULT '{}',
    hardware jsonb DEFAULT '{}',
    status text DEFAULT 'pending',
    started_at timestamptz,
    completed_at timestamptz,
    total_steps bigint,
    current_step bigint DEFAULT 0,
    best_metric_name text,
    best_metric_value float,
    final_metrics jsonb DEFAULT '{}',
    error_message text,
    created_at timestamptz DEFAULT now()
);
COMMENT ON TABLE neurondb.llm_training_runs IS 'Training/fine-tuning run tracking';

CREATE TABLE IF NOT EXISTS neurondb.llm_datasets (
    dataset_id BIGSERIAL PRIMARY KEY,
    name text UNIQUE NOT NULL,
    source text,
    description text,
    num_examples bigint,
    num_train bigint,
    num_val bigint,
    num_test bigint,
    schema_json jsonb DEFAULT '{}',
    filesystem_path text,
    splits jsonb DEFAULT '{}',
    preprocessing jsonb DEFAULT '{}',
    created_at timestamptz DEFAULT now()
);
COMMENT ON TABLE neurondb.llm_datasets IS 'Training dataset registry';

CREATE TABLE IF NOT EXISTS neurondb.llm_model_versions (
    version_id BIGSERIAL PRIMARY KEY,
    model_id bigint NOT NULL REFERENCES neurondb.llm_models(model_id) ON DELETE CASCADE,
    version text NOT NULL,
    parent_version_id bigint REFERENCES neurondb.llm_model_versions(version_id),
    change_type text,
    changelog text,
    training_run_id bigint REFERENCES neurondb.llm_training_runs(run_id) ON DELETE SET NULL,
    metrics jsonb DEFAULT '{}',
    is_current boolean DEFAULT true,
    created_at timestamptz DEFAULT now(),
    UNIQUE(model_id, version)
);
COMMENT ON TABLE neurondb.llm_model_versions IS 'Version history and lineage';

CREATE TABLE IF NOT EXISTS neurondb.llm_tokenizers (
    tokenizer_id BIGSERIAL PRIMARY KEY,
    model_id bigint NOT NULL REFERENCES neurondb.llm_models(model_id) ON DELETE CASCADE,
    tokenizer_type text NOT NULL,
    vocab_size integer,
    max_length integer,
    padding_side text DEFAULT 'right',
    truncation_side text DEFAULT 'right',
    special_tokens jsonb DEFAULT '{}',
    added_tokens jsonb DEFAULT '[]',
    tokenizer_config jsonb DEFAULT '{}',
    vocab_data bytea,
    merges_data bytea,
    filesystem_path text,
    created_at timestamptz DEFAULT now()
);
COMMENT ON TABLE neurondb.llm_tokenizers IS 'Tokenizer storage per model';

CREATE TABLE IF NOT EXISTS neurondb.llm_adapters (
    adapter_id BIGSERIAL PRIMARY KEY,
    name text NOT NULL,
    model_id bigint NOT NULL REFERENCES neurondb.llm_models(model_id) ON DELETE CASCADE,
    adapter_type text NOT NULL DEFAULT 'lora',
    rank integer,
    alpha float,
    dropout float DEFAULT 0.0,
    target_modules text[] DEFAULT '{}',
    config jsonb DEFAULT '{}',
    size_bytes bigint,
    storage_mode text DEFAULT 'database',
    filesystem_path text,
    adapter_data bytea,
    training_run_id bigint REFERENCES neurondb.llm_training_runs(run_id) ON DELETE SET NULL,
    metrics jsonb DEFAULT '{}',
    is_active boolean DEFAULT true,
    created_at timestamptz DEFAULT now(),
    UNIQUE(name)
);
COMMENT ON TABLE neurondb.llm_adapters IS 'LoRA/QLoRA/PEFT adapters';

CREATE TABLE IF NOT EXISTS neurondb.llm_model_tags (
    model_id bigint NOT NULL REFERENCES neurondb.llm_models(model_id) ON DELETE CASCADE,
    tag text NOT NULL,
    category text DEFAULT 'general',
    PRIMARY KEY(model_id, tag)
);
COMMENT ON TABLE neurondb.llm_model_tags IS 'Tagging system for models';

-- =========================================================================
-- Module 3: Prompt Management
-- =========================================================================

CREATE TABLE IF NOT EXISTS neurondb.llm_prompt_templates (
    template_id BIGSERIAL PRIMARY KEY,
    name text UNIQUE NOT NULL,
    template text NOT NULL,
    system_prompt text,
    variables jsonb DEFAULT '{}',
    model_name text,
    temperature float,
    max_tokens integer,
    stop_sequences text[] DEFAULT '{}',
    output_format text,
    output_schema jsonb DEFAULT '{}',
    version integer DEFAULT 1,
    tags text[] DEFAULT '{}',
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now()
);
COMMENT ON TABLE neurondb.llm_prompt_templates IS 'Reusable prompt templates with {{variable}} syntax';

CREATE TABLE IF NOT EXISTS neurondb.llm_prompt_chains (
    chain_id BIGSERIAL PRIMARY KEY,
    name text UNIQUE NOT NULL,
    description text,
    steps jsonb NOT NULL DEFAULT '[]',
    created_at timestamptz DEFAULT now()
);
COMMENT ON TABLE neurondb.llm_prompt_chains IS 'Multi-step prompt chains (pipelines)';

-- =========================================================================
-- Module 4: Conversation Management
-- =========================================================================

CREATE TABLE IF NOT EXISTS neurondb.llm_conversations (
    conversation_id BIGSERIAL PRIMARY KEY,
    session_id uuid DEFAULT gen_random_uuid(),
    title text,
    model_name text,
    system_prompt text,
    context_window integer,
    metadata jsonb DEFAULT '{}',
    user_id text DEFAULT current_user,
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now()
);
COMMENT ON TABLE neurondb.llm_conversations IS 'Chat sessions';

CREATE TABLE IF NOT EXISTS neurondb.llm_messages (
    message_id BIGSERIAL PRIMARY KEY,
    conversation_id bigint NOT NULL REFERENCES neurondb.llm_conversations(conversation_id) ON DELETE CASCADE,
    role text NOT NULL,
    content text NOT NULL DEFAULT '',
    name text,
    tool_calls jsonb DEFAULT '[]',
    tool_call_id text,
    tokens_in integer,
    tokens_out integer,
    latency_ms integer,
    model_name text,
    finish_reason text,
    metadata jsonb DEFAULT '{}',
    created_at timestamptz DEFAULT now()
);
COMMENT ON TABLE neurondb.llm_messages IS 'Individual messages in conversations';

-- =========================================================================
-- Module 5: Guardrails and Safety
-- =========================================================================

CREATE TABLE IF NOT EXISTS neurondb.llm_guardrails (
    guardrail_id BIGSERIAL PRIMARY KEY,
    name text UNIQUE NOT NULL,
    guardrail_type text NOT NULL,
    config jsonb NOT NULL DEFAULT '{}',
    action text DEFAULT 'block',
    severity text DEFAULT 'medium',
    is_active boolean DEFAULT true,
    applies_to text[],
    created_at timestamptz DEFAULT now()
);
COMMENT ON TABLE neurondb.llm_guardrails IS 'Content safety rules';

CREATE TABLE IF NOT EXISTS neurondb.llm_guardrail_log (
    log_id BIGSERIAL PRIMARY KEY,
    guardrail_id bigint REFERENCES neurondb.llm_guardrails(guardrail_id) ON DELETE SET NULL,
    conversation_id bigint,
    input_text text,
    triggered_rule text,
    action_taken text,
    details jsonb DEFAULT '{}',
    created_at timestamptz DEFAULT now()
);
COMMENT ON TABLE neurondb.llm_guardrail_log IS 'Guardrail trigger log';

-- =========================================================================
-- Module 6: Training Metrics and Checkpoints
-- =========================================================================

CREATE TABLE IF NOT EXISTS neurondb.llm_training_metrics (
    metric_id BIGSERIAL PRIMARY KEY,
    run_id bigint NOT NULL REFERENCES neurondb.llm_training_runs(run_id) ON DELETE CASCADE,
    step bigint NOT NULL,
    epoch float,
    metric_name text NOT NULL,
    metric_value float NOT NULL,
    recorded_at timestamptz DEFAULT now()
);
COMMENT ON TABLE neurondb.llm_training_metrics IS 'Step-level training metrics';

CREATE TABLE IF NOT EXISTS neurondb.llm_checkpoints (
    checkpoint_id BIGSERIAL PRIMARY KEY,
    run_id bigint NOT NULL REFERENCES neurondb.llm_training_runs(run_id) ON DELETE CASCADE,
    step bigint NOT NULL,
    epoch float,
    filesystem_path text NOT NULL,
    size_bytes bigint,
    metrics jsonb DEFAULT '{}',
    is_best boolean DEFAULT false,
    created_at timestamptz DEFAULT now()
);
COMMENT ON TABLE neurondb.llm_checkpoints IS 'Model checkpoints during training';

-- =========================================================================
-- Module 7: Evaluation and Benchmarks
-- =========================================================================

CREATE TABLE IF NOT EXISTS neurondb.llm_benchmarks (
    benchmark_id BIGSERIAL PRIMARY KEY,
    name text UNIQUE NOT NULL,
    description text,
    dataset_id bigint REFERENCES neurondb.llm_datasets(dataset_id) ON DELETE SET NULL,
    metric_names text[] DEFAULT '{}',
    num_examples integer,
    config jsonb DEFAULT '{}',
    created_at timestamptz DEFAULT now()
);
COMMENT ON TABLE neurondb.llm_benchmarks IS 'Benchmark definitions';

CREATE TABLE IF NOT EXISTS neurondb.llm_eval_results (
    result_id BIGSERIAL PRIMARY KEY,
    model_id bigint NOT NULL REFERENCES neurondb.llm_models(model_id) ON DELETE CASCADE,
    benchmark_id bigint NOT NULL REFERENCES neurondb.llm_benchmarks(benchmark_id) ON DELETE CASCADE,
    version text,
    adapter_id bigint REFERENCES neurondb.llm_adapters(adapter_id) ON DELETE SET NULL,
    metrics jsonb NOT NULL DEFAULT '{}',
    num_correct integer,
    num_total integer,
    latency_p50_ms float,
    latency_p95_ms float,
    latency_p99_ms float,
    tokens_per_second float,
    details jsonb DEFAULT '{}',
    evaluated_at timestamptz DEFAULT now(),
    evaluated_by text DEFAULT current_user
);
COMMENT ON TABLE neurondb.llm_eval_results IS 'Model evaluation results';

-- =========================================================================
-- Module 8: Cost and Token Usage
-- =========================================================================

CREATE TABLE IF NOT EXISTS neurondb.llm_token_usage (
    usage_id BIGSERIAL PRIMARY KEY,
    model_name text NOT NULL,
    provider_name text,
    operation text NOT NULL,
    tokens_input integer DEFAULT 0,
    tokens_output integer DEFAULT 0,
    tokens_total integer GENERATED ALWAYS AS (tokens_input + tokens_output) STORED,
    cost_usd numeric(12,8),
    latency_ms integer,
    user_id text DEFAULT current_user,
    conversation_id bigint,
    cached boolean DEFAULT false,
    created_at timestamptz DEFAULT now()
);
COMMENT ON TABLE neurondb.llm_token_usage IS 'Per-request token and cost tracking';

CREATE TABLE IF NOT EXISTS neurondb.llm_pricing (
    pricing_id BIGSERIAL PRIMARY KEY,
    model_name text NOT NULL,
    provider_name text NOT NULL,
    input_cost_per_1k numeric(12,8),
    output_cost_per_1k numeric(12,8),
    embedding_cost_per_1k numeric(12,8),
    effective_from timestamptz DEFAULT now(),
    effective_until timestamptz
);
COMMENT ON TABLE neurondb.llm_pricing IS 'Model pricing config';

CREATE TABLE IF NOT EXISTS neurondb.llm_budgets (
    budget_id BIGSERIAL PRIMARY KEY,
    name text UNIQUE NOT NULL,
    user_id text,
    model_name text,
    max_cost_usd numeric(12,2),
    max_tokens bigint,
    period text DEFAULT 'monthly',
    current_cost_usd numeric(12,2) DEFAULT 0,
    current_tokens bigint DEFAULT 0,
    period_start timestamptz DEFAULT now(),
    is_active boolean DEFAULT true,
    action_on_exceed text DEFAULT 'block'
);
COMMENT ON TABLE neurondb.llm_budgets IS 'Spending limits';

-- =========================================================================
-- Indexes
-- =========================================================================

CREATE INDEX IF NOT EXISTS idx_llm_models_name ON neurondb.llm_models(name);
CREATE INDEX IF NOT EXISTS idx_llm_models_family ON neurondb.llm_models(family);
CREATE INDEX IF NOT EXISTS idx_llm_models_model_type ON neurondb.llm_models(model_type);
CREATE INDEX IF NOT EXISTS idx_llm_models_tags ON neurondb.llm_models USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_llm_models_architecture ON neurondb.llm_models USING GIN(architecture);

CREATE INDEX IF NOT EXISTS idx_llm_model_files_model_id ON neurondb.llm_model_files(model_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_llm_model_files_model_filename ON neurondb.llm_model_files(model_id, filename);

CREATE INDEX IF NOT EXISTS idx_llm_model_versions_model_id ON neurondb.llm_model_versions(model_id);
CREATE INDEX IF NOT EXISTS idx_llm_model_versions_current ON neurondb.llm_model_versions(model_id) WHERE is_current;

CREATE INDEX IF NOT EXISTS idx_llm_tokenizers_model_id ON neurondb.llm_tokenizers(model_id);

CREATE INDEX IF NOT EXISTS idx_llm_adapters_model_id ON neurondb.llm_adapters(model_id);
CREATE INDEX IF NOT EXISTS idx_llm_adapters_name ON neurondb.llm_adapters(name);

CREATE INDEX IF NOT EXISTS idx_llm_model_deployments_model ON neurondb.llm_model_deployments(model_id);
CREATE INDEX IF NOT EXISTS idx_llm_model_deployments_provider ON neurondb.llm_model_deployments(provider_id);

CREATE INDEX IF NOT EXISTS idx_llm_messages_conversation_created ON neurondb.llm_messages(conversation_id, created_at);

CREATE INDEX IF NOT EXISTS idx_llm_token_usage_created_model ON neurondb.llm_token_usage(created_at, model_name);
CREATE INDEX IF NOT EXISTS idx_llm_token_usage_created ON neurondb.llm_token_usage(created_at);

CREATE INDEX IF NOT EXISTS idx_llm_training_metrics_run_step ON neurondb.llm_training_metrics(run_id, step, metric_name);
CREATE INDEX IF NOT EXISTS idx_llm_training_metrics_run_id ON neurondb.llm_training_metrics(run_id);

CREATE INDEX IF NOT EXISTS idx_llm_eval_results_model_benchmark ON neurondb.llm_eval_results(model_id, benchmark_id);

CREATE INDEX IF NOT EXISTS idx_llm_guardrail_log_created ON neurondb.llm_guardrail_log(created_at);

-- =========================================================================
-- Module 10: Views and Monitoring
-- =========================================================================

CREATE OR REPLACE VIEW neurondb.llm_model_overview AS
SELECT m.model_id, m.name, m.family, m.model_type, m.task, m.parameter_count,
       m.context_length, m.format, m.quantization, m.storage_mode, m.is_active,
       m.created_at,
       (SELECT count(*) FROM neurondb.llm_model_files f WHERE f.model_id = m.model_id) AS file_count,
       (SELECT coalesce(sum(f.size_bytes), 0) FROM neurondb.llm_model_files f WHERE f.model_id = m.model_id) AS total_size_bytes,
       (SELECT count(*) FROM neurondb.llm_model_deployments d WHERE d.model_id = m.model_id AND d.is_active) AS active_deployments,
       (SELECT er.metrics FROM neurondb.llm_eval_results er
        JOIN neurondb.llm_benchmarks b ON er.benchmark_id = b.benchmark_id
        WHERE er.model_id = m.model_id ORDER BY er.evaluated_at DESC LIMIT 1) AS latest_eval_metrics
FROM neurondb.llm_models m;

COMMENT ON VIEW neurondb.llm_model_overview IS 'Models with file count, size, active deployments, latest eval';

CREATE OR REPLACE VIEW neurondb.llm_provider_status AS
SELECT p.provider_id, p.name, p.provider_type, p.api_base, p.is_active, p.priority,
       count(d.deployment_id) AS deployment_count,
       (SELECT count(*) FROM neurondb.llm_token_usage u WHERE u.provider_name = p.name AND u.created_at > now() - interval '24 hours') AS requests_24h
FROM neurondb.llm_providers p
LEFT JOIN neurondb.llm_model_deployments d ON d.provider_id = p.provider_id AND d.is_active
GROUP BY p.provider_id, p.name, p.provider_type, p.api_base, p.is_active, p.priority;

COMMENT ON VIEW neurondb.llm_provider_status IS 'Provider health and request counts';

CREATE OR REPLACE VIEW neurondb.llm_usage_dashboard AS
SELECT date_trunc('day', created_at) AS day,
       model_name,
       count(*) AS requests,
       sum(tokens_input + tokens_output) AS total_tokens,
       sum(coalesce(cost_usd, 0)) AS total_cost_usd,
       avg(latency_ms) AS avg_latency_ms
FROM neurondb.llm_token_usage
WHERE created_at > now() - interval '30 days'
GROUP BY date_trunc('day', created_at), model_name;

COMMENT ON VIEW neurondb.llm_usage_dashboard IS 'Aggregated usage: requests, tokens, cost per model per day';

CREATE OR REPLACE VIEW neurondb.llm_training_dashboard AS
SELECT r.run_id, r.run_name, r.run_type, r.status, r.started_at, r.completed_at,
       r.current_step, r.total_steps, r.best_metric_name, r.best_metric_value,
       m.name AS model_name,
       (SELECT jsonb_object_agg(metric_name, metric_value)
        FROM (SELECT metric_name, metric_value FROM neurondb.llm_training_metrics tm
              WHERE tm.run_id = r.run_id AND tm.step = (SELECT max(step) FROM neurondb.llm_training_metrics WHERE run_id = r.run_id)) last_metrics) AS last_metrics
FROM neurondb.llm_training_runs r
LEFT JOIN neurondb.llm_models m ON m.model_id = r.model_id
ORDER BY r.created_at DESC;

COMMENT ON VIEW neurondb.llm_training_dashboard IS 'Active/completed training runs with metrics';

CREATE OR REPLACE VIEW neurondb.llm_budget_overview AS
SELECT b.budget_id, b.name, b.user_id, b.model_name, b.max_cost_usd, b.max_tokens,
       b.current_cost_usd, b.current_tokens, b.period, b.is_active,
       CASE WHEN b.max_cost_usd > 0 THEN round(100.0 * b.current_cost_usd / b.max_cost_usd, 2) ELSE NULL END AS cost_utilization_pct,
       CASE WHEN b.max_tokens > 0 THEN round(100.0 * b.current_tokens::numeric / b.max_tokens, 2) ELSE NULL END AS token_utilization_pct
FROM neurondb.llm_budgets b
WHERE b.is_active;

COMMENT ON VIEW neurondb.llm_budget_overview IS 'Budget utilization percentages';

CREATE OR REPLACE VIEW neurondb.llm_conversation_summary AS
SELECT c.conversation_id, c.session_id, c.title, c.model_name, c.user_id, c.created_at,
       count(msg.message_id) AS message_count,
       sum(coalesce(msg.tokens_in, 0) + coalesce(msg.tokens_out, 0)) AS total_tokens
FROM neurondb.llm_conversations c
LEFT JOIN neurondb.llm_messages msg ON msg.conversation_id = c.conversation_id
GROUP BY c.conversation_id, c.session_id, c.title, c.model_name, c.user_id, c.created_at;

COMMENT ON VIEW neurondb.llm_conversation_summary IS 'Conversations with message counts and token totals';

CREATE OR REPLACE VIEW neurondb.llm_adapter_overview AS
SELECT a.adapter_id, a.name, a.adapter_type, a.rank, a.alpha, a.is_active, a.created_at,
       m.name AS model_name,
       a.metrics
FROM neurondb.llm_adapters a
JOIN neurondb.llm_models m ON m.model_id = a.model_id;

COMMENT ON VIEW neurondb.llm_adapter_overview IS 'Adapters with model info and metrics';
