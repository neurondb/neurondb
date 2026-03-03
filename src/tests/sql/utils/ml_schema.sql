/*
 * ml_schema.sql
 *   Complete ML schema for NeuronDB
 *
 * All CREATE TABLE statements moved from C code to SQL
 * for proper database initialization.
 */

-- Projects and model management
CREATE TABLE IF NOT EXISTS neurondb.ml_projects (
    project_id SERIAL PRIMARY KEY,
    project_name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    owner TEXT DEFAULT CURRENT_USER,
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'archived', 'deleted'))
);

CREATE TABLE IF NOT EXISTS neurondb.ml_models (
    model_id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES neurondb.ml_projects(project_id),
    model_name TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    version INTEGER DEFAULT 1,
    model_data BYTEA,
    hyperparameters JSONB,
    metrics JSONB,
    training_table TEXT,
    feature_columns TEXT[],
    target_column TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'training' CHECK (status IN ('training', 'completed', 'failed', 'deployed')),
    UNIQUE(project_id, model_name, version)
);

CREATE TABLE IF NOT EXISTS neurondb.ml_experiments (
    experiment_id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES neurondb.ml_projects(project_id),
    experiment_name TEXT NOT NULL,
    description TEXT,
    config JSONB,
    results JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed'))
);

CREATE TABLE IF NOT EXISTS neurondb.ml_deployments (
    deployment_id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES neurondb.ml_models(model_id),
    deployment_name TEXT NOT NULL UNIQUE,
    endpoint TEXT,
    version INTEGER DEFAULT 1,
    deployed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'rollback'))
);

-- Feature store tables
CREATE TABLE IF NOT EXISTS neurondb.feature_stores (
    store_id SERIAL PRIMARY KEY,
    store_name TEXT NOT NULL UNIQUE,
    schema_name TEXT NOT NULL,
    table_name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS neurondb.feature_definitions (
    feature_id SERIAL PRIMARY KEY,
    store_id INTEGER REFERENCES neurondb.feature_stores(store_id),
    feature_name TEXT NOT NULL,
    feature_type TEXT NOT NULL,
    transformation TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(store_id, feature_name)
);

-- MLOps tables
CREATE TABLE IF NOT EXISTS neurondb.ab_tests (
    test_id SERIAL PRIMARY KEY,
    test_name TEXT NOT NULL UNIQUE,
    model_ids INTEGER[],
    traffic_split FLOAT[],
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    status TEXT DEFAULT 'running' CHECK (status IN ('running', 'completed', 'paused')),
    results JSONB
);

CREATE TABLE IF NOT EXISTS neurondb.model_monitoring (
    log_id BIGSERIAL PRIMARY KEY,
    model_id INTEGER,
    prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    input_data JSONB,
    prediction JSONB,
    confidence FLOAT,
    latency_ms INTEGER,
    success BOOLEAN DEFAULT true,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS neurondb.model_versions (
    version_id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES neurondb.ml_models(model_id),
    version_tag TEXT NOT NULL,
    model_snapshot BYTEA,
    metrics JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT DEFAULT CURRENT_USER,
    UNIQUE(model_id, version_tag)
);

CREATE TABLE IF NOT EXISTS neurondb.drift_detection (
    drift_id BIGSERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES neurondb.ml_models(model_id),
    detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    drift_score FLOAT,
    drift_type TEXT CHECK (drift_type IN ('data', 'concept', 'prediction')),
    baseline_period INTERVAL,
    current_period INTERVAL,
    details JSONB
);

CREATE TABLE IF NOT EXISTS neurondb.feature_flags (
    flag_id SERIAL PRIMARY KEY,
    flag_name TEXT NOT NULL UNIQUE,
    enabled BOOLEAN DEFAULT false,
    rollout_percentage FLOAT DEFAULT 0.0 CHECK (rollout_percentage >= 0 AND rollout_percentage <= 100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS neurondb.experiment_metrics (
    metric_id BIGSERIAL PRIMARY KEY,
    experiment_id INTEGER,
    variant TEXT,
    metric_name TEXT,
    metric_value FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS neurondb.model_audit_log (
    audit_id BIGSERIAL PRIMARY KEY,
    model_id INTEGER,
    action TEXT NOT NULL,
    user_id TEXT DEFAULT CURRENT_USER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    details JSONB
);

-- Recommender system tables
CREATE TABLE IF NOT EXISTS neurondb.collaborative_filter_models (
    model_id SERIAL PRIMARY KEY,
    model_name TEXT NOT NULL UNIQUE,
    n_factors INTEGER NOT NULL,
    user_factors BYTEA,
    item_factors BYTEA,
    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS neurondb.recommendations_cache (
    cache_id BIGSERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    item_id INTEGER NOT NULL,
    score FLOAT NOT NULL,
    model_id INTEGER,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP
);

-- Deep learning model registry
CREATE TABLE IF NOT EXISTS neurondb.dl_models (
    dl_model_id SERIAL PRIMARY KEY,
    model_name TEXT NOT NULL UNIQUE,
    framework TEXT CHECK (framework IN ('pytorch', 'tensorflow', 'onnx')),
    model_path TEXT,
    model_binary BYTEA,
    input_shape INTEGER[],
    output_shape INTEGER[],
    quantized BOOLEAN DEFAULT false,
    imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Time series models
CREATE TABLE IF NOT EXISTS neurondb.timeseries_models (
    ts_model_id SERIAL PRIMARY KEY,
    model_name TEXT NOT NULL UNIQUE,
    p INTEGER,  -- AR order
    d INTEGER,  -- Differencing
    q INTEGER,  -- MA order
    coefficients BYTEA,
    residuals BYTEA,
    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_ml_models_project ON neurondb.ml_models(project_id);
CREATE INDEX IF NOT EXISTS idx_ml_models_status ON neurondb.ml_models(status);
CREATE INDEX IF NOT EXISTS idx_monitoring_model_time ON neurondb.model_monitoring(model_id, prediction_time DESC);
CREATE INDEX IF NOT EXISTS idx_drift_model_time ON neurondb.drift_detection(model_id, detection_time DESC);
CREATE INDEX IF NOT EXISTS idx_recommendations_user ON neurondb.recommendations_cache(user_id, generated_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_model_time ON neurondb.model_audit_log(model_id, timestamp DESC);

