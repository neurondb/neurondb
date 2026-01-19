-- ============================================================================
-- ML Unified API Schema
-- PostgresML-compatible unified interface for NeuronDB
-- ============================================================================

-- Feature Stores Table
CREATE TABLE IF NOT EXISTS neurondb.feature_stores (
    store_id SERIAL PRIMARY KEY,
    store_name TEXT UNIQUE NOT NULL,
    entity_table TEXT NOT NULL,
    entity_key TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Feature Definitions Table
CREATE TABLE IF NOT EXISTS neurondb.features (
    feature_id SERIAL PRIMARY KEY,
    store_id INTEGER REFERENCES neurondb.feature_stores(store_id) ON DELETE CASCADE,
    feature_name TEXT NOT NULL,
    feature_type TEXT NOT NULL CHECK (feature_type IN ('numeric', 'categorical', 'vector', 'text')),
    transformation TEXT,
    version INTEGER DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(store_id, feature_name, version)
);

-- Hyperparameter Tuning Results
CREATE TABLE IF NOT EXISTS neurondb.hyperparameter_results (
    result_id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES neurondb.ml_projects(project_id) ON DELETE CASCADE,
    algorithm TEXT NOT NULL,
    parameters JSONB NOT NULL,
    score FLOAT NOT NULL,
    cv_scores FLOAT[] NOT NULL,
    training_time_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_hyperparam_project ON neurondb.hyperparameter_results(project_id);
CREATE INDEX IF NOT EXISTS idx_hyperparam_score ON neurondb.hyperparameter_results(score DESC);

-- Text ML Models
CREATE TABLE IF NOT EXISTS neurondb.text_models (
    model_id SERIAL PRIMARY KEY,
    model_name TEXT UNIQUE NOT NULL,
    model_type TEXT NOT NULL CHECK (model_type IN ('classification', 'sentiment', 'ner', 'summarization')),
    model_path TEXT,
    vocabulary_size INTEGER,
    embedding_dim INTEGER,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- RAG Pipeline Configurations
CREATE TABLE IF NOT EXISTS neurondb.rag_pipelines (
    pipeline_id SERIAL PRIMARY KEY,
    pipeline_name TEXT UNIQUE NOT NULL,
    chunk_size INTEGER DEFAULT 512,
    chunk_overlap INTEGER DEFAULT 128,
    embedding_model TEXT NOT NULL,
    reranking_model TEXT,
    configuration JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON neurondb.feature_stores TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON neurondb.features TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON neurondb.hyperparameter_results TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON neurondb.text_models TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON neurondb.rag_pipelines TO PUBLIC;

GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA neurondb TO PUBLIC;

-- Comments
COMMENT ON TABLE neurondb.feature_stores IS 'Feature store registry for ML feature management';
COMMENT ON TABLE neurondb.features IS 'Feature definitions with versioning';
COMMENT ON TABLE neurondb.hyperparameter_results IS 'Hyperparameter tuning results';
COMMENT ON TABLE neurondb.text_models IS 'Text ML model registry';
COMMENT ON TABLE neurondb.rag_pipelines IS 'RAG pipeline configurations';

