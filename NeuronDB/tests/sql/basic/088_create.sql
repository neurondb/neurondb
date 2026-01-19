-- ============================================================================
-- NeurondB Regression Test Suite - Setup and Initialization
-- This file creates all necessary database structures for testing
-- ============================================================================

-- Create test database (if not exists, should already be created by pg_regress)
-- Drop existing extension if present (clean slate)
DROP EXTENSION IF EXISTS neurondb CASCADE;

-- Create extension with all dependencies
CREATE EXTENSION neurondb CASCADE;

-- Verify extension is installed
SELECT extname, extversion FROM pg_extension WHERE extname = 'neurondb';

-- Verify schema exists
SELECT nspname FROM pg_namespace WHERE nspname = 'neurondb';

-- ============================================================================
-- Create Test Tables
-- ============================================================================

-- Test vectors table for basic operations
CREATE TABLE IF NOT EXISTS test_vectors (
    id SERIAL PRIMARY KEY,
    name TEXT,
    embedding vector(3),
    metadata JSONB
);

-- Large dimension test table
CREATE TABLE IF NOT EXISTS test_high_dim (
    id SERIAL PRIMARY KEY,
    vec_128 vector(128),
    vec_256 vector(256),
    vec_1536 vector(1536),
    category TEXT
);

-- Table for index testing
CREATE TABLE IF NOT EXISTS test_index_vectors (
    id SERIAL PRIMARY KEY,
    vec vector(4),
    label TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Table for aggregation testing
CREATE TABLE IF NOT EXISTS test_aggregates (
    id SERIAL PRIMARY KEY,
    vec vector(5),
    group_id INTEGER,
    value REAL
);

-- Table for distance metric testing
CREATE TABLE IF NOT EXISTS test_distances (
    id SERIAL PRIMARY KEY,
    vec1 vector(3),
    vec2 vector(3),
    expected_l2 REAL,
    expected_cosine REAL
);

-- Table for worker functionality testing
CREATE TABLE IF NOT EXISTS test_worker_data (
    id SERIAL PRIMARY KEY,
    job_type TEXT,
    status TEXT,
    data JSONB
);

-- ============================================================================
-- Populate Test Data
-- ============================================================================

-- Insert basic test vectors
INSERT INTO test_vectors (name, embedding, metadata) VALUES
    ('item1', '[1.0, 0.0, 0.0]', '{"type": "test", "priority": 1}'),
    ('item2', '[0.0, 1.0, 0.0]', '{"type": "test", "priority": 2}'),
    ('item3', '[0.0, 0.0, 1.0]', '{"type": "test", "priority": 3}'),
    ('item4', '[0.5, 0.5, 0.0]', '{"type": "test", "priority": 4}'),
    ('item5', '[0.3, 0.3, 0.4]', '{"type": "test", "priority": 5}');

-- Insert high dimension vectors (sample data)
INSERT INTO test_high_dim (vec_128, vec_256, vec_1536, category) VALUES
    (
        (SELECT array_fill(0.1::float4, ARRAY[128])::vector(128)),
        (SELECT array_fill(0.1::float4, ARRAY[256])::vector(256)),
        (SELECT array_fill(0.1::float4, ARRAY[1536])::vector(1536)),
        'category_a'
    ),
    (
        (SELECT array_fill(0.2::float4, ARRAY[128])::vector(128)),
        (SELECT array_fill(0.2::float4, ARRAY[256])::vector(256)),
        (SELECT array_fill(0.2::float4, ARRAY[1536])::vector(1536)),
        'category_b'
    );

-- Insert index test vectors
INSERT INTO test_index_vectors (vec, label) VALUES
    ('[1, 2, 3, 4]', 'label1'),
    ('[2, 3, 4, 5]', 'label2'),
    ('[3, 4, 5, 6]', 'label3'),
    ('[4, 5, 6, 7]', 'label4'),
    ('[5, 6, 7, 8]', 'label5');

-- Insert aggregation test data
INSERT INTO test_aggregates (vec, group_id, value) VALUES
    ('[1, 2, 3, 4, 5]', 1, 10.5),
    ('[2, 3, 4, 5, 6]', 1, 11.5),
    ('[3, 4, 5, 6, 7]', 2, 12.5),
    ('[4, 5, 6, 7, 8]', 2, 13.5),
    ('[5, 6, 7, 8, 9]', 3, 14.5);

-- Insert distance test data
INSERT INTO test_distances (vec1, vec2, expected_l2, expected_cosine) VALUES
    ('[1, 0, 0]', '[1, 0, 0]', 0.0, 0.0),
    ('[1, 0, 0]', '[0, 1, 0]', 1.4142135, 1.0),
    ('[1, 2, 3]', '[4, 5, 6]', 5.196152, 0.025368154);

-- Insert worker test data
INSERT INTO test_worker_data (job_type, status, data) VALUES
    ('vector_index', 'pending', '{"index_name": "test_idx1"}'::jsonb),
    ('embedding', 'processing', '{"model": "test_model"}'::jsonb),
    ('rerank', 'completed', '{"score": 0.95}'::jsonb);

-- ============================================================================
-- Create Indexes for Testing
-- ============================================================================

-- Note: HNSW index creation might fail if operator classes not fully implemented
-- This is expected and tests will handle gracefully

-- Try to create indexes (will skip if not supported)
DO $$
BEGIN
    -- Basic vector index attempt
    BEGIN
        CREATE INDEX idx_test_vectors ON test_vectors USING btree ((embedding::text));
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'Index creation skipped: %', SQLERRM;
    END;
END $$;

-- ============================================================================
-- Setup Worker Tables with Sample Data
-- ============================================================================

-- Populate job queue with test jobs
INSERT INTO neurondb.neurondb_job_queue (job_type, payload, status, tenant_id) VALUES
    ('vector_search', '{"query": "[1,2,3]", "k": 10}'::jsonb, 'pending', 1),
    ('index_build', '{"table": "test_vectors", "column": "embedding"}'::jsonb, 'pending', 1),
    ('embedding_generation', '{"text": "test query", "model": "ada-002"}'::jsonb, 'pending', 2);

-- Populate query metrics
INSERT INTO neurondb.neurondb_query_metrics (latency_ms, recall_at_k) VALUES
    (15.5, 0.95),
    (22.3, 0.87),
    (18.7, 0.92);

-- Populate embedding cache
INSERT INTO neurondb.neurondb_embedding_cache (cache_key, model_name, embedding) VALUES
    ('test_key_1', 'text-embedding-ada-002', '[0.1, 0.2, 0.3]'::vector),
    ('test_key_2', 'text-embedding-ada-002', '[0.4, 0.5, 0.6]'::vector);

-- Populate index maintenance
INSERT INTO neurondb.neurondb_index_maintenance (index_name, index_type, num_nodes, fragmentation_ratio) VALUES
    ('idx_test_vectors', 'hnsw', 5, 0.1),
    ('idx_test_index_vectors', 'ivf', 10, 0.05);

-- Populate LLM jobs
INSERT INTO neurondb.neurondb_llm_jobs (operation, model_name, input_text, status) VALUES
    ('completion', 'gpt-3.5-turbo', 'Test prompt 1', 'pending'),
    ('embedding', 'text-embedding-ada-002', 'Test text for embedding', 'processing'),
    ('rerank', 'cross-encoder', 'Query and document', 'completed');

-- Populate Prometheus metrics
INSERT INTO neurondb.neurondb_prometheus_metrics (metric_name, metric_value) VALUES
    ('queries_per_second', 150.5),
    ('avg_latency_ms', 18.2),
    ('total_queries', 10000),
    ('index_size_mb', 245.8);

-- ============================================================================
-- Verify Setup Complete
-- ============================================================================

-- Count all test tables
SELECT 
    COUNT(*) AS test_tables_count 
FROM pg_tables 
WHERE schemaname = 'public' 
    AND tablename LIKE 'test_%';

-- Count worker table entries
SELECT 
    'job_queue' AS table_name, COUNT(*) AS count FROM neurondb.neurondb_job_queue
UNION ALL
SELECT 'query_metrics', COUNT(*) FROM neurondb.neurondb_query_metrics
UNION ALL
SELECT 'embedding_cache', COUNT(*) FROM neurondb.neurondb_embedding_cache
UNION ALL
SELECT 'index_maintenance', COUNT(*) FROM neurondb.neurondb_index_maintenance
UNION ALL
SELECT 'llm_jobs', COUNT(*) FROM neurondb.neurondb_llm_jobs
UNION ALL
SELECT 'prometheus_metrics', COUNT(*) FROM neurondb.neurondb_prometheus_metrics
ORDER BY table_name;

-- Verify extension functions are available
SELECT 
    COUNT(*) AS available_functions
FROM pg_proc p
JOIN pg_namespace n ON p.pronamespace = n.oid
WHERE n.nspname = 'neurondb'
    OR p.proname LIKE 'vector_%'
    OR p.proname LIKE '%_distance%';

