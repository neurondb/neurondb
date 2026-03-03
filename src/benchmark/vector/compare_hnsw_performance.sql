-- ============================================================================
-- NeuronDB vs Baseline HNSW Performance Comparison
-- ============================================================================
-- This benchmark compares optimized NeuronDB HNSW performance against:
-- - Baseline: 59.49s for 50K vectors (128-dim L2)
-- - pgvector reference: 7.85s for 50K vectors (128-dim L2)
-- - Optimized NeuronDB: Target < 1s for 50K vectors
-- ============================================================================

\timing on
\pset footer off
\pset pager off
\set ON_ERROR_STOP on

SET client_min_messages TO WARNING;
SET maintenance_work_mem = '256MB';

\echo '=========================================================================='
\echo 'NeuronDB HNSW Performance Benchmark - Optimized vs Baseline'
\echo '=========================================================================='
\echo ''

-- ============================================================================
-- Setup: Create test table with 50K vectors (matching plan scenario)
-- ============================================================================

\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Setup: Creating test table with 50,000 vectors (128-dim)...'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE EXTENSION IF NOT EXISTS neurondb;

DROP TABLE IF EXISTS benchmark_vectors_50k CASCADE;
CREATE TABLE benchmark_vectors_50k (
    id SERIAL PRIMARY KEY,
    embedding vector(128)
);

-- Insert 50,000 vectors (same pattern as plan)
INSERT INTO benchmark_vectors_50k (embedding)
SELECT ARRAY(
    SELECT (random() + (i % 10) * 0.1)::real 
    FROM generate_series(1, 128)
)::vector(128)
FROM generate_series(1, 50000) i;

\echo '50,000 vectors inserted'
\echo ''

-- ============================================================================
-- Benchmark: HNSW Index Creation (50K vectors, 128-dim, L2 distance)
-- ============================================================================

\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Benchmark: HNSW Index Creation (50K vectors, 128-dim, L2)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DROP INDEX IF EXISTS idx_benchmark_50k_hnsw;

CREATE INDEX idx_benchmark_50k_hnsw 
    ON benchmark_vectors_50k USING hnsw (embedding vector_l2_ops) 
    WITH (m = 16, ef_construction = 200);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Benchmark: 100K vectors (128-dim, L2)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DROP TABLE IF EXISTS benchmark_vectors_100k CASCADE;
CREATE TABLE benchmark_vectors_100k (
    id SERIAL PRIMARY KEY,
    embedding vector(128)
);

INSERT INTO benchmark_vectors_100k (embedding)
SELECT ARRAY(
    SELECT (random() + (i % 20) * 0.05)::real 
    FROM generate_series(1, 128)
)::vector(128)
FROM generate_series(1, 100000) i;

CREATE INDEX idx_benchmark_100k_hnsw 
    ON benchmark_vectors_100k USING hnsw (embedding vector_l2_ops) 
    WITH (m = 16, ef_construction = 200);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Benchmark: 10K vectors (768-dim, L2)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DROP TABLE IF EXISTS benchmark_vectors_768 CASCADE;
CREATE TABLE benchmark_vectors_768 (
    id SERIAL PRIMARY KEY,
    embedding vector(768)
);

INSERT INTO benchmark_vectors_768 (embedding)
SELECT ARRAY(
    SELECT (random() + (i % 10) * 0.1)::real 
    FROM generate_series(1, 768)
)::vector(768)
FROM generate_series(1, 10000) i;

CREATE INDEX idx_benchmark_768_hnsw 
    ON benchmark_vectors_768 USING hnsw (embedding vector_l2_ops) 
    WITH (m = 16, ef_construction = 200);

\echo ''

-- ============================================================================
-- Performance Comparison Summary
-- ============================================================================

\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Performance Comparison Summary'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Get actual timings from PostgreSQL
-- Note: Timing output is captured by \timing on above
-- This section provides a summary view

SELECT 
    '50K vectors (128-dim)' AS test_case,
    'Baseline (before optimization)' AS version,
    59490 AS time_ms,
    '59.49s' AS time_formatted
UNION ALL
SELECT 
    '50K vectors (128-dim)' AS test_case,
    'pgvector reference' AS version,
    7850 AS time_ms,
    '7.85s' AS time_formatted
UNION ALL
SELECT 
    '50K vectors (128-dim)' AS test_case,
    'NeuronDB Optimized' AS version,
    NULL AS time_ms,  -- Will be filled from actual timing above
    'See timing above' AS time_formatted
UNION ALL
SELECT 
    '100K vectors (128-dim)' AS test_case,
    'NeuronDB Optimized' AS version,
    NULL AS time_ms,
    'See timing above' AS time_formatted
UNION ALL
SELECT 
    '10K vectors (768-dim)' AS test_case,
    'NeuronDB Optimized' AS version,
    NULL AS time_ms,
    'See timing above' AS time_formatted
ORDER BY test_case, time_ms NULLS LAST;

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Index Size Information'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT 
    schemaname,
    relname AS tablename,
    indexrelname AS indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size,
    pg_size_pretty(pg_relation_size(relid)) AS table_size
FROM pg_stat_user_indexes
WHERE relname LIKE 'benchmark_vectors%'
ORDER BY relname, indexrelname;

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Query Performance Test (KNN Search)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SET enable_seqscan = off;
SET hnsw.ef_search = 100;

-- Test query performance
WITH query_vector AS (
    SELECT ARRAY(
        SELECT random()::real 
        FROM generate_series(1, 128)
    )::vector(128) AS q
)
SELECT 
    v.id,
    v.embedding <-> q.q AS l2_distance
FROM benchmark_vectors_50k v, query_vector q
ORDER BY v.embedding <-> q.q
LIMIT 10;

RESET enable_seqscan;
RESET hnsw.ef_search;

\echo ''
\echo '=========================================================================='
\echo 'Benchmark Complete'
\echo '=========================================================================='
\echo ''
\echo 'Key Results:'
\echo '- 50K vectors (128-dim): Check timing above'
\echo '- 100K vectors (128-dim): Check timing above'
\echo '- 10K vectors (768-dim): Check timing above'
\echo ''
\echo 'Target from plan: < 1s for 50K vectors'
\echo 'Baseline: 59.49s for 50K vectors'
\echo 'pgvector reference: 7.85s for 50K vectors'
\echo ''

\timing off

