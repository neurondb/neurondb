-- ============================================================================
-- pgvector Performance Benchmark Suite
-- ============================================================================
-- This file benchmarks pgvector extension performance with:
-- - Large datasets (10K-100K vectors)
-- - Multiple index types (HNSW only - IVFFlat skipped for stability)
-- - Different distance metrics (L2, cosine, inner product)
-- - Query throughput and performance tests
-- ============================================================================

\timing on
\pset footer off
\pset pager off
\set ON_ERROR_STOP on

SET client_min_messages TO WARNING;
SET maintenance_work_mem = '256MB';

\echo '=========================================================================='
\echo 'pgvector Performance Benchmark Suite'
\echo '=========================================================================='
\echo ''

-- ============================================================================
-- Setup: Create test tables
-- ============================================================================

\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Setup: Creating test tables...'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Ensure pgvector extension is loaded
CREATE EXTENSION IF NOT EXISTS vector;

-- Clean up any existing tables to ensure clean run
DROP TABLE IF EXISTS vectors_128 CASCADE;
DROP TABLE IF EXISTS vectors_768 CASCADE;
DROP TABLE IF EXISTS vectors_100k CASCADE;

-- Table for 128-dimensional vectors (common embedding size)
CREATE TABLE vectors_128 (
    id SERIAL PRIMARY KEY,
    category TEXT,
    embedding vector(128),
    metadata JSONB
);

-- Table for 768-dimensional vectors (large embeddings like BERT)
CREATE TABLE vectors_768 (
    id SERIAL PRIMARY KEY,
    category TEXT,
    embedding vector(768),
    metadata JSONB
);

-- Table for 100K vectors (scale test)
CREATE TABLE vectors_100k (
    id SERIAL PRIMARY KEY,
    category TEXT,
    embedding vector(128),
    metadata JSONB
);

\echo 'Tables created successfully'
\echo ''

-- ============================================================================
-- Data Generation: Insert large datasets
-- ============================================================================

\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Data Generation: Inserting 50,000 vectors (128-dim)...'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

INSERT INTO vectors_128 (category, embedding, metadata)
SELECT 
    'cluster_' || ((i % 10) + 1),
    ARRAY(
        SELECT (random() + (i % 10) * 0.1)::real 
        FROM generate_series(1, 128)
    )::vector(128),
    jsonb_build_object('id', i, 'cluster', (i % 10) + 1, 'batch', (i / 1000)::int)
FROM generate_series(1, 50000) i;

\echo '50,000 vectors inserted'
\echo ''

\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Data Generation: Inserting 10,000 vectors (768-dim)...'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

INSERT INTO vectors_768 (category, embedding, metadata)
SELECT 
    'cluster_' || ((i % 10) + 1),
    ARRAY(
        SELECT (random() + (i % 10) * 0.1)::real 
        FROM generate_series(1, 768)
    )::vector(768),
    jsonb_build_object('id', i, 'cluster', (i % 10) + 1)
FROM generate_series(1, 10000) i;

\echo '10,000 vectors inserted'
\echo ''

\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Data Generation: Inserting 100,000 vectors for scale test...'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

INSERT INTO vectors_100k (category, embedding, metadata)
SELECT 
    'cluster_' || ((i % 20) + 1),
    ARRAY(
        SELECT (random() + (i % 20) * 0.05)::real 
        FROM generate_series(1, 128)
    )::vector(128),
    jsonb_build_object('id', i, 'cluster', (i % 20) + 1)
FROM generate_series(1, 100000) i;

\echo '100,000 vectors inserted'
\echo ''

-- ============================================================================
-- Benchmark 1: HNSW Index Creation Performance
-- ============================================================================

\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Benchmark 1: HNSW Index Creation (128-dim, L2 distance)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DROP INDEX IF EXISTS idx_vectors_128_hnsw_l2;
CREATE INDEX idx_vectors_128_hnsw_l2 
    ON vectors_128 USING hnsw (embedding vector_l2_ops) 
    WITH (m = 16, ef_construction = 200);

\echo ''

\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Benchmark 1b: HNSW Index Creation (128-dim, Cosine distance)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DROP INDEX IF EXISTS idx_vectors_128_hnsw_cosine;
CREATE INDEX idx_vectors_128_hnsw_cosine 
    ON vectors_128 USING hnsw (embedding vector_cosine_ops) 
    WITH (m = 16, ef_construction = 200);

\echo ''

\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Benchmark 1c: HNSW Index Creation (768-dim, L2 distance)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DROP INDEX IF EXISTS idx_vectors_768_hnsw_l2;
CREATE INDEX idx_vectors_768_hnsw_l2 
    ON vectors_768 USING hnsw (embedding vector_l2_ops) 
    WITH (m = 16, ef_construction = 200);

\echo ''

\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Benchmark 1d: HNSW Index Creation (100K vectors, 128-dim)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DROP INDEX IF EXISTS idx_vectors_100k_hnsw_l2;
CREATE INDEX idx_vectors_100k_hnsw_l2
    ON vectors_100k USING hnsw (embedding vector_l2_ops) 
    WITH (m = 16, ef_construction = 200);

\echo ''

-- ============================================================================
-- Benchmark 2: KNN Search Performance - L2 Distance
-- ============================================================================

\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Benchmark 3: Top-10 KNN Search (L2 distance, HNSW index)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SET enable_seqscan = off;
SET hnsw.ef_search = 100;

WITH query_vector AS (
    SELECT ARRAY(
        SELECT random()::real 
        FROM generate_series(1, 128)
    )::vector(128) AS q
)
SELECT 
    v.id,
    v.category,
    v.embedding <-> q.q AS l2_distance,
    RANK() OVER (ORDER BY v.embedding <-> q.q) AS rank
FROM vectors_128 v, query_vector q
ORDER BY v.embedding <-> q.q
LIMIT 10;

RESET enable_seqscan;
RESET hnsw.ef_search;

\echo ''

-- ============================================================================
-- Benchmark 4: KNN Search Performance - Cosine Distance
-- ============================================================================

\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Benchmark 4: Top-10 KNN Search (Cosine distance, HNSW index)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SET enable_seqscan = off;
SET hnsw.ef_search = 100;

WITH query_vector AS (
    SELECT ARRAY(
        SELECT random()::real 
        FROM generate_series(1, 128)
    )::vector(128) AS q
)
SELECT 
    v.id,
    v.category,
    v.embedding <=> q.q AS cosine_distance,
    1 - (v.embedding <=> q.q) AS cosine_similarity,
    RANK() OVER (ORDER BY v.embedding <=> q.q) AS rank
FROM vectors_128 v, query_vector q
ORDER BY v.embedding <=> q.q
LIMIT 10;

RESET enable_seqscan;
RESET hnsw.ef_search;

\echo ''

-- ============================================================================
-- Benchmark 5: KNN Search Performance - Inner Product
-- ============================================================================

\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Benchmark 5: Top-10 KNN Search (Inner product, HNSW index)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SET enable_seqscan = off;
SET hnsw.ef_search = 100;

WITH query_vector AS (
    SELECT ARRAY(
        SELECT random()::real 
        FROM generate_series(1, 128)
    )::vector(128) AS q
)
SELECT 
    v.id,
    v.category,
    v.embedding <#> q.q AS neg_inner_product,
    -(v.embedding <#> q.q) AS inner_product,
    RANK() OVER (ORDER BY v.embedding <#> q.q) AS rank
FROM vectors_128 v, query_vector q
ORDER BY v.embedding <#> q.q
LIMIT 10;

RESET enable_seqscan;
RESET hnsw.ef_search;

\echo ''

-- ============================================================================
-- Benchmark 6: Filtered KNN Search
-- ============================================================================

\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Benchmark 6: Filtered Top-10 KNN Search (category filter)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SET enable_seqscan = off;
SET hnsw.ef_search = 100;

WITH query_vector AS (
    SELECT ARRAY(
        SELECT (random() + 1 * 0.1)::real 
        FROM generate_series(1, 128)
    )::vector(128) AS q
)
SELECT 
    v.id,
    v.category,
    v.embedding <-> q.q AS distance
FROM vectors_128 v, query_vector q
WHERE v.category = 'cluster_1'
ORDER BY v.embedding <-> q.q
LIMIT 10;

RESET enable_seqscan;
RESET hnsw.ef_search;

\echo ''

-- ============================================================================
-- Benchmark 7: Large Vector Search (768 dimensions)
-- ============================================================================

\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Benchmark 7: Top-10 KNN Search (768-dim vectors)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SET enable_seqscan = off;
SET hnsw.ef_search = 100;

WITH query_vector AS (
    SELECT ARRAY(
        SELECT random()::real 
        FROM generate_series(1, 768)
    )::vector(768) AS q
)
SELECT 
    v.id,
    v.category,
    v.embedding <-> q.q AS l2_distance
FROM vectors_768 v, query_vector q
ORDER BY v.embedding <-> q.q
LIMIT 10;

RESET enable_seqscan;
RESET hnsw.ef_search;

\echo ''

-- ============================================================================
-- Benchmark 8: Batch Query Throughput (Multiple KNN queries)
-- ============================================================================

\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Benchmark 8: Batch Query Throughput (20 sequential queries)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SET enable_seqscan = off;
SET hnsw.ef_search = 100;

DO $$
DECLARE
    query_vec vector(128);
    result_count int;
    i int;
BEGIN
    FOR i IN 1..20 LOOP
        -- Generate random query vector
        SELECT ARRAY(
            SELECT random()::real 
            FROM generate_series(1, 128)
        )::vector(128) INTO query_vec;
        
        -- Execute KNN query
        SELECT COUNT(*) INTO result_count
        FROM (
            SELECT id
            FROM vectors_128
            ORDER BY embedding <-> query_vec
            LIMIT 10
        ) sub;
    END LOOP;
END $$;

RESET enable_seqscan;
RESET hnsw.ef_search;

\echo ''

-- ============================================================================
-- Benchmark 9: Aggregation Queries with Vector Operations
-- ============================================================================

\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Benchmark 10: Category Aggregation with Average Distance'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SET enable_seqscan = off;
SET hnsw.ef_search = 100;

WITH query_vector AS (
    SELECT ARRAY(
        SELECT random()::real 
        FROM generate_series(1, 128)
    )::vector(128) AS q
)
SELECT 
    v.category,
    COUNT(*) AS total_vectors,
    AVG(v.embedding <-> q.q) AS avg_l2_distance,
    MIN(v.embedding <-> q.q) AS min_distance,
    MAX(v.embedding <-> q.q) AS max_distance
FROM vectors_128 v, query_vector q
GROUP BY v.category
ORDER BY avg_l2_distance;

RESET enable_seqscan;
RESET hnsw.ef_search;

\echo ''

-- ============================================================================
-- Benchmark 11: Top-K Per Category
-- ============================================================================

\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Benchmark 11: Top-5 Per Category (Partitioned KNN)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SET enable_seqscan = off;
SET hnsw.ef_search = 100;

WITH query_vector AS (
    SELECT ARRAY(
        SELECT random()::real 
        FROM generate_series(1, 128)
    )::vector(128) AS q
),
ranked AS (
    SELECT 
        v.id,
        v.category,
        v.embedding <-> q.q AS distance,
        ROW_NUMBER() OVER (PARTITION BY v.category ORDER BY v.embedding <-> q.q) AS rn
    FROM vectors_128 v, query_vector q
)
SELECT id, category, distance
FROM ranked
WHERE rn <= 5
ORDER BY category, distance;

RESET enable_seqscan;
RESET hnsw.ef_search;

\echo ''

-- ============================================================================
-- Benchmark 12: Sequential Scan vs Index Comparison
-- ============================================================================

\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Benchmark 12a: Sequential Scan Performance (no index)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SET enable_seqscan = on;
SET enable_indexscan = off;
SET enable_bitmapscan = off;

WITH query_vector AS (
    SELECT ARRAY(
        SELECT random()::real 
        FROM generate_series(1, 128)
    )::vector(128) AS q
)
SELECT COUNT(*) AS result_count
FROM (
    SELECT id
    FROM vectors_128
    ORDER BY embedding <-> (SELECT q FROM query_vector)
    LIMIT 10
) sub;

RESET enable_seqscan;
RESET enable_indexscan;
RESET enable_bitmapscan;

\echo ''

\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Benchmark 12b: HNSW Index Scan Performance'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SET enable_seqscan = off;
SET hnsw.ef_search = 100;

WITH query_vector AS (
    SELECT ARRAY(
        SELECT random()::real 
        FROM generate_series(1, 128)
    )::vector(128) AS q
)
SELECT COUNT(*) AS result_count
FROM (
    SELECT id
    FROM vectors_128
    ORDER BY embedding <-> (SELECT q FROM query_vector)
    LIMIT 10
) sub;

RESET enable_seqscan;
RESET hnsw.ef_search;

\echo ''

-- ============================================================================
-- Summary Statistics
-- ============================================================================

\echo '=========================================================================='
\echo 'Benchmark Summary'
\echo '=========================================================================='

SELECT 
    'vectors_128' AS table_name,
    COUNT(*) AS total_rows,
    pg_size_pretty(pg_total_relation_size('vectors_128')) AS total_size,
    pg_size_pretty(pg_relation_size('vectors_128')) AS table_size
FROM vectors_128
UNION ALL
SELECT 
    'vectors_768' AS table_name,
    COUNT(*) AS total_rows,
    pg_size_pretty(pg_total_relation_size('vectors_768')) AS total_size,
    pg_size_pretty(pg_relation_size('vectors_768')) AS table_size
FROM vectors_768;

\echo ''
\echo 'Index Information:'
SELECT 
    schemaname,
    relname AS tablename,
    indexrelname AS indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE relname IN ('vectors_128', 'vectors_768')
ORDER BY relname, indexrelname;

\echo ''
\echo '=========================================================================='
\echo 'pgvector Benchmark Suite Completed'
\echo '=========================================================================='

\timing off

