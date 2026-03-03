-- ============================================================================
-- NeuronDB Vector Stress Benchmark Suite
-- ============================================================================
-- This file performs stress testing of NeuronDB extension with:
-- - High query throughput (concurrent KNN searches)
-- - Concurrent inserts under load
-- - Index maintenance operations
-- - Mixed read/write workloads
-- 
-- Designed to run for approximately 5 minutes
-- ============================================================================

\timing on
\pset footer off
\pset pager off
\set ON_ERROR_STOP on

SET client_min_messages TO NOTICE;
SET maintenance_work_mem = '256MB';

\echo '=========================================================================='
\echo 'NeuronDB Vector Stress Benchmark Suite'
\echo '=========================================================================='
\echo 'Target Duration: ~5 minutes (~300 seconds)'
\echo ''

-- ============================================================================
-- Setup Phase: Create test table with 100K vectors
-- ============================================================================

\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Setup: Creating test table with 100K vectors (128-dim)...'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE EXTENSION IF NOT EXISTS neurondb;

DROP TABLE IF EXISTS stress_vectors CASCADE;
CREATE TABLE stress_vectors (
    id SERIAL PRIMARY KEY,
    category TEXT,
    embedding vector(128),
    metadata JSONB
);

-- Disable autovacuum for this table to avoid crashes during stress test
ALTER TABLE stress_vectors SET (autovacuum_enabled = false);

-- Insert 100K initial vectors
INSERT INTO stress_vectors (category, embedding, metadata)
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

-- Create HNSW index
\echo 'Creating HNSW index...'
CREATE INDEX idx_stress_vectors_hnsw_l2 
    ON stress_vectors USING hnsw (embedding vector_l2_ops) 
    WITH (m = 16, ef_construction = 200);

\echo 'Index created'
\echo ''

-- ============================================================================
-- Stress Test: Scenario 1 - Query Stress (Progressive: 1000, 2000, 3000, 4000, 5000 queries)
-- ============================================================================

\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Scenario 1: Query Stress (Progressive batches: 1K, 2K, 3K, 4K, 5K queries)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SET enable_seqscan = off;
SET hnsw.ef_search = 100;

DO $$
DECLARE
    batch_start_time timestamp;
    batch_end_time timestamp;
    query_count int := 0;
    query_vec vector(128);
    result_count int;
    batch_size int;
    batch_num int;
    total_queries int := 0;
    total_time interval := '0 seconds'::interval;
BEGIN
    -- Run progressive batches: 1000, 2000, 3000, 4000, 5000 queries
    FOR batch_num IN 1..5 LOOP
        batch_size := batch_num * 1000;
        batch_start_time := clock_timestamp();
        query_count := 0;
        
        RAISE NOTICE 'Starting batch %: % queries', batch_num, batch_size;
        
        FOR i IN 1..batch_size LOOP
            -- Generate random query vector
            SELECT ARRAY(
                SELECT random()::real 
                FROM generate_series(1, 128)
            )::vector(128) INTO query_vec;
            
            -- Execute KNN query
            SELECT COUNT(*) INTO result_count
            FROM (
                SELECT id
                FROM stress_vectors
                ORDER BY embedding <-> query_vec
                LIMIT 10
            ) sub;
            
            query_count := query_count + 1;
        END LOOP;
        
        batch_end_time := clock_timestamp();
        total_queries := total_queries + query_count;
        total_time := total_time + (batch_end_time - batch_start_time);
        
        RAISE NOTICE 'Batch %: Completed % queries in % seconds (%) QPS', 
            batch_num,
            query_count,
            ROUND(EXTRACT(EPOCH FROM (batch_end_time - batch_start_time))::numeric, 2),
            ROUND((query_count / EXTRACT(EPOCH FROM (batch_end_time - batch_start_time)))::numeric, 2);
    END LOOP;
    
    RAISE NOTICE 'Query Stress Summary: Completed % total queries in % seconds (%) QPS', 
        total_queries,
        ROUND(EXTRACT(EPOCH FROM total_time)::numeric, 2),
        ROUND((total_queries / EXTRACT(EPOCH FROM total_time))::numeric, 2);
END $$;

RESET enable_seqscan;
RESET hnsw.ef_search;

\echo ''

-- ============================================================================
-- Stress Test: Scenario 2 - Insert Stress (Progressive: 50, 100, 150, 200, 250 batch inserts)
-- ============================================================================

\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Scenario 2: Insert Stress (Progressive batches: 50, 100, 150, 200, 250 batches of 100 vectors)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
    batch_start_time timestamp;
    batch_end_time timestamp;
    batch_count int := 0;
    base_id int := 100000;
    batch_size int;
    batch_num int;
    i int;
    total_batches int := 0;
    total_vectors int := 0;
    total_time interval := '0 seconds'::interval;
BEGIN
    -- Run progressive batches: 50, 100, 150, 200, 250 batches
    FOR batch_num IN 1..5 LOOP
        batch_size := batch_num * 50;
        batch_start_time := clock_timestamp();
        batch_count := 0;
        base_id := 100000 + (batch_num - 1) * 12500;  -- Offset for each batch
        
        RAISE NOTICE 'Starting batch %: % batches (% vectors)', batch_num, batch_size, batch_size * 100;
        
        FOR i IN 1..batch_size LOOP
            -- Insert batch of 100 vectors
            INSERT INTO stress_vectors (category, embedding, metadata)
            SELECT 
                'cluster_' || (((base_id + (i * 100) + gs.val) % 20) + 1),
                ARRAY(
                    SELECT (random() + ((base_id + (i * 100) + gs.val) % 20) * 0.05)::real 
                    FROM generate_series(1, 128) AS gs2
                )::vector(128),
                jsonb_build_object('id', base_id + (i * 100) + gs.val, 'cluster', ((base_id + (i * 100) + gs.val) % 20) + 1)
            FROM generate_series(0, 99) AS gs(val);
            
            batch_count := batch_count + 1;
        END LOOP;
        
        batch_end_time := clock_timestamp();
        total_batches := total_batches + batch_count;
        total_vectors := total_vectors + (batch_count * 100);
        total_time := total_time + (batch_end_time - batch_start_time);
        
        RAISE NOTICE 'Batch %: Completed % batches (% vectors) in % seconds (%) vectors/second', 
            batch_num,
            batch_count,
            batch_count * 100,
            ROUND(EXTRACT(EPOCH FROM (batch_end_time - batch_start_time))::numeric, 2),
            ROUND(((batch_count * 100) / EXTRACT(EPOCH FROM (batch_end_time - batch_start_time)))::numeric, 2);
    END LOOP;
    
    RAISE NOTICE 'Insert Stress Summary: Completed % batches (% vectors) in % seconds (%) vectors/second', 
        total_batches,
        total_vectors,
        ROUND(EXTRACT(EPOCH FROM total_time)::numeric, 2),
        ROUND((total_vectors / EXTRACT(EPOCH FROM total_time))::numeric, 2);
END $$;

\echo ''

-- ============================================================================
-- Stress Test: Scenario 3 - Mixed Workload (Progressive: queries + inserts)
-- ============================================================================

\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Scenario 3: Mixed Workload (Progressive: 1K+50, 2K+100, 3K+150, 4K+200, 5K+250)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SET enable_seqscan = off;
SET hnsw.ef_search = 100;

DO $$
DECLARE
    batch_start_time timestamp;
    batch_end_time timestamp;
    query_count int := 0;
    insert_count int := 0;
    query_vec vector(128);
    result_count int;
    base_id int := 200000;
    batch_num int;
    query_batch_size int;
    insert_batch_size int;
    i int;
    total_queries int := 0;
    total_inserts int := 0;
    total_vectors int := 0;
    total_time interval := '0 seconds'::interval;
BEGIN
    -- Run progressive batches: (1K queries + 50 inserts), (2K + 100), (3K + 150), (4K + 200), (5K + 250)
    FOR batch_num IN 1..5 LOOP
        query_batch_size := batch_num * 1000;
        insert_batch_size := batch_num * 50;
        batch_start_time := clock_timestamp();
        query_count := 0;
        insert_count := 0;
        base_id := 200000 + (batch_num - 1) * 12500;
        
        RAISE NOTICE 'Starting batch %: % queries + % batch inserts', batch_num, query_batch_size, insert_batch_size;
        
        -- First do queries
        FOR i IN 1..query_batch_size LOOP
            SELECT ARRAY(
                SELECT random()::real 
                FROM generate_series(1, 128)
            )::vector(128) INTO query_vec;
            
            SELECT COUNT(*) INTO result_count
            FROM (
                SELECT id
                FROM stress_vectors
                ORDER BY embedding <-> query_vec
                LIMIT 10
            ) sub;
            
            query_count := query_count + 1;
        END LOOP;
        
        -- Then do batch inserts
        FOR i IN 1..insert_batch_size LOOP
            INSERT INTO stress_vectors (category, embedding, metadata)
            SELECT 
                'cluster_' || (((base_id + (i * 100) + gs.val) % 20) + 1),
                ARRAY(
                    SELECT (random() + ((base_id + (i * 100) + gs.val) % 20) * 0.05)::real 
                    FROM generate_series(1, 128) AS gs2
                )::vector(128),
                jsonb_build_object('id', base_id + (i * 100) + gs.val, 'cluster', ((base_id + (i * 100) + gs.val) % 20) + 1)
            FROM generate_series(0, 99) AS gs(val);
            
            insert_count := insert_count + 1;
        END LOOP;
        
        batch_end_time := clock_timestamp();
        total_queries := total_queries + query_count;
        total_inserts := total_inserts + insert_count;
        total_vectors := total_vectors + (insert_count * 100);
        total_time := total_time + (batch_end_time - batch_start_time);
        
        RAISE NOTICE 'Batch %: Completed % queries + % batch inserts (% vectors) in % seconds', 
            batch_num,
            query_count,
            insert_count,
            insert_count * 100,
            ROUND(EXTRACT(EPOCH FROM (batch_end_time - batch_start_time))::numeric, 2);
    END LOOP;
    
    RAISE NOTICE 'Mixed Workload Summary: Completed % queries + % batch inserts (% vectors) in % seconds', 
        total_queries,
        total_inserts,
        total_vectors,
        ROUND(EXTRACT(EPOCH FROM total_time)::numeric, 2);
END $$;

RESET enable_seqscan;
RESET hnsw.ef_search;

\echo ''

-- ============================================================================
-- Stress Test: Scenario 4 - Index Maintenance (Progressive: 5, 10, 15, 20, 25 ANALYZE operations)
-- ============================================================================

\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Scenario 4: Index Maintenance (Progressive batches: 5, 10, 15, 20, 25 ANALYZE operations)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
    batch_start_time timestamp;
    batch_end_time timestamp;
    analyze_count int := 0;
    batch_size int;
    batch_num int;
    i int;
    total_analyzes int := 0;
    total_time interval := '0 seconds'::interval;
BEGIN
    -- Run progressive batches: 5, 10, 15, 20, 25 ANALYZE operations
    FOR batch_num IN 1..5 LOOP
        batch_size := batch_num * 5;
        batch_start_time := clock_timestamp();
        analyze_count := 0;
        
        RAISE NOTICE 'Starting batch %: % ANALYZE operations', batch_num, batch_size;
        
        FOR i IN 1..batch_size LOOP
            BEGIN
                ANALYZE stress_vectors;
                analyze_count := analyze_count + 1;
            EXCEPTION
                WHEN OTHERS THEN
                    RAISE WARNING 'ANALYZE operation % failed: %', i, SQLERRM;
                    -- Continue with next operation
            END;
        END LOOP;
        
        batch_end_time := clock_timestamp();
        total_analyzes := total_analyzes + analyze_count;
        total_time := total_time + (batch_end_time - batch_start_time);
        
        RAISE NOTICE 'Batch %: Completed % ANALYZE operations in % seconds (%) ops/second', 
            batch_num,
            analyze_count,
            ROUND(EXTRACT(EPOCH FROM (batch_end_time - batch_start_time))::numeric, 2),
            ROUND((analyze_count / EXTRACT(EPOCH FROM (batch_end_time - batch_start_time)))::numeric, 2);
    END LOOP;
    
    RAISE NOTICE 'Index Maintenance Summary: Completed % ANALYZE operations in % seconds (%) ops/second', 
        total_analyzes,
        ROUND(EXTRACT(EPOCH FROM total_time)::numeric, 2),
        ROUND((total_analyzes / EXTRACT(EPOCH FROM total_time))::numeric, 2);
END $$;

\echo ''

-- ============================================================================
-- Summary Statistics
-- ============================================================================

\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Stress Test Summary'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT 
    'stress_vectors' AS table_name,
    COUNT(*) AS total_rows,
    pg_size_pretty(pg_total_relation_size('stress_vectors')) AS total_size,
    pg_size_pretty(pg_relation_size('stress_vectors')) AS table_size
FROM stress_vectors;

\echo ''
\echo 'Index Information:'
SELECT 
    schemaname,
    relname AS tablename,
    indexrelname AS indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE relname = 'stress_vectors'
ORDER BY indexrelname;

\echo ''
\echo '=========================================================================='
\echo 'NeuronDB Vector Stress Benchmark Completed'
\echo '=========================================================================='

\timing off

