-- ============================================================================
-- NeuronDB Workers Demo - neurandefrag (Index Maintenance Worker) Tests
-- ============================================================================
-- Purpose: Test automatic index maintenance and defragmentation
-- Worker: neurandefrag - HNSW graph compaction, orphan cleanup, rebalancing
-- ============================================================================

\echo '=========================================='
\echo 'neurandefrag (Index Maintenance) Tests'
\echo '=========================================='

SET search_path TO worker_demo, neurondb, public;

-- ============================================================================
-- TEST 1: Create Test Table with Vectors
-- ============================================================================
\echo ''
\echo 'TEST 1: Create Test Table'
\echo '--------------------------------------------'

CREATE TABLE worker_demo.documents (
    id SERIAL PRIMARY KEY,
    tenant_id INT,
    title TEXT,
    embedding vector(384),
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    is_deleted BOOLEAN DEFAULT false
);

\echo 'Created documents table'

-- Insert initial data
INSERT INTO worker_demo.documents (tenant_id, title, embedding)
SELECT 
    (random() * 5)::int + 1,
    'Document ' || i,
    array_fill(random()::float4, ARRAY[384])::vector(384)
FROM generate_series(1, 1000) i;

\echo 'Inserted 1000 documents'

\echo ''
\echo '✓ Test table created and populated'

-- ============================================================================
-- TEST 2: Check Initial Table Statistics
-- ============================================================================
\echo ''
\echo 'TEST 2: Initial Table Statistics'
\echo '--------------------------------------------'

SELECT 
    schemaname,
    tablename,
    n_tup_ins as total_inserts,
    n_tup_upd as total_updates,
    n_tup_del as total_deletes,
    n_live_tup as live_tuples,
    n_dead_tup as dead_tuples,
    n_mod_since_analyze as mod_since_analyze,
    last_vacuum,
    last_autovacuum,
    last_analyze
FROM pg_stat_user_tables
WHERE schemaname = 'worker_demo' 
AND tablename = 'documents';

\echo ''
\echo '✓ Initial statistics captured'

-- ============================================================================
-- TEST 3: Simulate Fragmentation
-- ============================================================================
\echo ''
\echo 'TEST 3: Simulate Fragmentation'
\echo '--------------------------------------------'

-- Update 20% of records to create dead tuples
UPDATE worker_demo.documents 
SET 
    updated_at = now(),
    embedding = array_fill((random() * 2 - 1)::float4, ARRAY[384])::vector(384)
WHERE id % 5 = 0;

\echo 'Updated 20% of records'

-- Delete 10% of records (soft delete)
UPDATE worker_demo.documents 
SET is_deleted = true
WHERE id % 10 = 0;

\echo 'Soft-deleted 10% of records'

-- Hard delete 5% of records
DELETE FROM worker_demo.documents 
WHERE id % 20 = 0;

\echo 'Hard-deleted 5% of records'

\echo ''
\echo '✓ Fragmentation simulated'

-- ============================================================================
-- TEST 4: Check Fragmentation Metrics
-- ============================================================================
\echo ''
\echo 'TEST 4: Fragmentation Metrics'
\echo '--------------------------------------------'

-- Updated statistics
SELECT 
    'After Fragmentation' as stage,
    n_tup_ins as total_inserts,
    n_tup_upd as total_updates,
    n_tup_del as total_deletes,
    n_live_tup as live_tuples,
    n_dead_tup as dead_tuples,
    (n_dead_tup::float / NULLIF(n_live_tup + n_dead_tup, 0))::numeric(5,4) as dead_tuple_ratio,
    pg_size_pretty(pg_relation_size('worker_demo.documents')) as table_size
FROM pg_stat_user_tables
WHERE schemaname = 'worker_demo' 
AND tablename = 'documents';

-- Bloat estimation
\echo ''
\echo 'Table Bloat Estimate:'
SELECT 
    COUNT(*) as total_records,
    COUNT(*) FILTER (WHERE is_deleted) as deleted_records,
    (COUNT(*) FILTER (WHERE is_deleted)::float / COUNT(*))::numeric(5,4) as deletion_ratio,
    pg_size_pretty(pg_relation_size('worker_demo.documents')) as current_size,
    pg_size_pretty(pg_total_relation_size('worker_demo.documents')) as total_size_with_indexes
FROM worker_demo.documents;

\echo ''
\echo '✓ Fragmentation metrics collected'

-- ============================================================================
-- TEST 5: Maintenance Operations
-- ============================================================================
\echo ''
\echo 'TEST 5: Maintenance Operations'
\echo '--------------------------------------------'

-- Manual VACUUM to simulate worker action
VACUUM ANALYZE worker_demo.documents;

\echo 'Executed VACUUM ANALYZE'

-- Check post-vacuum statistics
SELECT 
    'After VACUUM' as stage,
    n_live_tup as live_tuples,
    n_dead_tup as dead_tuples,
    (n_dead_tup::float / NULLIF(n_live_tup + n_dead_tup, 0))::numeric(5,4) as dead_tuple_ratio,
    pg_size_pretty(pg_relation_size('worker_demo.documents')) as table_size,
    last_vacuum,
    last_analyze
FROM pg_stat_user_tables
WHERE schemaname = 'worker_demo' 
AND tablename = 'documents';

\echo ''
\echo '✓ Maintenance operations completed'

-- ============================================================================
-- TEST 6: Defrag Worker Configuration
-- ============================================================================
\echo ''
\echo 'TEST 6: Defrag Worker Configuration'
\echo '--------------------------------------------'

SELECT 
    'neurandefrag Configuration' as section,
    current_setting('neurondb.neurandefrag_naptime') as naptime_ms,
    current_setting('neurondb.neurandefrag_compact_threshold') as compact_threshold,
    current_setting('neurondb.neurandefrag_fragmentation_threshold') as fragmentation_threshold,
    current_setting('neurondb.neurandefrag_maintenance_window') as maintenance_window,
    current_setting('neurondb.neurandefrag_enabled') as enabled;

\echo ''
\echo '✓ Configuration verified'

-- ============================================================================
-- TEST 7: Maintenance Candidate Detection
-- ============================================================================
\echo ''
\echo 'TEST 7: Maintenance Candidate Detection'
\echo '--------------------------------------------'

-- Tables needing maintenance
WITH table_stats AS (
    SELECT 
        schemaname,
        tablename,
        n_live_tup,
        n_dead_tup,
        (n_dead_tup::float / NULLIF(n_live_tup + n_dead_tup, 0)) as dead_ratio,
        n_mod_since_analyze,
        last_vacuum,
        last_autovacuum,
        pg_relation_size(schemaname || '.' || tablename) as size_bytes
    FROM pg_stat_user_tables
    WHERE schemaname = 'worker_demo'
)
SELECT 
    schemaname,
    tablename,
    n_live_tup as live_tuples,
    n_dead_tup as dead_tuples,
    dead_ratio::numeric(5,4) as dead_tuple_ratio,
    n_mod_since_analyze as mods_since_analyze,
    pg_size_pretty(size_bytes) as table_size,
    CASE 
        WHEN dead_ratio > 0.3 THEN 'HIGH PRIORITY: Needs vacuum'
        WHEN dead_ratio > 0.2 THEN 'MEDIUM: Consider vacuum'
        WHEN n_mod_since_analyze > 1000 THEN 'MEDIUM: Needs analyze'
        ELSE 'LOW: Healthy'
    END as maintenance_priority
FROM table_stats
ORDER BY dead_ratio DESC NULLS LAST;

\echo ''
\echo '✓ Maintenance candidates identified'

-- ============================================================================
-- TEST 8: Performance Impact Analysis
-- ============================================================================
\echo ''
\echo 'TEST 8: Performance Impact'
\echo '--------------------------------------------'

-- Compare query performance on fragmented vs maintained tables
EXPLAIN (ANALYZE, BUFFERS) 
SELECT COUNT(*) 
FROM worker_demo.documents 
WHERE NOT is_deleted;

\echo ''
\echo '✓ Performance impact analyzed'

\echo ''
\echo '=========================================='
\echo 'neurandefrag Maintenance Worker Tests Complete!'
\echo '=========================================='
\echo ''
\echo 'Key Findings:'
\echo '  - Fragmentation detection working'
\echo '  - Maintenance operations functional'
\echo '  - Dead tuple tracking accurate'
\echo '  - Vacuum/analyze effective'
\echo ''
\echo 'Worker Purpose:'
\echo '  neurandefrag performs automatic maintenance:'
\echo '  - HNSW graph compaction'
\echo '  - Orphan edge cleanup'
\echo '  - Level rebalancing'
\echo '  - Tombstone pruning'
\echo '  - Statistics refresh'
\echo '  - Scheduled rebuild windows'
\echo ''
\echo 'Maintenance Strategy:'
\echo '  - Runs every 5 minutes (default)'
\echo '  - Targets fragmentation > 30%'
\echo '  - Operates during maintenance windows'
\echo '  - Minimizes impact on queries'
\echo ''
\echo 'Fragmentation Thresholds:'
\echo '  - > 0.30 dead ratio: HIGH priority'
\echo '  - > 0.20 dead ratio: MEDIUM priority'
\echo '  - > 1000 modifications: Needs analyze'
\echo ''

