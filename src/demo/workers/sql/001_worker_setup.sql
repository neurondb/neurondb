-- ============================================================================
-- NeuronDB Background Workers Demo - Setup & Verification
-- ============================================================================
-- Purpose: Verify all 4 background workers are properly configured
-- Workers: neuranq, neuranmon, neurandefrag, neuranllm
-- ============================================================================

\echo '=========================================='
\echo 'NeuronDB Workers - Setup & Verification'
\echo '=========================================='

-- Create demo schema
DROP SCHEMA IF EXISTS worker_demo CASCADE;
CREATE SCHEMA worker_demo;
SET search_path TO worker_demo, neurondb, public;

-- ============================================================================
-- STEP 1: Verify Worker GUC Configuration
-- ============================================================================
\echo ''
\echo 'STEP 1: Verify Worker GUC Configuration'
\echo '--------------------------------------------'

-- Show all neurondb worker settings
SELECT 
    name,
    setting,
    unit,
    category,
    short_desc
FROM pg_settings
WHERE name LIKE 'neurondb.neuran%'
ORDER BY name;

-- Verify key worker settings
\echo ''
\echo 'Key Worker Settings:'
SELECT 
    current_setting('neurondb.neuranq_naptime') as queue_naptime_ms,
    current_setting('neurondb.neuranmon_naptime') as tuner_naptime_ms,
    current_setting('neurondb.neurandefrag_naptime') as defrag_naptime_ms;

\echo ''
\echo '✓ Worker GUC configuration verified'

-- ============================================================================
-- STEP 2: Verify Required Tables Exist
-- ============================================================================
\echo ''
\echo 'STEP 2: Verify Required Tables'
\echo '--------------------------------------------'

SELECT 
    schemaname,
    tablename,
    CASE 
        WHEN tablename = 'neurondb_job_queue' THEN 'neuranq (Queue Worker)'
        WHEN tablename = 'neurondb_query_metrics' THEN 'neuranmon (Tuner Worker)'
        WHEN tablename = 'neurondb_llm_jobs' THEN 'neuranllm (LLM Worker)'
        ELSE 'General'
    END as used_by_worker
FROM pg_tables
WHERE schemaname = 'neurondb'
AND tablename IN ('neurondb_job_queue', 'neurondb_query_metrics', 'neurondb_llm_jobs')
ORDER BY tablename;

\echo ''
\echo '✓ Required tables verified'

-- ============================================================================
-- STEP 3: Check Background Worker Status
-- ============================================================================
\echo ''
\echo 'STEP 3: Check Background Worker Status'
\echo '--------------------------------------------'

-- Check if any neurondb workers are running
SELECT 
    pid,
    backend_type,
    backend_start,
    state,
    query
FROM pg_stat_activity
WHERE backend_type = 'background worker'
AND query LIKE '%neuron%'
ORDER BY backend_start DESC;

\echo ''
\echo 'Note: Workers may not be running if shared_preload_libraries is not configured'
\echo 'To enable workers, add to postgresql.conf:'
\echo '  shared_preload_libraries = ''neurondb'''
\echo 'Then restart PostgreSQL.'

\echo ''
\echo '✓ Background worker status checked'

-- ============================================================================
-- STEP 4: Verify Worker Functions
-- ============================================================================
\echo ''
\echo 'STEP 4: Verify Worker Functions'
\echo '--------------------------------------------'

-- List all worker-related functions
SELECT 
    proname as function_name,
    pronargs as num_args,
    pg_get_function_result(oid) as return_type
FROM pg_proc
WHERE pronamespace = 'neurondb'::regnamespace
AND (
    proname LIKE '%tenant%worker%'
    OR proname LIKE '%tenant%stats%'
    OR proname = 'create_tenant_worker'
    OR proname = 'get_tenant_stats'
)
ORDER BY proname;

\echo ''
\echo '✓ Worker functions verified'

\echo ''
\echo '=========================================='
\echo 'Setup Complete!'
\echo '=========================================='

