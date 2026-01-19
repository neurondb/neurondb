-- 030_worker_basic.sql
-- Basic test for worker module: worker status and manual execution

\timing on
\pset footer off
\pset pager off
\pset tuples_only off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Worker Module: Basic Functionality Tests'
\echo '=========================================================================='

/*-------------------------------------------------------------------
 * ---- WORKER STATUS ----
 *------------------------------------------------------------------*/
\echo ''
\echo 'Worker Status Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 1: List worker functions'
SELECT 
	proname as function_name,
	pronargs as num_args,
	pg_get_function_result(oid) as return_type
FROM pg_proc
WHERE pronamespace = 'neurondb'::regnamespace
AND (
	proname LIKE '%neuran%'
	OR proname LIKE '%worker%'
)
ORDER BY proname;

\echo 'Test 2: Check worker status (if available)'
-- This may return empty if workers are not running
SELECT 
	pid,
	backend_type,
	state
FROM pg_stat_activity
WHERE backend_type = 'background worker'
AND (query LIKE '%neuron%' OR application_name LIKE '%neuron%')
ORDER BY backend_start DESC;

/*-------------------------------------------------------------------
 * ---- MANUAL WORKER EXECUTION ----
 *------------------------------------------------------------------*/
\echo ''
\echo 'Manual Worker Execution Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 3: Queue worker manual execution'
SELECT neuranq_run_once() AS queue_processed;

\echo 'Test 4: Tuner worker manual execution'
SELECT neuranmon_sample() AS tuner_sampled;

\echo 'Test 5: Defrag worker manual execution'
SELECT neurandefrag_run() AS defrag_executed;

/*-------------------------------------------------------------------
 * ---- WORKER TABLES ----
 *------------------------------------------------------------------*/
\echo ''
\echo 'Worker Table Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 6: Check worker tables exist'
SELECT 
	schemaname,
	tablename,
	CASE 
		WHEN tablename = 'neurondb_job_queue' THEN 'Queue Worker'
		WHEN tablename = 'neurondb_query_metrics' THEN 'Tuner Worker'
		WHEN tablename = 'neurondb_llm_jobs' THEN 'LLM Worker'
		ELSE 'Other'
	END as worker_type
FROM pg_tables
WHERE schemaname = 'neurondb'
AND tablename IN ('neurondb_job_queue', 'neurondb_query_metrics', 'neurondb_llm_jobs')
ORDER BY tablename;

\echo ''
\echo '=========================================================================='
\echo '✓ Worker Module: Basic tests complete'
\echo '=========================================================================='

\echo 'Test completed successfully'
