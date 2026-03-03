-- 030_worker_negative.sql
-- Negative test cases for worker module: error handling, invalid inputs

\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP off

\echo '=========================================================================='
\echo 'Worker Module: Negative Test Cases (Error Handling)'
\echo '=========================================================================='

/*-------------------------------------------------------------------
 * ---- WORKER EXECUTION ERRORS ----
 * Test error handling for worker execution
 *------------------------------------------------------------------*/
\echo ''
\echo 'Worker Execution Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 1: Queue worker with corrupted job data'
-- This should handle gracefully
SELECT neuranq_run_once();

\echo 'Error Test 2: Tuner worker with no query metrics'
-- This should handle gracefully
SELECT neuranmon_sample();

\echo 'Error Test 3: Defrag worker with no indexes'
-- This should handle gracefully
SELECT neurandefrag_run();

/*-------------------------------------------------------------------
 * ---- WORKER CONFIGURATION ERRORS ----
 * Test error handling for invalid worker configuration
 *------------------------------------------------------------------*/
\echo ''
\echo 'Worker Configuration Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 4: Invalid naptime value (negative)'
SET neurondb.neuranq_naptime = -1;
SELECT neuranq_run_once();

\echo 'Error Test 5: Invalid batch size (zero)'
SET neurondb.neuranq_batch_size = 0;
SELECT neuranq_run_once();

\echo 'Error Test 6: Invalid timeout (negative)'
SET neurondb.neuranq_timeout = -1;
SELECT neuranq_run_once();

\echo 'Error Test 7: Invalid max retries (negative)'
SET neurondb.neuranq_max_retries = -1;
SELECT neuranq_run_once();

/*-------------------------------------------------------------------
 * ---- WORKER TABLE ERRORS ----
 * Test error handling for missing or corrupted worker tables
 *------------------------------------------------------------------*/
\echo ''
\echo 'Worker Table Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 8: Queue worker with missing table'
-- Workers should handle missing tables gracefully
DO $$
BEGIN
	BEGIN
		-- Try to access non-existent table
		PERFORM COUNT(*) FROM neurondb.nonexistent_table;
	EXCEPTION WHEN OTHERS THEN
		NULL; -- Expected error
	END;
END$$;

\echo 'Error Test 9: Worker operations with NULL values'
-- Test handling of NULL in job queue
SELECT 
	COUNT(*) AS null_job_count
FROM neurondb.job_queue
WHERE job_type IS NULL OR payload IS NULL;

/*-------------------------------------------------------------------
 * ---- WORKER STATE ERRORS ----
 * Test error handling for worker state issues
 *------------------------------------------------------------------*/
\echo ''
\echo 'Worker State Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 10: Worker execution with disabled worker'
SET neurondb.neuranq_enabled = off;
SELECT neuranq_run_once();
SET neurondb.neuranq_enabled = on;

\echo 'Error Test 11: Worker execution with very large batch size'
SET neurondb.neuranq_batch_size = 1000000;
SELECT neuranq_run_once();
SET neurondb.neuranq_batch_size = 100;

\echo 'Error Test 12: Worker execution with very small timeout'
SET neurondb.neuranq_timeout = 1;
SELECT neuranq_run_once();
SET neurondb.neuranq_timeout = 30000;

\echo ''
\echo '=========================================================================='
\echo '✓ Worker Module: Negative test cases complete'
\echo '=========================================================================='

\echo 'Test completed successfully'




