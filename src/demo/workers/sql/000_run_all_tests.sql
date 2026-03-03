-- ============================================================================
-- NeuronDB Background Workers - Complete Test Suite
-- ============================================================================
-- Run all worker tests in sequence
-- ============================================================================

\echo '=========================================='
\echo 'NeuronDB Background Workers Test Suite'
\echo 'Running all tests...'
\echo '=========================================='

\timing on

-- Setup & Verification
\echo ''
\echo 'Running: 001_worker_setup.sql'
\i 001_worker_setup.sql

-- Queue Worker Tests
\echo ''
\echo 'Running: 002_queue_worker.sql'
\i 002_queue_worker.sql

-- Auto-Tuner Worker Tests
\echo ''
\echo 'Running: 003_tuner_worker.sql'
\i 003_tuner_worker.sql

-- Defrag Worker Tests
\echo ''
\echo 'Running: 004_defrag_worker.sql'
\i 004_defrag_worker.sql

-- LLM Worker Tests
\echo ''
\echo 'Running: 005_llm_worker.sql'
\i 005_llm_worker.sql

\timing off

-- Final Summary
\echo ''
\echo '=========================================='
\echo 'All Worker Tests Complete!'
\echo '=========================================='
\echo ''
\echo 'Test Summary:'
\echo '  ✓ Worker Setup & Verification'
\echo '  ✓ neuranq (Queue Worker)'
\echo '  ✓ neuranmon (Auto-Tuner)'
\echo '  ✓ neurandefrag (Index Maintenance)'
\echo '  ✓ neuranllm (LLM Processing)'
\echo ''
\echo 'Next Steps:'
\echo '  1. Review test results above'
\echo '  2. Check worker logs for any errors'
\echo '  3. Monitor worker performance in production'
\echo ''
\echo 'Worker Monitoring Commands:'
\echo '  -- Check running workers'
\echo '  SELECT pid, backend_type, backend_start'
\echo '  FROM pg_stat_activity'
\echo '  WHERE backend_type = ''background worker'';'
\echo ''
\echo '  -- Job queue status'
\echo '  SELECT status, COUNT(*) FROM neurondb.job_queue GROUP BY status;'
\echo ''
\echo '  -- LLM job status'
\echo '  SELECT status, COUNT(*) FROM neurondb.llm_jobs GROUP BY status;'
\echo ''
\echo '  -- Performance metrics'
\echo '  SELECT * FROM neurondb.vector_stats;'
\echo ''

