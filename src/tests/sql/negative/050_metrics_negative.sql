-- 037_metrics_negative.sql
-- Negative test cases for metrics module: error handling

\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP off

\echo '=========================================================================='
\echo 'Metrics Module: Negative Test Cases (Error Handling)'
\echo '=========================================================================='

\echo ''
\echo 'Metrics Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 1: Statistics view with corrupted data'
-- Statistics view should handle gracefully
SELECT * FROM pg_stat_neurondb();

\echo 'Error Test 2: Statistics reset multiple times'
SELECT pg_neurondb_stat_reset() AS reset1;
SELECT pg_neurondb_stat_reset() AS reset2;
SELECT pg_neurondb_stat_reset() AS reset3;

\echo ''
\echo '=========================================================================='
\echo '✓ Metrics Module: Negative test cases complete'
\echo '=========================================================================='

\echo 'Test completed successfully'




