-- 037_metrics_basic.sql
-- Basic test for metrics module: statistics and Prometheus metrics

\timing on
\pset footer off
\pset pager off
\pset tuples_only off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Metrics Module: Basic Functionality Tests'
\echo '=========================================================================='

/*-------------------------------------------------------------------
 * ---- STATISTICS VIEWS ----
 *------------------------------------------------------------------*/
\echo ''
\echo 'Statistics Views Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 1: pg_stat_neurondb view'
SELECT * FROM pg_stat_neurondb();

\echo 'Test 2: Reset statistics'
SELECT pg_neurondb_stat_reset() AS stats_reset;

\echo 'Test 3: Statistics after reset'
SELECT * FROM pg_stat_neurondb();

/*-------------------------------------------------------------------
 * ---- PROMETHEUS METRICS ----
 *------------------------------------------------------------------*/
\echo ''
\echo 'Prometheus Metrics Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 4: Prometheus metrics export (if available)'
-- Prometheus metrics are typically accessed via HTTP endpoint
-- Test through internal functions if available
DO $$
BEGIN
	-- Metrics are typically exported via HTTP, test through views
	PERFORM COUNT(*) FROM pg_stat_neurondb();
END$$;

\echo ''
\echo '=========================================================================='
\echo '✓ Metrics Module: Basic tests complete'
\echo '=========================================================================='

\echo 'Test completed successfully'
