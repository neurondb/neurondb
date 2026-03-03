-- 035_tenant_negative.sql
-- Negative test cases for tenant module: error handling

\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP off

\echo '=========================================================================='
\echo 'Tenant Module: Negative Test Cases (Error Handling)'
\echo '=========================================================================='

/*-------------------------------------------------------------------
 * ---- TENANT WORKER ERRORS ----
 *------------------------------------------------------------------*/
\echo ''
\echo 'Tenant Worker Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 1: Create tenant worker with NULL tenant_id'
SELECT create_tenant_worker(NULL, 'queue', '{}');

\echo 'Error Test 2: Create tenant worker with NULL worker_type'
SELECT create_tenant_worker('tenant1', NULL, '{}');

\echo 'Error Test 3: Create tenant worker with NULL config'
SELECT create_tenant_worker('tenant1', 'queue', NULL);

\echo 'Error Test 4: Create tenant worker with invalid tenant_id (too long)'
SELECT create_tenant_worker('a' || repeat('b', 100), 'queue', '{}');

\echo 'Error Test 5: Get tenant stats with NULL tenant_id'
SELECT * FROM get_tenant_stats(NULL);

\echo 'Error Test 6: Get tenant stats for non-existent tenant'
SELECT * FROM get_tenant_stats('nonexistent_tenant_xyz');

\echo ''
\echo '=========================================================================='
\echo '✓ Tenant Module: Negative test cases complete'
\echo '=========================================================================='

\echo 'Test completed successfully'




