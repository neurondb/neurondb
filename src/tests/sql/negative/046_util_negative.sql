-- 033_util_negative.sql
-- Negative test cases for utility module: error handling, invalid inputs

\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP off

\echo '=========================================================================='
\echo 'Utility Module: Negative Test Cases (Error Handling)'
\echo '=========================================================================='

/*-------------------------------------------------------------------
 * ---- CONFIGURATION ERRORS ----
 * Test error handling for configuration operations
 *------------------------------------------------------------------*/
\echo ''
\echo 'Configuration Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 1: Get configuration for non-existent option'
SELECT get_vector_config('nonexistent_config_xyz');

\echo 'Error Test 2: Set configuration with NULL name'
SELECT set_vector_config(NULL, '200');

\echo 'Error Test 3: Set configuration with NULL value'
SELECT set_vector_config('ef_construction', NULL);

\echo 'Error Test 4: Set configuration with invalid value type'
SELECT set_vector_config('ef_construction', 'not_a_number');

\echo 'Error Test 5: Set configuration with out-of-range value'
SELECT set_vector_config('ef_construction', '-100');

\echo 'Error Test 6: Reset non-existent configuration'
SELECT reset_vector_config('nonexistent_config_xyz');

/*-------------------------------------------------------------------
 * ---- SECURITY OPERATION ERRORS ----
 * Test error handling for security operations
 *------------------------------------------------------------------*/
\echo ''
\echo 'Security Operation Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 7: Post-quantum encryption with NULL vector'
SELECT encrypt_postquantum(NULL::vector);

\echo 'Error Test 8: Access mask with NULL role'
SELECT set_access_mask(NULL, 'l2,cosine', 'hnsw');

\echo 'Error Test 9: Access mask with NULL metrics'
SELECT set_access_mask('role1', NULL, 'hnsw');

\echo 'Error Test 10: Access mask with NULL indexes'
SELECT set_access_mask('role1', 'l2,cosine', NULL);

/*-------------------------------------------------------------------
 * ---- HOOK OPERATION ERRORS ----
 * Test error handling for hook operations
 *------------------------------------------------------------------*/
\echo ''
\echo 'Hook Operation Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 11: Register custom operator with NULL name'
SELECT register_custom_operator(NULL, 'vector_custom_distance');

\echo 'Error Test 12: Register custom operator with NULL function'
SELECT register_custom_operator('custom_distance', NULL);

\echo 'Error Test 13: Enable replication with NULL publication'
SELECT enable_vector_replication(NULL);

\echo 'Error Test 14: Create FDW with NULL name'
SELECT create_vector_fdw(NULL, 'host', 5432);

\echo 'Error Test 15: Create FDW with NULL host'
SELECT create_vector_fdw('fdw_name', NULL, 5432);

\echo 'Error Test 16: Create FDW with invalid port'
SELECT create_vector_fdw('fdw_name', 'host', -1);

/*-------------------------------------------------------------------
 * ---- DISTRIBUTED OPERATION ERRORS ----
 * Test error handling for distributed operations
 *------------------------------------------------------------------*/
\echo ''
\echo 'Distributed Operation Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 17: Federated query with NULL host'
SELECT federated_vector_query(NULL, 'SELECT * FROM test');

\echo 'Error Test 18: Federated query with NULL query'
SELECT federated_vector_query('localhost', NULL);

\echo 'Error Test 19: Federated query with invalid host'
SELECT federated_vector_query('invalid_host_xyz:99999', 'SELECT * FROM test');

\echo 'Error Test 20: Federated query with empty query'
SELECT federated_vector_query('localhost', '');

\echo ''
\echo '=========================================================================='
\echo '✓ Utility Module: Negative test cases complete'
\echo '=========================================================================='

\echo 'Test completed successfully'




