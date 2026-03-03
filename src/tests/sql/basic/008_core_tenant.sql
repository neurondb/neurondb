-- 035_tenant_basic.sql
-- Basic test for tenant module: multi-tenancy operations

\timing on
\pset footer off
\pset pager off
\pset tuples_only off

\set ON_ERROR_STOP on

-- Create test_train table and view
DROP TABLE IF EXISTS test_train CASCADE;
CREATE TABLE test_train (features vector(28), label integer);
INSERT INTO test_train (features, label) SELECT array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 28)))::vector(28), (random() * 2)::integer FROM generate_series(1, 1000);
CREATE OR REPLACE VIEW test_train_view AS SELECT features, label FROM test_train;

CREATE TABLE IF NOT EXISTS test_train (features vector(28), label integer);
DELETE FROM test_train;
INSERT INTO test_train (features, label) SELECT array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 28)))::vector(28), (random() * 2)::integer FROM generate_series(1, 1000);
CREATE OR REPLACE VIEW test_train_view AS SELECT features, label FROM test_train;


\echo '=========================================================================='
\echo 'Tenant Module: Basic Functionality Tests'
\echo '=========================================================================='

/*-------------------------------------------------------------------
 * ---- TENANT WORKER OPERATIONS ----
 *------------------------------------------------------------------*/
\echo ''
\echo 'Tenant Worker Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 1: Create tenant worker'
SELECT create_tenant_worker('tenant1'::text, 'queue'::text, '{"batch_size": 100}'::text) AS worker_created;

\echo 'Test 2: Get tenant statistics'
SELECT * FROM get_tenant_stats('tenant1');

/*-------------------------------------------------------------------
 * ---- TENANT ISOLATION ----
 *------------------------------------------------------------------*/
\echo ''
\echo 'Tenant Isolation Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 3: Create table with tenant_id'
DROP TABLE IF EXISTS tenant_test_table;
-- Workaround for PostgreSQL 18 B-tree deduplication bug
DROP SEQUENCE IF EXISTS tenant_test_table_id_seq CASCADE;
CREATE SEQUENCE tenant_test_table_id_seq;
CREATE TABLE tenant_test_table (
	id INTEGER DEFAULT nextval('tenant_test_table_id_seq'),
	embedding vector(28),
	tenant_id integer,
	label integer
);

INSERT INTO tenant_test_table (embedding, tenant_id, label)
SELECT features, (i % 3) + 1 AS tenant_id, label
FROM test_train_view, generate_series(1, 3) i
LIMIT 300;

\echo 'Test 4: Query with tenant filter'
SELECT 
	tenant_id,
	COUNT(*) AS vector_count
FROM tenant_test_table
GROUP BY tenant_id
ORDER BY tenant_id;

\echo ''
\echo '=========================================================================='
\echo '✓ Tenant Module: Basic tests complete'
\echo '=========================================================================='

DROP TABLE IF EXISTS tenant_test_table CASCADE;

\echo 'Test completed successfully'
