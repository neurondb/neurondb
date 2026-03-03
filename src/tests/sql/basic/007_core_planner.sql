-- 034_planner_basic.sql
-- Basic test for planner module: query optimization paths

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
\echo 'Planner Module: Basic Functionality Tests'
\echo '=========================================================================='

/*-------------------------------------------------------------------
 * ---- QUERY OPTIMIZATION ----
 * Test planner optimization through EXPLAIN
 *------------------------------------------------------------------*/
\echo ''
\echo 'Query Optimization Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Create test table
DROP TABLE IF EXISTS planner_test_table;
-- Workaround for PostgreSQL 18 B-tree deduplication bug
DROP SEQUENCE IF EXISTS planner_test_table_id_seq CASCADE;
CREATE SEQUENCE planner_test_table_id_seq;
CREATE TABLE planner_test_table (
	id INTEGER DEFAULT nextval('planner_test_table_id_seq'),
	embedding vector(28),
	label integer
);

INSERT INTO planner_test_table (embedding, label)
SELECT features, label
FROM test_train_view
LIMIT 100;

CREATE INDEX idx_planner_hnsw ON planner_test_table 
USING hnsw (embedding vector_l2_ops);

\echo 'Test 1: EXPLAIN query plan for KNN search'
EXPLAIN (ANALYZE, BUFFERS) 
SELECT id, embedding <-> (SELECT embedding FROM planner_test_table LIMIT 1) AS distance
FROM planner_test_table
ORDER BY embedding <-> (SELECT embedding FROM planner_test_table LIMIT 1)
LIMIT 10;

\echo 'Test 2: EXPLAIN query plan with filter'
EXPLAIN (ANALYZE, BUFFERS)
SELECT id, embedding <-> (SELECT embedding FROM planner_test_table LIMIT 1) AS distance
FROM planner_test_table
WHERE label < 5
ORDER BY embedding <-> (SELECT embedding FROM planner_test_table LIMIT 1)
LIMIT 10;

\echo ''
\echo '=========================================================================='
\echo '✓ Planner Module: Basic tests complete'
\echo '=========================================================================='

DROP TABLE IF EXISTS planner_test_table CASCADE;

\echo 'Test completed successfully'
