-- 036_types_basic.sql
-- Basic test for types module: quantization and aggregates

\timing on
\pset footer off
\pset pager off
\pset tuples_only off

\set ON_ERROR_STOP on

-- Create test_train table and view if they don't exist (after extension is loaded)
DROP TABLE IF EXISTS test_train CASCADE;
CREATE TABLE test_train (features vector(28), label integer);
INSERT INTO test_train (features, label) 
SELECT array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 28)))::vector(28), (random() * 2)::integer 
FROM generate_series(1, 1000);
CREATE OR REPLACE VIEW test_train_view AS SELECT features, label FROM test_train;

\echo '=========================================================================='
\echo 'Types Module: Basic Functionality Tests'
\echo '=========================================================================='

/*-------------------------------------------------------------------
 * ---- QUANTIZATION TYPES ----
 *------------------------------------------------------------------*/
\echo ''
\echo 'Quantization Types Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 1: INT8 quantization'
SELECT 
	vector_quantize_int8(
		vector '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]'::vector,
		vector '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'::vector,
		vector '[10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]'::vector
	) AS int8_quantized;

\echo 'Test 2: FP16 quantization'
SELECT vector_quantize_fp16(vector '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]'::vector) AS fp16_quantized;

\echo 'Test 3: Binary quantization'
SELECT vector_quantize_binary(vector '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]'::vector) AS binary_quantized;

/*-------------------------------------------------------------------
 * ---- AGGREGATE FUNCTIONS ----
 *------------------------------------------------------------------*/
\echo ''
\echo 'Aggregate Functions Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 4: Vector aggregates (if available)'
DROP TABLE IF EXISTS types_test_table;
-- Workaround for PostgreSQL 18 B-tree deduplication bug
DROP SEQUENCE IF EXISTS types_test_table_id_seq CASCADE;
CREATE SEQUENCE types_test_table_id_seq;
CREATE TABLE types_test_table (
    id INTEGER DEFAULT nextval('types_test_table_id_seq'),
    embedding vector(28)
);

INSERT INTO types_test_table (embedding)
SELECT features FROM test_train_view LIMIT 100;

-- Test aggregates
SELECT 
	COUNT(*) AS count,
	AVG(vector_norm(embedding)) AS avg_norm,
	MIN(vector_norm(embedding)) AS min_norm,
	MAX(vector_norm(embedding)) AS max_norm
FROM types_test_table;

\echo ''
\echo '=========================================================================='
\echo '✓ Types Module: Basic tests complete'
\echo '=========================================================================='

DROP TABLE IF EXISTS types_test_table CASCADE;

\echo 'Test completed successfully'
