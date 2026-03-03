-- 031_storage_basic.sql
-- Basic test for storage module: buffer management, WAL operations, ANN buffer

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
\echo 'Storage Module: Basic Functionality Tests'
\echo '=========================================================================='

/*-------------------------------------------------------------------
 * ---- BUFFER MANAGEMENT ----
 *------------------------------------------------------------------*/
\echo ''
\echo 'Buffer Management Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 1: Create table with vector column for buffer testing'
DROP TABLE IF EXISTS storage_test_table;
-- Workaround for PostgreSQL 18 B-tree deduplication bug
DROP SEQUENCE IF EXISTS storage_test_table_id_seq CASCADE;
CREATE SEQUENCE storage_test_table_id_seq;
CREATE TABLE storage_test_table (
	id INTEGER DEFAULT nextval('storage_test_table_id_seq'),
	embedding vector(28),
	metadata jsonb
);

-- Insert test data
INSERT INTO storage_test_table (embedding, metadata)
SELECT features, '{"source": "test"}'::jsonb
FROM test_train_view
LIMIT 100;

\echo 'Test 2: Buffer operations through normal table operations'
SELECT COUNT(*) AS row_count FROM storage_test_table;
SELECT pg_size_pretty(pg_relation_size('storage_test_table'::regclass)) AS table_size;

\echo 'Test 3: Buffer statistics'
SELECT 
	schemaname,
	relname AS tablename,
	heap_blks_read,
	heap_blks_hit,
	idx_blks_read,
	idx_blks_hit
FROM pg_statio_user_tables
WHERE relname = 'storage_test_table';

/*-------------------------------------------------------------------
 * ---- WAL OPERATIONS ----
 *------------------------------------------------------------------*/
\echo ''
\echo 'WAL Operations Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 4: WAL compression (if available)'
SELECT 
	'WAL Compress' AS test_type,
	COUNT(*) AS n_compressed
FROM (
	SELECT vector_wal_compress(
		vector_out(embedding)::text,
		vector_out((SELECT embedding FROM storage_test_table LIMIT 1))::text
	) AS compressed
	FROM storage_test_table
	LIMIT 10
) sub;

\echo 'Test 5: WAL size estimation'
SELECT 
	'WAL Estimate' AS test_type,
	AVG(vector_wal_estimate_size(
		vector_out(embedding)::text,
		vector_out((SELECT embedding FROM storage_test_table LIMIT 1))::text
	)) AS avg_estimated_size
FROM storage_test_table
LIMIT 10;

\echo 'Test 6: WAL statistics'
SELECT vector_wal_get_stats() AS wal_stats;

/*-------------------------------------------------------------------
 * ---- ANN BUFFER OPERATIONS ----
 *------------------------------------------------------------------*/
\echo ''
\echo 'ANN Buffer Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 7: Create index for ANN buffer testing'
CREATE INDEX idx_storage_hnsw ON storage_test_table 
USING hnsw (embedding vector_l2_ops) 
WITH (m = 16, ef_construction = 200);

\echo 'Test 8: Query using ANN buffer'
SELECT 
	id,
	embedding <-> (SELECT embedding FROM storage_test_table LIMIT 1) AS distance
FROM storage_test_table
ORDER BY embedding <-> (SELECT embedding FROM storage_test_table LIMIT 1)
LIMIT 10;

\echo ''
\echo '=========================================================================='
\echo '✓ Storage Module: Basic tests complete'
\echo '=========================================================================='

DROP TABLE IF EXISTS storage_test_table CASCADE;

\echo 'Test completed successfully'
