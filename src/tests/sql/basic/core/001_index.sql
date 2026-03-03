-- 030_index_basic.sql
-- Basic test for index module: HNSW and IVF index creation and queries

\timing on
\pset footer off
\pset pager off
\pset tuples_only off

\set ON_ERROR_STOP on

-- Ensure neurondb types/operators (including vector) are available
CREATE EXTENSION IF NOT EXISTS neurondb;

-- Create test_train table and view
DROP TABLE IF EXISTS test_train CASCADE;
CREATE TABLE test_train (features vector(28), label integer);
INSERT INTO test_train (features, label) SELECT array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 28)))::vector(28), (random() * 2)::integer FROM generate_series(1, 1000);
CREATE OR REPLACE VIEW test_train_view AS SELECT features, label FROM test_train;
SET client_min_messages TO WARNING;

\echo '=========================================================================='
\echo 'Index Module: Basic Functionality Tests'
\echo '=========================================================================='

/*-------------------------------------------------------------------
 * ---- HNSW INDEX CREATION ----
 *------------------------------------------------------------------*/
\echo ''
\echo 'HNSW Index Creation'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Create test table
DROP TABLE IF EXISTS index_test_table;
CREATE TABLE index_test_table (
	id SERIAL PRIMARY KEY,
	embedding vector(28),
	label integer
);

-- Insert test data
INSERT INTO index_test_table (embedding, label)
SELECT features, label
FROM test_train_view
LIMIT 100;

\echo 'Test 1: Create HNSW index with default parameters'
CREATE INDEX idx_test_hnsw_default ON index_test_table 
USING hnsw (embedding vector_l2_ops);

\echo 'Test 2: Create HNSW index with custom parameters'
DROP INDEX IF EXISTS idx_test_hnsw_custom;
CREATE INDEX idx_test_hnsw_custom ON index_test_table 
USING hnsw (embedding vector_l2_ops) 
WITH (m = 16, ef_construction = 200);

\echo 'Test 2a: Verify HNSW index parameters were set correctly'
DO $$
DECLARE
	v_indexdef text;
	v_m_found boolean := false;
	v_ef_construction_found boolean := false;
BEGIN
	SELECT indexdef INTO v_indexdef
	FROM pg_indexes
	WHERE indexname = 'idx_test_hnsw_custom';
	
	IF v_indexdef IS NULL THEN
		RAISE EXCEPTION 'Index idx_test_hnsw_custom not found';
	END IF;
	
	v_m_found := v_indexdef LIKE '%m=''16''%';
	v_ef_construction_found := v_indexdef LIKE '%ef_construction=''200''%';
	
	IF v_m_found AND v_ef_construction_found THEN
		RAISE NOTICE 'PASS: Parameters correctly set (m=16, ef_construction=200)';
	ELSE
		RAISE EXCEPTION 'FAIL: Parameters not correctly set. Expected m=16, ef_construction=200. Got: %', v_indexdef;
	END IF;
END $$;

\echo 'Test 3: Create HNSW index with cosine distance'
DROP INDEX IF EXISTS idx_test_hnsw_cosine;
CREATE INDEX idx_test_hnsw_cosine ON index_test_table 
USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 200);

\echo 'Test 3a: Verify HNSW cosine index parameters were set correctly'
DO $$
DECLARE
	v_indexdef text;
	v_m_found boolean := false;
	v_ef_construction_found boolean := false;
BEGIN
	SELECT indexdef INTO v_indexdef
	FROM pg_indexes
	WHERE indexname = 'idx_test_hnsw_cosine';
	
	IF v_indexdef IS NULL THEN
		RAISE EXCEPTION 'Index idx_test_hnsw_cosine not found';
	END IF;
	
	v_m_found := v_indexdef LIKE '%m=''16''%';
	v_ef_construction_found := v_indexdef LIKE '%ef_construction=''200''%';
	
	IF v_m_found AND v_ef_construction_found THEN
		RAISE NOTICE 'PASS: Parameters correctly set (m=16, ef_construction=200)';
	ELSE
		RAISE EXCEPTION 'FAIL: Parameters not correctly set. Expected m=16, ef_construction=200. Got: %', v_indexdef;
	END IF;
END $$;

\echo 'Test 4: Create HNSW index with inner product'
DROP INDEX IF EXISTS idx_test_hnsw_ip;
CREATE INDEX idx_test_hnsw_ip ON index_test_table 
USING hnsw (embedding vector_ip_ops) 
WITH (m = 16, ef_construction = 200);

\echo 'Test 4a: Verify HNSW inner product index parameters were set correctly'
DO $$
DECLARE
	v_indexdef text;
	v_m_found boolean := false;
	v_ef_construction_found boolean := false;
BEGIN
	SELECT indexdef INTO v_indexdef
	FROM pg_indexes
	WHERE indexname = 'idx_test_hnsw_ip';
	
	IF v_indexdef IS NULL THEN
		RAISE EXCEPTION 'Index idx_test_hnsw_ip not found';
	END IF;
	
	v_m_found := v_indexdef LIKE '%m=''16''%';
	v_ef_construction_found := v_indexdef LIKE '%ef_construction=''200''%';
	
	IF v_m_found AND v_ef_construction_found THEN
		RAISE NOTICE 'PASS: Parameters correctly set (m=16, ef_construction=200)';
	ELSE
		RAISE EXCEPTION 'FAIL: Parameters not correctly set. Expected m=16, ef_construction=200. Got: %', v_indexdef;
	END IF;
END $$;

/*-------------------------------------------------------------------
 * ---- IVF INDEX CREATION ----
 *------------------------------------------------------------------*/
\echo ''
\echo 'IVF Index Creation'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 5: Create IVF index with default parameters'
-- Default nlists=100 may be too large for single page, use smaller value
CREATE INDEX idx_test_ivf_default ON index_test_table 
USING ivf (embedding vector_l2_ops) WITH (lists = 10);

\echo 'Test 6: Create IVF index with custom parameters'
DROP INDEX IF EXISTS idx_test_ivf_custom;
CREATE INDEX idx_test_ivf_custom ON index_test_table 
USING ivf (embedding vector_l2_ops) 
WITH (lists = 10);

\echo 'Test 6a: Verify IVF index parameters were set correctly'
DO $$
DECLARE
	v_indexdef text;
	v_lists_found boolean := false;
BEGIN
	SELECT indexdef INTO v_indexdef
	FROM pg_indexes
	WHERE indexname = 'idx_test_ivf_custom';
	
	IF v_indexdef IS NULL THEN
		RAISE EXCEPTION 'Index idx_test_ivf_custom not found';
	END IF;
	
	v_lists_found := v_indexdef LIKE '%lists=''10''%';
	
	IF v_lists_found THEN
		RAISE NOTICE 'PASS: Parameters correctly set (lists=10)';
	ELSE
		RAISE EXCEPTION 'FAIL: Parameters not correctly set. Expected lists=10. Got: %', v_indexdef;
	END IF;
END $$;

/*-------------------------------------------------------------------
 * ---- INDEX QUERIES ----
 *------------------------------------------------------------------*/
\echo ''
\echo 'Index Query Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 7: KNN query using HNSW index'
SELECT 
	id,
	embedding <-> (SELECT embedding FROM index_test_table LIMIT 1) AS distance
FROM index_test_table
ORDER BY embedding <-> (SELECT embedding FROM index_test_table LIMIT 1)
LIMIT 10;

\echo 'Test 8: KNN query using IVF index'
SELECT 
	id,
	embedding <-> (SELECT embedding FROM index_test_table LIMIT 1) AS distance
FROM index_test_table
ORDER BY embedding <-> (SELECT embedding FROM index_test_table LIMIT 1)
LIMIT 10;

\echo 'Test 9: Cosine distance query'
SELECT 
	id,
	embedding <=> (SELECT embedding FROM index_test_table LIMIT 1) AS distance
FROM index_test_table
ORDER BY embedding <=> (SELECT embedding FROM index_test_table LIMIT 1)
LIMIT 10;

\echo 'Test 10: Inner product query'
SELECT 
	id,
	embedding <#> (SELECT embedding FROM index_test_table LIMIT 1) AS distance
FROM index_test_table
ORDER BY embedding <#> (SELECT embedding FROM index_test_table LIMIT 1)
LIMIT 10;

/*-------------------------------------------------------------------
 * ---- INDEX METADATA ----
 *------------------------------------------------------------------*/
\echo ''
\echo 'Index Metadata Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 11: Index size and statistics'
SELECT 
	indexname,
	indexdef,
	pg_size_pretty(pg_relation_size(indexname::regclass)) AS index_size
FROM pg_indexes
WHERE tablename = 'index_test_table'
ORDER BY indexname;

\echo ''
\echo '=========================================================================='
\echo '✓ Index Module: Basic tests complete'
\echo '=========================================================================='

DROP TABLE IF EXISTS index_test_table CASCADE;

\echo 'Test completed successfully'
