-- 030_index_negative.sql
-- Negative test cases for index module: error handling, invalid inputs

\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP off

\echo '=========================================================================='
\echo 'Index Module: Negative Test Cases (Error Handling)'
\echo '=========================================================================='

/*-------------------------------------------------------------------
 * ---- INVALID INDEX PARAMETERS ----
 * Test error handling for invalid index parameters
 *------------------------------------------------------------------*/
\echo ''
\echo 'Invalid Index Parameter Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Create test table
DROP TABLE IF EXISTS index_negative_test;
CREATE TABLE index_negative_test (
	id SERIAL PRIMARY KEY,
	embedding vector(28),
	label integer
);

INSERT INTO index_negative_test (embedding, label)
SELECT features, label
FROM test_train_view
LIMIT 100;

\echo 'Error Test 1: HNSW with invalid m parameter (too small)'
CREATE INDEX idx_hnsw_invalid_m ON index_negative_test 
USING hnsw (embedding vector_l2_ops) 
WITH (m = 0, ef_construction = 200);

\echo 'Error Test 2: HNSW with invalid m parameter (too large)'
CREATE INDEX idx_hnsw_invalid_m_large ON index_negative_test 
USING hnsw (embedding vector_l2_ops) 
WITH (m = 1000, ef_construction = 200);

\echo 'Error Test 3: HNSW with invalid ef_construction (negative)'
CREATE INDEX idx_hnsw_invalid_ef ON index_negative_test 
USING hnsw (embedding vector_l2_ops) 
WITH (m = 16, ef_construction = -1);

\echo 'Error Test 4: IVF with invalid lists parameter (too small)'
CREATE INDEX idx_ivf_invalid_lists ON index_negative_test 
USING ivf (embedding vector_l2_ops) 
WITH (lists = 0);

\echo 'Error Test 5: IVF with invalid lists parameter (too large)'
CREATE INDEX idx_ivf_invalid_lists_large ON index_negative_test 
USING ivf (embedding vector_l2_ops) 
WITH (lists = 100000);

/*-------------------------------------------------------------------
 * ---- INVALID TABLE/COLUMN ERRORS ----
 * Test error handling for invalid table/column references
 *------------------------------------------------------------------*/
\echo ''
\echo 'Invalid Table/Column Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 6: Index on non-existent table'
CREATE INDEX idx_nonexistent ON nonexistent_table 
USING hnsw (embedding vector_l2_ops);

\echo 'Error Test 7: Index on non-existent column'
CREATE INDEX idx_nonexistent_col ON index_negative_test 
USING hnsw (nonexistent_column vector_l2_ops);

\echo 'Error Test 8: Index on wrong column type'
CREATE INDEX idx_wrong_type ON index_negative_test 
USING hnsw (label vector_l2_ops);

/*-------------------------------------------------------------------
 * ---- INDEX OPERATION ERRORS ----
 * Test error handling for index operations
 *------------------------------------------------------------------*/
\echo ''
\echo 'Index Operation Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 9: Query with NULL vector'
SELECT 
	id,
	embedding <-> NULL::vector AS distance
FROM index_negative_test
ORDER BY embedding <-> NULL::vector
LIMIT 10;

\echo 'Error Test 10: Query with dimension mismatch'
SELECT 
	id,
	embedding <-> vector '[1,2,3]'::vector AS distance
FROM index_negative_test
ORDER BY embedding <-> vector '[1,2,3]'::vector
LIMIT 10;

\echo 'Error Test 11: REINDEX on non-existent index'
REINDEX INDEX nonexistent_index;

\echo 'Error Test 12: Query on dropped index'
CREATE INDEX idx_temp_test ON index_negative_test 
USING hnsw (embedding vector_l2_ops);

DROP INDEX idx_temp_test;

SELECT 
	id,
	embedding <-> (SELECT embedding FROM index_negative_test LIMIT 1) AS distance
FROM index_negative_test
ORDER BY embedding <-> (SELECT embedding FROM index_negative_test LIMIT 1)
LIMIT 10;

/*-------------------------------------------------------------------
 * ---- EMPTY TABLE ERRORS ----
 * Test error handling for empty tables
 *------------------------------------------------------------------*/
\echo ''
\echo 'Empty Table Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 13: Index on empty table'
DROP TABLE IF EXISTS index_empty_test;
CREATE TABLE index_empty_test (
	id SERIAL PRIMARY KEY,
	embedding vector(28)
);

CREATE INDEX idx_empty_hnsw ON index_empty_test 
USING hnsw (embedding vector_l2_ops);

\echo 'Error Test 14: Query on empty indexed table'
SELECT 
	id,
	embedding <-> vector '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]'::vector AS distance
FROM index_empty_test
ORDER BY embedding <-> vector '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]'::vector
LIMIT 10;

DROP TABLE IF EXISTS index_empty_test;

/*-------------------------------------------------------------------
 * ---- INDEX CONSISTENCY ERRORS ----
 * Test error handling for index consistency issues
 *------------------------------------------------------------------*/
\echo ''
\echo 'Index Consistency Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 15: Index on table with NULL vectors'
DROP TABLE IF EXISTS index_null_test;
CREATE TABLE index_null_test (
	id SERIAL PRIMARY KEY,
	embedding vector(28)
);

INSERT INTO index_null_test (embedding) VALUES
	(NULL),
	(vector '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]'::vector);

CREATE INDEX idx_null_hnsw ON index_null_test 
USING hnsw (embedding vector_l2_ops);

\echo 'Error Test 16: Query with NULL in indexed column'
SELECT 
	id,
	embedding <-> (SELECT embedding FROM index_null_test WHERE embedding IS NOT NULL LIMIT 1) AS distance
FROM index_null_test
WHERE embedding IS NOT NULL
ORDER BY embedding <-> (SELECT embedding FROM index_null_test WHERE embedding IS NOT NULL LIMIT 1)
LIMIT 10;

DROP TABLE IF EXISTS index_null_test;

\echo ''
\echo '=========================================================================='
\echo '✓ Index Module: Negative test cases complete'
\echo '=========================================================================='

DROP TABLE IF EXISTS index_negative_test CASCADE;

\echo 'Test completed successfully'
