-- 031_storage_negative.sql
-- Negative test cases for storage module: error handling, invalid inputs

\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP off

\echo '=========================================================================='
\echo 'Storage Module: Negative Test Cases (Error Handling)'
\echo '=========================================================================='

/*-------------------------------------------------------------------
 * ---- WAL OPERATION ERRORS ----
 * Test error handling for WAL operations
 *------------------------------------------------------------------*/
\echo ''
\echo 'WAL Operation Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 1: WAL Compress with NULL vector'
SELECT vector_wal_compress(NULL::text, vector_out(vector '[1,2,3]'::vector)::text);

\echo 'Error Test 2: WAL Compress with NULL base vector'
SELECT vector_wal_compress(vector_out(vector '[1,2,3]'::vector)::text, NULL::text);

\echo 'Error Test 3: WAL Decompress with NULL compressed data'
SELECT vector_wal_decompress(NULL::text, vector_out(vector '[1,2,3]'::vector)::text);

\echo 'Error Test 4: WAL Decompress with NULL base vector'
SELECT vector_wal_decompress('compressed_data'::text, NULL::text);

\echo 'Error Test 5: WAL Estimate Size with NULL vector'
SELECT vector_wal_estimate_size(NULL::text, vector_out(vector '[1,2,3]'::vector)::text);

\echo 'Error Test 6: WAL Estimate Size with NULL base vector'
SELECT vector_wal_estimate_size(vector_out(vector '[1,2,3]'::vector)::text, NULL::text);

\echo 'Error Test 7: WAL Decompress with Invalid Compressed Data'
SELECT vector_wal_decompress('invalid_compressed_data_xyz'::text, vector_out(vector '[1,2,3]'::vector)::text);

/*-------------------------------------------------------------------
 * ---- BUFFER OPERATION ERRORS ----
 * Test error handling for buffer operations
 *------------------------------------------------------------------*/
\echo ''
\echo 'Buffer Operation Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 8: Buffer operations on non-existent table'
SELECT COUNT(*) FROM nonexistent_storage_table;

\echo 'Error Test 9: Buffer operations with corrupted data'
DROP TABLE IF EXISTS storage_corrupt_test;
CREATE TABLE storage_corrupt_test (
	id SERIAL PRIMARY KEY,
	embedding vector(28)
);

-- Try to insert potentially problematic data
INSERT INTO storage_corrupt_test (embedding) VALUES (NULL);

-- Try to query corrupted data
SELECT COUNT(*) FROM storage_corrupt_test WHERE embedding IS NULL;

DROP TABLE IF EXISTS storage_corrupt_test;

/*-------------------------------------------------------------------
 * ---- ANN BUFFER OPERATION ERRORS ----
 * Test error handling for ANN buffer operations
 *------------------------------------------------------------------*/
\echo ''
\echo 'ANN Buffer Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 10: ANN buffer query with NULL vector'
DROP TABLE IF EXISTS storage_ann_test;
CREATE TABLE storage_ann_test (
	id SERIAL PRIMARY KEY,
	embedding vector(28)
);

INSERT INTO storage_ann_test (embedding)
SELECT features FROM test_train_view LIMIT 10;

CREATE INDEX idx_ann_test ON storage_ann_test 
USING hnsw (embedding vector_l2_ops);

SELECT 
	id,
	embedding <-> NULL::vector AS distance
FROM storage_ann_test
ORDER BY embedding <-> NULL::vector
LIMIT 10;

\echo 'Error Test 11: ANN buffer query with dimension mismatch'
SELECT 
	id,
	embedding <-> vector '[1,2,3]'::vector AS distance
FROM storage_ann_test
ORDER BY embedding <-> vector '[1,2,3]'::vector
LIMIT 10;

\echo 'Error Test 12: ANN buffer query on dropped index'
DROP INDEX idx_ann_test;

SELECT 
	id,
	embedding <-> (SELECT embedding FROM storage_ann_test LIMIT 1) AS distance
FROM storage_ann_test
ORDER BY embedding <-> (SELECT embedding FROM storage_ann_test LIMIT 1)
LIMIT 10;

DROP TABLE IF EXISTS storage_ann_test;

\echo ''
\echo '=========================================================================='
\echo '✓ Storage Module: Negative test cases complete'
\echo '=========================================================================='

\echo 'Test completed successfully'




