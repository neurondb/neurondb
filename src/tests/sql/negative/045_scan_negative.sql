-- 032_scan_negative.sql
-- Negative test cases for scan module: error handling, invalid inputs

\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP off

\echo '=========================================================================='
\echo 'Scan Module: Negative Test Cases (Error Handling)'
\echo '=========================================================================='

/*-------------------------------------------------------------------
 * ---- SCAN OPERATION ERRORS ----
 * Test error handling for scan operations
 *------------------------------------------------------------------*/
\echo ''
\echo 'Scan Operation Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 1: HNSW scan with NULL query vector'
DROP TABLE IF EXISTS scan_error_test;
CREATE TABLE scan_error_test (
	id SERIAL PRIMARY KEY,
	embedding vector(28)
);

INSERT INTO scan_error_test (embedding)
SELECT features FROM test_train_view LIMIT 10;

CREATE INDEX idx_scan_error ON scan_error_test 
USING hnsw (embedding vector_l2_ops);

SELECT 
	id,
	embedding <-> NULL::vector AS distance
FROM scan_error_test
ORDER BY embedding <-> NULL::vector
LIMIT 10;

\echo 'Error Test 2: HNSW scan with dimension mismatch'
SELECT 
	id,
	embedding <-> vector '[1,2,3]'::vector AS distance
FROM scan_error_test
ORDER BY embedding <-> vector '[1,2,3]'::vector
LIMIT 10;

\echo 'Error Test 3: HNSW scan on dropped index'
DROP INDEX idx_scan_error;

SELECT 
	id,
	embedding <-> (SELECT embedding FROM scan_error_test LIMIT 1) AS distance
FROM scan_error_test
ORDER BY embedding <-> (SELECT embedding FROM scan_error_test LIMIT 1)
LIMIT 10;

/*-------------------------------------------------------------------
 * ---- RLS INTEGRATION ERRORS ----
 * Test error handling for RLS operations
 *------------------------------------------------------------------*/
\echo ''
\echo 'RLS Integration Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 4: RLS scan with invalid policy'
DROP TABLE IF EXISTS scan_rls_error;
CREATE TABLE scan_rls_error (
	id SERIAL PRIMARY KEY,
	embedding vector(28),
	tenant_id integer
);

ALTER TABLE scan_rls_error ENABLE ROW LEVEL SECURITY;

-- Try to create invalid policy
DO $$
BEGIN
	BEGIN
		CREATE POLICY invalid_policy ON scan_rls_error
			FOR SELECT
			USING (tenant_id = invalid_column_name);
	EXCEPTION WHEN OTHERS THEN
		NULL; -- Expected error
	END;
END$$;

DROP TABLE IF EXISTS scan_rls_error;

/*-------------------------------------------------------------------
 * ---- QUOTA ENFORCEMENT ERRORS ----
 * Test error handling for quota operations
 *------------------------------------------------------------------*/
\echo ''
\echo 'Quota Enforcement Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 5: Quota check with NULL tenant'
SELECT neurondb_check_quota(NULL, 'idx_scan_error'::regclass, 100);

\echo 'Error Test 6: Quota check with invalid index OID'
SELECT neurondb_check_quota('tenant1', 99999::regclass, 100);

\echo 'Error Test 7: Quota check with negative vector count'
SELECT neurondb_check_quota('tenant1', 'idx_scan_error'::regclass, -100);

\echo 'Error Test 8: Quota usage with NULL tenant'
SELECT * FROM neurondb_get_quota_usage(NULL, 'idx_scan_error'::regclass);

\echo 'Error Test 9: Quota usage with invalid index OID'
SELECT * FROM neurondb_get_quota_usage('tenant1', 99999::regclass);

DROP TABLE IF EXISTS scan_error_test CASCADE;

\echo ''
\echo '=========================================================================='
\echo '✓ Scan Module: Negative test cases complete'
\echo '=========================================================================='

\echo 'Test completed successfully'




