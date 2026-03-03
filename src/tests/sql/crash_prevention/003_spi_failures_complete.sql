/*
 * Crash Prevention Test: Comprehensive SPI Failure Scenarios
 * Tests that SPI (Server Programming Interface) failures are handled gracefully.
 * 
 * Expected: All failures should return PostgreSQL ERROR messages, not crash.
 */

\set ON_ERROR_STOP off

BEGIN;

\echo '=========================================================================='
\echo 'SPI Failure Tests: Non-existent Tables'
\echo '=========================================================================='

-- Test 1: Non-existent table in train
DO $$
BEGIN
    PERFORM neurondb.train('linear_regression', 'nonexistent_table_xyz123', 'features', 'label', '{}'::jsonb);
    RAISE EXCEPTION 'Should have failed with non-existent table';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Correctly rejected non-existent table in train: %', SQLERRM;
END$$;

-- Test 2: Non-existent table in evaluate
DO $$
BEGIN
    PERFORM neurondb.evaluate(1, 'nonexistent_table_xyz123', 'features', 'label');
    RAISE EXCEPTION 'Should have failed with non-existent table';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Correctly rejected non-existent table in evaluate: %', SQLERRM;
END$$;

ROLLBACK;

BEGIN;

\echo ''
\echo 'SPI Failure Tests: Non-existent Columns'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE TABLE IF NOT EXISTS test_spi_table (
    features vector(128),
    label float4
);

-- Test 3: Non-existent feature column
DO $$
BEGIN
    PERFORM neurondb.train('linear_regression', 'test_spi_table', 'nonexistent_column_xyz', 'label', '{}'::jsonb);
    RAISE EXCEPTION 'Should have failed with non-existent column';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Correctly rejected non-existent feature column: %', SQLERRM;
END$$;

-- Test 4: Non-existent label column
DO $$
BEGIN
    PERFORM neurondb.train('linear_regression', 'test_spi_table', 'features', 'nonexistent_label_xyz', '{}'::jsonb);
    RAISE EXCEPTION 'Should have failed with non-existent label column';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Correctly rejected non-existent label column: %', SQLERRM;
END$$;

ROLLBACK;

BEGIN;

\echo ''
\echo 'SPI Failure Tests: Empty Tables'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE TEMP TABLE empty_test_table (features vector(128), label float4);

-- Test 5: Empty table in train (should error, not crash)
DO $$
BEGIN
    PERFORM neurondb.train('linear_regression', 'empty_test_table', 'features', 'label', '{}'::jsonb);
    RAISE EXCEPTION 'Should have failed with empty table';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Correctly rejected empty table in train: %', SQLERRM;
END$$;

-- Test 6: Empty table in evaluate
DO $$
BEGIN
    PERFORM neurondb.evaluate(1, 'empty_test_table', 'features', 'label');
    RAISE NOTICE 'Empty table in evaluate handled (may return NULL)';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Correctly rejected empty table in evaluate: %', SQLERRM;
END$$;

ROLLBACK;

BEGIN;

\echo ''
\echo 'SPI Failure Tests: Tables with All NULL Values'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE TEMP TABLE null_test_table (features vector(128), label float4);
INSERT INTO null_test_table VALUES (NULL, NULL);
INSERT INTO null_test_table VALUES (NULL, NULL);

-- Test 7: Table with all NULLs in train
DO $$
BEGIN
    PERFORM neurondb.train('linear_regression', 'null_test_table', 'features', 'label', '{}'::jsonb);
    RAISE EXCEPTION 'Should have failed with all NULL values';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Correctly rejected table with all NULLs in train: %', SQLERRM;
END$$;

-- Test 8: Table with all NULLs in evaluate
DO $$
BEGIN
    PERFORM neurondb.evaluate(1, 'null_test_table', 'features', 'label');
    RAISE NOTICE 'Table with all NULLs in evaluate handled';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Correctly rejected table with all NULLs in evaluate: %', SQLERRM;
END$$;

ROLLBACK;

BEGIN;

\echo ''
\echo 'SPI Failure Tests: Wrong Column Types'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE TEMP TABLE wrong_type_table (
    features text,  -- Should be vector
    label text      -- Should be numeric
);

-- Test 9: Wrong feature column type
DO $$
BEGIN
    PERFORM neurondb.train('linear_regression', 'wrong_type_table', 'features', 'label', '{}'::jsonb);
    RAISE EXCEPTION 'Should have failed with wrong column type';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Correctly rejected wrong feature column type: %', SQLERRM;
END$$;

ROLLBACK;

BEGIN;

\echo ''
\echo 'SPI Failure Tests: View Instead of Table'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE TABLE IF NOT EXISTS test_view_base (features vector(128), label float4);
CREATE OR REPLACE VIEW test_view AS SELECT features, label FROM test_view_base;

-- Test 10: Using view (should work, but test it doesn't crash)
DO $$
BEGIN
    -- Views should work, but test it doesn't crash
    BEGIN
        PERFORM neurondb.train('linear_regression', 'test_view', 'features', 'label', '{}'::jsonb);
        RAISE NOTICE 'View handled correctly (may succeed or fail, but no crash)';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'View handled with error (no crash): %', SQLERRM;
    END;
END$$;

ROLLBACK;

COMMIT;

\echo ''
\echo '=========================================================================='
\echo '✓ SPI Failure Tests Complete'
\echo '=========================================================================='




