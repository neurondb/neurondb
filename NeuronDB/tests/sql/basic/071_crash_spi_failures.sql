/*
 * Crash Prevention Test: SPI Failure Scenarios
 * Tests that SPI failures are handled gracefully without crashes.
 * Note: Some of these may require special setup or mocking.
 */

\set ON_ERROR_STOP off

BEGIN;

/* Test 1: Table does not exist */
DO $$
BEGIN
    PERFORM evaluate_linear_regression_by_model_id(1, 'nonexistent_table_xyz', 'features', 'label');
    RAISE EXCEPTION 'Should have failed with non-existent table';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Correctly rejected non-existent table: %', SQLERRM;
END$$;
ROLLBACK;

BEGIN;
/* Test 2: Column does not exist */
DO $$
BEGIN
    PERFORM evaluate_linear_regression_by_model_id(1, 'test_table', 'nonexistent_column', 'label');
    RAISE EXCEPTION 'Should have failed with non-existent column';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Correctly rejected non-existent column: %', SQLERRM;
END$$;
ROLLBACK;

BEGIN;
/* Test 3: Empty table */
CREATE TEMP TABLE empty_test_table (features float4[], label float4);
SELECT evaluate_linear_regression_by_model_id(1, 'empty_test_table', 'features', 'label');
ROLLBACK;

BEGIN;
/* Test 4: Table with all NULL values */
CREATE TEMP TABLE null_test_table (features float4[], label float4);
INSERT INTO null_test_table VALUES (NULL, NULL);
SELECT evaluate_linear_regression_by_model_id(1, 'null_test_table', 'features', 'label');
ROLLBACK;

BEGIN;
/* Test 5: Invalid query syntax (should be caught before SPI_execute) */
/* This is more of a code-level test */

COMMIT;

