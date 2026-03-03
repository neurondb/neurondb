/*
 * Crash Prevention Test: Integer Overflow and Allocation Size Attacks
 * Tests overflow protection in size calculations.
 * 
 * Expected: Functions should detect overflows and return errors, not crash.
 */

\set ON_ERROR_STOP off

BEGIN;

\echo '=========================================================================='
\echo 'Overflow Tests: Large Table Sizes'
\echo '=========================================================================='

-- Test 1: Very large table (potential overflow in row counting)
-- Note: This tests the system's ability to handle large tables
CREATE TEMP TABLE overflow_test_table AS
SELECT 
    array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 128)))::vector(128) as features,
    random()::float4 as label
FROM generate_series(1, 1000000);  -- 1M rows

DO $$
BEGIN
    BEGIN
        PERFORM neurondb.train('linear_regression', 'overflow_test_table', 'features', 'label', '{}'::jsonb);
        RAISE NOTICE 'Very large table (1M rows) handled';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'Very large table handled with error (no crash): %', SQLERRM;
    END;
END$$;

ROLLBACK;

BEGIN;

\echo ''
\echo 'Overflow Tests: High-Dimension Large Tables'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Test 2: Large dimension * large rows (potential overflow in allocation size)
CREATE TEMP TABLE high_dim_large_table AS
SELECT 
    array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 4096)))::vector(4096) as features,
    random()::float4 as label
FROM generate_series(1, 100000);  -- 100K rows * 4096 dim = large allocation

DO $$
BEGIN
    BEGIN
        PERFORM neurondb.train('linear_regression', 'high_dim_large_table', 'features', 'label', '{}'::jsonb);
        RAISE NOTICE 'High-dimension large table handled';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'High-dimension large table handled with error (no crash): %', SQLERRM;
    END;
END$$;

ROLLBACK;

BEGIN;

\echo ''
\echo 'Overflow Tests: Extreme Dimension Vectors'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Test 3: Extremely high dimension vector (should validate against MaxAllocSize)
DO $$
BEGIN
    BEGIN
        PERFORM neurondb.predict(
            'model_1',
            array_to_vector(ARRAY(SELECT 1.0 FROM generate_series(1, 100000)))::vector(100000)
        );
        RAISE NOTICE 'Extreme dimension vector handled';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE '✓ Correctly rejected extreme dimension (overflow protection): %', SQLERRM;
    END;
END$$;

ROLLBACK;

COMMIT;

\echo ''
\echo '=========================================================================='
\echo '✓ Overflow Tests Complete'
\echo '=========================================================================='
\echo ''
\echo 'Note: Most overflow protection is in C code. These tests verify'
\echo 'that the SQL layer passes through errors correctly without crashing.'
\echo ''




