/*
 * Crash Prevention Test: Extreme Memory Stress Tests
 * Tests memory context handling under various stress conditions.
 * 
 * Expected: Functions should handle memory pressure gracefully, not crash.
 */

\set ON_ERROR_STOP off

BEGIN;

\echo '=========================================================================='
\echo 'Memory Stress Tests: Large Batch Operations'
\echo '=========================================================================='

-- Test 1: Large table (10K rows) - should not crash
CREATE TEMP TABLE large_test_table AS
SELECT 
    array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 128)))::vector(128) as features,
    random()::float4 as label
FROM generate_series(1, 10000);

DO $$
BEGIN
    BEGIN
        PERFORM neurondb.train('linear_regression', 'large_test_table', 'features', 'label', '{}'::jsonb);
        RAISE NOTICE 'Large table (10K rows) handled successfully';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'Large table handled with error (no crash): %', SQLERRM;
    END;
END$$;

ROLLBACK;

BEGIN;

\echo ''
\echo 'Memory Stress Tests: Very Large Table (100K rows)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Test 2: Very large table - may timeout but should not crash
CREATE TEMP TABLE very_large_test_table AS
SELECT 
    array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 128)))::vector(128) as features,
    random()::float4 as label
FROM generate_series(1, 100000);

DO $$
BEGIN
    BEGIN
        PERFORM neurondb.train('linear_regression', 'very_large_test_table', 'features', 'label', '{}'::jsonb);
        RAISE NOTICE 'Very large table (100K rows) handled successfully';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'Very large table handled with error (no crash): %', SQLERRM;
    END;
END$$;

ROLLBACK;

BEGIN;

\echo ''
\echo 'Memory Stress Tests: High-Dimensional Vectors'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Test 3: High-dimensional vectors (1536 dimensions)
CREATE TEMP TABLE high_dim_table AS
SELECT 
    array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 1536)))::vector(1536) as features,
    random()::float4 as label
FROM generate_series(1, 1000);

DO $$
BEGIN
    BEGIN
        PERFORM neurondb.train('linear_regression', 'high_dim_table', 'features', 'label', '{}'::jsonb);
        RAISE NOTICE 'High-dimensional vectors (1536D) handled successfully';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'High-dimensional vectors handled with error (no crash): %', SQLERRM;
    END;
END$$;

ROLLBACK;

BEGIN;

\echo ''
\echo 'Memory Stress Tests: Rapid Function Calls'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE TEMP TABLE rapid_test_table AS
SELECT 
    array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 128)))::vector(128) as features,
    random()::float4 as label
FROM generate_series(1, 100);

-- Test 4: Rapid evaluate calls (should not leak memory or crash)
DO $$
DECLARE
    i int;
    result float8;
BEGIN
    FOR i IN 1..100 LOOP
        BEGIN
            result := neurondb.evaluate(999999, 'rapid_test_table', 'features', 'label');
            -- Expected to fail, but should not crash
        EXCEPTION WHEN OTHERS THEN
            NULL;  -- Expected error, continue
        END;
    END LOOP;
    RAISE NOTICE '✓ Rapid function calls completed without crash';
END$$;

ROLLBACK;

BEGIN;

\echo ''
\echo 'Memory Stress Tests: Nested Function Calls'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Test 5: Nested function calls (evaluate within evaluate context)
DO $$
DECLARE
    result float8;
BEGIN
    BEGIN
        result := neurondb.evaluate(
            (SELECT model_id FROM neurondb.ml_models LIMIT 1),
            'test_crash_null_table',
            'features',
            'label'
        );
        RAISE NOTICE 'Nested function calls handled';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'Nested function calls handled with error (no crash): %', SQLERRM;
    END;
END$$;

ROLLBACK;

COMMIT;

\echo ''
\echo '=========================================================================='
\echo '✓ Memory Stress Tests Complete'
\echo '=========================================================================='




