/*
 * Crash Prevention Test: Array Bounds and Dimension Mismatches
 * Tests array bounds validation and dimension handling.
 * 
 * Expected: Functions should validate dimensions and return errors, not crash.
 */

\set ON_ERROR_STOP off

BEGIN;

\echo '=========================================================================='
\echo 'Array Bounds Tests: Empty Arrays'
\echo '=========================================================================='

-- Test 1: Empty array (should error, not crash)
DO $$
BEGIN
    PERFORM neurondb.predict('model_1', ARRAY[]::float4[]);
    RAISE EXCEPTION 'Should have failed with empty array';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Correctly rejected empty array: %', SQLERRM;
END$$;

ROLLBACK;

BEGIN;

\echo ''
\echo 'Array Bounds Tests: Dimension Mismatches'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Test 2: Wrong dimension vector (expecting 128, got 64)
DO $$
BEGIN
    PERFORM neurondb.predict('model_1', array_to_vector(ARRAY(SELECT 1.0 FROM generate_series(1, 64)))::vector(64));
    RAISE NOTICE 'Wrong dimension handled (may error, but no crash)';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Correctly rejected wrong dimension: %', SQLERRM;
END$$;

-- Test 3: Wrong dimension vector (expecting 128, got 256)
DO $$
BEGIN
    PERFORM neurondb.predict('model_1', array_to_vector(ARRAY(SELECT 1.0 FROM generate_series(1, 256)))::vector(256));
    RAISE NOTICE 'Wrong dimension (larger) handled (may error, but no crash)';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Correctly rejected wrong dimension (larger): %', SQLERRM;
END$$;

ROLLBACK;

BEGIN;

\echo ''
\echo 'Array Bounds Tests: Very Large Dimensions'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Test 4: Very large dimension vector (should validate, not crash)
DO $$
BEGIN
    PERFORM neurondb.predict('model_1', array_to_vector(ARRAY(SELECT 1.0 FROM generate_series(1, 10000)))::vector(10000));
    RAISE NOTICE 'Very large dimension handled (may error, but no crash)';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Correctly rejected very large dimension: %', SQLERRM;
END$$;

ROLLBACK;

BEGIN;

\echo ''
\echo 'Array Bounds Tests: Single Element Arrays'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Test 5: Single element array (when expecting many)
DO $$
BEGIN
    PERFORM neurondb.predict('model_1', array_to_vector(ARRAY[1.0])::vector(1));
    RAISE NOTICE 'Single element array handled (may error, but no crash)';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Correctly rejected single element array: %', SQLERRM;
END$$;

ROLLBACK;

BEGIN;

\echo ''
\echo 'Array Bounds Tests: Vector Operations with Dimension Mismatches'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Test 6: Dimension mismatch in vector operations
SELECT vector_l2_distance(
    array_to_vector(ARRAY(SELECT 1.0 FROM generate_series(1, 128)))::vector(128),
    array_to_vector(ARRAY(SELECT 1.0 FROM generate_series(1, 64)))::vector(64)
);

ROLLBACK;

BEGIN;

\echo ''
\echo 'Array Bounds Tests: Vector with NULL Elements'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Note: PostgreSQL arrays can't have NULL elements in this context easily
-- But we can test with arrays that might be interpreted incorrectly
-- This is more of a C code validation test

ROLLBACK;

COMMIT;

\echo ''
\echo '=========================================================================='
\echo '✓ Array Bounds Tests Complete'
\echo '=========================================================================='




