-- ============================================================================
-- Regression tests for vector correctness fixes
-- ============================================================================
-- Tests for:
-- 1. vector_eq NaN handling and brace fix
-- 2. binaryvec_in bounds checking fix
-- 3. Edge cases and error conditions

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Test: vector_eq correctness (NaN handling, brace fix)'
\echo '=========================================================================='

-- Test 1: Equal vectors
SELECT '[1,2,3]'::vector = '[1,2,3]'::vector AS should_be_true;
SELECT '[1,2,3]'::vector = '[1,2,4]'::vector AS should_be_false;

-- Test 2: Different dimensions
SELECT '[1,2]'::vector = '[1,2,3]'::vector AS should_be_false;

-- Test 3: Zero vectors
SELECT '[0,0,0]'::vector = '[0,0,0]'::vector AS should_be_true;

-- Test 4: Very small differences (within tolerance)
SELECT '[1.0,2.0,3.0]'::vector = '[1.0000001,2.0000001,3.0000001]'::vector AS should_be_true;

-- Test 5: Large differences (outside tolerance)
SELECT '[1.0,2.0,3.0]'::vector = '[1.1,2.1,3.1]'::vector AS should_be_false;

-- Test 6: NaN handling (should reject NaN during input)
DO $$
BEGIN
    BEGIN
        PERFORM '[NaN,1,2]'::vector;
        RAISE WARNING 'NaN was accepted (should be rejected)';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'NaN correctly rejected: %', SQLERRM;
    END;
END $$;

-- Test 7: NULL handling
SELECT NULL::vector = NULL::vector AS null_eq_null;
SELECT '[1,2,3]'::vector = NULL::vector AS vec_eq_null;
SELECT NULL::vector = '[1,2,3]'::vector AS null_eq_vec;

\echo ''
\echo '=========================================================================='
\echo 'Test: binaryvec_in bounds checking fix'
\echo '=========================================================================='

-- Test 1: Valid binary vector (array format)
SELECT '[1,0,1,0,1]'::binaryvec;

-- Test 2: Valid binary vector (string format)
SELECT '10101'::binaryvec;

-- Test 3: Edge case - exactly 8 bits (1 byte)
SELECT '[1,1,1,1,1,1,1,1]'::binaryvec;

-- Test 4: Edge case - 9 bits (2 bytes)
SELECT '[1,1,1,1,1,1,1,1,1]'::binaryvec;

-- Test 5: Large binary vector (should work within limits)
SELECT repeat('1', 64)::binaryvec;

-- Test 6: Bounds check - should not crash on valid input
DO $$
DECLARE
    bv binaryvec;
BEGIN
    -- This should work without bounds error
    bv := '[1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]'::binaryvec;
    RAISE NOTICE 'Binary vector created successfully with 16 bits';
END $$;

-- Test 7: Empty binary vector (should error)
DO $$
BEGIN
    BEGIN
        PERFORM '[]'::binaryvec;
        RAISE WARNING 'Empty binary vector was accepted (should be rejected)';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'Empty binary vector correctly rejected: %', SQLERRM;
    END;
END $$;

-- Test 8: Invalid characters (should error)
DO $$
BEGIN
    BEGIN
        PERFORM '[1,0,2,1]'::binaryvec;
        RAISE WARNING 'Invalid character was accepted (should be rejected)';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'Invalid character correctly rejected: %', SQLERRM;
    END;
END $$;

\echo ''
\echo '=========================================================================='
\echo 'Test: vector_ne (not equal) operator'
\echo '=========================================================================='

SELECT '[1,2,3]'::vector != '[1,2,3]'::vector AS should_be_false;
SELECT '[1,2,3]'::vector != '[1,2,4]'::vector AS should_be_true;
SELECT '[1,2,3]'::vector <> '[1,2,3]'::vector AS should_be_false;

\echo ''
\echo '=========================================================================='
\echo 'Test: vector hash function'
\echo '=========================================================================='

-- Hash should be consistent
SELECT vector_hash('[1,2,3]'::vector) = vector_hash('[1,2,3]'::vector) AS hash_consistent;
SELECT vector_hash('[1,2,3]'::vector) != vector_hash('[1,2,4]'::vector) AS hash_different;

\echo ''
\echo '=========================================================================='
\echo 'All correctness fix tests completed'
\echo '=========================================================================='



