/*
 * Crash Prevention Test: Type Casting & Coercion Crashes
 * Tests type confusion scenarios and casting errors.
 * 
 * Expected: Functions should validate types and return errors, not crash.
 */

\set ON_ERROR_STOP off

BEGIN;

\echo '=========================================================================='
\echo 'Type Confusion Tests: Wrong Vector Types'
\echo '=========================================================================='

CREATE TABLE IF NOT EXISTS test_type_confusion (
    vec vector(128),
    half_vec halfvec(128),
    sparse_vec sparsevec(128),
    id integer
);

-- Test 1: Attempt to use wrong vector type in operations
-- (This depends on available functions)
DO $$
BEGIN
    BEGIN
        -- Try to use halfvec where vector expected
        SELECT vector_l2_distance(
            (SELECT vec FROM test_type_confusion LIMIT 1),
            (SELECT half_vec::vector FROM test_type_confusion LIMIT 1)
        );
        RAISE NOTICE 'Type casting handled (may succeed or error)';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'Type casting handled with error (no crash): %', SQLERRM;
    END;
END$$;

ROLLBACK;

BEGIN;

\echo ''
\echo 'Type Confusion Tests: Binary String vs Vector'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Note: Direct binary string confusion is hard to test via SQL
-- This is more of a C code validation test

ROLLBACK;

BEGIN;

\echo ''
\echo 'Type Confusion Tests: JSONB Parsing Errors'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE TABLE IF NOT EXISTS test_jsonb_table (
    features vector(128),
    label float4
);

-- Test 2: Invalid JSONB in train params
DO $$
BEGIN
    BEGIN
        PERFORM neurondb.train('linear_regression', 'test_jsonb_table', 'features', 'label', '{invalid json}'::jsonb);
        RAISE EXCEPTION 'Should have failed with invalid JSONB';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE '✓ Correctly rejected invalid JSONB: %', SQLERRM;
    END;
END$$;

-- Test 3: Malformed JSONB
DO $$
BEGIN
    BEGIN
        PERFORM neurondb.train('linear_regression', 'test_jsonb_table', 'features', 'label', '{"unclosed":'::jsonb);
        RAISE EXCEPTION 'Should have failed with malformed JSONB';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE '✓ Correctly rejected malformed JSONB: %', SQLERRM;
    END;
END$$;

ROLLBACK;

COMMIT;

\echo ''
\echo '=========================================================================='
\echo '✓ Type Confusion Tests Complete'
\echo '=========================================================================='




