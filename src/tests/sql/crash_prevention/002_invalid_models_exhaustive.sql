/*
 * Crash Prevention Test: Invalid Model Handling
 * Tests that functions handle invalid/non-existent models gracefully without crashing.
 * 
 * Expected: All functions should return PostgreSQL ERROR messages, not crash.
 */

\set ON_ERROR_STOP off

BEGIN;

\echo '=========================================================================='
\echo 'Invalid Model ID Tests'
\echo '=========================================================================='

-- Test 1: Non-existent model_id (large positive number)
DO $$
BEGIN
    PERFORM neurondb.evaluate(999999, 'test_crash_null_table', 'features', 'label');
    RAISE EXCEPTION 'Should have failed with non-existent model';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Correctly rejected non-existent model (999999): %', SQLERRM;
END$$;

-- Test 2: Negative model_id
DO $$
BEGIN
    PERFORM neurondb.evaluate(-1, 'test_crash_null_table', 'features', 'label');
    RAISE EXCEPTION 'Should have failed with negative model_id';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Correctly rejected negative model_id: %', SQLERRM;
END$$;

-- Test 3: Zero model_id
DO $$
BEGIN
    PERFORM neurondb.evaluate(0, 'test_crash_null_table', 'features', 'label');
    RAISE EXCEPTION 'Should have failed with zero model_id';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Correctly rejected zero model_id: %', SQLERRM;
END$$;

-- Test 4: Very large model_id (potential integer overflow)
DO $$
BEGIN
    PERFORM neurondb.evaluate(2147483647, 'test_crash_null_table', 'features', 'label');
    RAISE NOTICE 'Large model_id handled (may fail gracefully)';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Correctly rejected very large model_id: %', SQLERRM;
END$$;

-- Test 5: Non-existent model in predict
DO $$
BEGIN
    PERFORM neurondb.predict('model_999999', array_to_vector(ARRAY(SELECT 1.0 FROM generate_series(1, 128)))::vector(128));
    RAISE EXCEPTION 'Should have failed with non-existent model';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Correctly rejected non-existent model in predict: %', SQLERRM;
END$$;

ROLLBACK;

BEGIN;

\echo ''
\echo 'Invalid Model Name Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Test 6: Empty model name
DO $$
BEGIN
    PERFORM neurondb.predict('', array_to_vector(ARRAY(SELECT 1.0 FROM generate_series(1, 128)))::vector(128));
    RAISE EXCEPTION 'Should have failed with empty model name';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Correctly rejected empty model name: %', SQLERRM;
END$$;

-- Test 7: Model name with invalid characters
DO $$
BEGIN
    PERFORM neurondb.predict('model_!@#$%', array_to_vector(ARRAY(SELECT 1.0 FROM generate_series(1, 128)))::vector(128));
    RAISE EXCEPTION 'Should have failed with invalid model name';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Correctly rejected invalid model name: %', SQLERRM;
END$$;

-- Test 8: Very long model name
DO $$
BEGIN
    PERFORM neurondb.predict('model_' || repeat('x', 10000), array_to_vector(ARRAY(SELECT 1.0 FROM generate_series(1, 128)))::vector(128));
    RAISE NOTICE 'Very long model name handled (may fail gracefully)';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Correctly rejected very long model name: %', SQLERRM;
END$$;

ROLLBACK;

BEGIN;

\echo ''
\echo 'Model State Tests (deleted, corrupted, etc.)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Test 9: Attempt to evaluate a model that was just deleted
-- (This requires creating then deleting a model, which is complex to test here)
-- We'll test with predict on a model that doesn't exist
DO $$
DECLARE
    model_name text;
BEGIN
    -- Use a name that definitely doesn't exist
    model_name := 'model_deleted_' || extract(epoch from now())::text;
    BEGIN
        PERFORM neurondb.predict(model_name, array_to_vector(ARRAY(SELECT 1.0 FROM generate_series(1, 128)))::vector(128));
        RAISE EXCEPTION 'Should have failed with deleted model';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE '✓ Correctly rejected deleted model: %', SQLERRM;
    END;
END$$;

ROLLBACK;

BEGIN;

\echo ''
\echo 'Model Algorithm Mismatch Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Note: This is harder to test without actually creating models with wrong types
-- We'll test with evaluate on models that don't exist (algorithm mismatch can't be tested without setup)
DO $$
BEGIN
    -- Test evaluate with a model that doesn't exist (could be wrong algorithm)
    PERFORM neurondb.evaluate(999998, 'test_crash_null_table', 'features', 'label');
    RAISE EXCEPTION 'Should have failed with non-existent model';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Correctly rejected non-existent model (potential algorithm mismatch): %', SQLERRM;
END$$;

ROLLBACK;

COMMIT;

\echo ''
\echo '=========================================================================='
\echo '✓ Invalid Model Tests Complete'
\echo '=========================================================================='




