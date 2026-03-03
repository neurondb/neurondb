/*
 * Crash Prevention Test: Concurrency & Race Condition Stress Tests
 * Tests concurrent operations to find race conditions.
 * 
 * Expected: Functions should handle concurrent access gracefully, not crash.
 * 
 * Note: These tests require multiple connections and are best run with
 * a concurrent test runner. This file provides basic concurrency scenarios.
 */

\set ON_ERROR_STOP off

BEGIN;

\echo '=========================================================================='
\echo 'Concurrency Tests: Basic Concurrent Operations'
\echo '=========================================================================='

CREATE TABLE IF NOT EXISTS test_concurrent_table (
    features vector(128),
    label float4,
    id integer
);

INSERT INTO test_concurrent_table (features, label, id)
SELECT array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 128)))::vector(128),
       random()::float4,
       generate_series(1, 1000)
ON CONFLICT DO NOTHING;

-- Test 1: Rapid sequential operations (simulating concurrent access)
DO $$
DECLARE
    i int;
    model_id int;
BEGIN
    -- Train a model
    BEGIN
        SELECT neurondb.train('linear_regression', 'test_concurrent_table', 'features', 'label', '{}'::jsonb) INTO model_id;
        
        -- Rapid predictions (simulating concurrent predictions)
        FOR i IN 1..100 LOOP
            BEGIN
                PERFORM neurondb.predict('model_' || model_id, 
                    array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 128)))::vector(128));
            EXCEPTION WHEN OTHERS THEN
                NULL;  -- Ignore errors, test doesn't crash
            END;
        END LOOP;
        
        RAISE NOTICE '✓ Rapid sequential operations completed without crash';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'Rapid operations handled with error (no crash): %', SQLERRM;
    END;
END$$;

ROLLBACK;

BEGIN;

\echo ''
\echo 'Concurrency Tests: Model Deletion During Use'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Note: True concurrent deletion requires multiple connections
-- This test simulates the scenario

DO $$
DECLARE
    model_id int;
BEGIN
    -- Train a model
    BEGIN
        SELECT neurondb.train('linear_regression', 'test_concurrent_table', 'features', 'label', '{}'::jsonb) INTO model_id;
        
        -- Try to use model immediately (in real concurrency, another thread might delete it)
        BEGIN
            PERFORM neurondb.predict('model_' || model_id,
                array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 128)))::vector(128));
            RAISE NOTICE 'Model prediction succeeded';
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'Model prediction handled with error (no crash): %', SQLERRM;
        END;
        
        -- Try to delete model
        BEGIN
            PERFORM neurondb.drop_model('model_' || model_id);
            RAISE NOTICE 'Model deletion succeeded';
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'Model deletion handled with error (no crash): %', SQLERRM;
        END;
        
        RAISE NOTICE '✓ Model lifecycle operations completed without crash';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'Model lifecycle handled with error (no crash): %', SQLERRM;
    END;
END$$;

ROLLBACK;

COMMIT;

\echo ''
\echo '=========================================================================='
\echo '✓ Concurrency Tests Complete'
\echo '=========================================================================='
\echo ''
\echo 'Note: True concurrency testing requires multiple database connections.'
\echo 'For comprehensive concurrency testing, use a concurrent test runner.'
\echo ''




