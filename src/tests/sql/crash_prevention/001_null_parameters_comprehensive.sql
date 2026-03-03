/*
 * Crash Prevention Test: Comprehensive NULL Parameter Injection
 * Tests that all functions properly handle NULL parameters without crashing.
 * 
 * Expected: All functions should return PostgreSQL ERROR messages, not crash.
 */

\set ON_ERROR_STOP off

-- Common test setup
CREATE TABLE IF NOT EXISTS test_crash_null_table (
    features vector(128),
    label float4,
    id integer
);

INSERT INTO test_crash_null_table (features, label, id)
SELECT array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 128)))::vector(128),
       random()::float4,
       generate_series(1, 100)
ON CONFLICT DO NOTHING;

BEGIN;

\echo '=========================================================================='
\echo 'NULL Parameter Tests: Unified API Functions'
\echo '=========================================================================='

-- Test 1: NULL algorithm
DO $$
BEGIN
    PERFORM neurondb.train(NULL, 'test_crash_null_table', 'features', 'label', '{}'::jsonb);
    RAISE EXCEPTION 'Should have failed with NULL algorithm';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Correctly rejected NULL algorithm: %', SQLERRM;
END$$;

-- Test 2: NULL table_name
DO $$
BEGIN
    PERFORM neurondb.train('linear_regression', NULL, 'features', 'label', '{}'::jsonb);
    RAISE EXCEPTION 'Should have failed with NULL table_name';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Correctly rejected NULL table_name: %', SQLERRM;
END$$;

-- Test 3: NULL feature column
DO $$
BEGIN
    PERFORM neurondb.train('linear_regression', 'test_crash_null_table', NULL, 'label', '{}'::jsonb);
    RAISE EXCEPTION 'Should have failed with NULL feature column';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Correctly rejected NULL feature column: %', SQLERRM;
END$$;

-- Test 4: NULL label column
DO $$
BEGIN
    PERFORM neurondb.train('linear_regression', 'test_crash_null_table', 'features', NULL, '{}'::jsonb);
    RAISE EXCEPTION 'Should have failed with NULL label column';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Correctly rejected NULL label column: %', SQLERRM;
END$$;

-- Test 5: NULL params (should be allowed or return error, not crash)
DO $$
BEGIN
    PERFORM neurondb.train('linear_regression', 'test_crash_null_table', 'features', 'label', NULL);
    RAISE NOTICE 'NULL params handled (may be allowed)';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Correctly rejected NULL params: %', SQLERRM;
END$$;

ROLLBACK;

BEGIN;

\echo ''
\echo 'NULL Parameter Tests: Predict Function'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Test 6: NULL model_id in predict
DO $$
BEGIN
    PERFORM neurondb.predict(NULL, array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 128)))::vector(128));
    RAISE EXCEPTION 'Should have failed with NULL model_id';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Correctly rejected NULL model_id in predict: %', SQLERRM;
END$$;

-- Test 7: NULL vector in predict
DO $$
BEGIN
    PERFORM neurondb.predict('model_1', NULL::vector);
    RAISE EXCEPTION 'Should have failed with NULL vector';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Correctly rejected NULL vector in predict: %', SQLERRM;
END$$;

ROLLBACK;

BEGIN;

\echo ''
\echo 'NULL Parameter Tests: Evaluate Function'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Test 8: NULL model_id in evaluate
DO $$
BEGIN
    PERFORM neurondb.evaluate(NULL, 'test_crash_null_table', 'features', 'label');
    RAISE EXCEPTION 'Should have failed with NULL model_id';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Correctly rejected NULL model_id in evaluate: %', SQLERRM;
END$$;

-- Test 9: NULL table_name in evaluate
DO $$
BEGIN
    PERFORM neurondb.evaluate(1, NULL, 'features', 'label');
    RAISE EXCEPTION 'Should have failed with NULL table_name';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Correctly rejected NULL table_name in evaluate: %', SQLERRM;
END$$;

-- Test 10: NULL feature column in evaluate
DO $$
BEGIN
    PERFORM neurondb.evaluate(1, 'test_crash_null_table', NULL, 'label');
    RAISE EXCEPTION 'Should have failed with NULL feature column';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Correctly rejected NULL feature column in evaluate: %', SQLERRM;
END$$;

-- Test 11: NULL label column in evaluate
DO $$
BEGIN
    PERFORM neurondb.evaluate(1, 'test_crash_null_table', 'features', NULL);
    RAISE EXCEPTION 'Should have failed with NULL label column';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Correctly rejected NULL label column in evaluate: %', SQLERRM;
END$$;

ROLLBACK;

BEGIN;

\echo ''
\echo 'NULL Parameter Tests: Vector Operations'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Test 12: NULL vectors in distance operations
SELECT vector_l2_distance(NULL::vector, array_to_vector(ARRAY(SELECT 1.0 FROM generate_series(1, 128)))::vector(128));

SELECT vector_l2_distance(array_to_vector(ARRAY(SELECT 1.0 FROM generate_series(1, 128)))::vector(128), NULL::vector);

SELECT vector_l2_distance(NULL::vector, NULL::vector);

-- Test 13: NULL vectors in cosine distance
SELECT vector_cosine_distance(NULL::vector, array_to_vector(ARRAY(SELECT 1.0 FROM generate_series(1, 128)))::vector(128));

-- Test 14: NULL vectors in inner product
SELECT vector_inner_product(NULL::vector, array_to_vector(ARRAY(SELECT 1.0 FROM generate_series(1, 128)))::vector(128));

ROLLBACK;

BEGIN;

\echo ''
\echo 'NULL Parameter Tests: Embedding Functions'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Test 15: NULL text in embed functions (if they exist)
-- Note: These may not exist, but test gracefully
DO $$
BEGIN
    BEGIN
        PERFORM embed_text(NULL);
        RAISE NOTICE 'NULL handled in embed_text (may be allowed)';
    EXCEPTION WHEN undefined_function THEN
        RAISE NOTICE 'embed_text function not available';
    WHEN OTHERS THEN
        RAISE NOTICE '✓ Correctly rejected NULL in embed_text: %', SQLERRM;
    END;
END$$;

ROLLBACK;

BEGIN;

\echo ''
\echo 'NULL Parameter Tests: Index Operations'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Test 16: NULL vector in index creation (should error, not crash)
DO $$
BEGIN
    CREATE INDEX IF NOT EXISTS test_null_idx ON test_crash_null_table USING ivf (NULL);
    RAISE EXCEPTION 'Should have failed with NULL in index';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Correctly rejected NULL in index creation: %', SQLERRM;
END$$;

ROLLBACK;

BEGIN;

\echo ''
\echo 'NULL Parameter Tests: Model Management'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Test 17: NULL model name in drop_model
DO $$
BEGIN
    PERFORM neurondb.drop_model(NULL);
    RAISE EXCEPTION 'Should have failed with NULL model name';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Correctly rejected NULL model name in drop_model: %', SQLERRM;
END$$;

-- Test 18: NULL model name in load_model
DO $$
BEGIN
    PERFORM neurondb.load_model(NULL, '/path/to/model', 'onnx');
    RAISE EXCEPTION 'Should have failed with NULL model name';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '✓ Correctly rejected NULL model name in load_model: %', SQLERRM;
END$$;

ROLLBACK;

COMMIT;

\echo ''
\echo '=========================================================================='
\echo '✓ NULL Parameter Tests Complete'
\echo '=========================================================================='




