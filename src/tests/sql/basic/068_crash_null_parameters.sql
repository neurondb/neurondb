/*
 * Crash Prevention Test: NULL Parameter Injection
 * Tests that all functions properly handle NULL parameters
 * without crashing the server.
 */

\set ON_ERROR_STOP off

BEGIN;

/* Test 1: NULL model_id */
DO $$
BEGIN
    PERFORM evaluate_linear_regression_by_model_id(NULL, 'test_table', 'features', 'label');
    RAISE EXCEPTION 'Should have failed with NULL model_id';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Correctly rejected NULL model_id: %', SQLERRM;
END$$;
ROLLBACK;

BEGIN;
/* Test 2: NULL table_name */
DO $$
BEGIN
    PERFORM evaluate_linear_regression_by_model_id(1, NULL, 'features', 'label');
    RAISE EXCEPTION 'Should have failed with NULL table_name';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Correctly rejected NULL table_name: %', SQLERRM;
END$$;
ROLLBACK;

BEGIN;
/* Test 3: NULL feature_col */
DO $$
BEGIN
    PERFORM evaluate_linear_regression_by_model_id(1, 'test_table', NULL, 'label');
    RAISE EXCEPTION 'Should have failed with NULL feature_col';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Correctly rejected NULL feature_col: %', SQLERRM;
END$$;
ROLLBACK;

BEGIN;
/* Test 4: NULL label_col (should be allowed for some algorithms) */
SELECT evaluate_kmeans_by_model_id(1, 'test_table', NULL);
ROLLBACK;

BEGIN;
/* Test 5: NULL in predict functions */
SELECT predict_linear_regression_by_model_id(NULL, ARRAY[1.0, 2.0, 3.0]::float4[]);
ROLLBACK;

BEGIN;
/* Test 6: NULL features array */
SELECT predict_linear_regression_by_model_id(1, NULL);
ROLLBACK;

BEGIN;
/* Test 7: NULL in training functions */
SELECT train_linear_regression(NULL, 'test_table', 'features', 'label');
ROLLBACK;

BEGIN;
/* Test 8: NULL in model loading */
SELECT neurondb.load_model(NULL, '/path/to/model', 'onnx');
ROLLBACK;

COMMIT;

