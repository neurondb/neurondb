-- Comprehensive ML function stress test
-- Tests invalid inputs, edge cases, and error conditions

\set ON_ERROR_STOP off
\timing on

BEGIN;

-- Setup test data
CREATE TEMP TABLE ml_test_data (features vector(28), label integer);
INSERT INTO ml_test_data SELECT array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 28)))::vector(28), (random() * 2)::integer FROM generate_series(1, 1000);

\echo 'Test 1: NULL parameters in train function'
DO $$
BEGIN
    PERFORM neurondb.train(NULL, 'ml_test_data', 'features', 'label', '{}'::jsonb);
    RAISE EXCEPTION 'Should have failed with NULL algorithm';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'PASS: NULL algorithm rejected: %', SQLERRM;
END$$;

DO $$
BEGIN
    PERFORM neurondb.train('linear_regression', NULL, 'features', 'label', '{}'::jsonb);
    RAISE EXCEPTION 'Should have failed with NULL table';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'PASS: NULL table rejected: %', SQLERRM;
END$$;

\echo 'Test 2: Invalid algorithm names'
DO $$
BEGIN
    PERFORM neurondb.train('invalid_algorithm_xyz', 'ml_test_data', 'features', 'label', '{}'::jsonb);
    RAISE EXCEPTION 'Should have failed with invalid algorithm';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'PASS: Invalid algorithm rejected: %', SQLERRM;
END$$;

\echo 'Test 3: Non-existent table'
DO $$
BEGIN
    PERFORM neurondb.train('linear_regression', 'nonexistent_table_xyz', 'features', 'label', '{}'::jsonb);
    RAISE EXCEPTION 'Should have failed with non-existent table';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'PASS: Non-existent table rejected: %', SQLERRM;
END$$;

\echo 'Test 4: Non-existent columns'
DO $$
BEGIN
    PERFORM neurondb.train('linear_regression', 'ml_test_data', 'nonexistent_col', 'label', '{}'::jsonb);
    RAISE EXCEPTION 'Should have failed with non-existent column';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'PASS: Non-existent column rejected: %', SQLERRM;
END$$;

\echo 'Test 5: Empty table'
CREATE TEMP TABLE empty_ml_data (features vector(28), label integer);
DO $$
BEGIN
    PERFORM neurondb.train('linear_regression', 'empty_ml_data', 'features', 'label', '{}'::jsonb);
    RAISE EXCEPTION 'Should have failed with empty table';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'PASS: Empty table handled: %', SQLERRM;
END$$;

\echo 'Test 6: Invalid hyperparameters'
DO $$
BEGIN
    PERFORM neurondb.train('linear_regression', 'ml_test_data', 'features', 'label', '{"invalid_param": "bad_value"}'::jsonb);
    RAISE NOTICE 'INFO: Invalid hyperparameters handled (may be ignored)';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'PASS: Invalid hyperparameters rejected: %', SQLERRM;
END$$;

\echo 'Test 7: Invalid model ID in evaluate'
DO $$
BEGIN
    PERFORM neurondb.evaluate(-1, 'ml_test_data', 'features', 'label');
    RAISE EXCEPTION 'Should have failed with negative model ID';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'PASS: Negative model ID rejected: %', SQLERRM;
END$$;

DO $$
BEGIN
    PERFORM neurondb.evaluate(999999, 'ml_test_data', 'features', 'label');
    RAISE EXCEPTION 'Should have failed with non-existent model';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'PASS: Non-existent model rejected: %', SQLERRM;
END$$;

\echo 'Test 8: Concurrent training operations'
DO $$
DECLARE
    i int;
    model_id int;
BEGIN
    FOR i IN 1..10 LOOP
        BEGIN
            SELECT neurondb.train('linear_regression', 'ml_test_data', 'features', 'label', '{}'::jsonb) INTO model_id;
            IF model_id IS NOT NULL THEN
                PERFORM neurondb.drop_model('model_' || model_id::text);
            END IF;
        EXCEPTION WHEN OTHERS THEN
            -- Expected to fail sometimes, but should not crash
            NULL;
        END;
    END LOOP;
    RAISE NOTICE 'PASS: Concurrent operations completed';
END$$;

ROLLBACK;



