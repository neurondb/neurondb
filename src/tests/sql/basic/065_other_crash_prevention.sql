-- 039_crash_prevention_basic.sql
-- Basic crash prevention tests for NeuronDB
-- Tests fundamental crash prevention mechanisms

\timing on
\pset footer off
\pset pager off
\pset tuples_only off

\set ON_ERROR_STOP on

-- Create test_train table and view
DROP TABLE IF EXISTS test_train CASCADE;
CREATE TABLE test_train (features vector(28), label integer);
INSERT INTO test_train (features, label) SELECT array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 28)))::vector(28), (random() * 2)::integer FROM generate_series(1, 1000);
CREATE OR REPLACE VIEW test_train_view AS SELECT features, label FROM test_train;

CREATE TABLE IF NOT EXISTS test_train (features vector(28), label integer);
DELETE FROM test_train;
INSERT INTO test_train (features, label) SELECT array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 28)))::vector(28), (random() * 2)::integer FROM generate_series(1, 1000);
CREATE OR REPLACE VIEW test_train_view AS SELECT features, label FROM test_train;


\echo '=========================================================================='
\echo 'Crash Prevention: Basic Functionality Tests'
\echo '=========================================================================='

/*-------------------------------------------------------------------
 * ---- BASIC NULL HANDLING ----
 * Test basic NULL parameter handling
 *------------------------------------------------------------------*/
\echo ''
\echo 'Basic NULL Handling Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 1: Vector operations with NULL handling'
SELECT 
	'NULL Test' AS test_type,
	vector_l2_distance(vector '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]'::vector, vector '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]'::vector) AS valid_distance,
	vector_l2_distance(NULL::vector, vector '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]'::vector) AS null_distance;

\echo 'Test 2: Model operations with NULL handling'
DO $$
DECLARE
	model_id int;
BEGIN
	-- Train model
	SELECT neurondb.train('linear_regression', 'test_train_view', 'features', 'label', '{}'::jsonb) INTO model_id;
	
	-- Test NULL handling in prediction
	BEGIN
		PERFORM neurondb.predict(NULL, vector '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]'::vector);
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'NULL model name correctly rejected';
	END;
	
	BEGIN
		PERFORM neurondb.predict('model_' || model_id::text, NULL::vector);
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'NULL input vector correctly rejected';
	END;
	
	RAISE NOTICE 'Model operations with NULL handling completed';
END$$;

/*-------------------------------------------------------------------
 * ---- BASIC ERROR HANDLING ----
 * Test basic error handling mechanisms
 *------------------------------------------------------------------*/
\echo ''
\echo 'Basic Error Handling Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 3: Invalid table name handling'
DO $$
BEGIN
	PERFORM neurondb.train('linear_regression', 'nonexistent_table_xyz', 'features', 'label', '{}'::jsonb);
EXCEPTION WHEN OTHERS THEN
	RAISE NOTICE 'Invalid table name correctly rejected: %', SQLERRM;
END$$;

\echo 'Test 4: Invalid column name handling'
DO $$
BEGIN
	PERFORM neurondb.train('linear_regression', 'test_train_view', 'nonexistent_column', 'label', '{}'::jsonb);
EXCEPTION WHEN OTHERS THEN
	RAISE NOTICE 'Invalid column name correctly rejected: %', SQLERRM;
END$$;

\echo 'Test 5: Invalid model ID handling'
DO $$
DECLARE
	result float8;
BEGIN
	result := neurondb.evaluate(-999, 'test_train_view', 'features', 'label');
	IF result IS NULL THEN
		RAISE NOTICE 'Invalid model ID correctly handled (returned NULL)';
	ELSE
		RAISE NOTICE 'Invalid model ID returned result: %', result;
	END IF;
EXCEPTION WHEN OTHERS THEN
	RAISE NOTICE 'Invalid model ID correctly rejected: %', SQLERRM;
END$$;

/*-------------------------------------------------------------------
 * ---- BASIC MEMORY SAFETY ----
 * Test basic memory safety mechanisms
 *------------------------------------------------------------------*/
\echo ''
\echo 'Basic Memory Safety Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 6: Vector operations with various sizes'
SELECT 
	vector_l2_distance(vector '[1,2,3]'::vector, vector '[4,5,6]'::vector) AS small_vectors,
	vector_l2_distance(vector '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]'::vector, 
	                   vector '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]'::vector) AS large_vectors;

\echo 'Test 7: Model training and cleanup'
DO $$
DECLARE
	model_id int;
BEGIN
	-- Train model
	SELECT neurondb.train('linear_regression', 'test_train_view', 'features', 'label', '{}'::jsonb) INTO model_id;
	
	-- Use model (handle GPU prediction failures gracefully)
	BEGIN
		PERFORM neurondb.predict('model_' || model_id::text, vector '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]'::vector);
	EXCEPTION WHEN OTHERS THEN
		-- GPU prediction may fail if model is GPU-only and GPU unavailable
		RAISE NOTICE 'Prediction failed (may be GPU-only model): %', SQLERRM;
	END;
	
	-- Cleanup (drop model)
	BEGIN
		PERFORM neurondb.drop_model('model_' || model_id::text);
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'Model drop failed: %', SQLERRM;
	END;
	
	RAISE NOTICE 'Model training and cleanup completed: model_id = %', model_id;
END$$;

\echo ''
\echo '=========================================================================='
\echo '✓ Crash Prevention: Basic tests complete'
\echo '=========================================================================='

\echo 'Test completed successfully'
