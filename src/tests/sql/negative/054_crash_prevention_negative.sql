-- 039_crash_prevention_negative.sql
-- Comprehensive crash prevention tests for NeuronDB
-- Tests all 6 crash categories from crash-proofing plan:
-- 1. Memory Context Issues
-- 2. Invalid pfree Calls
-- 3. Wrong Context pfree
-- 4. NULL Return Values
-- 5. NULL Input Parameters
-- 6. SPI Context Crashes

\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP off

\echo '=========================================================================='
\echo 'Crash Prevention: Comprehensive Negative Test Cases'
\echo '=========================================================================='

/*-------------------------------------------------------------------
 * ---- CATEGORY 1: NULL INPUT PARAMETERS ----
 * Test all functions with NULL inputs to prevent crashes
 *------------------------------------------------------------------*/
\echo ''
\echo 'Category 1: NULL Input Parameter Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 1.1: ML Training with NULL table name'
DO $$
BEGIN
	PERFORM neurondb.train('linear_regression', NULL, 'features', 'label', '{}'::jsonb);
EXCEPTION WHEN OTHERS THEN
	NULL; -- Expected error
END$$;

\echo 'Test 1.2: ML Training with NULL feature column'
DO $$
BEGIN
	PERFORM neurondb.train('linear_regression', 'test_train_view', NULL, 'label', '{}'::jsonb);
EXCEPTION WHEN OTHERS THEN
	NULL; -- Expected error
END$$;

\echo 'Test 1.3: ML Training with NULL label column'
DO $$
BEGIN
	PERFORM neurondb.train('linear_regression', 'test_train_view', 'features', NULL, '{}'::jsonb);
EXCEPTION WHEN OTHERS THEN
	NULL; -- Expected error
END$$;

\echo 'Test 1.4: Vector operations with NULL vectors'
SELECT vector_l2_distance(NULL::vector, vector '[1,2,3]'::vector);
SELECT vector_cosine_distance(vector '[1,2,3]'::vector, NULL::vector);
SELECT vector_inner_product(NULL::vector, NULL::vector);

\echo 'Test 1.5: Model prediction with NULL model name'
DO $$
BEGIN
	PERFORM neurondb.predict(NULL, vector '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]'::vector);
EXCEPTION WHEN OTHERS THEN
	NULL; -- Expected error
END$$;

\echo 'Test 1.6: Model prediction with NULL input vector'
DO $$
BEGIN
	PERFORM neurondb.predict('test_model', NULL::vector);
EXCEPTION WHEN OTHERS THEN
	NULL; -- Expected error
END$$;

\echo 'Test 1.7: GPU functions with NULL parameters'
SELECT neurondb_gpu_enable(NULL);
SELECT vector_l2_distance_gpu(NULL::vector, vector '[1,2,3]'::vector);
SELECT vector_l2_distance_gpu(vector '[1,2,3]'::vector, NULL::vector);

\echo 'Test 1.8: Index operations with NULL vectors'
DROP TABLE IF EXISTS crash_test_index;
CREATE TABLE crash_test_index (
	id SERIAL PRIMARY KEY,
	embedding vector(28)
);

INSERT INTO crash_test_index (embedding)
SELECT features FROM test_train_view LIMIT 10;

CREATE INDEX idx_crash_test ON crash_test_index 
USING hnsw (embedding vector_l2_ops);

SELECT id, embedding <-> NULL::vector AS distance
FROM crash_test_index
ORDER BY embedding <-> NULL::vector
LIMIT 10;

DROP TABLE IF EXISTS crash_test_index CASCADE;

/*-------------------------------------------------------------------
 * ---- CATEGORY 2: NULL RETURN VALUES ----
 * Test handling of functions that may return NULL
 *------------------------------------------------------------------*/
\echo ''
\echo 'Category 2: NULL Return Value Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 2.1: Model loading with non-existent model'
DO $$
DECLARE
	result vector;
BEGIN
	result := neurondb.predict('nonexistent_model_xyz', vector '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]'::vector);
	IF result IS NULL THEN
		RAISE NOTICE 'Model correctly returned NULL for non-existent model';
	ELSE
		RAISE NOTICE 'Model returned non-NULL result';
	END IF;
EXCEPTION WHEN OTHERS THEN
	NULL; -- Expected error
END$$;

\echo 'Test 2.2: Model evaluation with invalid model ID'
DO $$
DECLARE
	result float8;
BEGIN
	result := neurondb.evaluate(-999, 'test_train_view', 'features', 'label');
	IF result IS NULL THEN
		RAISE NOTICE 'Evaluation correctly returned NULL for invalid model';
	ELSE
		RAISE NOTICE 'Evaluation returned result: %', result;
	END IF;
EXCEPTION WHEN OTHERS THEN
	NULL; -- Expected error
END$$;

\echo 'Test 2.3: GPU info with GPU unavailable'
SELECT neurondb_gpu_info() AS gpu_info;

\echo 'Test 2.4: Vector operations with empty results'
SELECT 
	id,
	embedding <-> (SELECT embedding FROM test_train_view LIMIT 1) AS distance
FROM test_train_view
WHERE 1=0  -- Empty result set
ORDER BY embedding <-> (SELECT embedding FROM test_train_view LIMIT 1)
LIMIT 10;

/*-------------------------------------------------------------------
 * ---- CATEGORY 3: SPI CONTEXT CRASHES ----
 * Test SPI operations that could crash
 *------------------------------------------------------------------*/
\echo ''
\echo 'Category 3: SPI Context Crash Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 3.1: SPI query with invalid SQL'
DO $$
DECLARE
	ret int;
BEGIN
	-- This should be handled gracefully by SPI safe wrappers
	PERFORM neurondb.train('linear_regression', 'test_train_view', 'features', 'label', '{}'::jsonb);
	-- Try to evaluate with invalid table
	PERFORM neurondb.evaluate(
		(SELECT model_id FROM neurondb.models LIMIT 1),
		'nonexistent_table_xyz',
		'features',
		'label'
	);
EXCEPTION WHEN OTHERS THEN
	NULL; -- Expected error
END$$;

\echo 'Test 3.2: SPI query with syntax error'
DO $$
BEGIN
	-- This should be caught by SPI safe wrappers
	PERFORM neurondb.train('linear_regression', 'test_train_view', 'features', 'label', 'invalid_json'::jsonb);
EXCEPTION WHEN OTHERS THEN
	NULL; -- Expected error
END$$;

\echo 'Test 3.3: SPI query with missing columns'
DO $$
BEGIN
	PERFORM neurondb.train('linear_regression', 'test_train_view', 'nonexistent_col', 'label', '{}'::jsonb);
EXCEPTION WHEN OTHERS THEN
	NULL; -- Expected error
END$$;

\echo 'Test 3.4: SPI query with type mismatch'
DO $$
BEGIN
	PERFORM neurondb.train('linear_regression', 'test_train_view', 'id', 'label', '{}'::jsonb);
EXCEPTION WHEN OTHERS THEN
	NULL; -- Expected error
END$$;

/*-------------------------------------------------------------------
 * ---- CATEGORY 4: MEMORY ALLOCATION FAILURES ----
 * Test scenarios that could cause memory issues
 *------------------------------------------------------------------*/
\echo ''
\echo 'Category 4: Memory Allocation Failure Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 4.1: Extremely large feature vectors'
DO $$
DECLARE
	large_features float8[];
	model_id int;
BEGIN
	-- Create very large feature array (simulate memory pressure)
	large_features := ARRAY(SELECT random() FROM generate_series(1, 10000))::float8[];
	
	-- Try to train with large features
	CREATE TEMP TABLE large_feature_test AS
	SELECT 
		1 AS id,
		large_features AS features,
		1.0 AS label;
	
	SELECT neurondb.train('linear_regression', 'large_feature_test', 'features', 'label', '{}'::jsonb) INTO model_id;
	
	RAISE NOTICE 'Large feature vector handled: model_id = %', model_id;
EXCEPTION WHEN OTHERS THEN
	RAISE NOTICE 'Large feature vector handled gracefully: %', SQLERRM;
END$$;

\echo 'Test 4.2: Extremely large dataset (memory pressure)'
DO $$
DECLARE
	model_id int;
BEGIN
	-- Create large dataset
	CREATE TEMP TABLE large_dataset AS
	SELECT 
		i AS id,
		ARRAY[random(), random(), random(), random()]::float8[] AS features,
		random() * 100 AS label
	FROM generate_series(1, 100000) i;
	
	-- Try to train (may hit memory limits)
	SELECT neurondb.train('linear_regression', 'large_dataset', 'features', 'label', '{}'::jsonb) INTO model_id;
	
	RAISE NOTICE 'Large dataset handled: model_id = %', model_id;
EXCEPTION WHEN OTHERS THEN
	RAISE NOTICE 'Large dataset handled gracefully: %', SQLERRM;
END$$;

\echo 'Test 4.3: Very high dimensional vectors'
DO $$
DECLARE
	high_dim_vector vector;
BEGIN
	-- Try to create very high dimensional vector
	-- This tests vector dimension limits
	BEGIN
		high_dim_vector := (SELECT array_agg(random()) FROM generate_series(1, 10000))::vector;
		RAISE NOTICE 'High dimensional vector created';
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'High dimensional vector rejected: %', SQLERRM;
	END;
END$$;

/*-------------------------------------------------------------------
 * ---- CATEGORY 5: INVALID MEMORY OPERATIONS ----
 * Test operations that could access invalid memory
 *------------------------------------------------------------------*/
\echo ''
\echo 'Category 5: Invalid Memory Operation Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 5.1: Vector operations with dimension mismatches'
SELECT vector_l2_distance(
	vector '[1,2,3]'::vector,
	vector '[1,2,3,4,5]'::vector
);

\echo 'Test 5.2: Vector operations with zero-length vectors'
SELECT vector_l2_distance(
	vector '[]'::vector,
	vector '[1,2,3]'::vector
);

\echo 'Test 5.3: Vector operations with negative dimensions'
DO $$
BEGIN
	-- This should be caught by validation
	PERFORM vector_l2_distance(
		vector '[1,2,3]'::vector(-1),
		vector '[1,2,3]'::vector
	);
EXCEPTION WHEN OTHERS THEN
	NULL; -- Expected error
END$$;

\echo 'Test 5.4: Index operations on dropped index'
DROP TABLE IF EXISTS crash_test_dropped;
CREATE TABLE crash_test_dropped (
	id SERIAL PRIMARY KEY,
	embedding vector(28)
);

INSERT INTO crash_test_dropped (embedding)
SELECT features FROM test_train_view LIMIT 10;

CREATE INDEX idx_crash_dropped ON crash_test_dropped 
USING hnsw (embedding vector_l2_ops);

DROP INDEX idx_crash_dropped;

-- Try to query after index dropped
SELECT id, embedding <-> (SELECT embedding FROM crash_test_dropped LIMIT 1) AS distance
FROM crash_test_dropped
ORDER BY embedding <-> (SELECT embedding FROM crash_test_dropped LIMIT 1)
LIMIT 10;

DROP TABLE IF EXISTS crash_test_dropped CASCADE;

/*-------------------------------------------------------------------
 * ---- CATEGORY 6: RESOURCE EXHAUSTION ----
 * Test scenarios that could exhaust resources
 *------------------------------------------------------------------*/
\echo ''
\echo 'Category 6: Resource Exhaustion Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 6.1: Multiple concurrent model training'
DO $$
DECLARE
	model_ids int[];
	i int;
BEGIN
	-- Try to train multiple models concurrently
	FOR i IN 1..10 LOOP
		BEGIN
			SELECT neurondb.train('linear_regression', 'test_train_view', 'features', 'label', '{}'::jsonb) INTO model_ids[i];
		EXCEPTION WHEN OTHERS THEN
			NULL; -- May hit resource limits
		END;
	END LOOP;
	
	RAISE NOTICE 'Concurrent training handled';
END$$;

\echo 'Test 6.2: Large batch operations'
DO $$
DECLARE
	texts text[];
	i int;
BEGIN
	-- Create large batch
	texts := ARRAY[]::text[];
	FOR i IN 1..1000 LOOP
		texts := array_append(texts, 'Batch text item ' || i::text);
	END LOOP;
	
	-- Try batch embedding
	BEGIN
		PERFORM embed_text_batch(texts);
		RAISE NOTICE 'Large batch handled';
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'Large batch handled gracefully: %', SQLERRM;
	END;
END$$;

\echo 'Test 6.3: Deeply nested operations'
DO $$
DECLARE
	result vector;
BEGIN
	-- Try deeply nested vector operations
	result := vector_normalize(
		vector_add(
			vector_multiply(
				vector '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]'::vector,
				2.0
			),
			vector '[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]'::vector
		)
	);
	
	RAISE NOTICE 'Deeply nested operations handled: % dimensions', vector_dims(result);
EXCEPTION WHEN OTHERS THEN
	RAISE NOTICE 'Deeply nested operations handled gracefully: %', SQLERRM;
END$$;

/*-------------------------------------------------------------------
 * ---- EDGE CASES AND STRESS TESTS ----
 * Test edge cases that could cause crashes
 *------------------------------------------------------------------*/
\echo ''
\echo 'Edge Cases and Stress Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 7.1: Empty string inputs'
DO $$
BEGIN
	PERFORM neurondb.train('linear_regression', '', 'features', 'label', '{}'::jsonb);
EXCEPTION WHEN OTHERS THEN
	NULL; -- Expected error
END$$;

\echo 'Test 7.2: Very long string inputs'
DO $$
DECLARE
	long_string text;
BEGIN
	long_string := repeat('a', 100000);
	-- Try to use in function that might not handle it
	PERFORM neurondb.train('linear_regression', long_string, 'features', 'label', '{}'::jsonb);
EXCEPTION WHEN OTHERS THEN
	NULL; -- Expected error
END$$;

\echo 'Test 7.3: Special characters in inputs'
DO $$
BEGIN
	PERFORM neurondb.train('linear_regression', 'test_train_view', 'features; DROP TABLE', 'label', '{}'::jsonb);
EXCEPTION WHEN OTHERS THEN
	NULL; -- Expected error
END$$;

\echo 'Test 7.4: NaN and Infinity values'
DO $$
DECLARE
	nan_vector vector;
	inf_vector vector;
BEGIN
	-- Try to create vectors with NaN/Inf
	nan_vector := vector '[NaN, 1, 2]'::vector;
	inf_vector := vector '[Infinity, 1, 2]'::vector;
	
	-- Try operations
	SELECT vector_l2_distance(nan_vector, vector '[1,2,3]'::vector);
	SELECT vector_l2_distance(inf_vector, vector '[1,2,3]'::vector);
EXCEPTION WHEN OTHERS THEN
	NULL; -- May be handled or rejected
END$$;

\echo 'Test 7.5: Zero vectors and edge values'
SELECT 
	vector_l2_distance(vector '[0,0,0]'::vector, vector '[0,0,0]'::vector) AS zero_distance,
	vector_cosine_distance(vector '[0,0,0]'::vector, vector '[1,1,1]'::vector) AS zero_cosine,
	vector_norm(vector '[0,0,0]'::vector) AS zero_norm;

\echo ''
\echo '=========================================================================='
\echo '✓ Crash Prevention: All negative test cases complete'
\echo '=========================================================================='

\echo 'Test completed successfully'




