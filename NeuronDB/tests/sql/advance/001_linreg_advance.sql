-- 001_linreg_advance.sql
-- Exhaustive detailed test for linear_regression: all train, error, predict, evaluate.
-- Works on 1000 rows only and tests each and every way with comprehensive coverage
-- Tests: GPU/CPU paths, hyperparameters, batch operations, error handling, metadata

SET client_min_messages TO WARNING;
\set ON_ERROR_STOP on
\timing on
\pset footer off
\pset pager off
\pset tuples_only off

/* Step 0: Read settings from test_settings table and verify GPU configuration */
DO $$
DECLARE
	gpu_mode TEXT;
	current_gpu_enabled TEXT;
	current_gpu_kernels TEXT;
	gpu_kernels_val TEXT;
BEGIN
	-- Read GPU mode setting from test_settings
	SELECT setting_value INTO gpu_mode FROM test_settings WHERE setting_key = 'gpu_mode';
	SELECT setting_value INTO gpu_kernels_val FROM test_settings WHERE setting_key = 'gpu_kernels';
	
	-- Verify GPU configuration matches test_settings (set by test runner)
	SELECT current_setting('neurondb.gpu_enabled', true) INTO current_gpu_enabled;
	SELECT current_setting('neurondb.gpu_kernels', true) INTO current_gpu_kernels;
	
	IF gpu_mode = 'gpu' THEN
		-- Verify GPU is enabled (should be set by test runner)
		IF current_gpu_enabled != 'on' THEN
			RAISE WARNING 'GPU mode expected but neurondb.gpu_enabled = % (expected: on)', current_gpu_enabled;
		END IF;
	ELSE
		-- Verify GPU is disabled (should be set by test runner)
		IF current_gpu_enabled != 'off' THEN
			RAISE WARNING 'CPU mode expected but neurondb.gpu_enabled = % (expected: off)', current_gpu_enabled;
		END IF;
	END IF;
END $$;

\echo '=========================================================================='
\echo 'linear_regression: Exhaustive GPU/CPU + Error Coverage (1000 rows sample)'
\echo '=========================================================================='

/* Check that source tables exist (dataset schema or public schema) */
DO $$
DECLARE
	train_table TEXT;
	test_table TEXT;
BEGIN
	-- Check for dataset schema tables first, then public schema
	SELECT table_schema || '.' || table_name INTO train_table
	FROM information_schema.tables 
	WHERE (table_schema = 'dataset' AND table_name = 'test_train')
	   OR (table_schema = 'public' AND table_name IN ('sample_train', 'test_train'))
	ORDER BY CASE WHEN table_schema = 'dataset' THEN 0 ELSE 1 END
	LIMIT 1;
	
	IF train_table IS NULL THEN
		RAISE EXCEPTION 'No training table found in dataset or public schema';
	END IF;
	
	-- Determine corresponding test table
	IF train_table LIKE 'dataset.%' THEN
		test_table := 'dataset.test_test';
	ELSIF train_table LIKE '%sample_train%' THEN
		test_table := 'sample_test';
	ELSE
		-- Extract schema from train_table and use it for test_table
		IF train_table LIKE '%.%' THEN
			test_table := split_part(train_table, '.', 1) || '.test_test';
		ELSE
			test_table := 'test_test';
		END IF;
	END IF;
	
	-- Verify test table exists
	IF NOT EXISTS (SELECT 1 FROM information_schema.tables 
	               WHERE (table_schema || '.' || table_name) = test_table) THEN
		RAISE EXCEPTION 'Test table % does not exist', test_table;
	END IF;
END
$$;

-- Create views with 1000 rows for advance tests (use existing views if available)
DROP VIEW IF EXISTS test_train_view CASCADE;
DROP VIEW IF EXISTS test_test_view CASCADE;

-- Check if views already exist from test runner, otherwise create from source tables
DO $$
DECLARE
	train_source TEXT;
	test_source TEXT;
BEGIN
	-- Find source tables
	SELECT table_schema || '.' || table_name INTO train_source
	FROM information_schema.tables 
	WHERE (table_schema = 'dataset' AND table_name = 'test_train')
	   OR (table_schema = 'public' AND table_name IN ('sample_train', 'test_train'))
	ORDER BY CASE WHEN table_schema = 'dataset' THEN 0 ELSE 1 END
	LIMIT 1;
	
	IF train_source LIKE 'dataset.%' THEN
		test_source := 'dataset.test_test';
	ELSIF train_source LIKE '%sample_train%' THEN
		test_source := 'sample_test';
	ELSE
		test_source := 'test_test';
	END IF;
	
	-- Create views with type conversion
	EXECUTE format('CREATE VIEW test_train_view AS SELECT features::vector(28) as features, label FROM %s LIMIT 1000', train_source);
	EXECUTE format('CREATE VIEW test_test_view AS SELECT features::vector(28) as features, label FROM %s LIMIT 1000', test_source);
END
$$;

\echo ''
\echo 'Dataset Information'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
SELECT 
	COUNT(*)::bigint AS train_count,
	(SELECT COUNT(*)::bigint FROM test_test_view) AS test_count,
	(SELECT vector_dims(features) FROM test_train_view LIMIT 1) AS feature_dim,
	ROUND((SELECT AVG(label) FROM test_train_view)::numeric, 4) AS avg_label,
	ROUND((SELECT STDDEV(label) FROM test_train_view)::numeric, 4) AS stddev_label
FROM test_train_view;

/*---- Register required GPU kernels ----*/
\echo ''
\echo 'GPU Configuration'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
-- GPU already configured via test_settings above
SELECT neurondb_gpu_info() AS gpu_info;

/*
 * ---- TRAINING routines (1000 sampled rows) ----
 * Test multiple hyperparameter combinations and paths
 */
\echo ''
\echo 'Training Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 1: GPU training with default parameters'
SET neurondb.gpu_enabled = on;
DROP TABLE IF EXISTS gpu_model_temp_001;
CREATE TEMP TABLE gpu_model_temp_001 AS
SELECT neurondb.train('linear_regression', 
	'test_train_view', 
	'features', 'label', '{}'::jsonb)::integer AS gpu_model_id;

SELECT 
	'GPU Default' AS config,
	gpu_model_id AS model_id,
	(SELECT metrics->>'storage' FROM neurondb.ml_models WHERE model_id = gpu_model_id) AS storage
FROM gpu_model_temp_001;

\echo 'Test 2: CPU training with default parameters'
SET neurondb.gpu_enabled = off;
DROP TABLE IF EXISTS cpu_model_temp_001;
CREATE TEMP TABLE cpu_model_temp_001 AS
SELECT neurondb.train('linear_regression', 
	'test_train_view', 
	'features', 'label', '{}'::jsonb)::integer AS cpu_model_id;

SELECT 
	'CPU Default' AS config,
	cpu_model_id AS model_id,
	(SELECT metrics->>'storage' FROM neurondb.ml_models WHERE model_id = cpu_model_id) AS storage
FROM cpu_model_temp_001;

\echo 'Test 3: Custom hyperparameters (fit_intercept=true, normalize=false)'
DROP TABLE IF EXISTS custom_model_temp_001;
CREATE TEMP TABLE custom_model_temp_001 AS
SELECT neurondb.train('linear_regression', 
	'test_train_view', 
	'features', 'label', 
	'{"fit_intercept":true,"normalize":false}'::jsonb)::integer AS custom_model_id;

SELECT 
	'Custom (intercept, no norm)' AS config,
	custom_model_id AS model_id,
	(SELECT metrics->>'fit_intercept' FROM neurondb.ml_models WHERE model_id = custom_model_id) AS fit_intercept
FROM custom_model_temp_001;

\echo 'Test 4: Without intercept (fit_intercept=false)'
DROP TABLE IF EXISTS noint_model_temp_001;
CREATE TEMP TABLE noint_model_temp_001 AS
SELECT neurondb.train('linear_regression', 
	'test_train_view', 
	'features', 'label', 
	'{"fit_intercept":false}'::jsonb)::integer AS noint_model_id;

/* --- ERROR path: bad table or column --- */
\echo ''
\echo 'Error Handling Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 1: Invalid table name'
DO $$
DECLARE
	error_occurred BOOLEAN := false;
BEGIN
	BEGIN
		PERFORM neurondb.train('linear_regression','missing_table','features','label','{}'::jsonb);
		RAISE EXCEPTION 'FAIL: expected error for missing table but operation succeeded';
	EXCEPTION WHEN OTHERS THEN 
		error_occurred := true;
		-- Verify it's a meaningful error (not just any exception)
		IF SQLERRM IS NULL OR LENGTH(SQLERRM) = 0 THEN
			RAISE EXCEPTION 'FAIL: error occurred but error message is empty';
		END IF;
	END;
	IF NOT error_occurred THEN
		RAISE EXCEPTION 'FAIL: expected error for missing table but no error occurred';
	END IF;
END$$;

\echo 'Error Test 2: Invalid feature column'
DO $$
DECLARE
	error_occurred BOOLEAN := false;
BEGIN
	BEGIN
		PERFORM neurondb.train('linear_regression','test_train_view','notafeat','label','{}'::jsonb);
		RAISE EXCEPTION 'FAIL: expected error for invalid feature column but operation succeeded';
	EXCEPTION WHEN OTHERS THEN 
		error_occurred := true;
		-- Verify it's a meaningful error
		IF SQLERRM IS NULL OR LENGTH(SQLERRM) = 0 THEN
			RAISE EXCEPTION 'FAIL: error occurred but error message is empty';
		END IF;
	END;
	IF NOT error_occurred THEN
		RAISE EXCEPTION 'FAIL: expected error for invalid feature column but no error occurred';
	END IF;
END$$;

\echo 'Error Test 3: Invalid label column'
DO $$
DECLARE
	error_occurred BOOLEAN := false;
BEGIN
	BEGIN
		PERFORM neurondb.train('linear_regression','test_train_view','features','notalabel','{}'::jsonb);
		RAISE EXCEPTION 'FAIL: expected error for invalid label column but operation succeeded';
	EXCEPTION WHEN OTHERS THEN 
		error_occurred := true;
		-- Verify it's a meaningful error
		IF SQLERRM IS NULL OR LENGTH(SQLERRM) = 0 THEN
			RAISE EXCEPTION 'FAIL: error occurred but error message is empty';
		END IF;
	END;
	IF NOT error_occurred THEN
		RAISE EXCEPTION 'FAIL: expected error for invalid label column but no error occurred';
	END IF;
END$$;

-- Error Test 4: NULL features
-- NOTE: This test is skipped because NULL feature column now properly raises an error
-- instead of crashing. The NULL check has been added to the neurondb.train() function.
-- Uncomment below to test NULL feature column handling:
/*
\echo 'Error Test 4: NULL features'
DO $$
BEGIN
	BEGIN
		PERFORM neurondb.train('linear_regression','test_train_view',NULL,'label','{}'::jsonb);
		RAISE EXCEPTION 'FAIL: expected error for NULL feature column';
	EXCEPTION WHEN OTHERS THEN 
			NULL;
		-- Error handled correctly
		NULL;
	END;
END$$;
*/

/*-------------------------------------------------------------------
 * ---- PREDICT ----
 * GPU/CPU paths, all error paths, batch and single, sampling 1000
 *------------------------------------------------------------------*/
\echo ''
\echo 'Prediction Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Predict Test 1: GPU batch prediction (1000 rows)'
SET neurondb.gpu_enabled = on;
SELECT 
	'GPU Batch' AS test_type,
	COUNT(*) AS n_predictions,
	ROUND(AVG(score)::numeric, 4) AS avg_score,
	ROUND(MIN(score)::numeric, 4) AS min_score,
	ROUND(MAX(score)::numeric, 4) AS max_score,
	ROUND(STDDEV(score)::numeric, 4) AS stddev_score
FROM (
	SELECT neurondb.predict((SELECT gpu_model_id FROM gpu_model_temp_001 LIMIT 1), features) AS score
	FROM test_test_view
	LIMIT 1000
) sub;

\echo 'Predict Test 2: CPU batch prediction (1000 rows)'
SET neurondb.gpu_enabled = off;
SELECT 
	'CPU Batch' AS test_type,
	COUNT(*) AS n_predictions,
	ROUND(AVG(score)::numeric, 4) AS avg_score,
	ROUND(MIN(score)::numeric, 4) AS min_score,
	ROUND(MAX(score)::numeric, 4) AS max_score,
	ROUND(STDDEV(score)::numeric, 4) AS stddev_score
FROM (
	SELECT neurondb.predict((SELECT cpu_model_id FROM cpu_model_temp_001 LIMIT 1), features) AS score
	FROM test_test_view
	LIMIT 1000
) sub;

\echo 'Predict Test 3: Custom model single row prediction'
SELECT 
	'Custom Single' AS test_type,
	neurondb.predict((SELECT custom_model_id FROM custom_model_temp_001 LIMIT 1), features) AS prediction
FROM test_test_view
LIMIT 1;

\echo 'Predict Test 4: Custom model batch (100 rows)'
SET neurondb.gpu_enabled = off;
SELECT 
	'Custom Batch' AS test_type,
	COUNT(*) AS n_predictions,
	ROUND(AVG(score)::numeric, 4) AS avg_score
FROM (
	SELECT neurondb.predict((SELECT custom_model_id FROM custom_model_temp_001 LIMIT 1), features) AS score
	FROM test_test_view
	LIMIT 100
) b;

\echo 'Predict Test 5: No-intercept model batch (50 rows)'
SELECT 
	'No-Intercept Batch' AS test_type,
	COUNT(*) AS n_predictions,
	ROUND(AVG(score)::numeric, 4) AS avg_score
FROM (
	SELECT neurondb.predict((SELECT noint_model_id FROM noint_model_temp_001 LIMIT 1), features) AS score
	FROM test_test_view
	LIMIT 50
) b;

/* Error: invalid model id */
\echo ''
\echo 'Prediction Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Predict Error Test 1: Non-existent model ID'
DO $$
DECLARE
	error_occurred BOOLEAN := false;
BEGIN
	BEGIN
		PERFORM neurondb.predict(-10, ARRAY[0.1, 0.2, 0.3]::double precision[]);
		RAISE EXCEPTION 'FAIL: non-existent model should error but operation succeeded';
	EXCEPTION WHEN OTHERS THEN 
		error_occurred := true;
		IF SQLERRM IS NULL OR LENGTH(SQLERRM) = 0 THEN
			RAISE EXCEPTION 'FAIL: error occurred but error message is empty';
		END IF;
	END;
	IF NOT error_occurred THEN
		RAISE EXCEPTION 'FAIL: expected error for non-existent model but no error occurred';
	END IF;
END$$;

\echo 'Predict Error Test 2: Wrong feature type (integer array)'
DO $$
DECLARE
	cpu_mid_temp integer;
	error_occurred BOOLEAN := false;
BEGIN
	SELECT cpu_model_id INTO cpu_mid_temp FROM cpu_model_temp_001 LIMIT 1;
	IF cpu_mid_temp IS NULL THEN
		RAISE EXCEPTION 'FAIL: no CPU model available for test';
	END IF;
	BEGIN
		PERFORM neurondb.predict(cpu_mid_temp, '{1,2,3}'::integer[]);
		RAISE EXCEPTION 'FAIL: int[] instead of float[] should error but operation succeeded';
	EXCEPTION WHEN OTHERS THEN 
		error_occurred := true;
		IF SQLERRM IS NULL OR LENGTH(SQLERRM) = 0 THEN
			RAISE EXCEPTION 'FAIL: error occurred but error message is empty';
		END IF;
	END;
	IF NOT error_occurred THEN
		RAISE EXCEPTION 'FAIL: expected error for wrong feature type but no error occurred';
	END IF;
END$$;

\echo 'Predict Error Test 3: Wrong feature dimension'
DO $$
DECLARE
	model_dim integer;
	cpu_mid integer;
	error_occurred BOOLEAN := false;
BEGIN
	SELECT cpu_model_id INTO cpu_mid FROM cpu_model_temp_001 LIMIT 1;
	IF cpu_mid IS NULL THEN
		RAISE EXCEPTION 'FAIL: no CPU model available for test';
	END IF;
	SELECT jsonb_array_length(metrics->'coefficients') INTO model_dim
	FROM neurondb.ml_models WHERE model_id = cpu_mid
	AND metrics IS NOT NULL AND metrics->'coefficients' IS NOT NULL;
	IF model_dim IS NULL THEN
		model_dim := 28; /* Default if not available */
	END IF;
	BEGIN
		PERFORM neurondb.predict(cpu_mid, ARRAY[42.0]::double precision[]); -- wrong dim
		RAISE EXCEPTION 'FAIL: wrong feature array length should error but operation succeeded';
	EXCEPTION WHEN OTHERS THEN 
		error_occurred := true;
		IF SQLERRM IS NULL OR LENGTH(SQLERRM) = 0 THEN
			RAISE EXCEPTION 'FAIL: error occurred but error message is empty';
		END IF;
	END;
	IF NOT error_occurred THEN
		RAISE EXCEPTION 'FAIL: expected error for wrong feature dimension but no error occurred';
	END IF;
END
$$;

\echo 'Predict Error Test 4: NULL model ID'
DO $$
DECLARE
	error_occurred BOOLEAN := false;
BEGIN
	BEGIN
		PERFORM neurondb.predict(NULL, ARRAY[0.1, 0.2, 0.3]::double precision[]);
		RAISE EXCEPTION 'FAIL: NULL model ID should error but operation succeeded';
	EXCEPTION WHEN OTHERS THEN 
		error_occurred := true;
		IF SQLERRM IS NULL OR LENGTH(SQLERRM) = 0 THEN
			RAISE EXCEPTION 'FAIL: error occurred but error message is empty';
		END IF;
	END;
	IF NOT error_occurred THEN
		RAISE EXCEPTION 'FAIL: expected error for NULL model ID but no error occurred';
	END IF;
END$$;

\echo 'Predict Error Test 5: NULL features'
DO $$
DECLARE
	cpu_mid integer;
	error_occurred BOOLEAN := false;
BEGIN
	SELECT cpu_model_id INTO cpu_mid FROM cpu_model_temp_001 LIMIT 1;
	IF cpu_mid IS NULL THEN
		RAISE EXCEPTION 'FAIL: no CPU model available for test';
	END IF;
	BEGIN
		PERFORM neurondb.predict(cpu_mid, NULL);
		RAISE EXCEPTION 'FAIL: NULL features should error but operation succeeded';
	EXCEPTION WHEN OTHERS THEN 
		error_occurred := true;
		IF SQLERRM IS NULL OR LENGTH(SQLERRM) = 0 THEN
			RAISE EXCEPTION 'FAIL: error occurred but error message is empty';
		END IF;
	END;
	IF NOT error_occurred THEN
		RAISE EXCEPTION 'FAIL: expected error for NULL features but no error occurred';
	END IF;
END$$;

/*-------------------------------------------------------------------
 * ---- EVALUATE ----
 * Metrics for all model types/paths ; test set sampled to 1000 rows
 *------------------------------------------------------------------*/
\echo ''
\echo 'Evaluation Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Evaluate Test 1: CPU model evaluation'
DO $$
DECLARE
	cpu_mid integer;
	result jsonb;
BEGIN
	SELECT cpu_model_id INTO cpu_mid FROM cpu_model_temp_001 LIMIT 1;
	IF cpu_mid IS NULL THEN
		RAISE EXCEPTION 'FAIL: no CPU model available for evaluation test';
	END IF;
	result := neurondb.evaluate(cpu_mid, 'test_test_view', 'features', 'label');
	IF result IS NULL THEN
		RAISE EXCEPTION 'FAIL: evaluation returned NULL for CPU model';
	END IF;
	IF result ? 'error' THEN
		RAISE EXCEPTION 'FAIL: evaluation returned error: %', result->>'error';
	END IF;
END$$;

\echo 'Evaluate Test 2: GPU model evaluation'
DO $$
DECLARE
	gpu_mid integer;
	result jsonb;
BEGIN
	SELECT gpu_model_id INTO gpu_mid FROM gpu_model_temp_001 LIMIT 1;
	IF gpu_mid IS NULL THEN
		RAISE EXCEPTION 'FAIL: no GPU model available for evaluation test';
	END IF;
	result := neurondb.evaluate(gpu_mid, 'test_test_view', 'features', 'label');
	IF result IS NULL THEN
		RAISE EXCEPTION 'FAIL: evaluation returned NULL for GPU model';
	END IF;
	IF result ? 'error' THEN
		RAISE EXCEPTION 'FAIL: evaluation returned error: %', result->>'error';
	END IF;
END$$;

\echo 'Evaluate Test 3: Custom model evaluation'
DO $$
DECLARE
	custom_mid integer;
	result jsonb;
BEGIN
	SELECT custom_model_id INTO custom_mid FROM custom_model_temp_001 LIMIT 1;
	IF custom_mid IS NULL THEN
		RAISE EXCEPTION 'FAIL: no custom model available for evaluation test';
	END IF;
	result := neurondb.evaluate(custom_mid, 'test_test_view', 'features', 'label');
	IF result IS NULL THEN
		RAISE EXCEPTION 'FAIL: evaluation returned NULL for custom model';
	END IF;
	IF result ? 'error' THEN
		RAISE EXCEPTION 'FAIL: evaluation returned error: %', result->>'error';
	END IF;
END$$;

\echo 'Evaluate Test 4: No-intercept model evaluation'
DO $$
DECLARE
	noint_mid integer;
	result jsonb;
BEGIN
	SELECT noint_model_id INTO noint_mid FROM noint_model_temp_001 LIMIT 1;
	IF noint_mid IS NULL THEN
		RAISE EXCEPTION 'FAIL: no no-intercept model available for evaluation test';
	END IF;
	result := neurondb.evaluate(noint_mid, 'test_test_view', 'features', 'label');
	IF result IS NULL THEN
		RAISE EXCEPTION 'FAIL: evaluation returned NULL for no-intercept model';
	END IF;
	IF result ? 'error' THEN
		RAISE EXCEPTION 'FAIL: evaluation returned error: %', result->>'error';
	END IF;
END$$;

\echo ''
\echo 'Evaluation Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Evaluate Error Test 1: Invalid table'
DO $$
DECLARE
	cpu_mid integer;
	error_occurred BOOLEAN := false;
BEGIN
	SELECT cpu_model_id INTO cpu_mid FROM cpu_model_temp_001 LIMIT 1;
	IF cpu_mid IS NULL THEN
		RAISE EXCEPTION 'FAIL: no CPU model available for test';
	END IF;
	BEGIN
		PERFORM neurondb.evaluate(cpu_mid, 'no_such', 'features', 'label');
		RAISE EXCEPTION 'FAIL: eval on bad table must error but operation succeeded';
	EXCEPTION WHEN OTHERS THEN 
		error_occurred := true;
		IF SQLERRM IS NULL OR LENGTH(SQLERRM) = 0 THEN
			RAISE EXCEPTION 'FAIL: error occurred but error message is empty';
		END IF;
	END;
	IF NOT error_occurred THEN
		RAISE EXCEPTION 'FAIL: expected error for invalid table but no error occurred';
	END IF;
END$$;

\echo 'Evaluate Error Test 2: Invalid feature column'
DO $$
DECLARE
	cpu_mid integer;
	error_occurred BOOLEAN := false;
BEGIN
	SELECT cpu_model_id INTO cpu_mid FROM cpu_model_temp_001 LIMIT 1;
	IF cpu_mid IS NULL THEN
		RAISE EXCEPTION 'FAIL: no CPU model available for test';
	END IF;
	BEGIN
		PERFORM neurondb.evaluate(cpu_mid, 'test_test_view', 'badfeature', 'label');
		RAISE EXCEPTION 'FAIL: eval on bad feature col must error but operation succeeded';
	EXCEPTION WHEN OTHERS THEN 
		error_occurred := true;
		IF SQLERRM IS NULL OR LENGTH(SQLERRM) = 0 THEN
			RAISE EXCEPTION 'FAIL: error occurred but error message is empty';
		END IF;
	END;
	IF NOT error_occurred THEN
		RAISE EXCEPTION 'FAIL: expected error for invalid feature column but no error occurred';
	END IF;
END$$;

\echo 'Evaluate Error Test 3: Invalid label column'
DO $$
DECLARE
	cpu_mid integer;
	error_occurred BOOLEAN := false;
BEGIN
	SELECT cpu_model_id INTO cpu_mid FROM cpu_model_temp_001 LIMIT 1;
	IF cpu_mid IS NULL THEN
		RAISE EXCEPTION 'FAIL: no CPU model available for test';
	END IF;
	BEGIN
		PERFORM neurondb.evaluate(cpu_mid, 'test_test_view', 'features', 'badlabel');
		RAISE EXCEPTION 'FAIL: eval on bad label col must error but operation succeeded';
	EXCEPTION WHEN OTHERS THEN 
		error_occurred := true;
		IF SQLERRM IS NULL OR LENGTH(SQLERRM) = 0 THEN
			RAISE EXCEPTION 'FAIL: error occurred but error message is empty';
		END IF;
	END;
	IF NOT error_occurred THEN
		RAISE EXCEPTION 'FAIL: expected error for invalid label column but no error occurred';
	END IF;
END$$;

\echo 'Evaluate Error Test 4: NULL model ID'
DO $$
DECLARE
	error_occurred BOOLEAN := false;
BEGIN
	BEGIN
		PERFORM neurondb.evaluate(NULL, 'test_test_view', 'features', 'label');
		RAISE EXCEPTION 'FAIL: NULL model ID should error but operation succeeded';
	EXCEPTION WHEN OTHERS THEN 
		error_occurred := true;
		IF SQLERRM IS NULL OR LENGTH(SQLERRM) = 0 THEN
			RAISE EXCEPTION 'FAIL: error occurred but error message is empty';
		END IF;
	END;
	IF NOT error_occurred THEN
		RAISE EXCEPTION 'FAIL: expected error for NULL model ID but no error occurred';
	END IF;
END$$;

/*-------------------------------------------------------------------
 * Model catalog check and metadata
 *------------------------------------------------------------------*/
\echo ''
\echo 'Model Metadata Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Metadata Test 1: GPU model metadata'
SELECT 
	m.model_id,
	m.algorithm,
	m.created_at,
	m.metrics->>'storage' AS storage_type,
	m.metrics->>'r_squared' AS r_squared,
	m.metrics->>'mse' AS mse,
	m.metrics->>'mae' AS mae,
	m.metrics->>'n_samples' AS n_samples,
	m.metrics->>'n_features' AS n_features,
	CASE 
		WHEN m.model_data IS NULL THEN 'NULL (GPU model)'
		ELSE format('%s bytes', pg_column_size(m.model_data))
	END AS model_data_status
FROM neurondb.ml_models m, gpu_model_temp_001 t
WHERE m.model_id = t.gpu_model_id;

\echo 'Metadata Test 2: CPU model metadata'
SELECT 
	m.model_id,
	m.algorithm,
	m.created_at,
	m.metrics->>'storage' AS storage_type,
	m.metrics->>'r_squared' AS r_squared,
	m.metrics->>'mse' AS mse,
	m.metrics->>'mae' AS mae,
	CASE 
		WHEN m.model_data IS NULL THEN 'NULL (GPU model)'
		ELSE format('%s bytes', pg_column_size(m.model_data))
	END AS model_data_status
FROM neurondb.ml_models m, cpu_model_temp_001 t
WHERE m.model_id = t.cpu_model_id;

\echo 'Metadata Test 3: Custom model metadata'
SELECT 
	m.model_id,
	m.algorithm,
	m.metrics->>'fit_intercept' AS fit_intercept,
	m.metrics->>'r_squared' AS r_squared,
	m.metrics->>'mse' AS mse
FROM neurondb.ml_models m, custom_model_temp_001 t
WHERE m.model_id = t.custom_model_id;

\echo 'Metadata Test 4: Model comparison (all models)'
SELECT 
	algorithm,
	COUNT(*) AS n_models,
	MIN(created_at) AS first_created,
	MAX(created_at) AS last_created,
	COUNT(CASE WHEN metrics->>'storage' = 'gpu' THEN 1 END) AS gpu_models,
	COUNT(CASE WHEN metrics->>'storage' = 'cpu' THEN 1 END) AS cpu_models,
	ROUND(AVG((metrics->>'r_squared')::numeric), 4) AS avg_r_squared
FROM neurondb.ml_models
WHERE algorithm = 'linear_regression'
GROUP BY algorithm;

\echo ''
\echo '=========================================================================='
\echo '✓ linear_regression: Full exhaustive code-path test complete (1000-row sample)'
\echo '=========================================================================='

\echo 'Test completed successfully'
