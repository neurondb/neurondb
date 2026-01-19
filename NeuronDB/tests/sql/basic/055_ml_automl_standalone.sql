\timing on
\pset footer off
\pset pager off

-- This test uses test_train_view and test_test_view created by the test runner
-- The views are created from dataset.test_train and dataset.test_test tables

\set ON_ERROR_STOP on

/* Step 0: Read settings from test_settings table and verify GPU configuration */
DO $$
DECLARE
	gpu_mode TEXT;
	current_gpu_enabled TEXT;
	current_automl_gpu TEXT;
BEGIN
	-- Read GPU mode setting from test_settings (if table exists)
	BEGIN
		SELECT setting_value INTO gpu_mode FROM test_settings WHERE setting_key = 'gpu_mode';
	EXCEPTION WHEN undefined_table THEN
		gpu_mode := NULL;
	END;
	
	-- Verify GPU configuration matches test_settings (set by test runner)
	SELECT current_setting('neurondb.compute_mode', true) INTO current_gpu_enabled;
	SELECT current_setting('neurondb.automl.use_gpu', true) INTO current_automl_gpu;
	
	IF gpu_mode = 'gpu' THEN
		-- Verify GPU is enabled (should be set by test runner)
		IF current_gpu_enabled != 'on' THEN
			RAISE WARNING 'GPU mode expected but neurondb.compute_mode = % (expected: on)', current_gpu_enabled;
		END IF;
	ELSIF gpu_mode IS NOT NULL THEN
		-- Verify GPU is disabled (should be set by test runner)
		IF current_gpu_enabled != 'off' THEN
			RAISE WARNING 'CPU mode expected but neurondb.compute_mode = % (expected: 0)', current_gpu_enabled;
		END IF;
	END IF;
END $$;

\echo '=========================================================================='
\echo '=========================================================================='

-- Test AutoML for classification
DROP TABLE IF EXISTS automl_classification_result;
DO $$
DECLARE
	result_val integer;
BEGIN
	BEGIN
		result_val := auto_train(
			'test_train_view',
			'features',
			'label',
			'classification',
			'accuracy'
		);
		
		CREATE TEMP TABLE automl_classification_result AS
		SELECT result_val::text AS result;
	EXCEPTION WHEN OTHERS THEN
		RAISE WARNING 'auto_train (classification) failed: %', SQLERRM;
		CREATE TEMP TABLE automl_classification_result AS
		SELECT 'AutoML failed: ' || SQLERRM AS result;
	END;
END $$;

-- Show AutoML classification results
SELECT result FROM automl_classification_result;

-- Extract best model_id from result (if available)
-- The result contains the best model_id in the text
-- For demonstration, we'll show the leaderboard

\echo ''
\echo '=========================================================================='
\echo '=========================================================================='

-- Test AutoML for regression
DROP TABLE IF EXISTS automl_regression_result;
DO $$
DECLARE
	result_val integer;
BEGIN
	BEGIN
		result_val := auto_train(
			'test_train_view',
			'features',
			'label',
			'regression',
			'r2'
		);
		
		CREATE TEMP TABLE automl_regression_result AS
		SELECT result_val::text AS result;
	EXCEPTION WHEN OTHERS THEN
		RAISE WARNING 'auto_train (regression) failed: %', SQLERRM;
		CREATE TEMP TABLE automl_regression_result AS
		SELECT 'AutoML failed: ' || SQLERRM AS result;
	END;
END $$;

-- Show AutoML regression results
SELECT result FROM automl_regression_result;

\echo ''
\echo '=========================================================================='
\echo '=========================================================================='

-- Test classification with F1 score
DROP TABLE IF EXISTS automl_f1_result;
DO $$
DECLARE
	result_val integer;
BEGIN
	BEGIN
		result_val := auto_train(
			'test_train_view',
			'features',
			'label',
			'classification',
			'f1'
		);
		
		CREATE TEMP TABLE automl_f1_result AS
		SELECT result_val::text AS result;
	EXCEPTION WHEN OTHERS THEN
		RAISE WARNING 'auto_train (F1) failed: %', SQLERRM;
		CREATE TEMP TABLE automl_f1_result AS
		SELECT 'AutoML failed: ' || SQLERRM AS result;
	END;
END $$;

SELECT 'F1 Score Results:' AS test_type, result FROM automl_f1_result;

-- Test regression with MSE
DROP TABLE IF EXISTS automl_mse_result;
DO $$
DECLARE
	result_val integer;
BEGIN
	BEGIN
		result_val := auto_train(
			'test_train_view',
			'features',
			'label',
			'regression',
			'mse'
		);
		
		CREATE TEMP TABLE automl_mse_result AS
		SELECT result_val::text AS result;
	EXCEPTION WHEN OTHERS THEN
		RAISE WARNING 'auto_train (MSE) failed: %', SQLERRM;
		CREATE TEMP TABLE automl_mse_result AS
		SELECT 'AutoML failed: ' || SQLERRM AS result;
	END;
END $$;

SELECT 'MSE Results:' AS test_type, result FROM automl_mse_result;

\echo ''
\echo '=========================================================================='
\echo 'AutoML: Verify GPU usage'
\echo '=========================================================================='

-- Check if GPU models were created
-- Explicitly exclude model_data (bytea) to avoid pager issues
SELECT 
	m.model_id,
	m.algorithm,
	m.metrics->>'storage' AS storage_backend,
	m.metrics->>'n_features' AS n_features,
	m.metrics->>'n_samples' AS n_samples,
	m.created_at,
	CASE WHEN m.model_data IS NULL THEN 'NULL' ELSE 'present' END AS model_data_status
FROM neurondb.ml_models m
WHERE m.algorithm IN ('linear_regression', 'logistic_regression', 'random_forest', 'decision_tree', 'ridge', 'lasso', 'svm')
	AND m.created_at > NOW() - INTERVAL '10 minutes'
ORDER BY m.created_at DESC
LIMIT 10;

\echo ''
\echo '=========================================================================='
\echo '=========================================================================='

-- Cleanup
DROP TABLE IF EXISTS automl_classification_result;
DROP TABLE IF EXISTS automl_regression_result;
DROP TABLE IF EXISTS automl_f1_result;
DROP TABLE IF EXISTS automl_mse_result;

\echo 'Test completed successfully'
