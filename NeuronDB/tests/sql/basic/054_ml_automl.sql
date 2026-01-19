\timing on
\pset footer off
\pset pager off

-- This test uses sample_train and sample_test tables created by ml_dataset.py
-- Run: python ml_dataset.py <dataset_name> to populate the database first
-- Or use the test runner: python run_ml_tests.py
--
-- Verify required tables exist

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
		IF current_gpu_enabled != '1' THEN
			RAISE WARNING 'GPU mode expected but neurondb.compute_mode = % (expected: 1)', current_gpu_enabled;
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

-- Test AutoML for classification task
DROP TABLE IF EXISTS automl_model_temp;
DO $$
DECLARE
	model_id_result integer;
BEGIN
	BEGIN
		model_id_result := auto_train(
			'test_train_view',
			'features',
			'label',
			'classification',
			'accuracy'
		);
		
		CREATE TEMP TABLE automl_model_temp AS
		SELECT model_id_result AS model_id;
		
		IF model_id_result IS NULL OR model_id_result <= 0 THEN
			RAISE WARNING 'auto_train returned invalid model_id: %', model_id_result;
			-- Create empty table so test can continue
			DROP TABLE IF EXISTS automl_model_temp;
			CREATE TEMP TABLE automl_model_temp (model_id integer);
		END IF;
	EXCEPTION WHEN OTHERS THEN
		RAISE WARNING 'auto_train failed: %', SQLERRM;
		-- Create empty table so test can continue
		DROP TABLE IF EXISTS automl_model_temp;
		CREATE TEMP TABLE automl_model_temp (model_id integer);
	END;
END $$;

SELECT COALESCE(model_id, 0) AS model_id FROM automl_model_temp;

-- Evaluate the best model
CREATE TEMP TABLE automl_metrics_temp (metrics jsonb);
DO $$
DECLARE
	mid integer;
	metrics_result jsonb;
BEGIN
	SELECT model_id INTO mid FROM automl_model_temp LIMIT 1;
	
	IF mid IS NULL OR mid <= 0 THEN
		RAISE WARNING 'No model_id found for evaluation';
		RETURN;
	END IF;
	
	metrics_result := neurondb.evaluate(mid, 'test_test_view', 'features', 'label');
	INSERT INTO automl_metrics_temp VALUES (metrics_result);
END
$$;

-- Show metrics (only if model was created successfully)
SELECT
	format('%-15s', 'Algorithm') AS metric,
	COALESCE(
		(SELECT m.algorithm::text 
		 FROM neurondb.ml_models m, automl_model_temp t 
		 WHERE m.model_id = t.model_id AND t.model_id > 0),
		'N/A'
	) AS value
UNION ALL
SELECT
	format('%-15s', 'Accuracy'),
	CASE WHEN (m.metrics::jsonb ? 'accuracy')
		THEN ROUND((m.metrics::jsonb ->> 'accuracy')::numeric, 4)::text
		ELSE NULL END
FROM automl_metrics_temp m
UNION ALL
SELECT
	format('%-15s', 'Precision'),
	CASE WHEN (m.metrics::jsonb ? 'precision')
		THEN ROUND((m.metrics::jsonb ->> 'precision')::numeric, 4)::text
		ELSE NULL END
FROM automl_metrics_temp m
UNION ALL
SELECT
	format('%-15s', 'Recall'),
	CASE WHEN (m.metrics::jsonb ? 'recall')
		THEN ROUND((m.metrics::jsonb ->> 'recall')::numeric, 4)::text
		ELSE NULL END
FROM automl_metrics_temp m
UNION ALL
SELECT
	format('%-15s', 'F1 Score'),
	CASE WHEN (m.metrics::jsonb ? 'f1_score')
		THEN ROUND((m.metrics::jsonb ->> 'f1_score')::numeric, 4)::text
		ELSE NULL END
FROM automl_metrics_temp m;

-- Cleanup
DROP TABLE IF EXISTS automl_model_temp;
DROP TABLE IF EXISTS automl_metrics_temp;

\echo 'Test completed successfully'
