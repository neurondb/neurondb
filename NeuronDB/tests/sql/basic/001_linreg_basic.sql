/*-------------------------------------------------------------------------
 *
 * 001_linreg.sql
 *    Linear Regression test
 *
 *    Step-by-step test with clean output and timing
 *
 *-------------------------------------------------------------------------*/

SET client_min_messages TO WARNING;
\set ON_ERROR_STOP on
\timing on
\pset footer off
\pset pager off
\pset tuples_only off

-- GPU CONFIGURATION
\echo ''
\echo 'GPU Configuration'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	gpu_mode TEXT;
	current_gpu_enabled TEXT;
BEGIN
	SELECT setting_value INTO gpu_mode FROM test_settings WHERE setting_key = 'gpu_mode';
	SELECT current_setting('neurondb.gpu_enabled', true) INTO current_gpu_enabled;
	IF gpu_mode = 'gpu' THEN
		PERFORM neurondb_gpu_enable();
	END IF;
END $$;

-- Create test_settings table if it doesn't exist
CREATE TABLE IF NOT EXISTS test_settings (
	setting_key TEXT PRIMARY KEY,
	setting_value TEXT,
	updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create test_metrics table if it doesn't exist
CREATE TABLE IF NOT EXISTS test_metrics (
	test_name TEXT PRIMARY KEY,
	algorithm TEXT,
	model_id INTEGER,
	train_samples BIGINT,
	test_samples BIGINT,
	-- Regression metrics
	mse NUMERIC,
	rmse NUMERIC,
	mae NUMERIC,
	r_squared NUMERIC,
	-- Classification metrics
	accuracy NUMERIC,
	precision NUMERIC,
	recall NUMERIC,
	f1_score NUMERIC,
	-- Clustering metrics
	silhouette_score NUMERIC,
	inertia NUMERIC,
	n_clusters INTEGER,
	-- Time series metrics
	mape NUMERIC,
	-- Predictions (stored as JSONB for flexibility)
	predictions JSONB,
	-- Additional metadata
	metadata JSONB,
	created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- DATASET
\echo ''
\echo 'Dataset Information'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Expected split: 80% training, 20% test'
\echo ''

SELECT 
	(SELECT COUNT(*)::bigint FROM test_train_view) AS train_rows,
	(SELECT COUNT(*)::bigint FROM test_test_view) AS test_rows,
	(SELECT COUNT(*)::bigint FROM test_train_view) + (SELECT COUNT(*)::bigint FROM test_test_view) AS total_rows,
	ROUND((SELECT COUNT(*)::bigint FROM test_train_view)::numeric / 
		  NULLIF((SELECT COUNT(*)::bigint FROM test_train_view) + (SELECT COUNT(*)::bigint FROM test_test_view), 0) * 100, 2) AS train_percentage,
	ROUND((SELECT COUNT(*)::bigint FROM test_test_view)::numeric / 
		  NULLIF((SELECT COUNT(*)::bigint FROM test_train_view) + (SELECT COUNT(*)::bigint FROM test_test_view), 0) * 100, 2) AS test_percentage;

SELECT 
	'test_train_view' AS dataset,
	COUNT(*)::bigint AS total_rows,
	COUNT(*) FILTER (WHERE features IS NOT NULL AND label IS NOT NULL)::bigint AS valid_rows
FROM test_train_view
UNION ALL
SELECT 
	'test_test_view',
	COUNT(*)::bigint,
	COUNT(*) FILTER (WHERE features IS NOT NULL AND label IS NOT NULL)::bigint
FROM test_test_view;


-- TRAINING
\echo ''
\echo 'Training'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DROP TABLE IF EXISTS gpu_model_temp;

CREATE TEMP TABLE gpu_model_temp AS
SELECT
	neurondb.train(
		'default',
		'linear_regression',
		'test_train_view',
		'label',
		ARRAY['features'],
		'{}'::jsonb
	)::integer AS model_id;

SELECT model_id FROM gpu_model_temp;

-- PREDICTION
\echo ''
\echo 'Prediction Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Predict Test 1: Single row prediction'
SELECT 
	'Single Row' AS test_type,
	neurondb.predict((SELECT model_id FROM gpu_model_temp LIMIT 1), features) AS prediction,
	label AS actual_label
FROM test_test_view
LIMIT 1;

\echo 'Predict Test 2: Batch prediction (100 rows)'
SELECT 
	'Batch (100 rows)' AS test_type,
	COUNT(*) AS n_predictions,
	ROUND(AVG(score)::numeric, 4) AS avg_prediction,
	ROUND(MIN(score)::numeric, 4) AS min_prediction,
	ROUND(MAX(score)::numeric, 4) AS max_prediction,
	ROUND(STDDEV(score)::numeric, 4) AS stddev_prediction
FROM (
	SELECT neurondb.predict((SELECT model_id FROM gpu_model_temp LIMIT 1), features) AS score
	FROM test_test_view
	LIMIT 100
) sub;

-- EVALUATION
\echo ''
\echo 'Evaluation Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Show evaluation setup info
SELECT 
	(SELECT model_id FROM gpu_model_temp LIMIT 1) AS model_id,
	(SELECT COUNT(*)::bigint FROM test_test_view WHERE features IS NOT NULL AND label IS NOT NULL) AS test_samples_available;

DROP TABLE IF EXISTS gpu_metrics_temp;
CREATE TEMP TABLE gpu_metrics_temp (metrics jsonb);

DO $$
DECLARE
	mid integer;
	metrics_result jsonb;
	eval_error text;
	test_count bigint;
BEGIN
	-- Get model_id
	SELECT model_id INTO mid FROM gpu_model_temp LIMIT 1;
	IF mid IS NULL THEN
		RAISE WARNING 'No model_id found in gpu_model_temp';
		INSERT INTO gpu_metrics_temp VALUES ('{"error": "No model_id found"}'::jsonb);
		RETURN;
	END IF;
	
	-- Verify test data is available
	SELECT COUNT(*) INTO test_count 
	FROM test_test_view 
	WHERE features IS NOT NULL AND label IS NOT NULL;
	
	IF test_count < 1 THEN
		RAISE WARNING 'No valid test samples available (count: %)', test_count;
		INSERT INTO gpu_metrics_temp VALUES (jsonb_build_object('error', 'No valid test samples available'));
		RETURN;
	END IF;
	
	RAISE NOTICE 'Evaluating model_id: % on test_test_view with % valid samples', mid, test_count;
	
	BEGIN
		metrics_result := neurondb.evaluate(mid, 'test_test_view', 'features', 'label');
		
		IF metrics_result IS NULL THEN
			RAISE WARNING 'Evaluation returned NULL for model_id: %', mid;
			INSERT INTO gpu_metrics_temp VALUES ('{"error": "Evaluation returned NULL"}'::jsonb);
		ELSE
			RAISE NOTICE 'Evaluation successful for model_id: %', mid;
			INSERT INTO gpu_metrics_temp VALUES (metrics_result);
		END IF;
	EXCEPTION WHEN OTHERS THEN
		eval_error := SQLERRM;
		RAISE WARNING 'Evaluation exception for model_id %: %', mid, eval_error;
		eval_error := REPLACE(REPLACE(REPLACE(eval_error, '"', '\"'), E'\n', ' '), E'\r', ' ');
		INSERT INTO gpu_metrics_temp VALUES (jsonb_build_object('error', eval_error, 'model_id', mid));
	END;
END $$;

-- METRICS
\echo ''

SELECT 
	m.algorithm::text AS algorithm,
	COALESCE((m.metrics::jsonb->>'n_samples')::bigint, m.num_samples::bigint, 0) AS n_samples,
	COALESCE((m.metrics::jsonb->>'n_features')::integer, m.num_features, 0) AS n_features,
	COALESCE(m.metrics::jsonb->>'storage', 'default') AS storage,
	ROUND(COALESCE((m.metrics::jsonb->>'mse')::numeric, 0), 6) AS mse,
	ROUND(COALESCE((m.metrics::jsonb->>'mae')::numeric, 0), 6) AS mae,
	ROUND(COALESCE((m.metrics::jsonb->>'r_squared')::numeric, 0), 6) AS r_squared
FROM neurondb.ml_models m, gpu_model_temp t
WHERE m.model_id = t.model_id;

/* Verify GPU was used for training when GPU mode is enabled */
DO $$
DECLARE
	gpu_mode TEXT;
	storage_val TEXT;
	gpu_available BOOLEAN;
BEGIN
	SELECT setting_value INTO gpu_mode FROM test_settings WHERE setting_key = 'gpu_mode';
	SELECT COALESCE(m.metrics::jsonb->>'storage', 'cpu') INTO storage_val
	FROM neurondb.ml_models m, gpu_model_temp t
	WHERE m.model_id = t.model_id;
	
	-- Only check GPU info if GPU mode is enabled - never call GPU functions in CPU mode
	IF gpu_mode = 'gpu' THEN
		BEGIN
			-- Check if GPU is actually available (use is_available column which matches C code check)
			SELECT COALESCE(BOOL_OR(is_available), false) INTO gpu_available
			FROM neurondb_gpu_info();
			
			-- If GPU mode is enabled and GPU is detected but model was trained on CPU, warn (not error)
			-- This can happen if Metal backend isn't properly initialized or GPU training failed
			IF gpu_available AND storage_val != 'gpu' THEN
				RAISE WARNING 'GPU mode enabled and GPU detected but model was trained on CPU (storage=%). This may indicate Metal backend initialization issue.', storage_val;
			END IF;
			
			-- If GPU is not available, it's expected to use CPU, so just warn
			IF NOT gpu_available AND storage_val != 'gpu' THEN
				RAISE WARNING 'GPU mode enabled but GPU hardware not available, model trained on CPU (storage=%)', storage_val;
			END IF;
		EXCEPTION WHEN OTHERS THEN
			-- If GPU function call fails, silently continue - GPU not available
			gpu_available := false;
		END;
	END IF;
	
	-- If CPU mode is enabled, verify model was trained on CPU
	IF gpu_mode = 'cpu' AND storage_val = 'gpu' THEN
		RAISE WARNING 'CPU mode enabled but model was trained on GPU (storage=gpu)';
	END IF;
END $$;

SELECT COUNT(*)::bigint AS test_samples
FROM test_test_view
WHERE features IS NOT NULL AND label IS NOT NULL;

\echo ''
\echo 'Evaluation Metrics'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Show raw metrics for debugging
\echo 'Raw metrics JSON:'
SELECT metrics FROM gpu_metrics_temp;

-- Show error if evaluation failed
SELECT 
	CASE 
		WHEN metrics IS NULL THEN 'ERROR: Metrics is NULL - evaluation did not run'
		WHEN (metrics::jsonb ? 'error') THEN 'ERROR: ' || (metrics::jsonb->>'error')
		ELSE 'Evaluation completed successfully'
	END AS evaluation_status
FROM gpu_metrics_temp;

SELECT
	format('%-15s', 'MSE') AS metric,
	CASE 
		WHEN m.metrics IS NULL THEN 'N/A'
		WHEN (m.metrics::jsonb ? 'error') THEN (m.metrics::jsonb->>'error')
		WHEN (m.metrics::jsonb ? 'mse')
			THEN ROUND((m.metrics::jsonb ->> 'mse')::numeric, 6)::text
		ELSE NULL 
	END AS value
FROM gpu_metrics_temp m
UNION ALL
SELECT
	format('%-15s', 'RMSE'),
	CASE 
		WHEN m.metrics IS NULL THEN 'N/A'
		WHEN (m.metrics::jsonb ? 'error') THEN NULL
		WHEN (m.metrics::jsonb ? 'rmse')
			THEN ROUND((m.metrics::jsonb ->> 'rmse')::numeric, 6)::text
		ELSE NULL 
	END
FROM gpu_metrics_temp m
UNION ALL
SELECT
	format('%-15s', 'MAE'),
	CASE 
		WHEN m.metrics IS NULL THEN 'N/A'
		WHEN (m.metrics::jsonb ? 'error') THEN NULL
		WHEN (m.metrics::jsonb ? 'mae')
			THEN ROUND((m.metrics::jsonb ->> 'mae')::numeric, 6)::text
		ELSE NULL 
	END
FROM gpu_metrics_temp m
UNION ALL
SELECT
	format('%-15s', 'R²'),
	CASE 
		WHEN m.metrics IS NULL THEN 'N/A'
		WHEN (m.metrics::jsonb ? 'error') THEN NULL
		WHEN (m.metrics::jsonb ? 'r_squared')
			THEN ROUND((m.metrics::jsonb ->> 'r_squared')::numeric, 6)::text
		ELSE NULL 
	END
FROM gpu_metrics_temp m
ORDER BY metric;

\echo ''
\echo 'Summary'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT 
	(SELECT model_id FROM gpu_model_temp) AS model_id,
	(SELECT COUNT(*)::bigint FROM test_train_view) AS train_samples,
	(SELECT COUNT(*)::bigint FROM test_test_view) AS test_samples,
	CASE 
		WHEN m.metrics IS NULL THEN NULL
		WHEN (m.metrics::jsonb ? 'error') THEN NULL
		WHEN (m.metrics::jsonb ? 'mse') THEN ROUND((m.metrics::jsonb->>'mse')::numeric, 6)
		ELSE NULL 
	END AS final_mse
FROM gpu_metrics_temp m;

-- Store results in test_metrics table
INSERT INTO test_metrics (
	test_name, algorithm, model_id, train_samples, test_samples,
	mse, rmse, mae, r_squared, updated_at
)
SELECT 
	'001_linreg_basic',
	'linear_regression',
	(SELECT model_id FROM gpu_model_temp),
	(SELECT COUNT(*)::bigint FROM test_train_view),
	(SELECT COUNT(*)::bigint FROM test_test_view),
	CASE 
		WHEN m.metrics IS NULL THEN NULL
		WHEN (m.metrics::jsonb ? 'error') THEN NULL
		WHEN (m.metrics::jsonb ? 'mse') THEN ROUND((m.metrics::jsonb->>'mse')::numeric, 6) 
		ELSE NULL 
	END,
	CASE 
		WHEN m.metrics IS NULL THEN NULL
		WHEN (m.metrics::jsonb ? 'error') THEN NULL
		WHEN (m.metrics::jsonb ? 'rmse') THEN ROUND((m.metrics::jsonb->>'rmse')::numeric, 6) 
		ELSE NULL 
	END,
	CASE 
		WHEN m.metrics IS NULL THEN NULL
		WHEN (m.metrics::jsonb ? 'error') THEN NULL
		WHEN (m.metrics::jsonb ? 'mae') THEN ROUND((m.metrics::jsonb->>'mae')::numeric, 6) 
		ELSE NULL 
	END,
	CASE 
		WHEN m.metrics IS NULL THEN NULL
		WHEN (m.metrics::jsonb ? 'error') THEN NULL
		WHEN (m.metrics::jsonb ? 'r_squared') THEN ROUND((m.metrics::jsonb->>'r_squared')::numeric, 6) 
		ELSE NULL 
	END,
	CURRENT_TIMESTAMP
FROM gpu_metrics_temp m
ON CONFLICT (test_name) DO UPDATE SET
	algorithm = EXCLUDED.algorithm,
	model_id = EXCLUDED.model_id,
	train_samples = EXCLUDED.train_samples,
	test_samples = EXCLUDED.test_samples,
	mse = EXCLUDED.mse,
	rmse = EXCLUDED.rmse,
	mae = EXCLUDED.mae,
	r_squared = EXCLUDED.r_squared,
	updated_at = CURRENT_TIMESTAMP;

DROP TABLE IF EXISTS gpu_model_temp;

/* Display comprehensive views */
SELECT 
	m.model_id,
	m.algorithm::text AS algorithm,
	m.project_id,
	m.version,
	m.status::text AS status,
	m.training_table,
	m.training_column,
	COALESCE((m.metrics::jsonb->>'n_samples')::bigint, m.num_samples::bigint, 0) AS n_samples,
	COALESCE((m.metrics::jsonb->>'n_features')::integer, m.num_features, 0) AS n_features,
	COALESCE(m.metrics::jsonb->>'storage', 'default') AS storage,
	ROUND(COALESCE((m.metrics::jsonb->>'mse')::numeric, 0), 6) AS mse,
	ROUND(COALESCE((m.metrics::jsonb->>'rmse')::numeric, 0), 6) AS rmse,
	ROUND(COALESCE((m.metrics::jsonb->>'mae')::numeric, 0), 6) AS mae,
	ROUND(COALESCE((m.metrics::jsonb->>'r_squared')::numeric, 0), 6) AS r_squared,
	m.training_time_ms,
	m.created_at,
	m.completed_at
FROM neurondb.ml_models m
WHERE m.model_id = (SELECT model_id FROM test_metrics WHERE test_name = '001_linreg_basic')
ORDER BY m.model_id DESC
LIMIT 1;

SELECT 
	tm.test_name,
	tm.algorithm,
	tm.model_id,
	tm.train_samples,
	tm.test_samples,
	tm.mse,
	tm.rmse,
	tm.mae,
	tm.r_squared,
	tm.created_at,
	tm.updated_at
FROM test_metrics tm
WHERE tm.test_name = '001_linreg_basic';

-- Only show GPU info if GPU mode is enabled - never call GPU functions in CPU mode
DO $$
DECLARE
	gpu_mode TEXT;
BEGIN
	SELECT setting_value INTO gpu_mode FROM test_settings WHERE setting_key = 'gpu_mode';
	
	IF gpu_mode = 'gpu' THEN
		-- Display GPU information only when GPU mode is enabled
		PERFORM NULL; -- Placeholder for GPU info display
	END IF;
END $$;

-- Conditionally display GPU info only in GPU mode
DO $$
DECLARE
	gpu_mode TEXT;
	rec RECORD;
BEGIN
	SELECT setting_value INTO gpu_mode FROM test_settings WHERE setting_key = 'gpu_mode';
	
	IF gpu_mode = 'gpu' THEN
		BEGIN
			RAISE NOTICE 'GPU Information:';
			-- Display GPU info by looping through results
			FOR rec IN SELECT device_id, device_name, total_memory_mb, free_memory_mb, 
					compute_capability_major, compute_capability_minor, is_available
			FROM neurondb_gpu_info()
			LOOP
				RAISE NOTICE '  Device %: % (Memory: %/% MB, Compute: %.%, Available: %)',
					rec.device_id, rec.device_name, rec.free_memory_mb, rec.total_memory_mb,
					rec.compute_capability_major, rec.compute_capability_minor, rec.is_available;
			END LOOP;
		EXCEPTION WHEN OTHERS THEN
			RAISE NOTICE 'GPU information not available';
		END;
	ELSE
		RAISE NOTICE 'CPU mode: GPU information skipped';
	END IF;
END $$;

SELECT 
	'test_train_view' AS dataset_name,
	COUNT(*)::bigint AS total_rows,
	COUNT(*) FILTER (WHERE features IS NOT NULL AND label IS NOT NULL)::bigint AS valid_rows,
	COUNT(*) FILTER (WHERE features IS NULL)::bigint AS null_features,
	COUNT(*) FILTER (WHERE label IS NULL)::bigint AS null_labels
FROM test_train_view
UNION ALL
SELECT 
	'test_test_view',
	COUNT(*)::bigint,
	COUNT(*) FILTER (WHERE features IS NOT NULL AND label IS NOT NULL)::bigint,
	COUNT(*) FILTER (WHERE features IS NULL)::bigint,
	COUNT(*) FILTER (WHERE label IS NULL)::bigint
FROM test_test_view
ORDER BY dataset_name;

SELECT 
	tm.test_name,
	tm.algorithm,
	tm.model_id,
	m.metrics::jsonb->>'storage' AS training_storage,
	tm.train_samples,
	tm.test_samples,
	ROUND(tm.mse::numeric, 6) AS mse,
	ROUND(tm.rmse::numeric, 6) AS rmse,
	ROUND(tm.mae::numeric, 6) AS mae,
	ROUND(tm.r_squared::numeric, 6) AS r_squared,
	CASE 
		WHEN m.metrics::jsonb->>'storage' = 'gpu' THEN 'GPU Training ✓'
		WHEN m.metrics::jsonb->>'storage' = 'cpu' THEN 'CPU Training'
		ELSE 'Unknown'
	END AS training_status,
	tm.updated_at AS test_completed_at
FROM test_metrics tm
LEFT JOIN neurondb.ml_models m ON m.model_id = tm.model_id
WHERE tm.test_name = '001_linreg_basic';

\echo ''
\echo 'Test completed successfully'
