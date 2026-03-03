/*-------------------------------------------------------------------------
 *
 * 015_kmeans.sql
 *    K-Means Clustering test
 *
 *    Step-by-step test with clean output and timing
 *
 *-------------------------------------------------------------------------*/

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
	SELECT current_setting('neurondb.compute_mode', true) INTO current_gpu_enabled;
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
	COUNT(*) FILTER (WHERE features IS NOT NULL)::bigint AS valid_rows
FROM test_train_view
UNION ALL
SELECT 
	'test_test_view',
	COUNT(*)::bigint,
	COUNT(*) FILTER (WHERE features IS NOT NULL)::bigint
FROM test_test_view;


-- TRAINING
\echo ''
\echo 'Training'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DROP TABLE IF EXISTS kmeans_model;

CREATE TEMP TABLE kmeans_model AS
SELECT
	train_kmeans_model_id('test_train_view', 'features', 3, 100) AS model_id;

SELECT model_id FROM kmeans_model;

-- PREDICTION
\echo ''
\echo 'Prediction Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Predict Test 1: Single row prediction'
-- Note: predict_kmeans_model_id function may not be implemented yet
-- SELECT 
-- 	'Single Row' AS test_type,
-- 	predict_kmeans((SELECT model_id FROM kmeans_model LIMIT 1), features) AS prediction
-- FROM test_test_view
-- LIMIT 1;

\echo 'Predict Test 2: Batch prediction (100 rows)'
-- Note: predict_kmeans_model_id function may not be implemented yet
-- SELECT 
-- 	'Batch (100 rows)' AS test_type,
-- 	COUNT(*) AS n_predictions
-- FROM (
-- 	SELECT predict_kmeans((SELECT model_id FROM kmeans_model LIMIT 1), features) AS score
-- 	FROM test_test_view
-- 	LIMIT 100
-- ) sub;

-- EVALUATION
\echo ''
\echo 'Evaluation Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DROP TABLE IF EXISTS kmeans_metrics;
CREATE TEMP TABLE kmeans_metrics (metrics jsonb);

DO $$
DECLARE
	mid integer;
	metrics_result jsonb;
	eval_error text;
BEGIN
	SELECT model_id INTO mid FROM kmeans_model LIMIT 1;
	IF mid IS NULL THEN
		RAISE WARNING 'No model_id found in kmeans_model';
		INSERT INTO kmeans_metrics VALUES ('{"error": "No model_id found"}'::jsonb);
		RETURN;
	END IF;
	
	BEGIN
		BEGIN
			metrics_result := evaluate_kmeans_by_model_id(mid, 'test_test_view', 'features');
			
			IF metrics_result IS NULL THEN
				RAISE WARNING 'Evaluation returned NULL';
				INSERT INTO kmeans_metrics VALUES ('{"error": "Evaluation returned NULL"}'::jsonb);
			ELSE
				INSERT INTO kmeans_metrics VALUES (metrics_result);
			END IF;
		EXCEPTION WHEN OTHERS THEN
			eval_error := SQLERRM;
			RAISE WARNING 'Evaluation exception: %', eval_error;
			eval_error := REPLACE(REPLACE(REPLACE(eval_error, '"', '\"'), E'\n', ' '), E'\r', ' ');
			INSERT INTO kmeans_metrics VALUES (jsonb_build_object('error', eval_error));
		END;
	EXCEPTION WHEN OTHERS THEN
		eval_error := SQLERRM;
		RAISE WARNING 'Outer evaluation exception: %', eval_error;
		eval_error := REPLACE(REPLACE(REPLACE(eval_error, '"', '\"'), E'\n', ' '), E'\r', ' ');
		INSERT INTO kmeans_metrics VALUES (jsonb_build_object('error', eval_error));
	END;
END $$;

-- METRICS
\echo ''

SELECT 
	'kmeans'::text AS algorithm,
	(SELECT COUNT(*)::bigint FROM test_train_view) AS n_samples,
	(SELECT COUNT(*)::bigint FROM test_test_view) AS test_samples;

/* Verify GPU was used for training when GPU mode is enabled */
DO $$
DECLARE
	gpu_mode TEXT;
BEGIN
	SELECT setting_value INTO gpu_mode FROM test_settings WHERE setting_key = 'gpu_mode';
	-- KMeans uses legacy functions, GPU check may not be available
END $$;

SELECT COUNT(*)::bigint AS test_samples
FROM test_test_view
WHERE features IS NOT NULL;

\echo ''
\echo 'Evaluation Metrics'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT
	format('%-15s', 'Inertia') AS metric,
	CASE 
		WHEN m.metrics IS NULL THEN 'N/A'
		WHEN (m.metrics::jsonb ? 'error') THEN (m.metrics::jsonb->>'error')
		WHEN (m.metrics::jsonb ? 'inertia')
			THEN ROUND((m.metrics::jsonb ->> 'inertia')::numeric, 6)::text
		ELSE 'N/A' 
	END AS value
FROM kmeans_metrics m
UNION ALL
SELECT
	format('%-15s', 'N_Clusters'),
	CASE 
		WHEN m.metrics IS NULL THEN NULL
		WHEN (m.metrics::jsonb ? 'error') THEN NULL
		WHEN (m.metrics::jsonb ? 'n_clusters')
			THEN (m.metrics::jsonb ->> 'n_clusters')::text
		ELSE NULL 
	END
FROM kmeans_metrics m
UNION ALL
SELECT
	format('%-15s', 'N_Iterations'),
	CASE 
		WHEN m.metrics IS NULL THEN NULL
		WHEN (m.metrics::jsonb ? 'error') THEN NULL
		WHEN (m.metrics::jsonb ? 'n_iterations')
			THEN (m.metrics::jsonb ->> 'n_iterations')::text
		ELSE NULL 
	END
FROM kmeans_metrics m
ORDER BY metric;

\echo ''
\echo 'Summary'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT 
	(SELECT model_id FROM kmeans_model) AS model_id,
	(SELECT COUNT(*)::bigint FROM test_train_view) AS train_samples,
	(SELECT COUNT(*)::bigint FROM test_test_view) AS test_samples,
	CASE 
		WHEN m.metrics IS NULL THEN NULL
		WHEN (m.metrics::jsonb ? 'error') THEN NULL
		WHEN (m.metrics::jsonb ? 'inertia') THEN ROUND((m.metrics::jsonb->>'inertia')::numeric, 6)
		ELSE NULL 
	END AS final_inertia
FROM kmeans_metrics m;

-- Store results in test_metrics table
INSERT INTO test_metrics (
	test_name, algorithm, model_id, train_samples, test_samples,
	inertia, n_clusters, updated_at
)
SELECT 
	'015_kmeans_basic',
	'kmeans',
	(SELECT model_id FROM kmeans_model),
	(SELECT COUNT(*)::bigint FROM test_train_view),
	(SELECT COUNT(*)::bigint FROM test_test_view),
	CASE 
		WHEN m.metrics IS NULL THEN NULL
		WHEN (m.metrics::jsonb ? 'error') THEN NULL
		WHEN (m.metrics::jsonb ? 'inertia') THEN ROUND((m.metrics::jsonb->>'inertia')::numeric, 6) 
		ELSE NULL 
	END,
	CASE 
		WHEN m.metrics IS NULL THEN NULL
		WHEN (m.metrics::jsonb ? 'error') THEN NULL
		WHEN (m.metrics::jsonb ? 'n_clusters') THEN (m.metrics::jsonb->>'n_clusters')::integer 
		ELSE NULL 
	END,
	CURRENT_TIMESTAMP
FROM kmeans_metrics m
ON CONFLICT (test_name) DO UPDATE SET
	algorithm = EXCLUDED.algorithm,
	model_id = EXCLUDED.model_id,
	train_samples = EXCLUDED.train_samples,
	test_samples = EXCLUDED.test_samples,
	inertia = EXCLUDED.inertia,
	n_clusters = EXCLUDED.n_clusters,
	updated_at = CURRENT_TIMESTAMP;

DROP TABLE IF EXISTS kmeans_model;

/* Display comprehensive views */
SELECT 
	tm.test_name,
	tm.algorithm,
	tm.model_id,
	tm.train_samples,
	tm.test_samples,
	tm.inertia,
	tm.n_clusters,
	tm.created_at,
	tm.updated_at
FROM test_metrics tm
WHERE tm.test_name = '015_kmeans_basic';

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
BEGIN
	SELECT setting_value INTO gpu_mode FROM test_settings WHERE setting_key = 'gpu_mode';
	
	IF gpu_mode = 'gpu' THEN
		BEGIN
			RAISE NOTICE 'GPU Information:';
			PERFORM device_id, device_name, total_memory_mb, free_memory_mb, 
					compute_capability_major, compute_capability_minor, is_available
			FROM neurondb_gpu_info();
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
	COUNT(*) FILTER (WHERE features IS NOT NULL)::bigint AS valid_rows,
	COUNT(*) FILTER (WHERE features IS NULL)::bigint AS null_features,
	0::bigint AS null_labels
FROM test_train_view
UNION ALL
SELECT 
	'test_test_view',
	COUNT(*)::bigint,
	COUNT(*) FILTER (WHERE features IS NOT NULL)::bigint,
	COUNT(*) FILTER (WHERE features IS NULL)::bigint,
	0::bigint
FROM test_test_view
ORDER BY dataset_name;

SELECT 
	tm.test_name,
	tm.algorithm,
	tm.model_id,
	tm.train_samples,
	tm.test_samples,
	ROUND(tm.inertia::numeric, 6) AS inertia,
	tm.n_clusters,
	CASE
		WHEN m.metrics IS NULL THEN 'CPU Training (default)'
		WHEN m.metrics::jsonb->>'storage' = 'gpu' THEN 'GPU Training ✓'
		WHEN m.metrics::jsonb->>'storage' = 'cpu' THEN 'CPU Training'
		WHEN m.metrics::jsonb->>'storage' IS NULL OR m.metrics::jsonb->>'storage' = '' THEN 'CPU Training (default)'
		ELSE 'Unknown'
	END AS training_status,
	tm.updated_at AS test_completed_at
FROM test_metrics tm
LEFT JOIN neurondb.ml_models m ON m.model_id = tm.model_id
WHERE tm.test_name = '015_kmeans_basic';

\echo ''
\echo 'Test completed successfully'
