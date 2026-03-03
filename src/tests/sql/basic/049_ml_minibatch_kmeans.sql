/*-------------------------------------------------------------------------
 *
 * 016_minibatch_kmeans.sql
 *    Mini-Batch K-Means Clustering test
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

-- Test Mini-Batch K-Means clustering with k=3, batch_size=100
WITH clusters AS (
	SELECT unnest(cluster_minibatch_kmeans('test_train_view', 'features', 3, 100, 100)) AS cluster_id
)
SELECT 
	(SELECT COUNT(*) FROM test_train_view) AS total_rows,
	COUNT(DISTINCT cluster_id) AS num_clusters
FROM clusters;

-- PREDICTION
\echo ''
\echo 'Prediction Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Mini-Batch K-Means uses direct clustering function, prediction not available'

-- EVALUATION
\echo ''
\echo 'Evaluation Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Mini-Batch K-Means uses direct clustering function, evaluation not available'

-- METRICS
\echo ''

\echo ''
\echo 'Evaluation Metrics'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Mini-Batch K-Means metrics: See training output above'

\echo ''
\echo 'Summary'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT 
	NULL::integer AS model_id,
	(SELECT COUNT(*)::bigint FROM test_train_view) AS train_samples,
	(SELECT COUNT(*)::bigint FROM test_test_view) AS test_samples,
	NULL::numeric AS final_inertia;

-- Store results in test_metrics table (minimal for direct clustering functions)
INSERT INTO test_metrics (
	test_name, algorithm, model_id, train_samples, test_samples,
	updated_at
)
SELECT 
	'016_minibatch_kmeans_basic',
	'minibatch_kmeans',
	NULL::integer,
	(SELECT COUNT(*)::bigint FROM test_train_view),
	(SELECT COUNT(*)::bigint FROM test_test_view),
	CURRENT_TIMESTAMP
ON CONFLICT (test_name) DO UPDATE SET
	algorithm = EXCLUDED.algorithm,
	train_samples = EXCLUDED.train_samples,
	test_samples = EXCLUDED.test_samples,
	updated_at = CURRENT_TIMESTAMP;

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
	CASE
		WHEN ts.setting_value = 'gpu' THEN 'GPU Training ✓'
		WHEN ts.setting_value = 'cpu' OR ts.setting_value IS NULL THEN 'CPU Training'
		WHEN current_setting('neurondb.compute_mode', true) = 'gpu' THEN 'GPU Training ✓'
		ELSE 'CPU Training (default)'
	END AS training_status,
	tm.updated_at AS test_completed_at
FROM test_metrics tm
LEFT JOIN test_settings ts ON ts.setting_key = 'gpu_mode'
WHERE tm.test_name = '016_minibatch_kmeans_basic';

\echo ''
\echo 'Test completed successfully'
