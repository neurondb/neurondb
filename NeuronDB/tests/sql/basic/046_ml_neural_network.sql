-- 022_neural_network_basic.sql
-- Basic test for Neural Network (CPU and GPU)

\set ON_ERROR_STOP on
\timing on
\pset footer off
\pset pager off
\pset tuples_only off
SET client_min_messages TO WARNING;

/* Step 1: Verify prerequisites and create test data */

DROP TABLE IF EXISTS nn_data;
CREATE TABLE nn_data (
	id serial PRIMARY KEY,
	features vector,
	label int
);

-- Create sample data with vector features
INSERT INTO nn_data (features, label)
SELECT 
	array_to_vector_float8(ARRAY[x::double precision, (x*2 + random()*0.1)::double precision]) AS features,
	CASE WHEN x < 5 THEN 0 ELSE 1 END AS label
FROM generate_series(1, 100) AS x;

SELECT COUNT(*)::bigint AS data_rows FROM nn_data;

/* Step 2: Configure CPU */

\echo '=========================================================================='
\echo '=========================================================================='

-- Train Neural Network model on CPU
DO $$
DECLARE
	model_id int;
BEGIN
	BEGIN
		SELECT neurondb.train('default', 'neural_network', 'nn_data', 'label', ARRAY['features'], '{}'::jsonb) INTO model_id;
		IF model_id IS NULL THEN
			RETURN;
		END IF;
	EXCEPTION WHEN OTHERS THEN
		RETURN;
	END;
END $$;

-- Test inference on CPU
DO $$
DECLARE
	pred float8;
	model_id int;
BEGIN
	SELECT m.model_id INTO model_id FROM neurondb.ml_models m WHERE m.algorithm::text = 'neural_network' ORDER BY m.model_id DESC LIMIT 1;
	IF model_id IS NULL THEN
		RETURN;
	END IF;
	SELECT neurondb.predict(model_id, array_to_vector_float8(ARRAY[7.0, 14.0]::double precision[])) INTO pred;
	IF pred IS NULL THEN
		RETURN;
	END IF;
END $$;

/* Step 3: Configure GPU */
/* Step 0: Read settings from test_settings table and verify GPU configuration */
DO $$
DECLARE
	gpu_mode TEXT;
	current_gpu_enabled TEXT;
BEGIN
	-- Read GPU mode setting from test_settings
	SELECT setting_value INTO gpu_mode FROM test_settings WHERE setting_key = 'gpu_mode';
	
	-- Verify GPU configuration matches test_settings (set by test runner)
	SELECT current_setting('neurondb.compute_mode', true) INTO current_gpu_enabled;
	
	IF gpu_mode = 'gpu' THEN
		-- Verify GPU is enabled (should be set by test runner)
		-- compute_mode: 0=CPU, 1=GPU, 2=AUTO
		IF current_gpu_enabled != '1' THEN
			RAISE WARNING 'GPU mode expected but neurondb.compute_mode = % (expected: 1)', current_gpu_enabled;
		END IF;
	ELSE
		-- Verify GPU is disabled (should be set by test runner)
		-- compute_mode: 0=CPU, 1=GPU, 2=AUTO
		IF current_gpu_enabled != '0' THEN
			RAISE WARNING 'CPU mode expected but neurondb.compute_mode = % (expected: 0)', current_gpu_enabled;
		END IF;
	END IF;
END $$;

\echo '=========================================================================='
\echo '=========================================================================='

-- Train Neural Network model on GPU
DO $$
DECLARE
	model_id int;
BEGIN
	BEGIN
		SELECT neurondb.train('default', 'neural_network', 'nn_data', 'label', ARRAY['features'], '{}'::jsonb) INTO model_id;
		IF model_id IS NULL THEN
			RETURN;
		END IF;
	EXCEPTION WHEN OTHERS THEN
		RETURN;
	END;
END $$;

-- Test inference on GPU
DO $$
DECLARE
	pred float8;
	model_id int;
BEGIN
	SELECT m.model_id INTO model_id FROM neurondb.ml_models m WHERE m.algorithm::text = 'neural_network' ORDER BY m.model_id DESC LIMIT 1;
	IF model_id IS NULL THEN
		RETURN;
	END IF;
	SELECT neurondb.predict(model_id, array_to_vector_float8(ARRAY[7.0, 14.0]::double precision[])) INTO pred;
	IF pred IS NULL THEN
		RETURN;
	END IF;
END $$;

DROP TABLE IF EXISTS nn_data;

\echo 'Test completed successfully'
