-- 024_timeseries_basic.sql
-- Basic test for Time Series (CPU and GPU)

\set ON_ERROR_STOP on
\timing on
\pset footer off
\pset pager off
\pset tuples_only off
SET client_min_messages TO WARNING;

/* Step 1: Verify prerequisites and create test data */

DROP TABLE IF EXISTS ts_data;
CREATE TABLE ts_data (
	id serial PRIMARY KEY,
	features vector,
	label double precision
);

-- Create sample time series data
INSERT INTO ts_data (features, label)
SELECT 
	array_to_vector_float8(ARRAY[x::double precision]) AS features,
	(x::double precision + random()*0.1) AS label
FROM generate_series(1, 30) AS x;

SELECT COUNT(*)::bigint AS data_rows FROM ts_data;

/* Step 2: Configure CPU */

\echo '=========================================================================='
\echo '=========================================================================='

-- Train time series model on CPU
DO $$
DECLARE
	model_id int;
	data_count int;
BEGIN
	-- Verify we have training data
	SELECT COUNT(*) INTO data_count FROM ts_data;
	IF data_count IS NULL OR data_count = 0 THEN
		RAISE EXCEPTION 'No training data in ts_data table';
	END IF;
	
	-- Train the model
	SELECT neurondb.train('default', 'timeseries', 'ts_data', 'label', ARRAY['features'], '{}'::jsonb) INTO model_id;
	
	-- Verify model was created
	IF model_id IS NULL THEN
		RAISE EXCEPTION 'Model training failed: model_id is NULL';
	END IF;
	
	-- Verify model exists in ml_models table
	IF NOT EXISTS (SELECT 1 FROM neurondb.ml_models m WHERE m.model_id = model_id) THEN
		RAISE EXCEPTION 'Model % not found in ml_models table', model_id;
	END IF;
END $$;

-- Test inference on CPU
DO $$
DECLARE
	pred float8;
	model_id int;
	test_input vector;
BEGIN
	-- Get the trained model
	SELECT m.model_id INTO model_id FROM neurondb.ml_models m WHERE m.algorithm::text = 'timeseries' ORDER BY m.model_id DESC LIMIT 1;
	
	IF model_id IS NULL THEN
		RAISE EXCEPTION 'No trained timeseries model found';
	END IF;
	
	-- Create test input
	test_input := array_to_vector_float8(ARRAY[31::double precision]);
	
	-- Make prediction
	SELECT neurondb.predict(model_id, test_input) INTO pred;
	
	-- Verify prediction is not NULL
	IF pred IS NULL THEN
		RAISE EXCEPTION 'Prediction returned NULL for model_id %', model_id;
	END IF;
	
	-- Verify prediction is a valid number (not NaN or infinite)
	IF pred != pred OR pred = 'Infinity'::float8 OR pred = '-Infinity'::float8 THEN
		RAISE EXCEPTION 'Prediction is invalid (NaN or infinite): %', pred;
	END IF;
	
	-- Verify prediction is within reasonable range (for time series with values around 1-30, prediction should be reasonable)
	-- Allow some flexibility but check it's not completely out of range
	IF pred < -1000 OR pred > 1000 THEN
		RAISE WARNING 'Prediction value seems out of range: % (expected values around 1-30)', pred;
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
	SELECT current_setting('neurondb.gpu_enabled', true) INTO current_gpu_enabled;
	
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
\echo '=========================================================================='

-- Train time series model on GPU
DO $$
DECLARE
	model_id int;
	data_count int;
	gpu_model_id int;
BEGIN
	-- Verify we have training data
	SELECT COUNT(*) INTO data_count FROM ts_data;
	IF data_count IS NULL OR data_count = 0 THEN
		RAISE EXCEPTION 'No training data in ts_data table';
	END IF;
	
	-- Get previous model count to verify new model is created
	SELECT COUNT(*) INTO gpu_model_id FROM neurondb.ml_models WHERE algorithm::text = 'timeseries';
	
	-- Train the model
	SELECT neurondb.train('default', 'timeseries', 'ts_data', 'label', ARRAY['features'], '{}'::jsonb) INTO model_id;
	
	-- Verify model was created
	IF model_id IS NULL THEN
		RAISE EXCEPTION 'GPU model training failed: model_id is NULL';
	END IF;
	
	-- Verify model exists in ml_models table
	IF NOT EXISTS (SELECT 1 FROM neurondb.ml_models m WHERE m.model_id = model_id) THEN
		RAISE EXCEPTION 'GPU model % not found in ml_models table', model_id;
	END IF;
	
	-- Verify a new model was created (count should increase)
	IF (SELECT COUNT(*) FROM neurondb.ml_models WHERE algorithm::text = 'timeseries') <= gpu_model_id THEN
		RAISE WARNING 'No new model was created during GPU training';
	END IF;
END $$;

-- Test inference on GPU
DO $$
DECLARE
	pred float8;
	model_id int;
	test_input vector;
BEGIN
	-- Get the most recently trained model
	SELECT m.model_id INTO model_id FROM neurondb.ml_models m WHERE m.algorithm::text = 'timeseries' ORDER BY m.model_id DESC LIMIT 1;
	
	IF model_id IS NULL THEN
		RAISE EXCEPTION 'No trained timeseries model found for GPU inference';
	END IF;
	
	-- Create test input (different from CPU test)
	test_input := array_to_vector_float8(ARRAY[32::double precision]);
	
	-- Make prediction
	SELECT neurondb.predict(model_id, test_input) INTO pred;
	
	-- Verify prediction is not NULL
	IF pred IS NULL THEN
		RAISE EXCEPTION 'GPU prediction returned NULL for model_id %', model_id;
	END IF;
	
	-- Verify prediction is a valid number (not NaN or infinite)
	IF pred != pred OR pred = 'Infinity'::float8 OR pred = '-Infinity'::float8 THEN
		RAISE EXCEPTION 'GPU prediction is invalid (NaN or infinite): %', pred;
	END IF;
	
	-- Verify prediction is within reasonable range
	IF pred < -1000 OR pred > 1000 THEN
		RAISE WARNING 'GPU prediction value seems out of range: % (expected values around 1-30)', pred;
	END IF;
END $$;

DROP TABLE IF EXISTS ts_data;

\echo 'Test completed successfully'
