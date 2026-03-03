-- 023_pca_basic.sql
-- Basic test for PCA (Principal Component Analysis) with GPU acceleration

\set ON_ERROR_STOP on
\timing on
\pset footer off
\pset pager off
\pset tuples_only off
SET client_min_messages TO WARNING;

/* Step 1: Verify prerequisites and create test data */

DROP TABLE IF EXISTS pca_data;
CREATE TABLE pca_data (
	id serial PRIMARY KEY,
	features vector
);

-- Create sample data with vector features
INSERT INTO pca_data (features)
SELECT 
	array_to_vector_float8(ARRAY[
		x::double precision, 
		(x*1.5 + random()*0.1)::double precision, 
		(x*2.0 + random()*0.1)::double precision
	]) AS features
FROM generate_series(1, 100) AS x;

SELECT COUNT(*)::bigint AS data_rows FROM pca_data;

/* Step 2: Configure GPU */
\echo 'Step 2: Configuring GPU acceleration...'

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
		-- compute_mode is an integer: 0=cpu, 1=gpu, 2=auto
		IF current_gpu_enabled != '1' AND current_gpu_enabled != '2' THEN
			RAISE WARNING 'GPU mode expected but neurondb.compute_mode = % (expected: 1 or 2)', current_gpu_enabled;
		END IF;
	ELSE
		-- Verify GPU is disabled (should be set by test runner)
		IF current_gpu_enabled != '0' THEN
			RAISE WARNING 'CPU mode expected but neurondb.compute_mode = % (expected: 0)', current_gpu_enabled;
		END IF;
	END IF;
END $$;

\echo '=========================================================================='
\echo '=========================================================================='

/* Step 3: Test PCA transformation */

-- Test PCA transformation with proper validation
-- Note: reduce_pca returns real[][] (array of real arrays), not a table
-- We need to use proper array access for nested arrays
DO $$
DECLARE
	result vector[];
	original_count integer;
	original_dims integer;
	transformed_dims integer;
	result_count integer;
	pca_result real[][];  -- This is real[][] (array of arrays)
	vec_array vector;
	i integer;
	j integer;
	arr_elem real[];
BEGIN
	-- Get original data info
	SELECT COUNT(*) INTO original_count FROM pca_data;
	SELECT vector_dims(features) INTO original_dims FROM pca_data LIMIT 1;
	
	-- Verify we have data
	IF original_count IS NULL OR original_count = 0 THEN
		RAISE EXCEPTION 'No data in pca_data table';
	END IF;
	
	IF original_dims IS NULL OR original_dims = 0 THEN
		RAISE EXCEPTION 'Invalid feature dimensions: %', original_dims;
	END IF;
	
	-- Perform PCA transformation using reduce_pca
	-- reduce_pca returns real[][] (array of real arrays)
	SELECT reduce_pca('pca_data'::text, 'features'::text, 2) INTO pca_result;
	
	-- Validate result
	IF pca_result IS NULL OR array_length(pca_result, 1) IS NULL THEN
		RAISE EXCEPTION 'reduce_pca returned NULL or empty';
	END IF;
	
	result_count := array_length(pca_result, 1);
	IF result_count != original_count THEN
		RAISE EXCEPTION 'PCA result count (%) does not match input count (%)', result_count, original_count;
	END IF;
	
	-- Convert 2D array rows to vector[]
	-- pca_result is a 2D array real[][], so we need to extract rows as 1D arrays
	-- Use unnest to flatten the 2D slice to 1D, then aggregate back to array
	FOR i IN 1..result_count LOOP
		-- Extract the i-th row as a 1D array by unnesting the slice
		-- pca_result[i:i][1:2] gives a 1x2 2D slice, unnest flattens it to 1D
		SELECT array_agg(val ORDER BY ord) INTO arr_elem
		FROM unnest(pca_result[i:i][1:2]) WITH ORDINALITY AS t(val, ord);
		
		IF arr_elem IS NULL OR array_length(arr_elem, 1) IS NULL THEN
			RAISE EXCEPTION 'PCA result element % is NULL or invalid', i;
		END IF;
		
		-- Convert real[] to vector using array_to_vector
		-- Note: array_to_vector expects a one-dimensional real[] array
		vec_array := array_to_vector(arr_elem);
		
		IF vec_array IS NULL THEN
			RAISE EXCEPTION 'Failed to convert PCA result element % to vector', i;
		END IF;
		
		-- Append to result array
		result := array_append(result, vec_array);
	END LOOP;
	
	-- Validate result
	IF result IS NULL THEN
		RAISE EXCEPTION 'PCA transform returned NULL';
	END IF;
	
	result_count := array_length(result, 1);
	IF result_count IS NULL OR result_count = 0 THEN
		RAISE EXCEPTION 'PCA transform returned empty array';
	END IF;
	
	-- Verify transformed dimensions
	SELECT vector_dims(result[1]) INTO transformed_dims;
	IF transformed_dims IS NULL OR transformed_dims != 2 THEN
		RAISE EXCEPTION 'PCA transformed dimensions (%) should be 2', transformed_dims;
	END IF;
	
	-- Verify n_components is less than or equal to original dimensions
	IF 2 > original_dims THEN
		RAISE EXCEPTION 'n_components (2) cannot be greater than original dimensions (%)', original_dims;
	END IF;
	
	-- Verify all vectors in result have correct dimensions
	FOR i IN 1..LEAST(result_count, 10) LOOP
		IF vector_dims(result[i]) != 2 THEN
			RAISE EXCEPTION 'Result vector at index % has wrong dimensions: % (expected 2)', i, vector_dims(result[i]);
		END IF;
	END LOOP;
END $$;

DROP TABLE IF EXISTS pca_data;

\echo 'Test completed successfully'
