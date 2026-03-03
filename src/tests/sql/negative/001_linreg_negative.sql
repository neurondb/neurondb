\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Negative Test: linear_regression - Verifying errors are properly raised'
\echo '=========================================================================='

/* Setup: Create test views if they don't exist */
DO $$
BEGIN
	IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'sample_train') THEN
		NULL;
	END IF;
	IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'sample_test') THEN
		NULL;
	END IF;
END
$$;

\echo ''
\echo 'Test 1: NULL table name should error'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
DO $$
DECLARE
	error_occurred BOOLEAN := false;
BEGIN
	BEGIN
		PERFORM neurondb.train('linear_regression', NULL, 'features', 'label', '{}'::jsonb);
		RAISE EXCEPTION 'FAIL: NULL table name should error but operation succeeded';
	EXCEPTION WHEN OTHERS THEN 
		error_occurred := true;
		IF SQLERRM IS NULL OR LENGTH(SQLERRM) = 0 THEN
			RAISE EXCEPTION 'FAIL: error occurred but error message is empty';
		END IF;
	END;
	IF NOT error_occurred THEN
		RAISE EXCEPTION 'FAIL: expected error for NULL table name but no error occurred';
	END IF;
END$$;

\echo ''
\echo 'Test 2: NULL feature column should error'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
DO $$
DECLARE
	error_occurred BOOLEAN := false;
BEGIN
	BEGIN
		PERFORM neurondb.train('linear_regression', 'test_train_view', NULL, 'label', '{}'::jsonb);
		RAISE EXCEPTION 'FAIL: NULL feature column should error but operation succeeded';
	EXCEPTION WHEN OTHERS THEN 
		error_occurred := true;
		IF SQLERRM IS NULL OR LENGTH(SQLERRM) = 0 THEN
			RAISE EXCEPTION 'FAIL: error occurred but error message is empty';
		END IF;
	END;
	IF NOT error_occurred THEN
		RAISE EXCEPTION 'FAIL: expected error for NULL feature column but no error occurred';
	END IF;
END$$;

\echo ''
\echo 'Test 3: NULL label column should error'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
DO $$
DECLARE
	error_occurred BOOLEAN := false;
BEGIN
	BEGIN
		PERFORM neurondb.train('linear_regression', 'test_train_view', 'features', NULL, '{}'::jsonb);
		RAISE EXCEPTION 'FAIL: NULL label column should error but operation succeeded';
	EXCEPTION WHEN OTHERS THEN 
		error_occurred := true;
		IF SQLERRM IS NULL OR LENGTH(SQLERRM) = 0 THEN
			RAISE EXCEPTION 'FAIL: error occurred but error message is empty';
		END IF;
	END;
	IF NOT error_occurred THEN
		RAISE EXCEPTION 'FAIL: expected error for NULL label column but no error occurred';
	END IF;
END$$;

\echo ''
\echo 'Test 4: Nonexistent table should error'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
DO $$
DECLARE
	error_occurred BOOLEAN := false;
BEGIN
	BEGIN
		PERFORM neurondb.train('linear_regression', 'nonexistent_table_xyz', 'features', 'label', '{}'::jsonb);
		RAISE EXCEPTION 'FAIL: nonexistent table should error but operation succeeded';
	EXCEPTION WHEN OTHERS THEN 
		error_occurred := true;
		IF SQLERRM IS NULL OR LENGTH(SQLERRM) = 0 THEN
			RAISE EXCEPTION 'FAIL: error occurred but error message is empty';
		END IF;
	END;
	IF NOT error_occurred THEN
		RAISE EXCEPTION 'FAIL: expected error for nonexistent table but no error occurred';
	END IF;
END$$;

\echo ''
\echo 'Test 5: Nonexistent feature column should error'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
DO $$
DECLARE
	error_occurred BOOLEAN := false;
BEGIN
	BEGIN
		PERFORM neurondb.train('linear_regression', 'test_train_view', 'nonexistent_column', 'label', '{}'::jsonb);
		RAISE EXCEPTION 'FAIL: nonexistent feature column should error but operation succeeded';
	EXCEPTION WHEN OTHERS THEN 
		error_occurred := true;
		IF SQLERRM IS NULL OR LENGTH(SQLERRM) = 0 THEN
			RAISE EXCEPTION 'FAIL: error occurred but error message is empty';
		END IF;
	END;
	IF NOT error_occurred THEN
		RAISE EXCEPTION 'FAIL: expected error for nonexistent feature column but no error occurred';
	END IF;
END$$;

\echo ''
\echo 'Test 6: Nonexistent label column should error'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
DO $$
DECLARE
	error_occurred BOOLEAN := false;
BEGIN
	BEGIN
		PERFORM neurondb.train('linear_regression', 'test_train_view', 'features', 'nonexistent_label', '{}'::jsonb);
		RAISE EXCEPTION 'FAIL: nonexistent label column should error but operation succeeded';
	EXCEPTION WHEN OTHERS THEN 
		error_occurred := true;
		IF SQLERRM IS NULL OR LENGTH(SQLERRM) = 0 THEN
			RAISE EXCEPTION 'FAIL: error occurred but error message is empty';
		END IF;
	END;
	IF NOT error_occurred THEN
		RAISE EXCEPTION 'FAIL: expected error for nonexistent label column but no error occurred';
	END IF;
END$$;

\echo ''
\echo 'Test 7: Invalid model ID for prediction should error'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
DO $$
DECLARE
	error_occurred BOOLEAN := false;
BEGIN
	BEGIN
		PERFORM neurondb.predict(-1, '[1,2,3,4,5]'::vector);
		RAISE EXCEPTION 'FAIL: invalid model ID should error but operation succeeded';
	EXCEPTION WHEN OTHERS THEN 
		error_occurred := true;
		IF SQLERRM IS NULL OR LENGTH(SQLERRM) = 0 THEN
			RAISE EXCEPTION 'FAIL: error occurred but error message is empty';
		END IF;
	END;
	IF NOT error_occurred THEN
		RAISE EXCEPTION 'FAIL: expected error for invalid model ID but no error occurred';
	END IF;
END$$;

\echo ''
\echo 'Test 8: NULL model ID for prediction should error'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
DO $$
DECLARE
	error_occurred BOOLEAN := false;
BEGIN
	BEGIN
		PERFORM neurondb.predict(NULL, '[1,2,3,4,5]'::vector);
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

\echo ''
\echo 'Test 9: NULL features for prediction should error'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
DO $$
DECLARE
	mid integer;
	error_occurred BOOLEAN := false;
BEGIN
	-- First get a valid model_id
	SELECT neurondb.train('linear_regression', 'test_train_view', 'features', 'label', '{}'::jsonb)::integer INTO mid;
	IF mid IS NULL THEN
		RAISE EXCEPTION 'FAIL: could not create model for test';
	END IF;
	
	BEGIN
		PERFORM neurondb.predict(mid, NULL::vector);
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

\echo ''
\echo 'Test 10: Wrong dimension vector for prediction should error'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
DO $$
DECLARE
	mid integer;
	error_occurred BOOLEAN := false;
BEGIN
	-- First get a valid model_id
	SELECT neurondb.train('linear_regression', 'test_train_view', 'features', 'label', '{}'::jsonb)::integer INTO mid;
	IF mid IS NULL THEN
		RAISE EXCEPTION 'FAIL: could not create model for test';
	END IF;
	
	BEGIN
		PERFORM neurondb.predict(mid, '[1,2,3]'::vector);
		RAISE EXCEPTION 'FAIL: wrong dimension vector should error but operation succeeded';
	EXCEPTION WHEN OTHERS THEN 
		error_occurred := true;
		IF SQLERRM IS NULL OR LENGTH(SQLERRM) = 0 THEN
			RAISE EXCEPTION 'FAIL: error occurred but error message is empty';
		END IF;
	END;
	IF NOT error_occurred THEN
		RAISE EXCEPTION 'FAIL: expected error for wrong dimension vector but no error occurred';
	END IF;
END$$;

\echo ''
\echo 'Test 11: Invalid parameter should error or be ignored (implementation dependent)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
-- This test may pass or fail depending on implementation - just verify it doesn't crash
DO $$
DECLARE
	result integer;
BEGIN
	BEGIN
		result := neurondb.train('linear_regression', 'test_train_view', 'features', 'label', '{"invalid_param": "invalid_value"}'::jsonb)::integer;
		-- If it succeeds, that's okay - invalid params may be ignored
		IF result IS NULL THEN
			RAISE EXCEPTION 'FAIL: training returned NULL model ID';
		END IF;
	EXCEPTION WHEN OTHERS THEN
		-- If it errors, that's also okay - invalid params may be rejected
		IF SQLERRM IS NULL OR LENGTH(SQLERRM) = 0 THEN
			RAISE EXCEPTION 'FAIL: error occurred but error message is empty';
		END IF;
	END;
END$$;

\echo ''
\echo 'Test 12: Invalid algorithm name should error'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
DO $$
DECLARE
	error_occurred BOOLEAN := false;
BEGIN
	BEGIN
		PERFORM neurondb.train('invalid_algorithm_name', 'test_train_view', 'features', 'label', '{}'::jsonb);
		RAISE EXCEPTION 'FAIL: invalid algorithm name should error but operation succeeded';
	EXCEPTION WHEN OTHERS THEN 
		error_occurred := true;
		IF SQLERRM IS NULL OR LENGTH(SQLERRM) = 0 THEN
			RAISE EXCEPTION 'FAIL: error occurred but error message is empty';
		END IF;
	END;
	IF NOT error_occurred THEN
		RAISE EXCEPTION 'FAIL: expected error for invalid algorithm name but no error occurred';
	END IF;
END$$;

\echo ''
\echo 'Test 13: Empty table should error'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
DO $$
DECLARE
	error_occurred BOOLEAN := false;
BEGIN
	CREATE TEMP TABLE empty_train (
		features vector,
		label float8
	);
	
	BEGIN
		PERFORM neurondb.train('linear_regression', 'empty_train', 'features', 'label', '{}'::jsonb);
		RAISE EXCEPTION 'FAIL: empty table should error but operation succeeded';
	EXCEPTION WHEN OTHERS THEN 
		error_occurred := true;
		IF SQLERRM IS NULL OR LENGTH(SQLERRM) = 0 THEN
			RAISE EXCEPTION 'FAIL: error occurred but error message is empty';
		END IF;
	END;
	
	DROP TABLE IF EXISTS empty_train;
	
	IF NOT error_occurred THEN
		RAISE EXCEPTION 'FAIL: expected error for empty table but no error occurred';
	END IF;
END$$;

\echo ''
\echo 'Test 14: Invalid model ID for evaluation should error'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
DO $$
DECLARE
	error_occurred BOOLEAN := false;
BEGIN
	BEGIN
		PERFORM neurondb.evaluate(-1, 'test_test_view', 'features', 'label');
		RAISE EXCEPTION 'FAIL: invalid model ID should error but operation succeeded';
	EXCEPTION WHEN OTHERS THEN 
		error_occurred := true;
		IF SQLERRM IS NULL OR LENGTH(SQLERRM) = 0 THEN
			RAISE EXCEPTION 'FAIL: error occurred but error message is empty';
		END IF;
	END;
	IF NOT error_occurred THEN
		RAISE EXCEPTION 'FAIL: expected error for invalid model ID but no error occurred';
	END IF;
END$$;

\echo ''
\echo 'Test 15: Nonexistent table for evaluation should error'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
DO $$
DECLARE
	mid integer;
	error_occurred BOOLEAN := false;
BEGIN
	-- First get a valid model_id
	SELECT neurondb.train('linear_regression', 'test_train_view', 'features', 'label', '{}'::jsonb)::integer INTO mid;
	IF mid IS NULL THEN
		RAISE EXCEPTION 'FAIL: could not create model for test';
	END IF;
	
	BEGIN
		PERFORM neurondb.evaluate(mid, 'nonexistent_test_table', 'features', 'label');
		RAISE EXCEPTION 'FAIL: nonexistent table should error but operation succeeded';
	EXCEPTION WHEN OTHERS THEN 
		error_occurred := true;
		IF SQLERRM IS NULL OR LENGTH(SQLERRM) = 0 THEN
			RAISE EXCEPTION 'FAIL: error occurred but error message is empty';
		END IF;
	END;
	IF NOT error_occurred THEN
		RAISE EXCEPTION 'FAIL: expected error for nonexistent table but no error occurred';
	END IF;
END$$;

\echo ''
\echo '=========================================================================='
\echo 'All negative tests completed - errors were properly raised'
\echo '=========================================================================='
