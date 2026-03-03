-- 005_dt_negative.sql
-- Negative test for decision_tree - Verifying errors are properly raised

SET client_min_messages TO WARNING;
\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Negative Test: decision_tree - Verifying errors are properly raised'
\echo '=========================================================================='

\echo ''
\echo 'Test 1: Invalid table should error'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
DO $$
DECLARE
	error_occurred BOOLEAN := false;
BEGIN
	BEGIN
		PERFORM neurondb.train('decision_tree', 'nonexistent_table', 'features', 'label', '{}'::jsonb);
		RAISE EXCEPTION 'FAIL: invalid table should error but operation succeeded';
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

\echo ''
\echo 'Test 2: Invalid column should error'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
DO $$
DECLARE
	error_occurred BOOLEAN := false;
BEGIN
	BEGIN
		PERFORM neurondb.train('decision_tree', 'test_train_view', 'invalid_col', 'label', '{}'::jsonb);
		RAISE EXCEPTION 'FAIL: invalid column should error but operation succeeded';
	EXCEPTION WHEN OTHERS THEN
		error_occurred := true;
		IF SQLERRM IS NULL OR LENGTH(SQLERRM) = 0 THEN
			RAISE EXCEPTION 'FAIL: error occurred but error message is empty';
		END IF;
	END;
	IF NOT error_occurred THEN
		RAISE EXCEPTION 'FAIL: expected error for invalid column but no error occurred';
	END IF;
END$$;

\echo ''
\echo '=========================================================================='
\echo 'All negative tests completed - errors were properly raised'
\echo '=========================================================================='
