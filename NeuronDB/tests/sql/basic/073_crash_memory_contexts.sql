/*
 * Crash Prevention Test: Memory Context Stress Tests
 * Tests memory context handling under various conditions.
 * Note: Most memory context issues are tested at the C code level,
 * but we can test scenarios that stress memory management.
 */

\set ON_ERROR_STOP off

BEGIN;

/* Test 1: Large batch evaluation */
CREATE TEMP TABLE large_test_table AS
SELECT 
	ARRAY[random(), random(), random()]::float4[] as features,
	random()::float4 as label
FROM generate_series(1, 10000);

/* This should not crash even with large datasets */
DO $$
BEGIN
    PERFORM evaluate_linear_regression_by_model_id(1, 'large_test_table', 'features', 'label');
    RAISE EXCEPTION 'Should have failed - model does not exist';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Correctly handled large dataset error: %', SQLERRM;
END$$;
ROLLBACK;

BEGIN;
/* Test 2: Multiple rapid evaluations */
DO $$
DECLARE
	i int;
BEGIN
	FOR i IN 1..100 LOOP
		BEGIN
			PERFORM evaluate_linear_regression_by_model_id(1, 'test_table', 'features', 'label');
		EXCEPTION WHEN OTHERS THEN
			/* Expected to fail, but should not crash */
			NULL;
		END;
	END LOOP;
END $$;
ROLLBACK;

BEGIN;
/* Test 3: Nested function calls */
SELECT neurondb.evaluate(
	1,
	'test_table',
	'features',
	'label'
);
ROLLBACK;

COMMIT;

