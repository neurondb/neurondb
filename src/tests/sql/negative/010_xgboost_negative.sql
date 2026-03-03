-- 019_xgboost_negative.sql
-- Negative test for xgboost
-- All possible negative tests with 1000 rows only

\timing on
\pset footer off
\pset pager off
\pset tuples_only off
\set ON_ERROR_STOP off
SET client_min_messages TO WARNING;

\echo '=========================================================================='
\echo '=========================================================================='

/* Setup: Create test views if they don't exist */
DO $$
BEGIN
	IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'sample_train') THEN
		
		
	END IF;
	IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'sample_test') THEN
		
		
	END IF;
END
$$;

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train('xgboost', NULL, 'features', 'label', '{}'::jsonb);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train('xgboost', 'test_train_view', NULL, 'label', '{}'::jsonb);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train('xgboost', 'test_train_view', 'features', NULL, '{}'::jsonb);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train('xgboost', 'nonexistent_table', 'features', 'label', '{}'::jsonb);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train('xgboost', 'test_train_view', 'invalid_feature', 'label', '{}'::jsonb);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train('xgboost', 'test_train_view', 'features', 'invalid_label', '{}'::jsonb);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train('xgboost', 'test_train_view', 'features', 'label', '{"n_estimators":-1}'::jsonb);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.predict(NULL, array_to_vector_float8(ARRAY[1.0, 2.0]::double precision[]));

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.predict(-1, array_to_vector_float8(ARRAY[1.0, 2.0]::double precision[]));

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	model_id int;
BEGIN
	SELECT m.model_id INTO model_id FROM neurondb.ml_models m WHERE m.algorithm::text = 'xgboost' ORDER BY m.model_id DESC LIMIT 1;
	IF model_id IS NOT NULL THEN
		SELECT neurondb.predict(model_id, NULL);
	END IF;
END $$;

\echo ''

\echo 'Test completed successfully'
