-- 023_pca_negative.sql
-- Negative test for PCA (Principal Component Analysis)
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

-- Create test data
DROP TABLE IF EXISTS pca_test_data;
CREATE TEMP TABLE pca_test_data AS
SELECT features[1:5]::vector(5) AS feat_5d
FROM test_train_view
LIMIT 100;

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.transform_pca('nonexistent_table', ARRAY['feat_5d'], 2);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.transform_pca('pca_test_data', ARRAY['invalid_col'], 2);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.transform_pca('pca_test_data', ARRAY['feat_5d'], 0);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.transform_pca('pca_test_data', ARRAY['feat_5d'], -1);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.transform_pca(NULL, ARRAY['feat_5d'], 2);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.transform_pca('pca_test_data', NULL, 2);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.transform_pca('pca_test_data', ARRAY[]::text[], 2);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.transform_pca('pca_test_data', ARRAY['feat_5d'], NULL);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.transform_pca('pca_test_data', ARRAY['feat_5d'], 10);

DROP TABLE IF EXISTS pca_test_data;

\echo ''

\echo 'Test completed successfully'
