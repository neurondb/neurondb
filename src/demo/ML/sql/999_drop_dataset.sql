\set ON_ERROR_STOP on
\set QUIET on
-- ============================================================================
-- NeuronDB Fraud Detection Demo - Cleanup
-- Drops dataset and projects
-- ============================================================================

\echo '=========================================================================='
\echo '|       NeuronDB - Cleanup Fraud Detection Demo                           |'
\echo '=========================================================================='
\echo ''

\echo 'Cleaning up fraud detection demo...'
\echo ''

-- Drop the transactions table and cascading views
DROP TABLE IF EXISTS transactions CASCADE;

-- Drop ML projects (this will cascade to models)
DO $$
DECLARE
    proj_id integer;
BEGIN
    FOR proj_id IN 
        SELECT project_id FROM neurondb.ml_projects WHERE project_name LIKE 'fraud_%'
    LOOP
        PERFORM neurondb_delete_ml_project(
            (SELECT project_name FROM neurondb.ml_projects WHERE project_id = proj_id)
        );
    END LOOP;
END $$;

-- Drop helper functions
DROP FUNCTION IF EXISTS sigmoid(float);
DROP FUNCTION IF EXISTS gmm_to_clusters(float8[][]);

\echo 'Cleanup complete!'
\echo ''
\echo 'All fraud detection data and ML projects removed.'
\echo ''

