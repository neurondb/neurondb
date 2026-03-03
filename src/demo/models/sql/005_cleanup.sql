-- ============================================================================
-- NeuronDB Model Catalog Demo - Cleanup
-- ============================================================================
\set ON_ERROR_STOP on
\echo '=========================================='
\echo 'STEP 5: Cleanup Demo Artifacts'
\echo '=========================================='

SET search_path TO neurondb, public;

\echo ''
\echo 'Removing demo model catalog rows'
DELETE FROM neurondb.model_events
WHERE version_id IN (
    SELECT version_id FROM neurondb.model_versions WHERE model_id IN (
        SELECT model_id FROM neurondb.models WHERE model_name LIKE 'demo_%'
    )
);
DELETE FROM neurondb.model_versions
WHERE model_id IN (
    SELECT model_id FROM neurondb.models WHERE model_name LIKE 'demo_%'
);
DELETE FROM neurondb.models
WHERE model_name LIKE 'demo_%';

\echo ''
\echo 'Verifying catalog is clean'
SELECT COUNT(*) AS remaining_models
FROM neurondb.models
WHERE model_name LIKE 'demo_%';

\echo ''
\echo 'Cleaning up temporary filesystem artifacts'
\! rm -rf /tmp/neurondb_models_demo

\echo ''
\echo 'STEP 5 complete -- Demo environment reset'
\echo '=========================================='
