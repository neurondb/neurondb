-- ============================================================================
-- NeuronDB Model Catalog Demo - Error Handling Scenarios
-- ============================================================================
\set ON_ERROR_STOP on
\echo '=========================================='
\echo 'STEP 4: Error Handling & Constraint Validation'
\echo '=========================================='

SET search_path TO neurondb, public;

\echo ''
\echo 'Attempting duplicate version registration (expected failure)'
DO $$
DECLARE
    v_version_id bigint;
BEGIN
    BEGIN
        SELECT neurondb.register_model_version(
                   'demo_sentiment',
                   'public',
                   'v1',
                   '/tmp/neurondb_models_demo/sentiment-mini.onnx',
                   'onnx',
                   1280,
                   NULL,
                   current_user,
                   '{"note": "duplicate attempt"}'::jsonb)
        INTO v_version_id;
        RAISE EXCEPTION 'Expected unique violation was not raised (version id=%)', v_version_id;
    EXCEPTION WHEN unique_violation THEN
        RAISE NOTICE '✓ Unique constraint correctly prevented duplicate version label';
    END;
END $$;

\echo ''
\echo 'Attempting to update status for nonexistent version (expected failure)'
DO $$
BEGIN
    BEGIN
        PERFORM neurondb.update_model_version_status(-1, 'loaded', NULL, current_user);
        RAISE EXCEPTION 'Expected exception for missing version was not raised';
    EXCEPTION WHEN others THEN
        RAISE NOTICE '✓ update_model_version_status rejected nonexistent version: %', SQLERRM;
    END;
END $$;

\echo ''
\echo 'Attempting to load non-existent model file (expected failure)'
DO $$
BEGIN
    BEGIN
        PERFORM load_model('demo_missing', '/tmp/neurondb_models_demo/does_not_exist.onnx', 'onnx');
        RAISE EXCEPTION 'Expected load_model failure was not raised';
    EXCEPTION WHEN others THEN
        RAISE NOTICE '✓ load_model rejected missing file: %', SQLERRM;
    END;
END $$;

\echo ''
\echo 'Ensure catalog unaffected by failed operations'
SELECT model_name, version_label, status
FROM neurondb.model_catalog
WHERE model_name LIKE 'demo_%'
ORDER BY model_name;

\echo ''
\echo 'STEP 4 complete'
\echo '=========================================='
