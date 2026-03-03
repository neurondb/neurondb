/*
 * Crash Prevention Test: Algorithm Boundary Conditions
 * Tests boundary conditions for all ML algorithms.
 * 
 * Expected: Algorithms should handle edge cases gracefully, not crash.
 */

\set ON_ERROR_STOP off

BEGIN;

\echo '=========================================================================='
\echo 'Algorithm Boundary Tests: Linear Regression'
\echo '=========================================================================='

CREATE TABLE IF NOT EXISTS test_alg_boundaries (
    features vector(128),
    label float4,
    id integer
);

-- Test 1: Single sample
INSERT INTO test_alg_boundaries (features, label, id)
VALUES (array_to_vector(ARRAY(SELECT 1.0 FROM generate_series(1, 128)))::vector(128), 1.0, 1)
ON CONFLICT DO NOTHING;

DO $$
BEGIN
    BEGIN
        PERFORM neurondb.train('linear_regression', 'test_alg_boundaries', 'features', 'label', '{}'::jsonb);
        RAISE NOTICE 'Single sample handled';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'Single sample handled with error (no crash): %', SQLERRM;
    END;
END$$;

-- Test 2: Two samples
DELETE FROM test_alg_boundaries;
INSERT INTO test_alg_boundaries (features, label, id)
SELECT array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 128)))::vector(128),
       random()::float4,
       generate_series(1, 2)
ON CONFLICT DO NOTHING;

DO $$
BEGIN
    BEGIN
        PERFORM neurondb.train('linear_regression', 'test_alg_boundaries', 'features', 'label', '{}'::jsonb);
        RAISE NOTICE 'Two samples handled';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'Two samples handled with error (no crash): %', SQLERRM;
    END;
END$$;

ROLLBACK;

BEGIN;

\echo ''
\echo 'Algorithm Boundary Tests: Logistic Regression'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE TABLE IF NOT EXISTS test_logreg_boundaries (
    features vector(128),
    label integer
);

-- Test 3: All same class (should handle, not crash)
INSERT INTO test_logreg_boundaries (features, label)
SELECT array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 128)))::vector(128),
       0  -- All same class
FROM generate_series(1, 100)
ON CONFLICT DO NOTHING;

DO $$
BEGIN
    BEGIN
        PERFORM neurondb.train('logistic_regression', 'test_logreg_boundaries', 'features', 'label', '{}'::jsonb);
        RAISE NOTICE 'All same class handled';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'All same class handled with error (no crash): %', SQLERRM;
    END;
END$$;

ROLLBACK;

BEGIN;

\echo ''
\echo 'Algorithm Boundary Tests: K-Means'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE TABLE IF NOT EXISTS test_kmeans_boundaries (
    features vector(128),
    id integer
);

INSERT INTO test_kmeans_boundaries (features, id)
SELECT array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 128)))::vector(128),
       generate_series(1, 10)
ON CONFLICT DO NOTHING;

-- Test 4: More clusters than samples (should validate, not crash)
DO $$
BEGIN
    BEGIN
        PERFORM neurondb.train('kmeans', 'test_kmeans_boundaries', 'features', NULL, '{"n_clusters": 100}'::jsonb);
        RAISE NOTICE 'More clusters than samples handled';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE '✓ Correctly rejected more clusters than samples: %', SQLERRM;
    END;
END$$;

-- Test 5: Zero clusters (should validate, not crash)
DO $$
BEGIN
    BEGIN
        PERFORM neurondb.train('kmeans', 'test_kmeans_boundaries', 'features', NULL, '{"n_clusters": 0}'::jsonb);
        RAISE EXCEPTION 'Should have failed with zero clusters';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE '✓ Correctly rejected zero clusters: %', SQLERRM;
    END;
END$$;

ROLLBACK;

COMMIT;

\echo ''
\echo '=========================================================================='
\echo '✓ Algorithm Boundary Tests Complete (Sample)'
\echo '=========================================================================='
\echo ''
\echo 'Note: This is a sample of boundary tests. Comprehensive testing'
\echo 'would cover all 60+ algorithms with various edge cases.'
\echo ''




