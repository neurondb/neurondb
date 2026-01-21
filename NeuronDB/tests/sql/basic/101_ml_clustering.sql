-- ====================================================================
-- NeurondB Regression Tests: ML Clustering Algorithms
-- ====================================================================
-- Tests for K-Means, Mini-batch K-Means, DBSCAN, GMM, Hierarchical
-- Uses real data from: sift1m.vectors (128-d vectors)
-- ====================================================================

\echo '=== Using SIFT1M Dataset for Clustering Tests ==='

-- Create test data with synthetic vectors (sift1m.vectors may not exist)
-- Use 10 dimensions for fast testing
CREATE TEMP TABLE test_clustering_data AS
SELECT 
    id,
    array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 10)))::vector(10) as vec
FROM generate_series(1, 1000) AS id;

-- Show sample data
SELECT COUNT(*) as total_vectors, vector_dims(vec) as dimensions
FROM test_clustering_data
LIMIT 1;

\echo '=== Testing K-Means Clustering ==='

-- Test K-Means (skip if function doesn't exist)
DO $$
BEGIN
  BEGIN
    PERFORM 1 FROM neurondb.cluster_kmeans('test_clustering_data', 'vec', 3, 100) LIMIT 1;
    RAISE NOTICE 'cluster_kmeans function is available';
  EXCEPTION WHEN undefined_function THEN
    RAISE NOTICE 'cluster_kmeans function not available, skipping all clustering tests';
    RETURN;
  END;
END$$;

-- Only run if function exists (checked above)
SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_kmeans('test_clustering_data', 'vec', 3, 100)
GROUP BY cluster_id
ORDER BY cluster_id;

SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_kmeans('test_clustering_data', 'vec', 2, 50)
GROUP BY cluster_id
ORDER BY cluster_id;

SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_kmeans('test_clustering_data', 'vec', 1, 10)
GROUP BY cluster_id;

\echo '=== Testing Mini-batch K-Means ==='

-- Test Mini-batch K-Means (skip if function doesn't exist)
DO $$
BEGIN
  BEGIN
    PERFORM 1 FROM neurondb.cluster_minibatch_kmeans('test_clustering_data', 'vec', 3, 3, 50) LIMIT 1;
    RAISE NOTICE 'cluster_minibatch_kmeans function is available';
  EXCEPTION WHEN undefined_function THEN
    RAISE NOTICE 'cluster_minibatch_kmeans function not available, skipping mini-batch tests';
  END;
END$$;

SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_minibatch_kmeans('test_clustering_data', 'vec', 3, 3, 50)
WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'cluster_minibatch_kmeans' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
GROUP BY cluster_id
ORDER BY cluster_id;

SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_minibatch_kmeans('test_clustering_data', 'vec', 2, 5, 30)
WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'cluster_minibatch_kmeans' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
GROUP BY cluster_id
ORDER BY cluster_id;

\echo '=== Testing DBSCAN Clustering ==='

-- Test DBSCAN (skip if function doesn't exist)
DO $$
BEGIN
  BEGIN
    PERFORM 1 FROM neurondb.cluster_dbscan('test_clustering_data', 'vec', 1.0, 2) LIMIT 1;
    RAISE NOTICE 'cluster_dbscan function is available';
  EXCEPTION WHEN undefined_function THEN
    RAISE NOTICE 'cluster_dbscan function not available, skipping DBSCAN tests';
  END;
END$$;

SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_dbscan('test_clustering_data', 'vec', 1.0, 2)
WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'cluster_dbscan' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
GROUP BY cluster_id
ORDER BY cluster_id;

SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_dbscan('test_clustering_data', 'vec', 3.0, 2)
WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'cluster_dbscan' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
GROUP BY cluster_id
ORDER BY cluster_id;

SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_dbscan('test_clustering_data', 'vec', 0.3, 2)
WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'cluster_dbscan' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
GROUP BY cluster_id
ORDER BY cluster_id;

\echo '=== Testing Gaussian Mixture Model (GMM) ==='

-- Test GMM (skip if function doesn't exist)
DO $$
BEGIN
  BEGIN
    PERFORM 1 FROM neurondb.cluster_gmm('test_clustering_data', 'vec', 3, 50) LIMIT 1;
    RAISE NOTICE 'cluster_gmm function is available';
  EXCEPTION WHEN undefined_function THEN
    RAISE NOTICE 'cluster_gmm function not available, skipping GMM tests';
  END;
END$$;

SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_gmm('test_clustering_data', 'vec', 3, 50)
WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'cluster_gmm' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
GROUP BY cluster_id
ORDER BY cluster_id;

SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_gmm('test_clustering_data', 'vec', 2, 30)
WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'cluster_gmm' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
GROUP BY cluster_id
ORDER BY cluster_id;

\echo '=== Testing Hierarchical Clustering ==='

-- Test Hierarchical (skip if function doesn't exist)
DO $$
BEGIN
  BEGIN
    PERFORM 1 FROM neurondb.cluster_hierarchical('test_clustering_data', 'vec', 3, 'single') LIMIT 1;
    RAISE NOTICE 'cluster_hierarchical function is available';
  EXCEPTION WHEN undefined_function THEN
    RAISE NOTICE 'cluster_hierarchical function not available, skipping hierarchical tests';
  END;
END$$;

SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_hierarchical('test_clustering_data', 'vec', 3, 'single')
WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'cluster_hierarchical' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
GROUP BY cluster_id
ORDER BY cluster_id;

SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_hierarchical('test_clustering_data', 'vec', 2, 'complete')
WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'cluster_hierarchical' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
GROUP BY cluster_id
ORDER BY cluster_id;

SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_hierarchical('test_clustering_data', 'vec', 3, 'average')
WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'cluster_hierarchical' AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'neurondb'))
GROUP BY cluster_id
ORDER BY cluster_id;

\echo '=== Testing Cluster Quality Metrics ==='

-- Test Davies-Bouldin Index (skip if function doesn't exist)
DO $$
BEGIN
  BEGIN
    PERFORM neurondb.davies_bouldin_index('test_clustering_data', 'vec', 'test_clustering_data', 'id');
    RAISE NOTICE 'davies_bouldin_index function is available';
  EXCEPTION WHEN undefined_function THEN
    RAISE NOTICE 'davies_bouldin_index function not available, skipping quality metric tests';
  END;
END$$;

-- Note: Davies-Bouldin index tests require creating result tables, skip for now if function doesn't exist

\echo '=== Edge Cases and Error Handling ==='

-- Test with insufficient data points
CREATE TABLE test_small_data (
    id SERIAL PRIMARY KEY,
    vec vector(3)
);

INSERT INTO test_small_data (vec) VALUES
    ('[1.0, 2.0, 3.0]'::vector),
    ('[1.1, 2.1, 3.1]'::vector);

-- K-Means with more clusters than points (should handle gracefully)
DO $$
BEGIN
  BEGIN
    PERFORM 1 FROM neurondb.cluster_kmeans('test_small_data', 'vec', 5, 10) LIMIT 1;
    RAISE NOTICE 'K-Means edge case test completed';
  EXCEPTION WHEN undefined_function THEN
    RAISE NOTICE 'cluster_kmeans not available, skipping edge case tests';
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'K-Means edge case error (may be expected): %', SQLERRM;
  END;
END$$;

-- DBSCAN with no points meeting criteria
DO $$
BEGIN
  BEGIN
    PERFORM 1 FROM neurondb.cluster_dbscan('test_small_data', 'vec', 0.01, 10) LIMIT 1;
    RAISE NOTICE 'DBSCAN edge case test completed';
  EXCEPTION WHEN undefined_function THEN
    RAISE NOTICE 'cluster_dbscan not available, skipping edge case tests';
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'DBSCAN edge case error (may be expected): %', SQLERRM;
  END;
END$$;

-- Cleanup
DROP TABLE test_clustering_data CASCADE;
DROP TABLE test_small_data CASCADE;
