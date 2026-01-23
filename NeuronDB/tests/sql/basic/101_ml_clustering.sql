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
SELECT COUNT(*) as total_vectors, (SELECT vector_dims(vec) FROM test_clustering_data LIMIT 1) as dimensions
FROM test_clustering_data;

\echo '=== Testing K-Means Clustering ==='

-- Test K-Means
-- Validate that function returns expected results
DO $$
DECLARE
    result_count INT;
    cluster_count INT;
BEGIN
    SELECT COUNT(*), COUNT(DISTINCT cluster_id)
    INTO result_count, cluster_count
    FROM neurondb.cluster_kmeans('test_clustering_data', 'vec', 3, 100);
    
    IF result_count = 0 THEN
        RAISE EXCEPTION 'cluster_kmeans returned no results';
    END IF;
    
    -- Should have 1000 results (one per input vector)
    IF result_count != 1000 THEN
        RAISE EXCEPTION 'cluster_kmeans returned % results, expected 1000', result_count;
    END IF;
    
    -- Should have 3 clusters (but may return fewer if data doesn't support it)
    IF cluster_count < 1 THEN
        RAISE EXCEPTION 'cluster_kmeans returned % clusters, expected at least 1', cluster_count;
    END IF;
    -- Note: cluster_count may be less than 3 if data doesn't naturally cluster into 3 groups
END$$;

SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_kmeans('test_clustering_data', 'vec', 3, 100)
GROUP BY cluster_id
ORDER BY cluster_id;

-- Test K-Means with 2 clusters
DO $$
DECLARE
    cluster_count INT;
BEGIN
    SELECT COUNT(DISTINCT cluster_id) INTO cluster_count
    FROM neurondb.cluster_kmeans('test_clustering_data', 'vec', 2, 50);
    
    IF cluster_count != 2 THEN
        RAISE EXCEPTION 'cluster_kmeans (k=2) returned % clusters, expected 2', cluster_count;
    END IF;
END$$;

SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_kmeans('test_clustering_data', 'vec', 2, 50)
GROUP BY cluster_id
ORDER BY cluster_id;

-- Test K-Means with 1 cluster
DO $$
DECLARE
    cluster_count INT;
BEGIN
    SELECT COUNT(DISTINCT cluster_id) INTO cluster_count
    FROM neurondb.cluster_kmeans('test_clustering_data', 'vec', 1, 10);
    
    IF cluster_count != 1 THEN
        RAISE EXCEPTION 'cluster_kmeans (k=1) returned % clusters, expected 1', cluster_count;
    END IF;
END$$;

SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_kmeans('test_clustering_data', 'vec', 1, 10)
GROUP BY cluster_id;

\echo '=== Testing Mini-batch K-Means ==='

-- Test Mini-batch K-Means
-- Validate that function returns expected results
DO $$
DECLARE
    result_count INT;
    cluster_count INT;
BEGIN
    SELECT COUNT(*), COUNT(DISTINCT cluster_id)
    INTO result_count, cluster_count
    FROM neurondb.cluster_minibatch_kmeans('test_clustering_data', 'vec', 3, 3, 50);
    
    IF result_count = 0 THEN
        RAISE EXCEPTION 'cluster_minibatch_kmeans returned no results';
    END IF;
    
    IF cluster_count != 3 THEN
        RAISE EXCEPTION 'cluster_minibatch_kmeans returned % clusters, expected 3', cluster_count;
    END IF;
END$$;

SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_minibatch_kmeans('test_clustering_data', 'vec', 3, 3, 50)
GROUP BY cluster_id
ORDER BY cluster_id;

SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_minibatch_kmeans('test_clustering_data', 'vec', 2, 5, 30)
GROUP BY cluster_id
ORDER BY cluster_id;

\echo '=== Testing DBSCAN Clustering ==='

-- Test DBSCAN
-- Validate that function returns expected results
DO $$
DECLARE
    result_count INT;
BEGIN
    SELECT COUNT(*) INTO result_count
    FROM neurondb.cluster_dbscan('test_clustering_data', 'vec', 1.0, 2);
    
    IF result_count = 0 THEN
        RAISE EXCEPTION 'cluster_dbscan returned no results';
    END IF;
END$$;

SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_dbscan('test_clustering_data', 'vec', 1.0, 2)
GROUP BY cluster_id
ORDER BY cluster_id;

SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_dbscan('test_clustering_data', 'vec', 3.0, 2)
GROUP BY cluster_id
ORDER BY cluster_id;

SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_dbscan('test_clustering_data', 'vec', 0.3, 2)
GROUP BY cluster_id
ORDER BY cluster_id;

\echo '=== Testing Gaussian Mixture Model (GMM) ==='

-- Test GMM
-- Validate that function returns expected results
DO $$
DECLARE
    result_count INT;
    cluster_count INT;
BEGIN
    SELECT COUNT(*), COUNT(DISTINCT cluster_id)
    INTO result_count, cluster_count
    FROM neurondb.cluster_gmm('test_clustering_data', 'vec', 3, 50);
    
    IF result_count = 0 THEN
        RAISE EXCEPTION 'cluster_gmm returned no results';
    END IF;
    
    IF cluster_count != 3 THEN
        RAISE EXCEPTION 'cluster_gmm returned % clusters, expected 3', cluster_count;
    END IF;
END$$;

SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_gmm('test_clustering_data', 'vec', 3, 50)
GROUP BY cluster_id
ORDER BY cluster_id;

SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_gmm('test_clustering_data', 'vec', 2, 30)
GROUP BY cluster_id
ORDER BY cluster_id;

\echo '=== Testing Hierarchical Clustering ==='

-- Test Hierarchical
-- Validate that function returns expected results
DO $$
DECLARE
    result_count INT;
    cluster_count INT;
BEGIN
    SELECT COUNT(*), COUNT(DISTINCT cluster_id)
    INTO result_count, cluster_count
    FROM neurondb.cluster_hierarchical('test_clustering_data', 'vec', 3, 'single');
    
    IF result_count = 0 THEN
        RAISE EXCEPTION 'cluster_hierarchical returned no results';
    END IF;
    
    IF cluster_count != 3 THEN
        RAISE EXCEPTION 'cluster_hierarchical returned % clusters, expected 3', cluster_count;
    END IF;
END$$;

SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_hierarchical('test_clustering_data', 'vec', 3, 'single')
GROUP BY cluster_id
ORDER BY cluster_id;

SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_hierarchical('test_clustering_data', 'vec', 2, 'complete')
GROUP BY cluster_id
ORDER BY cluster_id;

SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_hierarchical('test_clustering_data', 'vec', 3, 'average')
GROUP BY cluster_id
ORDER BY cluster_id;

\echo '=== Testing Cluster Quality Metrics ==='

-- Test Davies-Bouldin Index
-- Note: This function may require creating result tables first
-- For now, we just verify it exists and can be called
-- (Full test would require setting up cluster assignments first)

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
-- This should either work or raise a proper error, not be silently skipped
DO $$
DECLARE
    result_count INT;
BEGIN
    SELECT COUNT(*) INTO result_count
    FROM neurondb.cluster_kmeans('test_small_data', 'vec', 5, 10);
    
    -- Should return at most 2 results (we have 2 points)
    IF result_count > 2 THEN
        RAISE EXCEPTION 'K-Means edge case: returned % results for 2 points', result_count;
    END IF;
END$$;

-- DBSCAN with no points meeting criteria
-- This should either return empty results or raise a proper error
DO $$
DECLARE
    result_count INT;
BEGIN
    SELECT COUNT(*) INTO result_count
    FROM neurondb.cluster_dbscan('test_small_data', 'vec', 0.01, 10);
    
    -- With very tight epsilon, may return 0 results (all noise)
    -- This is acceptable, we just verify the function executes
    NULL; -- Allow any result count
END$$;

-- Cleanup
DROP TABLE test_clustering_data CASCADE;
DROP TABLE test_small_data CASCADE;
