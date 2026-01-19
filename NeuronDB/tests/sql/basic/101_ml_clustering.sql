-- ====================================================================
-- NeurondB Regression Tests: ML Clustering Algorithms
-- ====================================================================
-- Tests for K-Means, Mini-batch K-Means, DBSCAN, GMM, Hierarchical
-- Uses real data from: sift1m.vectors (128-d vectors)
-- ====================================================================

\echo '=== Using SIFT1M Dataset for Clustering Tests ==='

-- Create test data from SIFT1M vectors (take first 1000 for speed)
-- Convert REAL[] to vector type, use first 10 dimensions for fast testing
CREATE TEMP TABLE test_clustering_data AS
SELECT 
    id,
    array_to_vector(embedding[1:10])::vector(10) as vec
FROM sift1m.vectors
WHERE id <= 1000
LIMIT 1000;

-- Show sample data
SELECT COUNT(*) as total_vectors, vector_dims(vec) as dimensions
FROM test_clustering_data
LIMIT 1;

\echo '=== Testing K-Means Clustering ==='

-- Test K-Means with 3 clusters
SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_kmeans('test_clustering_data', 'vec', 3, 100)
GROUP BY cluster_id
ORDER BY cluster_id;

-- Test K-Means with 2 clusters
SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_kmeans('test_clustering_data', 'vec', 2, 50)
GROUP BY cluster_id
ORDER BY cluster_id;

-- Test K-Means with single cluster (should assign all to cluster 0)
SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_kmeans('test_clustering_data', 'vec', 1, 10)
GROUP BY cluster_id;

\echo '=== Testing Mini-batch K-Means ==='

-- Test Mini-batch K-Means with 3 clusters, batch size 3
SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_minibatch_kmeans('test_clustering_data', 'vec', 3, 3, 50)
GROUP BY cluster_id
ORDER BY cluster_id;

-- Test Mini-batch K-Means with 2 clusters, batch size 5
SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_minibatch_kmeans('test_clustering_data', 'vec', 2, 5, 30)
GROUP BY cluster_id
ORDER BY cluster_id;

\echo '=== Testing DBSCAN Clustering ==='

-- Test DBSCAN with eps=1.0, min_pts=2
SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_dbscan('test_clustering_data', 'vec', 1.0, 2)
GROUP BY cluster_id
ORDER BY cluster_id;

-- Test DBSCAN with larger eps (should group more points)
SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_dbscan('test_clustering_data', 'vec', 3.0, 2)
GROUP BY cluster_id
ORDER BY cluster_id;

-- Test DBSCAN with tight eps (should create noise points: cluster_id = -1)
SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_dbscan('test_clustering_data', 'vec', 0.3, 2)
GROUP BY cluster_id
ORDER BY cluster_id;

\echo '=== Testing Gaussian Mixture Model (GMM) ==='

-- Test GMM with 3 components
SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_gmm('test_clustering_data', 'vec', 3, 50)
GROUP BY cluster_id
ORDER BY cluster_id;

-- Test GMM with 2 components
SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_gmm('test_clustering_data', 'vec', 2, 30)
GROUP BY cluster_id
ORDER BY cluster_id;

\echo '=== Testing Hierarchical Clustering ==='

-- Test Hierarchical clustering with 3 clusters, single linkage
SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_hierarchical('test_clustering_data', 'vec', 3, 'single')
GROUP BY cluster_id
ORDER BY cluster_id;

-- Test Hierarchical clustering with 2 clusters, complete linkage
SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_hierarchical('test_clustering_data', 'vec', 2, 'complete')
GROUP BY cluster_id
ORDER BY cluster_id;

-- Test Hierarchical clustering with 3 clusters, average linkage
SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_hierarchical('test_clustering_data', 'vec', 3, 'average')
GROUP BY cluster_id
ORDER BY cluster_id;

\echo '=== Testing Cluster Quality Metrics ==='

-- Test Davies-Bouldin Index for K-Means results
-- Lower values indicate better clustering
WITH kmeans_results AS (
    SELECT * FROM neurondb.cluster_kmeans('test_clustering_data', 'vec', 3, 100)
)
SELECT 
    neurondb.davies_bouldin_index('test_clustering_data', 'vec', 'kmeans_results', 'cluster_id') AS db_index;

-- Test with 2 clusters
WITH kmeans_2 AS (
    SELECT * FROM neurondb.cluster_kmeans('test_clustering_data', 'vec', 2, 50)
)
SELECT 
    neurondb.davies_bouldin_index('test_clustering_data', 'vec', 'kmeans_2', 'cluster_id') AS db_index_2;

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
SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_kmeans('test_small_data', 'vec', 5, 10)
GROUP BY cluster_id
ORDER BY cluster_id;

-- DBSCAN with no points meeting criteria
SELECT 
    cluster_id, 
    COUNT(*) as cluster_size
FROM neurondb.cluster_dbscan('test_small_data', 'vec', 0.01, 10)
GROUP BY cluster_id
ORDER BY cluster_id;

-- Cleanup
DROP TABLE test_clustering_data CASCADE;
DROP TABLE test_small_data CASCADE;

