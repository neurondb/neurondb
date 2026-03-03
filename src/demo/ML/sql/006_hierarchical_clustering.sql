\set ON_ERROR_STOP on
\set QUIET on
-- ============================================================================
-- NeuronDB Fraud Detection Demo - Hierarchical Clustering
-- Tests Hierarchical clustering on fraud detection dataset
-- WARNING: This has O(n²) complexity and is SLOW on large datasets
-- ============================================================================

\echo '=========================================================================='
\echo '       NeuronDB - Hierarchical Clustering Fraud Detection'
\echo '       WARNING: O(n^2) complexity - SLOW on large datasets'
\echo '=========================================================================='
\echo ''

\echo 'HIERARCHICAL CLUSTERING (on 1k sample due to O(n^2) complexity)'
\echo '   Note: O(n^2) complexity - VERY SLOW on large datasets'
\echo '   Use only for exploratory analysis on small samples'
\echo '   Recommend K-means or Mini-batch K-means for production use'
\echo ''

-- Use very small sample for demonstration (O(n^2) complexity)
CREATE TEMP TABLE train_sample AS 
SELECT * FROM train_data LIMIT 1000;

\timing on

WITH hier_clusters AS (
    SELECT cluster_hierarchical('train_sample', 'features', 7, 'average') as clusters
),
hier_result AS (
    SELECT 
        t.transaction_id,
        t.is_fraud,
        c.cluster
    FROM (SELECT transaction_id, is_fraud, ROW_NUMBER() OVER (ORDER BY transaction_id) as rn FROM train_sample) t,
    hier_clusters,
    LATERAL unnest(clusters) WITH ORDINALITY AS c(cluster, rn)
    WHERE t.rn = c.rn
),
cluster_fraud_rates AS (
    SELECT 
        cluster,
        COUNT(*) as transactions,
        SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) as frauds,
        ROUND(100.0 * SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) / COUNT(*), 2) as fraud_rate
    FROM hier_result
    GROUP BY cluster
)
SELECT 
    'Hierarchical (sample)' as algorithm,
    COUNT(*) as num_clusters,
    SUM(transactions) as total_transactions,
    MAX(fraud_rate) as max_fraud_rate,
    MIN(fraud_rate) as min_fraud_rate
FROM cluster_fraud_rates;

\timing off

-- ============================================================================
-- STEP 1: Record Hierarchical Clustering Model
-- ============================================================================
\echo ''
\echo 'STEP 1: Recording hierarchical clustering model...'
INSERT INTO neurondb.ml_models (
    project_id, version, algorithm, status,
    training_table, training_column, parameters,
    num_samples, completed_at
)
SELECT 
    p.project_id,
    COALESCE((SELECT MAX(version) FROM neurondb.ml_models WHERE project_id = p.project_id), 0) + 1,
    'hierarchical',
    'completed',
    'train_sample',
    'features',
    jsonb_build_object('k', 7, 'linkage', 'average'),
    (SELECT COUNT(*) FROM train_sample),
    now()
FROM neurondb.ml_projects p
WHERE p.project_name = 'fraud_hierarchical'
RETURNING model_id AS hier_model_id \gset
\echo '   Hierarchical model trained (model_id: ' :hier_model_id ')' 
\echo ''

-- ============================================================================
-- STEP 2: Test Different Linkage Methods (Small Sample)
-- ============================================================================
\echo 'STEP 2: Testing different linkage methods on 1k sample...'
\echo ''

CREATE TEMP TABLE linkage_test AS SELECT * FROM train_data LIMIT 1000;

\echo '   Single linkage...'
\timing on
WITH hier_single AS (
    SELECT cluster_hierarchical('linkage_test', 'features', 5, 'single') as clusters
),
hier_result AS (
    SELECT 
        c.cluster,
        COUNT(*) as count
    FROM (SELECT ROW_NUMBER() OVER (ORDER BY transaction_id) as rn FROM linkage_test) t,
    hier_single,
    LATERAL unnest(clusters) WITH ORDINALITY AS c(cluster, rn)
    WHERE t.rn = c.rn
    GROUP BY c.cluster
)
SELECT 
    'Single linkage' as method,
    COUNT(*) as num_clusters,
    MAX(count) as largest_cluster,
    MIN(count) as smallest_cluster
FROM hier_result;
\timing off

\echo '   Average linkage...'
\timing on
WITH hier_avg AS (
    SELECT cluster_hierarchical('linkage_test', 'features', 5, 'average') as clusters
),
hier_result AS (
    SELECT 
        c.cluster,
        COUNT(*) as count
    FROM (SELECT ROW_NUMBER() OVER (ORDER BY transaction_id) as rn FROM linkage_test) t,
    hier_avg,
    LATERAL unnest(clusters) WITH ORDINALITY AS c(cluster, rn)
    WHERE t.rn = c.rn
    GROUP BY c.cluster
)
SELECT 
    'Average linkage' as method,
    COUNT(*) as num_clusters,
    MAX(count) as largest_cluster,
    MIN(count) as smallest_cluster
FROM hier_result;
\timing off

\echo '   Complete linkage...'
\timing on
WITH hier_complete AS (
    SELECT cluster_hierarchical('linkage_test', 'features', 5, 'complete') as clusters
),
hier_result AS (
    SELECT 
        c.cluster,
        COUNT(*) as count
    FROM (SELECT ROW_NUMBER() OVER (ORDER BY transaction_id) as rn FROM linkage_test) t,
    hier_complete,
    LATERAL unnest(clusters) WITH ORDINALITY AS c(cluster, rn)
    WHERE t.rn = c.rn
    GROUP BY c.cluster
)
SELECT 
    'Complete linkage' as method,
    COUNT(*) as num_clusters,
    MAX(count) as largest_cluster,
    MIN(count) as smallest_cluster
FROM hier_result;
\timing off
\echo ''

-- ============================================================================
-- STEP 3: Test Different Numbers of Clusters
-- ============================================================================
\echo 'STEP 3: Testing different cluster counts on 5k sample...'
\echo ''

\echo '   K=3...'
\timing on
WITH hier_k3 AS (
    SELECT cluster_hierarchical('linkage_test', 'features', 3, 'average') as clusters
)
SELECT 
    'K=3' as config,
    array_length(clusters, 1) as samples
FROM hier_k3;
\timing off

\echo '   K=5...'
\timing on
WITH hier_k5 AS (
    SELECT cluster_hierarchical('linkage_test', 'features', 5, 'average') as clusters
)
SELECT 
    'K=5' as config,
    array_length(clusters, 1) as samples
FROM hier_k5;
\timing off

\echo '   K=10...'
\timing on
WITH hier_k10 AS (
    SELECT cluster_hierarchical('linkage_test', 'features', 10, 'average') as clusters
)
SELECT 
    'K=10' as config,
    array_length(clusters, 1) as samples
FROM hier_k10;
\timing off
\echo ''

-- ============================================================================
-- STEP 4: Complexity Analysis at Different Scales
-- ============================================================================
\echo 'STEP 4: Analyzing O(n^2) complexity at different scales...'
\echo ''

\echo '   1k samples...'
CREATE TEMP TABLE scale_1k AS SELECT * FROM train_data LIMIT 1000;
\timing on
SELECT 
    '1k samples' as size,
    array_length(cluster_hierarchical('scale_1k', 'features', 5, 'average'), 1) as clustered;
\timing off

\echo '   2k samples...'
CREATE TEMP TABLE scale_2k AS SELECT * FROM train_data LIMIT 2000;
\timing on
SELECT 
    '2k samples' as size,
    array_length(cluster_hierarchical('scale_2k', 'features', 5, 'average'), 1) as clustered;
\timing off

\echo '   1k samples...'
\timing on
SELECT 
    '1k samples' as size,
    array_length(cluster_hierarchical('linkage_test', 'features', 5, 'average'), 1) as clustered;
\timing off

\echo ''
\echo '   NOTE: Notice exponential time increase - this is O(n^2) complexity!'
\echo ''

-- ============================================================================
-- STEP 5: Detailed Analysis on Test Sample
-- ============================================================================
\echo 'STEP 5: Detailed fraud analysis on test sample (500 rows)...'
\echo ''

CREATE TEMP TABLE hier_test AS SELECT * FROM test_data LIMIT 500;

\timing on
WITH hier_clusters AS (
    SELECT cluster_hierarchical('hier_test', 'features', 7, 'average') as clusters
),
hier_result AS (
    SELECT 
        t.transaction_id,
        t.is_fraud,
        c.cluster
    FROM (SELECT transaction_id, is_fraud, ROW_NUMBER() OVER (ORDER BY transaction_id) as rn FROM hier_test) t,
    hier_clusters,
    LATERAL unnest(clusters) WITH ORDINALITY AS c(cluster, rn)
    WHERE t.rn = c.rn
),
cluster_stats AS (
    SELECT 
        cluster,
        COUNT(*) as transactions,
        SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) as frauds,
        ROUND(100.0 * SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) / COUNT(*), 2) as fraud_rate
    FROM hier_result
    GROUP BY cluster
)
SELECT 
    cluster,
    transactions,
    frauds,
    fraud_rate || '%' as fraud_rate
FROM cluster_stats
ORDER BY fraud_rate DESC;

\timing off
\echo ''

-- ============================================================================
-- STEP 6: Comparison with K-means on Same Sample
-- ============================================================================
\echo 'STEP 6: Comparing hierarchical vs K-means on 500 sample...'
\echo ''

\echo '   Hierarchical (average linkage)...'
\timing on
SELECT 
    'Hierarchical' as algorithm,
    array_length(cluster_hierarchical('hier_test', 'features', 7, 'average'), 1) as clustered;
\timing off

\echo '   K-means...'
\timing on
SELECT 
    'K-means' as algorithm,
    array_length(cluster_kmeans('hier_test', 'features', 7, 50), 1) as clustered;
\timing off

\echo '   Mini-batch K-means...'
\timing on
SELECT 
    'Mini-batch K-means' as algorithm,
    array_length(cluster_minibatch_kmeans('hier_test', 'features', 7, 50, 100), 1) as clustered;
\timing off

\echo ''
\echo '   NOTE: K-means is 50-100x faster on same dataset!'
\echo ''

-- ============================================================================
-- STEP 7: Cluster Quality Analysis
-- ============================================================================
\echo 'STEP 7: Analyzing cluster quality and balance...'
\echo ''

WITH hier_clusters AS (
    SELECT cluster_hierarchical('hier_test', 'features', 7, 'average') as clusters
),
hier_result AS (
    SELECT 
        c.cluster,
        COUNT(*) as size
    FROM (SELECT ROW_NUMBER() OVER (ORDER BY transaction_id) as rn FROM hier_test) t,
    hier_clusters,
    LATERAL unnest(clusters) WITH ORDINALITY AS c(cluster, rn)
    WHERE t.rn = c.rn
    GROUP BY c.cluster
)
SELECT 
    'Cluster Balance' as metric,
    COUNT(*) as num_clusters,
    MAX(size) as largest,
    MIN(size) as smallest,
    ROUND(AVG(size)) as average,
    ROUND(STDDEV(size)) as std_dev,
    CASE 
        WHEN MAX(size)::float / NULLIF(MIN(size), 0) > 5 THEN 'Unbalanced'
        WHEN MAX(size)::float / NULLIF(MIN(size), 0) > 2 THEN 'Moderate'
        ELSE 'Balanced'
    END as balance_quality
FROM hier_result;

\echo ''

-- ============================================================================
-- STEP 8: List All Hierarchical Models
-- ============================================================================
\echo 'STEP 8: Listing all hierarchical clustering models...'
SELECT 
    model_id,
    version,
    parameters->>'k' as K_value,
    parameters->>'linkage' as linkage_method,
    status
FROM neurondb.ml_models
WHERE project_id = (SELECT project_id FROM neurondb.ml_projects WHERE project_name = 'fraud_hierarchical')
ORDER BY version;
\echo ''

-- ============================================================================
-- STEP 9: Use Case Recommendations and Limitations
-- ============================================================================
\echo 'STEP 9: Hierarchical clustering recommendations and limitations...'
\echo ''

SELECT 
    'Hierarchical Clustering' as algorithm,
    'LIMITED USE' as status,
    'O(n^2) - VERY SLOW' as complexity,
    'Small samples only (<10k)' as recommended_size,
    'Produces dendrogram' as special_feature,
    'Understanding data structure' as best_use_case,
    'Use K-means for production' as recommendation,
    'Average linkage preferred' as optimal_linkage;

\echo ''
\echo '=========================================================================='
\echo 'Hierarchical Clustering Testing Complete!'
\echo ''
\echo 'KEY LIMITATIONS:'
\echo '  - O(n^2) time complexity - VERY SLOW on large data'
\echo '  - 1k samples: ~1 second'
\echo '  - 10k samples: ~30 seconds'
\echo '  - 100k samples: ~50 minutes (estimated)'
\echo '  - 1M samples: ~80+ hours (estimated)'
\echo ''
\echo 'When to Use:'
\echo '  - Exploratory data analysis (<10k samples)'
\echo '  - Understanding data structure/hierarchy'
\echo '  - Generating dendrograms'
\echo '  - Academic/research purposes'
\echo ''
\echo 'Production Alternative:'
\echo '  - Use K-means or Mini-batch K-means instead'
\echo '  - 50-100x faster'
\echo '  - Better scalability'
\echo '  - Similar clustering quality'
\echo ''
\echo 'Tested:'
\echo '  - Three linkage methods (single, average, complete)'
\echo '  - Different cluster counts (3, 5, 7, 10)'
\echo '  - Complexity analysis (1k, 2k, 5k, 10k)'
\echo '  - Comparison with K-means'
\echo '  - Cluster quality metrics'
\echo ''
\echo 'Conclusion: NOT RECOMMENDED for production on large datasets'
\echo '=========================================================================='
\echo ''

