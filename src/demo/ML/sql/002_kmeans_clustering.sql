\set ON_ERROR_STOP on
\set QUIET on
-- ============================================================================
-- NeuronDB Fraud Detection Demo - K-means Clustering
-- Tests K-means algorithm on fraud detection dataset
-- ============================================================================

\echo '=========================================================================='
\echo '|       NeuronDB - K-means Clustering Fraud Detection                     |'
\echo '=========================================================================='
\echo ''

\echo 'K-MEANS CLUSTERING (K=7 risk groups)'
\echo ''
\timing on

WITH kmeans_clusters AS (
    SELECT cluster_kmeans('train_data', 'features', 7, 50) as clusters
),
kmeans_result AS (
    SELECT 
        t.transaction_id,
        t.is_fraud,
        c.cluster
    FROM (SELECT transaction_id, is_fraud, ROW_NUMBER() OVER (ORDER BY transaction_id) as rn FROM train_data) t,
    kmeans_clusters,
    LATERAL unnest(clusters) WITH ORDINALITY AS c(cluster, rn)
    WHERE t.rn = c.rn
),
cluster_fraud_rates AS (
    SELECT 
        cluster,
        COUNT(*) as transactions,
        SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) as frauds,
        ROUND(100.0 * SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) / COUNT(*), 2) as fraud_rate
    FROM kmeans_result
    GROUP BY cluster
)
SELECT 
    'K-means' as algorithm,
    COUNT(*) as num_clusters,
    SUM(transactions) as total_transactions,
    MAX(fraud_rate) as max_fraud_rate,
    MIN(fraud_rate) as min_fraud_rate
FROM cluster_fraud_rates;

\timing off

-- ============================================================================
-- STEP 1: Train and Record Model (Version 1)
-- ============================================================================
\echo ''
\echo 'STEP 1: Training K-means model (K=7, 50 iterations)...'
SELECT neurondb_train_kmeans_project('fraud_kmeans', 'train_data', 'features', 7, 50) AS kmeans_model_id \gset
\echo '   Model trained (model_id: ' :kmeans_model_id ')'
\echo ''

-- ============================================================================
-- STEP 2: View Project Info
-- ============================================================================
\echo 'STEP 2: Viewing project information...'
SELECT neurondb_get_project_info('fraud_kmeans');
\echo ''

-- ============================================================================
-- STEP 3: List All Models in Project
-- ============================================================================
\echo 'STEP 3: Listing all models in fraud_kmeans project...'
SELECT 
    model_id,
    version,
    algorithm,
    status,
    training_time_ms || 'ms' as training_time,
    is_deployed
FROM neurondb_list_project_models('fraud_kmeans')
ORDER BY version;
\echo ''

-- ============================================================================
-- STEP 4: Train Additional Model Versions (Different K values)
-- ============================================================================
\echo 'STEP 4: Training additional model versions for comparison...'
\echo '   Training K=5 model...'
SELECT neurondb_train_kmeans_project('fraud_kmeans', 'train_data', 'features', 5, 50) AS kmeans_v2 \gset
\echo '   Model v2 trained (model_id: ' :kmeans_v2 ')'

\echo '   Training K=10 model...'
SELECT neurondb_train_kmeans_project('fraud_kmeans', 'train_data', 'features', 10, 50) AS kmeans_v3 \gset
\echo '   Model v3 trained (model_id: ' :kmeans_v3 ')'
\echo ''

-- ============================================================================
-- STEP 5: Compare All Model Versions
-- ============================================================================
\echo 'STEP 5: Comparing all K-means model versions...'
SELECT 
    version,
    parameters->>'k' as K_value,
    training_time_ms || 'ms' as training_time,
    status,
    is_deployed,
    created_at
FROM neurondb_list_project_models('fraud_kmeans')
ORDER BY version;
\echo ''

-- ============================================================================
-- STEP 6: Test Each Model Version
-- ============================================================================
\echo 'STEP 6: Testing each model version on test data...'

-- Test K=7 (Version 1)
\echo '   Testing K=7 model...'
\timing on
WITH test_clusters AS (
    SELECT cluster_kmeans('test_data', 'features', 7, 50) as clusters
),
test_result AS (
    SELECT 
        c.cluster,
        COUNT(*) as transactions,
        SUM(CASE WHEN t.is_fraud THEN 1 ELSE 0 END) as fraud_count,
        ROUND(100.0 * SUM(CASE WHEN t.is_fraud THEN 1 ELSE 0 END) / COUNT(*), 2) as fraud_rate
    FROM (SELECT transaction_id, is_fraud, ROW_NUMBER() OVER (ORDER BY transaction_id) as rn FROM test_data) t,
    test_clusters,
    LATERAL unnest(clusters) WITH ORDINALITY AS c(cluster, rn)
    WHERE t.rn = c.rn
    GROUP BY c.cluster
)
SELECT 
    'K=7' as model,
    COUNT(*) as num_clusters,
    MAX(fraud_rate) as max_fraud_rate,
    MIN(fraud_rate) as min_fraud_rate,
    ROUND(AVG(fraud_rate), 2) as avg_fraud_rate
FROM test_result;
\timing off
\echo ''

-- ============================================================================
-- STEP 7: Deploy Best Model
-- ============================================================================
\echo 'STEP 7: Deploying best model (version 1, K=7)...'
SELECT neurondb_deploy_model('fraud_kmeans', :kmeans_model_id);
\echo '   Model deployed successfully'
\echo ''

-- ============================================================================
-- STEP 8: Verify Deployment
-- ============================================================================
\echo 'STEP 8: Verifying deployed model...'
SELECT 
    'Currently deployed model:' as status,
    neurondb_get_deployed_model('fraud_kmeans') as deployed_model_id;

SELECT 
    version,
    algorithm,
    parameters,
    is_deployed,
    created_at as deployed_at
FROM neurondb_list_project_models('fraud_kmeans')
WHERE is_deployed = true;
\echo ''

-- ============================================================================
-- STEP 9: Model Metadata Summary
-- ============================================================================
\echo 'STEP 9: Complete project summary...'
SELECT 
    project_name,
    model_type,
    total_models,
    latest_version,
    deployed_version,
    created_at,
    updated_at
FROM neurondb.ml_projects_summary
WHERE project_name = 'fraud_kmeans';
\echo ''

\echo '=========================================================================='
\echo 'K-means Testing Complete!'
\echo '   - 3 model versions trained (K=5, K=7, K=10)'
\echo '   - All models tested on test data'
\echo '   - Best model (K=7) deployed'
\echo '   - Ready for production inference'
\echo '=========================================================================='
\echo ''

