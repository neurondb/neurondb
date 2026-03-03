\set ON_ERROR_STOP on
\set QUIET on
-- ============================================================================
-- NeuronDB Fraud Detection Demo - GMM (Gaussian Mixture Models) Clustering
-- Tests GMM algorithm on fraud detection dataset
-- ============================================================================

\echo '=========================================================================='
\echo '|       NeuronDB - GMM Clustering Fraud Detection                         |'
\echo '=========================================================================='
\echo ''

\echo 'GMM CLUSTERING (Gaussian Mixture Models)'
\echo ''
\echo 'Note: GMM returns probabilities, converting to cluster assignments...'
\echo ''
\timing on

-- GMM returns 2D array of probabilities, convert to cluster IDs using helper function
WITH gmm_probabilities AS (
    SELECT cluster_gmm('train_data', 'features', 7, 30) as probs
),
gmm_clusters AS (
    SELECT gmm_to_clusters(probs) as clusters
    FROM gmm_probabilities
),
gmm_result AS (
    SELECT 
        t.transaction_id,
        t.is_fraud,
        c.cluster
    FROM (SELECT transaction_id, is_fraud, ROW_NUMBER() OVER (ORDER BY transaction_id) as rn FROM train_data) t,
    gmm_clusters,
    LATERAL unnest(clusters) WITH ORDINALITY AS c(cluster, rn)
    WHERE t.rn = c.rn
),
cluster_fraud_rates AS (
    SELECT 
        cluster,
        COUNT(*) as transactions,
        SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) as frauds,
        ROUND(100.0 * SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) / COUNT(*), 2) as fraud_rate
    FROM gmm_result
    GROUP BY cluster
)
SELECT 
    'GMM' as algorithm,
    COUNT(*) as num_clusters,
    SUM(transactions) as total_transactions,
    MAX(fraud_rate) as max_fraud_rate,
    MIN(fraud_rate) as min_fraud_rate
FROM cluster_fraud_rates;

\timing off

-- ============================================================================
-- STEP 1: Record GMM Model Performance
-- ============================================================================
\echo ''
\echo 'STEP 1: Recording GMM model in project...'
INSERT INTO neurondb.ml_models (
    project_id, version, algorithm, status,
    training_table, training_column, parameters,
    num_samples, completed_at
)
SELECT 
    p.project_id,
    COALESCE((SELECT MAX(version) FROM neurondb.ml_models WHERE project_id = p.project_id), 0) + 1,
    'gmm',
    'completed',
    'train_data',
    'features',
    jsonb_build_object('k', 7, 'max_iters', 30),
    (SELECT COUNT(*) FROM train_data),
    now()
FROM neurondb.ml_projects p
WHERE p.project_name = 'fraud_gmm'
RETURNING model_id AS gmm_model_id \gset
\echo '   GMM model trained and recorded (model_id: ' :gmm_model_id ')'
\echo ''

-- ============================================================================
-- STEP 2: Train Additional GMM Versions
-- ============================================================================
\echo 'STEP 2: Training additional GMM models with different parameters...'
\echo '   Training GMM with K=5, 50 iterations...'
\timing on

WITH gmm_probabilities AS (
    SELECT cluster_gmm('train_data', 'features', 5, 50) as probs
),
gmm_clusters AS (
    SELECT gmm_to_clusters(probs) as clusters
    FROM gmm_probabilities
),
gmm_result AS (
    SELECT 
        t.transaction_id,
        t.is_fraud,
        c.cluster
    FROM (SELECT transaction_id, is_fraud, ROW_NUMBER() OVER (ORDER BY transaction_id) as rn FROM train_data) t,
    gmm_clusters,
    LATERAL unnest(clusters) WITH ORDINALITY AS c(cluster, rn)
    WHERE t.rn = c.rn
),
cluster_fraud_rates AS (
    SELECT 
        cluster,
        COUNT(*) as transactions,
        SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) as frauds,
        ROUND(100.0 * SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) / COUNT(*), 2) as fraud_rate
    FROM gmm_result
    GROUP BY cluster
)
SELECT 
    'GMM K=5' as algorithm,
    COUNT(*) as num_clusters,
    MAX(fraud_rate) as max_fraud_rate,
    MIN(fraud_rate) as min_fraud_rate
FROM cluster_fraud_rates;

\timing off
\echo '   GMM K=5 model trained'
\echo ''

-- ============================================================================
-- STEP 3: List All GMM Models
-- ============================================================================
\echo 'STEP 3: Listing all GMM models in project...'
SELECT 
    model_id,
    version,
    parameters->>'k' as K_value,
    parameters->>'max_iters' as iterations,
    status
FROM neurondb.ml_models
WHERE project_id = (SELECT project_id FROM neurondb.ml_projects WHERE project_name = 'fraud_gmm')
ORDER BY version;
\echo ''

-- ============================================================================
-- STEP 4: Test GMM on Test Dataset
-- ============================================================================
\echo 'STEP 4: Testing GMM model on test data...'
\timing on

WITH gmm_probabilities AS (
    SELECT cluster_gmm('test_data', 'features', 7, 30) as probs
),
gmm_clusters AS (
    SELECT gmm_to_clusters(probs) as clusters
    FROM gmm_probabilities
),
gmm_result AS (
    SELECT 
        t.transaction_id,
        t.is_fraud,
        c.cluster
    FROM (SELECT transaction_id, is_fraud, ROW_NUMBER() OVER (ORDER BY transaction_id) as rn FROM test_data) t,
    gmm_clusters,
    LATERAL unnest(clusters) WITH ORDINALITY AS c(cluster, rn)
    WHERE t.rn = c.rn
),
cluster_fraud_rates AS (
    SELECT 
        cluster,
        COUNT(*) as transactions,
        SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) as frauds,
        ROUND(100.0 * SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) / COUNT(*), 2) as fraud_rate
    FROM gmm_result
    GROUP BY cluster
)
SELECT 
    'GMM Test Data' as dataset,
    COUNT(*) as num_clusters,
    SUM(transactions) as total_samples,
    MAX(fraud_rate) as max_fraud_rate,
    MIN(fraud_rate) as min_fraud_rate,
    ROUND(AVG(fraud_rate), 2) as avg_fraud_rate
FROM cluster_fraud_rates;

\timing off
\echo ''

\echo '=========================================================================='
\echo 'GMM Testing Complete!'
\echo '   - 2 model versions trained (K=5, K=7)'
\echo '   - Models tested on test data'
\echo '   - Probabilistic clustering working'
\echo '=========================================================================='
\echo ''

