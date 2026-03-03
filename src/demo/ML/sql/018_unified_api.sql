-- ============================================================================
-- NeuronDB Unified ML API Demo
-- PostgresML-compatible unified training interface
-- ============================================================================

\set ON_ERROR_STOP on
\set QUIET on

\echo ''
\echo '══════════════════════════════════════════════════════════════════'
\echo '  Demo 18: Unified ML API (PostgresML-compatible)'
\echo '══════════════================================================================'
\echo ''

-- Create training data
DROP TABLE IF EXISTS unified_train_data CASCADE;
CREATE TEMP TABLE unified_train_data AS
SELECT 
    i as transaction_id,
    ARRAY[
        (random() * 100)::real,
        (random() * 50)::real,
        (random() * 10)::real
    ]::real[] as features,
    CASE WHEN random() > 0.7 THEN 1 ELSE 0 END as is_fraud
FROM generate_series(1, 10000) i;

\echo 'Training data created: 10,000 transactions'
\echo ''

-- Test 1: neurondb.train() - Unified training interface
\echo 'Test 1: neurondb.train() - Train multiple algorithms'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo ''

\echo 'Training Linear Regression...'
SELECT neurondb.train(
    'fraud_detection',
    'linear_regression',
    'unified_train_data',
    'is_fraud'
) as linear_model_id;

\echo ''
\echo 'Training Logistic Regression...'
SELECT neurondb.train(
    'fraud_detection',
    'logistic_regression',
    'unified_train_data',
    'is_fraud'
) as logistic_model_id;

\echo ''
\echo 'Training Random Forest with hyperparameters...'
SELECT neurondb.train(
    'fraud_detection',
    'random_forest',
    'unified_train_data',
    'is_fraud',
    NULL,
    '{"n_trees": 20, "max_depth": 8, "min_samples": 50}'::jsonb
) as rf_model_id;

\echo ''
\echo 'Training KNN...'
SELECT neurondb.train(
    'fraud_detection',
    'knn',
    'unified_train_data',
    'is_fraud',
    NULL,
    '{"k": 3}'::jsonb
) as knn_model_id;

\echo ''

-- Test 2: neurondb.predict() - Unified prediction
\echo 'Test 2: neurondb.predict() - Make predictions'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo ''

WITH test_features AS (
    SELECT ARRAY[75.5, 25.3, 5.2]::real[] as features
)
SELECT 
    'Linear Regression' as model_type,
    neurondb.predict(1, features) as prediction
FROM test_features

UNION ALL

SELECT 
    'Logistic Regression' as model_type,
    neurondb.predict(2, features) as prediction
FROM test_features

UNION ALL

SELECT 
    'Random Forest' as model_type,
    neurondb.predict(3, features) as prediction
FROM test_features;

\echo ''

-- Test 3: neurondb.deploy() - Model deployment
\echo 'Test 3: neurondb.deploy() - Deploy models to production'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo ''

SELECT 
    'Replace Strategy' as deployment_type,
    neurondb.deploy(1, 'replace') as deployment_id

UNION ALL

SELECT 
    'Blue-Green Strategy' as deployment_type,
    neurondb.deploy(2, 'blue_green') as deployment_id

UNION ALL

SELECT 
    'Canary Strategy' as deployment_type,
    neurondb.deploy(3, 'canary') as deployment_id;

\echo ''

-- Test 4: List models
\echo 'Test 4: List all trained models'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo ''

SELECT 
    model_id,
    model_name,
    algorithm,
    status
FROM neurondb.ml_models
ORDER BY model_id;

\echo ''

-- Test 5: List deployments
\echo 'Test 5: List all deployments'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo ''

SELECT 
    deployment_id,
    model_id,
    deployment_name,
    strategy,
    status
FROM neurondb.ml_deployments
ORDER BY deployment_id;

\echo ''
\echo '══════════════════════════════════════════════════════════════════'
\echo '  ✅ Unified ML API Demo Complete'
\echo '══════════════================================================================'
\echo ''

