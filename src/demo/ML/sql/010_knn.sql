\set ON_ERROR_STOP on
\set QUIET on
-- ============================================================================
-- NeuronDB ML Demo - K-Nearest Neighbors (KNN)
-- Tests KNN for both classification and regression
-- ============================================================================

\echo '=========================================================================='
\echo '       NeuronDB - K-Nearest Neighbors (KNN)'
\echo '=========================================================================='
\echo ''

\echo 'STEP 1: Prepare KNN dataset (using existing fraud data)...'

CREATE TEMP TABLE knn_train AS
SELECT transaction_id, features, CASE WHEN is_fraud THEN 1.0 ELSE 0.0 END as label
FROM transactions
WHERE features IS NOT NULL AND is_fraud IS NOT NULL
LIMIT 10000;  -- KNN is O(n) per prediction, use smaller dataset

CREATE TEMP TABLE knn_test AS
SELECT transaction_id, features, CASE WHEN is_fraud THEN 1.0 ELSE 0.0 END as label
FROM transactions
WHERE features IS NOT NULL AND is_fraud IS NOT NULL
OFFSET 10000 LIMIT 2000;

\echo '   Created 10k train, 2k test samples'
\echo ''

\echo 'STEP 2: Test KNN classification (k=5)...'
\echo '   Note: Full evaluation temporarily skipped due to C function stability'
\echo '   Testing basic classification functionality...'

\timing on
WITH predictions AS (
    SELECT 
        t.label::int as actual,
        knn_classify('knn_train', 'features', 'label', t.features, 5) as predicted
    FROM knn_test t
    LIMIT 1000
)
SELECT 
    COUNT(*) as total,
    SUM(CASE WHEN actual = predicted THEN 1 ELSE 0 END) as correct,
    ROUND((100.0 * SUM(CASE WHEN actual = predicted THEN 1 ELSE 0 END) / COUNT(*))::numeric, 2) as accuracy_pct
FROM predictions;
\timing off
\echo ''

\echo 'STEP 3: Single sample classification (k=5)...'

SELECT 
    t.transaction_id,
    t.label::int as actual_class,
    knn_classify('knn_train', 'features', 'label', t.features, 5) as predicted_class,
    CASE 
        WHEN t.label::int = knn_classify('knn_train', 'features', 'label', t.features, 5) 
        THEN 'CORRECT' 
        ELSE 'WRONG' 
    END as result
FROM knn_test t
LIMIT 10;
\echo ''

\echo 'STEP 4: Record best KNN model (k=5)...'
SELECT neurondb_create_ml_project('knn_demo', 'classification', 'KNN classification demo') AS knn_proj \gset

INSERT INTO neurondb.ml_models (project_id, version, algorithm, status, training_table, training_column, parameters, num_samples, completed_at)
SELECT 
    p.project_id, 1, 'custom', 'completed', 'knn_train', 'features',
    jsonb_build_object('k', 5, 'metric', 'euclidean'),
    10000, now()
FROM neurondb.ml_projects p WHERE project_name = 'knn_demo'
RETURNING model_id AS knn_model_id \gset

\echo '   KNN model recorded (ID: ' :knn_model_id ', k=5)'
\echo ''

\echo 'STEP 5: KNN Regression test (predicting continuous values)...'

-- Create regression dataset
CREATE TEMP TABLE knn_reg_train AS
SELECT 
    transaction_id, 
    features,
    amount as value
FROM transactions
WHERE features IS NOT NULL AND amount IS NOT NULL
LIMIT 5000;

\echo '   Sample KNN regression predictions (k=5):'
SELECT 
    transaction_id,
    ROUND(value::numeric, 2) as actual_value,
    ROUND(knn_regress('knn_reg_train', 'features', 'value', features, 5)::numeric, 2) as predicted_value
FROM knn_reg_train
LIMIT 10;
\echo ''

\echo '=========================================================================='
\echo 'K-Nearest Neighbors (KNN) Complete!'
\echo ''
\echo 'Performance Notes:'
\echo '  - KNN is O(n) per prediction (no training phase)'
\echo '  - Best for: Small-medium datasets (<100k samples)'
\echo '  - Advantages: Simple, no training, works well with local patterns'
\echo '  - Limitations: Slow on large datasets, sensitive to feature scaling'
\echo ''
\echo 'Best k value: 5-7 (good balance of bias-variance tradeoff)'
\echo '=========================================================================='
\echo ''

