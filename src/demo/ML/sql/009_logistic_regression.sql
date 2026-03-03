\set ON_ERROR_STOP on
\set QUIET on
-- ============================================================================
-- NeuronDB ML Demo - Logistic Regression (Binary Classification)
-- Tests logistic regression for fraud detection
-- ============================================================================

\echo '=========================================================================='
\echo '       NeuronDB - Logistic Regression (Binary Classification)'
\echo '=========================================================================='
\echo ''

\echo 'STEP 1: Prepare binary classification dataset...'

-- Use existing fraud labels from transactions
CREATE TEMP TABLE classification_data AS
SELECT 
    transaction_id,
    features,
    CASE WHEN is_fraud THEN 1 ELSE 0 END as label
FROM transactions
WHERE features IS NOT NULL AND is_fraud IS NOT NULL
LIMIT 100000;

-- Check class distribution
\echo '   Class distribution:'
SELECT 
    label::int as class,
    COUNT(*) as count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) as percentage
FROM classification_data
GROUP BY label
ORDER BY label;
\echo ''

-- Split into train (80%) and test (20%)
CREATE TEMP TABLE logistic_train AS SELECT * FROM classification_data LIMIT 80000;
CREATE TEMP TABLE logistic_test AS SELECT * FROM classification_data OFFSET 80000;

\echo '   Created 80k train, 20k test samples'
\echo ''

\echo 'STEP 2: Train logistic regression model...'
\echo '   (This may take 30-60 seconds with gradient descent...)'
\timing on
SELECT train_logistic_regression('logistic_train', 'features', 'label', 500, 0.01, 0.01) AS coefficients \gset
\timing off
\echo '   Training complete!'
\echo '   Coefficients (bias + weights):'
SELECT unnest(:'coefficients'::float8[]) as coef, generate_series(0, array_length(:'coefficients'::float8[], 1)-1) as idx
ORDER BY idx LIMIT 5;
\echo ''

\echo 'STEP 3: Evaluate on training data...'
\timing on
SELECT evaluate_logistic_regression('logistic_train', 'features', 'label', :'coefficients', 0.5) AS train_metrics \gset
\timing off

WITH metrics AS (
    SELECT unnest(ARRAY['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Log Loss']) as metric,
           unnest(:'train_metrics'::float8[]) as value,
           generate_series(1,5) as idx
)
SELECT metric, ROUND(value::numeric, 4) as value FROM metrics ORDER BY idx;
\echo ''

\echo 'STEP 4: Evaluate on test data...'
\timing on
SELECT evaluate_logistic_regression('logistic_test', 'features', 'label', :'coefficients', 0.5) AS test_metrics \gset
\timing off

WITH metrics AS (
    SELECT unnest(ARRAY['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Log Loss']) as metric,
           unnest(:'test_metrics'::float8[]) as value,
           generate_series(1,5) as idx
)
SELECT metric, ROUND(value::numeric, 4) as value FROM metrics ORDER BY idx;
\echo ''

\echo 'STEP 5: Sample predictions with probabilities...'
SELECT 
    transaction_id,
    label::int as actual_class,
    ROUND(predict_logistic_regression(:'coefficients', features)::numeric, 4) as probability,
    CASE WHEN predict_logistic_regression(:'coefficients', features) >= 0.5 THEN 1 ELSE 0 END as predicted_class,
    CASE 
        WHEN label::int = 1 AND predict_logistic_regression(:'coefficients', features) >= 0.5 THEN 'TP'
        WHEN label::int = 0 AND predict_logistic_regression(:'coefficients', features) < 0.5 THEN 'TN'
        WHEN label::int = 1 AND predict_logistic_regression(:'coefficients', features) < 0.5 THEN 'FN'
        ELSE 'FP'
    END as classification
FROM logistic_test
LIMIT 10;
\echo ''

\echo 'STEP 6: Test different classification thresholds...'
\echo '   Threshold sensitivity analysis:'

SELECT 
    threshold,
    ROUND((metrics[1])::numeric, 4) as accuracy,
    ROUND((metrics[2])::numeric, 4) as precision,
    ROUND((metrics[3])::numeric, 4) as recall,
    ROUND((metrics[4])::numeric, 4) as f1_score
FROM (
    SELECT 
        t as threshold,
        evaluate_logistic_regression('logistic_test', 'features', 'label', :'coefficients', t) as metrics
    FROM generate_series(0.3, 0.7, 0.1) t
) thresholds
ORDER BY threshold;
\echo ''

\echo 'STEP 7: Record model in ML project...'
SELECT neurondb_create_ml_project('classification_demo', 'classification', 'Logistic regression demo') AS proj_id \gset

INSERT INTO neurondb.ml_models (project_id, version, algorithm, status, training_table, training_column, parameters, num_samples, completed_at)
SELECT 
    p.project_id, 1, 'custom', 'completed', 'logistic_train', 'features',
    jsonb_build_object('coefficients', :'coefficients', 'accuracy', (:'train_metrics'::float8[])[1], 'f1_score', (:'train_metrics'::float8[])[4]),
    80000, now()
FROM neurondb.ml_projects p WHERE project_name = 'classification_demo'
RETURNING model_id AS logistic_model_id \gset

\echo '   Model recorded (ID: ' :logistic_model_id ')'
\echo ''

\echo '=========================================================================='
\echo 'Logistic Regression Complete!'
\echo '=========================================================================='
\echo ''

