\set ON_ERROR_STOP on
\set QUIET on
-- ============================================================================
-- NeuronDB ML Demo - Linear Regression (Supervised Learning)
-- Tests linear regression for predicting continuous values
-- ============================================================================

\echo '=========================================================================='
\echo '       NeuronDB - Linear Regression (Supervised Learning)'
\echo '=========================================================================='
\echo ''

\echo 'STEP 1: Create regression dataset (predicting transaction amount)...'

-- Create regression dataset with synthetic target
CREATE TEMP TABLE regression_data AS
SELECT 
    transaction_id,
    features,
    -- Synthetic target: simple linear combination
    random() * 1000.0 as amount
FROM transactions
WHERE features IS NOT NULL
LIMIT 100000;

-- Split into train (80%) and test (20%)
CREATE TEMP TABLE linear_train AS SELECT * FROM regression_data LIMIT 80000;
CREATE TEMP TABLE linear_test AS SELECT * FROM regression_data OFFSET 80000;

\echo '   Created 80k train, 20k test samples'
\echo ''

\echo 'STEP 2: Train linear regression model...'
\timing on
SELECT train_linear_regression('linear_train', 'features', 'amount') AS coefficients \gset
\timing off
\echo '   Coefficients (intercept + weights):'
SELECT unnest(:'coefficients'::float8[]) as coef, generate_series(0, array_length(:'coefficients'::float8[], 1)-1) as idx
ORDER BY idx LIMIT 5;
\echo ''

\echo 'STEP 3: Evaluate on training data...'
\timing on
SELECT evaluate_linear_regression('linear_train', 'features', 'amount', :'coefficients') AS train_metrics \gset
\timing off

WITH metrics AS (
    SELECT unnest(ARRAY['R²', 'MSE', 'MAE', 'RMSE']) as metric,
           unnest(:'train_metrics'::float8[]) as value,
           generate_series(1,4) as idx
)
SELECT metric, ROUND(value::numeric, 4) as value FROM metrics ORDER BY idx;
\echo ''

\echo 'STEP 4: Evaluate on test data...'
\timing on
SELECT evaluate_linear_regression('linear_test', 'features', 'amount', :'coefficients') AS test_metrics \gset
\timing off

WITH metrics AS (
    SELECT unnest(ARRAY['R²', 'MSE', 'MAE', 'RMSE']) as metric,
           unnest(:'test_metrics'::float8[]) as value,
           generate_series(1,4) as idx
)
SELECT metric, ROUND(value::numeric, 4) as value FROM metrics ORDER BY idx;
\echo ''

\echo 'STEP 5: Sample predictions...'
SELECT 
    transaction_id,
    ROUND(amount::numeric, 2) as actual,
    ROUND(predict_linear_regression(:'coefficients', features)::numeric, 2) as predicted,
    ROUND(ABS(amount - predict_linear_regression(:'coefficients', features))::numeric, 2) as error
FROM linear_test
LIMIT 10;
\echo ''

\echo 'STEP 6: Record model in ML project...'
SELECT neurondb_create_ml_project('regression_demo', 'regression', 'Linear regression demo') AS proj_id \gset

INSERT INTO neurondb.ml_models (project_id, version, algorithm, status, training_table, training_column, parameters, num_samples, completed_at)
SELECT 
    p.project_id, 1, 'custom', 'completed', 'linear_train', 'features',
    jsonb_build_object('coefficients', :'coefficients', 'r_squared', (:'train_metrics'::float8[])[1]),
    80000, now()
FROM neurondb.ml_projects p WHERE project_name = 'regression_demo'
RETURNING model_id AS linear_model_id \gset

\echo '   Model recorded (ID: ' :linear_model_id ')'
\echo ''

\echo '=========================================================================='
\echo 'Linear Regression Complete!'
\echo '=========================================================================='
\echo ''
