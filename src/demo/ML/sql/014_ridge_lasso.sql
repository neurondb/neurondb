\set ON_ERROR_STOP on
\set QUIET on
-- ============================================================================
-- NeuronDB ML Demo - Ridge, Lasso, and Elastic Net Regression
-- Tests regularized regression techniques to prevent overfitting
-- ============================================================================

\echo '=========================================================================='
\echo '       NeuronDB - Regularized Regression (Ridge, Lasso, Elastic Net)'
\echo '=========================================================================='
\echo ''

\echo 'STEP 1: Prepare regression dataset...'

-- Create regression dataset
CREATE TEMP TABLE reg_data AS
SELECT 
    transaction_id,
    features,
    amount as target
FROM transactions
WHERE features IS NOT NULL AND amount IS NOT NULL
LIMIT 50000;

-- Split into train (80%) and test (20%)
CREATE TEMP TABLE reg_train AS SELECT * FROM reg_data LIMIT 40000;
CREATE TEMP TABLE reg_test AS SELECT * FROM reg_data OFFSET 40000;

\echo '   Created 40k train, 10k test samples'
\echo ''

\echo '=========================================================================='
\echo 'PART 1: Ridge Regression (L2 Regularization)'
\echo '=========================================================================='
\echo ''

\echo 'STEP 2: Train Ridge regression with different lambda values...'
\echo '   Lambda controls regularization strength (higher = more regularization)'
\echo ''

\echo '   Training with lambda=0.1 (weak regularization)...'
\timing on
SELECT train_ridge_regression('reg_train', 'features', 'target', 0.1) AS ridge_weak \gset
\timing off

\echo '   Training with lambda=1.0 (moderate regularization)...'
\timing on
SELECT train_ridge_regression('reg_train', 'features', 'target', 1.0) AS ridge_moderate \gset
\timing off

\echo '   Training with lambda=10.0 (strong regularization)...'
\timing on
SELECT train_ridge_regression('reg_train', 'features', 'target', 10.0) AS ridge_strong \gset
\timing off
\echo ''

\echo 'STEP 3: Compare Ridge coefficients...'
\echo '   Weak regularization (λ=0.1):'
SELECT unnest(:'ridge_weak'::float8[]) as coef, generate_series(0, array_length(:'ridge_weak'::float8[], 1)-1) as idx
ORDER BY idx LIMIT 3;
\echo ''
\echo '   Moderate regularization (λ=1.0):'
SELECT unnest(:'ridge_moderate'::float8[]) as coef, generate_series(0, array_length(:'ridge_moderate'::float8[], 1)-1) as idx
ORDER BY idx LIMIT 3;
\echo ''
\echo '   Strong regularization (λ=10.0):'
SELECT unnest(:'ridge_strong'::float8[]) as coef, generate_series(0, array_length(:'ridge_strong'::float8[], 1)-1) as idx
ORDER BY idx LIMIT 3;
\echo ''

\echo '   Note: As lambda increases, coefficients shrink toward zero (preventing overfitting)'
\echo ''

\echo '=========================================================================='
\echo 'PART 2: Lasso Regression (L1 Regularization)'
\echo '=========================================================================='
\echo ''

\echo 'STEP 4: Train Lasso regression with different lambda values...'
\echo '   Lasso performs feature selection by setting some coefficients to exactly zero'
\echo ''

\echo '   Training with lambda=0.1 (weak regularization)...'
\timing on
SELECT train_lasso_regression('reg_train', 'features', 'target', 0.1, 1000) AS lasso_weak \gset
\timing off

\echo '   Training with lambda=1.0 (moderate regularization)...'
\timing on
SELECT train_lasso_regression('reg_train', 'features', 'target', 1.0, 1000) AS lasso_moderate \gset
\timing off

\echo '   Training with lambda=5.0 (strong regularization)...'
\timing on
SELECT train_lasso_regression('reg_train', 'features', 'target', 5.0, 1000) AS lasso_strong \gset
\timing off
\echo ''

\echo 'STEP 5: Compare Lasso coefficients and count non-zero features...'
\echo '   Weak regularization (λ=0.1):'
WITH coeffs AS (
    SELECT unnest(:'lasso_weak'::float8[]) as coef, 
           generate_series(0, array_length(:'lasso_weak'::float8[], 1)-1) as idx
)
SELECT idx, ROUND(coef::numeric, 4) as coef FROM coeffs WHERE idx <= 3 ORDER BY idx;

SELECT COUNT(*) as non_zero_features 
FROM (SELECT unnest(:'lasso_weak'::float8[]) as coef OFFSET 1) sub 
WHERE ABS(coef) > 1e-6;
\echo ''

\echo '   Moderate regularization (λ=1.0):'
WITH coeffs AS (
    SELECT unnest(:'lasso_moderate'::float8[]) as coef,
           generate_series(0, array_length(:'lasso_moderate'::float8[], 1)-1) as idx
)
SELECT idx, ROUND(coef::numeric, 4) as coef FROM coeffs WHERE idx <= 3 ORDER BY idx;

SELECT COUNT(*) as non_zero_features 
FROM (SELECT unnest(:'lasso_moderate'::float8[]) as coef OFFSET 1) sub 
WHERE ABS(coef) > 1e-6;
\echo ''

\echo '   Strong regularization (λ=5.0):'
WITH coeffs AS (
    SELECT unnest(:'lasso_strong'::float8[]) as coef,
           generate_series(0, array_length(:'lasso_strong'::float8[], 1)-1) as idx
)
SELECT idx, ROUND(coef::numeric, 4) as coef FROM coeffs WHERE idx <= 3 ORDER BY idx;

SELECT COUNT(*) as non_zero_features 
FROM (SELECT unnest(:'lasso_strong'::float8[]) as coef OFFSET 1) sub 
WHERE ABS(coef) > 1e-6;
\echo ''

\echo '   Note: As lambda increases, Lasso sets more coefficients to EXACTLY zero (sparse model)'
\echo ''

\echo '=========================================================================='
\echo 'PART 3: Elastic Net (L1 + L2 Regularization)'
\echo '=========================================================================='
\echo ''

\echo 'STEP 6: Train Elastic Net (combines Ridge and Lasso)...'
\echo '   alpha: overall regularization strength'
\echo '   l1_ratio: 0=Ridge only, 1=Lasso only, 0.5=equal mix'
\echo ''

SELECT train_elastic_net('reg_train', 'features', 'target', 1.0, 0.5) AS elastic_net \gset
\echo ''

\echo '=========================================================================='
\echo 'Regularized Regression Complete!'
\echo '=========================================================================='
\echo ''
\echo 'Summary:'
\echo '  ✅ Ridge Regression (L2): Shrinks coefficients, reduces overfitting'
\echo '  ✅ Lasso Regression (L1): Performs feature selection, creates sparse models'
\echo '  ✅ Elastic Net (L1+L2): Combines benefits of both Ridge and Lasso'
\echo ''
\echo 'Use Cases:'
\echo '  - Ridge: When all features are important, prevent overfitting'
\echo '  - Lasso: When feature selection is needed, high-dimensional data'
\echo '  - Elastic Net: When you need both regularization and feature selection'
\echo ''

