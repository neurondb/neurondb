\set ON_ERROR_STOP on
\set QUIET on

-- ============================================================================
-- Random Forest Classifier Demo - Ensemble Learning
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS neurondb;

\echo '══════════════════════════════════════════════════════════════════'
\echo '  Random Forest Classifier - Ensemble ML Algorithm'
\echo '══════════════════════════════════════════════════════════════════'
\echo ''

-- Step 1: Generate synthetic classification dataset
\echo 'Step 1: Generating classification dataset...'
\echo '  • 5,000 training samples'
\echo '  • 4 features per sample'
\echo '  • Binary classification (fraud/not fraud)'
\echo ''

DROP TABLE IF EXISTS rf_train CASCADE;
CREATE TABLE rf_train AS
SELECT 
    i as id,
    ARRAY[
        (random() * 100)::real,
        (random() * 50)::real,
        (random() * 200)::real,
        (random() * 10)::real
    ]::real[] as features,
    CASE 
        WHEN random() > 0.7 THEN 1
        ELSE 0
    END as label
FROM generate_series(1, 5000) i;

DROP TABLE IF EXISTS rf_test CASCADE;
CREATE TABLE rf_test AS
SELECT 
    i as id,
    ARRAY[
        (random() * 100)::real,
        (random() * 50)::real,
        (random() * 200)::real,
        (random() * 10)::real
    ]::real[] as features,
    CASE 
        WHEN random() > 0.7 THEN 1
        ELSE 0
    END as label
FROM generate_series(1, 1000) i;

\echo '  ✓ Dataset ready'
\echo ''

-- Step 2: Train Random Forest Classifier
\echo 'Step 2: Training Random Forest...'
\echo '  • Algorithm: Ensemble of Decision Trees'
\echo '  • n_trees: 10'
\echo '  • max_depth: 10'
\echo '  • Bootstrap sampling enabled'
\echo ''

\timing on
SELECT train_random_forest_classifier(
    'rf_train',
    'features',
    'label',
    10,    -- n_trees
    10,    -- max_depth
    100    -- min_samples_split
) as model_id;
\timing off

\echo ''

-- Step 3: Make Predictions
\echo 'Step 3: Making predictions on test set...'
\echo ''

\timing on
CREATE TEMP TABLE rf_predictions AS
SELECT 
    t.id,
    t.label as actual,
    predict_random_forest(
        (SELECT train_random_forest_classifier('rf_train', 'features', 'label', 10, 10, 100)),
        t.features
    ) as predicted
FROM rf_test t
LIMIT 1000;
\timing off

\echo ''

-- Step 4: Evaluate Model
\echo 'Step 4: Evaluating Random Forest model...'
\echo ''

\timing on
SELECT 
    COUNT(*) as total_predictions,
    SUM(CASE WHEN actual = predicted THEN 1 ELSE 0 END) as correct,
    ROUND((100.0 * SUM(CASE WHEN actual = predicted THEN 1 ELSE 0 END) / COUNT(*))::numeric, 2) as accuracy_pct,
    SUM(CASE WHEN actual = 1 AND predicted = 1 THEN 1 ELSE 0 END) as true_positives,
    SUM(CASE WHEN actual = 0 AND predicted = 0 THEN 1 ELSE 0 END) as true_negatives,
    SUM(CASE WHEN actual = 0 AND predicted = 1 THEN 1 ELSE 0 END) as false_positives,
    SUM(CASE WHEN actual = 1 AND predicted = 0 THEN 1 ELSE 0 END) as false_negatives
FROM rf_predictions;
\timing off

\echo ''

-- Step 5: Precision and Recall
\echo 'Step 5: Computing Precision and Recall...'
\echo ''

WITH metrics AS (
    SELECT 
        SUM(CASE WHEN actual = 1 AND predicted = 1 THEN 1 ELSE 0 END)::float as tp,
        SUM(CASE WHEN actual = 0 AND predicted = 0 THEN 1 ELSE 0 END)::float as tn,
        SUM(CASE WHEN actual = 0 AND predicted = 1 THEN 1 ELSE 0 END)::float as fp,
        SUM(CASE WHEN actual = 1 AND predicted = 0 THEN 1 ELSE 0 END)::float as fn
    FROM rf_predictions
)
SELECT 
    ROUND((tp / NULLIF(tp + fp, 0))::numeric, 4) as precision,
    ROUND((tp / NULLIF(tp + fn, 0))::numeric, 4) as recall,
    ROUND((2 * tp / NULLIF(2 * tp + fp + fn, 0))::numeric, 4) as f1_score
FROM metrics;

\echo ''

-- Step 6: Feature Importance (simulated)
\echo 'Step 6: Feature Importance Analysis...'
\echo ''

SELECT 
    'Feature ' || i as feature_name,
    ROUND((random() * 100)::numeric, 2) as importance_pct
FROM generate_series(1, 4) i
ORDER BY 2 DESC;

\echo ''
\echo '══════════════════════════════════════════════════════════════════'
\echo '  Random Forest Demo Complete!'
\echo '══════════════════════════════════════════════════════════════════'
\echo ''
\echo 'Key Advantages of Random Forest:'
\echo '  • Ensemble method reduces overfitting'
\echo '  • Handles non-linear relationships well'
\echo '  • Robust to outliers'
\echo '  • Provides feature importance'
\echo '  • No need for feature scaling'
\echo ''
\echo '══════════════════════════════════════════════════════════════════'

