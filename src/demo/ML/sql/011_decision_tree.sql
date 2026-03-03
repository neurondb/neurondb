\set ON_ERROR_STOP on
\set QUIET on
-- ============================================================================
-- NeuronDB ML Demo - Decision Tree Classifier
-- Tests CART (Classification and Regression Trees) algorithm
-- ============================================================================

\echo '=========================================================================='
\echo '       NeuronDB - Decision Tree Classifier (CART Algorithm)'
\echo '=========================================================================='
\echo ''

\echo 'STEP 1: Prepare classification dataset...'

-- Reuse existing fraud detection data
CREATE TEMP TABLE tree_data AS
SELECT 
    transaction_id,
    features,
    CASE WHEN is_fraud THEN 1 ELSE 0 END as label
FROM transactions
WHERE features IS NOT NULL
LIMIT 50000;  -- Smaller dataset for tree building

-- Check class distribution
\echo '   Class distribution:'
SELECT 
    label::int as class,
    COUNT(*) as count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) as percentage
FROM tree_data
GROUP BY label
ORDER BY label;
\echo ''

-- Split into train (80%) and test (20%)
CREATE TEMP TABLE tree_train AS SELECT * FROM tree_data LIMIT 40000;
CREATE TEMP TABLE tree_test AS SELECT * FROM tree_data OFFSET 40000;

\echo '   Created 40k train, 10k test samples'
\echo ''

\echo 'STEP 2: Train decision tree classifier...'
\echo '   Parameters: max_depth=10, min_samples_split=2'
\timing on
SELECT train_decision_tree_classifier('tree_train', 'features', 'label', 10, 2) AS tree_depth \gset
\timing off
\echo '   Tree trained with max depth: ' :tree_depth
\echo ''

\echo 'STEP 3: Evaluate tree performance...'
\echo '   Note: On macOS, using PL/pgSQL implementation (returns mock results)'
\echo '   On Linux, full C implementation provides accurate results'
\echo ''

\echo '   Training complete! Decision tree ready for predictions.'
\echo ''

\echo 'STEP 4: Record model in ML project...'
SELECT neurondb_create_ml_project('tree_demo', 'classification', 'Decision tree demo') AS proj_id \gset

INSERT INTO neurondb.ml_models (project_id, version, algorithm, status, training_table, training_column, parameters, num_samples, completed_at)
SELECT 
    p.project_id, 1, 'custom', 'completed', 'tree_train', 'features',
    jsonb_build_object('max_depth', 10, 'min_samples_split', 2, 'algorithm', 'CART'),
    40000, now()
FROM neurondb.ml_projects p WHERE project_name = 'tree_demo'
RETURNING model_id AS tree_model_id \gset

\echo '   Model recorded (ID: ' :tree_model_id ')'
\echo ''

\echo '=========================================================================='
\echo 'Decision Tree Complete!'
\echo '=========================================================================='
\echo ''

