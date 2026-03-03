\set ON_ERROR_STOP on
\set QUIET on
-- ============================================================================
-- NeuronDB ML Demo - Naive Bayes Classifier
-- Tests Gaussian Naive Bayes for continuous features
-- ============================================================================

\echo '=========================================================================='
\echo '       NeuronDB - Gaussian Naive Bayes Classifier'
\echo '=========================================================================='
\echo ''

\echo 'STEP 1: Prepare classification dataset...'

-- Reuse existing fraud detection data
CREATE TEMP TABLE nb_data AS
SELECT 
    transaction_id,
    features,
    CASE WHEN is_fraud THEN 1 ELSE 0 END as label
FROM transactions
WHERE features IS NOT NULL
LIMIT 50000;

-- Check class distribution
\echo '   Class distribution:'
SELECT 
    label::int as class,
    COUNT(*) as count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) as percentage
FROM nb_data
GROUP BY label
ORDER BY label;
\echo ''

-- Split into train (80%) and test (20%)
CREATE TEMP TABLE nb_train AS SELECT * FROM nb_data LIMIT 40000;
CREATE TEMP TABLE nb_test AS SELECT * FROM nb_data OFFSET 40000;

\echo '   Created 40k train, 10k test samples'
\echo ''

\echo 'STEP 2: Train Gaussian Naive Bayes classifier...'
\echo '   Assumes features follow Gaussian distribution within each class'
\timing on
SELECT train_naive_bayes_classifier('nb_train', 'features', 'label') AS nb_params \gset
\timing off
\echo '   Model trained with parameters array length: ' 
SELECT array_length(:'nb_params'::float8[], 1) as param_count;
\echo ''

\echo 'STEP 3: Model characteristics...'
\echo '   Algorithm: Gaussian Naive Bayes'
\echo '   Assumptions: Feature independence, Gaussian distribution'
\echo '   Output: Class probabilities via Bayes theorem'
\echo ''

\echo '   Note: On macOS, using PL/pgSQL implementation (returns mock results)'
\echo '   On Linux, full C implementation provides accurate Gaussian NB'
\echo ''

\echo 'STEP 4: Record model in ML project...'
SELECT neurondb_create_ml_project('nb_demo', 'classification', 'Naive Bayes demo') AS proj_id \gset

INSERT INTO neurondb.ml_models (project_id, version, algorithm, status, training_table, training_column, parameters, num_samples, completed_at)
SELECT 
    p.project_id, 1, 'custom', 'completed', 'nb_train', 'features',
    jsonb_build_object('algorithm', 'gaussian_naive_bayes', 'n_features', 5, 'n_classes', 2),
    40000, now()
FROM neurondb.ml_projects p WHERE project_name = 'nb_demo'
RETURNING model_id AS nb_model_id \gset

\echo '   Model recorded (ID: ' :nb_model_id ')'
\echo ''

\echo '=========================================================================='
\echo 'Naive Bayes Complete!'
\echo '=========================================================================='
\echo ''

