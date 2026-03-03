\set ON_ERROR_STOP on
\set QUIET on
-- ============================================================================
-- NeuronDB ML Demo - Support Vector Machine (SVM)
-- Tests Linear SVM with SMO algorithm
-- ============================================================================

\echo '=========================================================================='
\echo '       NeuronDB - Support Vector Machine (Linear SVM)'
\echo '=========================================================================='
\echo ''

\echo 'STEP 1: Prepare classification dataset...'

-- Reuse existing fraud detection data
CREATE TEMP TABLE svm_data AS
SELECT 
    transaction_id,
    features,
    CASE WHEN is_fraud THEN 1 ELSE 0 END as label
FROM transactions
WHERE features IS NOT NULL
LIMIT 20000;  -- Smaller dataset for SVM (computationally expensive)

-- Check class distribution
\echo '   Class distribution:'
SELECT 
    label::int as class,
    COUNT(*) as count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) as percentage
FROM svm_data
GROUP BY label
ORDER BY label;
\echo ''

-- Split into train (80%) and test (20%)
CREATE TEMP TABLE svm_train AS SELECT * FROM svm_data LIMIT 16000;
CREATE TEMP TABLE svm_test AS SELECT * FROM svm_data OFFSET 16000;

\echo '   Created 16k train, 4k test samples'
\echo ''

\echo 'STEP 2: Train linear SVM classifier...'
\echo '   Parameters: C=1.0 (regularization), max_iters=1000'
\echo '   Algorithm: SMO (Sequential Minimal Optimization)'
\timing on
SELECT train_svm_classifier('svm_train', 'features', 'label', 1.0, 1000) AS svm_alphas \gset
\timing off
\echo '   SVM trained with ' 
SELECT array_length(:'svm_alphas'::float8[], 1) as support_vectors;
\echo ''

\echo 'STEP 3: SVM characteristics...'
\echo '   Type: Linear SVM'
\echo '   Finds: Maximum margin hyperplane'
\echo '   Kernel: Linear (can be extended to RBF, polynomial, etc.)'
\echo '   Output: Support vectors and alpha coefficients'
\echo ''

\echo '   Note: On macOS, using PL/pgSQL implementation (returns mock results)'
\echo '   On Linux, full C implementation provides accurate SVM with SMO'
\echo ''

\echo 'STEP 4: Record model in ML project...'
SELECT neurondb_create_ml_project('svm_demo', 'classification', 'SVM demo') AS proj_id \gset

INSERT INTO neurondb.ml_models (project_id, version, algorithm, status, training_table, training_column, parameters, num_samples, completed_at)
SELECT 
    p.project_id, 1, 'custom', 'completed', 'svm_train', 'features',
    jsonb_build_object('algorithm', 'linear_svm', 'C', 1.0, 'max_iters', 1000, 'kernel', 'linear'),
    16000, now()
FROM neurondb.ml_projects p WHERE project_name = 'svm_demo'
RETURNING model_id AS svm_model_id \gset

\echo '   Model recorded (ID: ' :svm_model_id ')'
\echo ''

\echo '=========================================================================='
\echo 'SVM Complete!'
\echo '=========================================================================='
\echo ''

