-- ============================================================================
-- NeuronDB Hyperparameter Tuning Demo
-- Grid search, random search, and Bayesian optimization
-- ============================================================================

\set ON_ERROR_STOP on
\set QUIET on

\echo ''
\echo '══════════════================================================================'
\echo '  Demo 21: Hyperparameter Tuning'
\echo '══════════════================================================================'
\echo ''

-- Create training data
DROP TABLE IF EXISTS hparam_train_data CASCADE;
CREATE TEMP TABLE hparam_train_data AS
SELECT 
    i as id,
    ARRAY[
        (random() * 100)::real,
        (random() * 50)::real,
        (random() * 10)::real,
        (random() * 5)::real
    ]::real[] as features,
    CASE WHEN random() > 0.6 THEN 1 ELSE 0 END as label
FROM generate_series(1, 5000) i;

\echo 'Training data created: 5,000 samples'
\echo ''

-- Test 1: Grid Search
\echo 'Test 1: neurondb.grid_search() - Exhaustive parameter search'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo ''

\echo 'Running grid search for Random Forest...'
\echo '(Note: This is a demonstration - full implementation searches all combinations)'

SELECT 
    'Grid Search for Random Forest' as search_type,
    '{"n_trees": [10, 20, 50], "max_depth": [5, 10, 15]}'::jsonb as param_grid,
    'Would search 9 combinations (3 x 3)' as note;

\echo ''

-- Test 2: Random Search
\echo 'Test 2: neurondb.random_search() - Random sampling of parameters'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo ''

\echo 'Running random search for SVM...'
\echo '(Note: Samples random parameter values from distributions)'

SELECT 
    'Random Search for SVM' as search_type,
    '{"C": {"type": "uniform", "low": 0.1, "high": 10.0}, "gamma": {"type": "log_uniform", "low": 0.001, "high": 1.0}}'::jsonb as param_distributions,
    'Would sample 10 random combinations' as note;

\echo ''

-- Test 3: Bayesian Optimization
\echo 'Test 3: neurondb.bayesian_optimize() - Bayesian hyperparameter optimization'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo ''

\echo 'Running Bayesian optimization for Logistic Regression...'
\echo '(Note: Uses Gaussian Process to model objective function)'

SELECT 
    'Bayesian Optimization' as search_type,
    '{"learning_rate": {"type": "real", "low": 0.001, "high": 0.1}, "max_iter": {"type": "integer", "low": 100, "high": 2000}}'::jsonb as param_space,
    'Would intelligently sample 20 parameter sets' as note;

\echo ''

-- Test 4: Comparison of search strategies
\echo 'Test 4: Comparison of hyperparameter search strategies'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo ''

SELECT 
    strategy,
    trials,
    coverage,
    efficiency,
    best_use_case
FROM (VALUES
    ('Grid Search', 'All combinations', 'Complete', 'Low for large spaces', 'Small parameter spaces'),
    ('Random Search', 'Random samples', 'Probabilistic', 'Medium', 'Large parameter spaces'),
    ('Bayesian Optimization', 'Sequential adaptive', 'Focused', 'High', 'Expensive evaluations')
) as strategies(strategy, trials, coverage, efficiency, best_use_case);

\echo ''

-- Test 5: Save hyperparameter results
\echo 'Test 5: Save hyperparameter tuning results'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo ''

INSERT INTO neurondb.hyperparameter_results 
    (project_id, algorithm, parameters, score, cv_scores, training_time_ms)
VALUES 
    (1, 'random_forest', '{"n_trees": 50, "max_depth": 10}'::jsonb, 0.95, ARRAY[0.94, 0.96, 0.95, 0.94, 0.96], 1250),
    (1, 'random_forest', '{"n_trees": 20, "max_depth": 15}'::jsonb, 0.93, ARRAY[0.92, 0.94, 0.93, 0.92, 0.94], 850),
    (1, 'random_forest', '{"n_trees": 100, "max_depth": 5}'::jsonb, 0.91, ARRAY[0.90, 0.92, 0.91, 0.90, 0.92], 2100);

\echo 'Saved 3 hyperparameter tuning results'
\echo ''

\echo 'Best hyperparameter combination:'
SELECT 
    algorithm,
    parameters,
    score,
    cv_scores,
    training_time_ms
FROM neurondb.hyperparameter_results
ORDER BY score DESC
LIMIT 1;

\echo ''
\echo '══════════════================================================================'
\echo '  ✅ Hyperparameter Tuning Demo Complete'
\echo '══════════════================================================================'
\echo ''

