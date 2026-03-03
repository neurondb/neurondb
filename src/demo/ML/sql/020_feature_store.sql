-- ============================================================================
-- NeuronDB Feature Store Demo
-- Feature management, versioning, and serving
-- ============================================================================

\set ON_ERROR_STOP on
\set QUIET on

\echo ''
\echo '══════════════================================================================'
\echo '  Demo 20: Feature Store'
\echo '══════════════================================================================'
\echo ''

-- Create entity data
DROP TABLE IF EXISTS users CASCADE;
CREATE TEMP TABLE users AS
SELECT 
    i as user_id,
    'user_' || i as username,
    (random() * 1000)::numeric(10,2) as total_spent,
    (random() * 100)::integer as num_transactions,
    (random() * 5)::numeric(3,2) as avg_rating
FROM generate_series(1, 1000) i;

\echo 'Entity data created: 1,000 users'
\echo ''

-- Test 1: Create feature store
\echo 'Test 1: neurondb.create_feature_store() - Initialize feature store'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo ''

SELECT neurondb.create_feature_store(
    'user_features',
    'users',
    'user_id'
) as store_id;

\echo ''

-- Test 2: Register features
\echo 'Test 2: neurondb.register_feature() - Register feature definitions'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo ''

SELECT 
    'Total Spent' as feature_name,
    neurondb.register_feature(
        1,
        'total_spent',
        'numeric',
        'total_spent'
    ) as feature_id

UNION ALL

SELECT 
    'Transaction Count' as feature_name,
    neurondb.register_feature(
        1,
        'num_transactions',
        'numeric',
        'num_transactions'
    ) as feature_id

UNION ALL

SELECT 
    'Average Rating' as feature_name,
    neurondb.register_feature(
        1,
        'avg_rating',
        'numeric',
        'avg_rating'
    ) as feature_id

UNION ALL

SELECT 
    'Spending per Transaction' as feature_name,
    neurondb.register_feature(
        1,
        'spending_per_txn',
        'numeric',
        'CASE WHEN num_transactions > 0 THEN total_spent / num_transactions ELSE 0 END'
    ) as feature_id;

\echo ''

-- Test 3: Feature engineering
\echo 'Test 3: neurondb.feature_engineering() - Apply transformations'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo ''

SELECT neurondb.feature_engineering(
    1,
    '{}'::jsonb,
    'users'
) as features_generated;

\echo ''

-- Test 4: Verify feature store tables
\echo 'Test 4: Verify feature store schema'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo ''

\echo 'Feature stores:'
SELECT * FROM neurondb.feature_stores;

\echo ''
\echo 'Feature definitions:'
SELECT * FROM neurondb.features ORDER BY feature_id;

\echo ''
\echo '══════════════================================================================'
\echo '  ✅ Feature Store Demo Complete'
\echo '══════════════================================================================'
\echo ''

