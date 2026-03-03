\set ON_ERROR_STOP on
\set QUIET on
-- ============================================================================
-- NeuronDB Fraud Detection Demo - Step 1: Dataset Generation
-- Generates ~100MB dataset (~1.5M transactions) for ML training
-- ============================================================================


\echo '=========================================================================='
\echo '       NeuronDB Fraud Detection - Dataset Generation (100MB)'
\echo '=========================================================================='
\echo ''

-- Cleanup (if needed)
DROP TABLE IF EXISTS transactions CASCADE;
DROP EXTENSION IF EXISTS neurondb CASCADE;

-- Create extensions
CREATE EXTENSION neurondb;

\echo 'Extension loaded'
\echo ''

-- ============================================================================
-- Step 1: Create ML Projects
-- ============================================================================

\echo '========================================================================'
\echo 'Step 1: Creating ML Projects'
\echo '========================================================================'
\echo ''

SELECT neurondb_create_ml_project('fraud_kmeans', 'clustering', 'K-means fraud detection') AS p1 \gset
SELECT neurondb_create_ml_project('fraud_gmm', 'clustering', 'GMM fraud detection') AS p2 \gset
SELECT neurondb_create_ml_project('fraud_hierarchical', 'clustering', 'Hierarchical fraud detection') AS p3 \gset
SELECT neurondb_create_ml_project('fraud_minibatch', 'clustering', 'Mini-batch K-means fraud detection') AS p4 \gset
SELECT neurondb_create_ml_project('fraud_outliers', 'outlier_detection', 'Outlier-based fraud detection') AS p5 \gset

\echo 'Created 5 ML fraud detection projects'
\echo ''

-- ============================================================================
-- Step 2: Generate 100MB Dataset (~1.5M transactions)
-- ============================================================================

\echo '========================================================================'
\echo 'Step 2: Generating ~100MB Dataset (1.5M transactions)'
\echo '========================================================================'
\echo ''

CREATE TABLE transactions (
    transaction_id SERIAL PRIMARY KEY,
    user_id INTEGER,
    amount NUMERIC(10,2),
    transaction_time TIMESTAMP,
    merchant_category TEXT,
    location_distance NUMERIC(10,2),
    is_fraud BOOLEAN,
    features vector(5)
);

CREATE OR REPLACE FUNCTION sigmoid(x float) RETURNS float AS $$
BEGIN
    RETURN 1.0 / (1.0 + exp(-x));
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Helper function to convert GMM probabilities to cluster assignments
CREATE OR REPLACE FUNCTION gmm_to_clusters(probs float8[][])
RETURNS integer[]
LANGUAGE plpgsql IMMUTABLE AS $$
DECLARE
    result integer[];
    n integer;
    k integer;
    i integer;
    j integer;
    max_prob float8;
    best_cluster integer;
BEGIN
    n := array_length(probs, 1);  -- number of vectors
    k := array_length(probs, 2);  -- number of clusters
    result := ARRAY[]::integer[];
    
    FOR i IN 1..n LOOP
        max_prob := -1;
        best_cluster := 1;
        
        FOR j IN 1..k LOOP
            IF probs[i][j] > max_prob THEN
                max_prob := probs[i][j];
                best_cluster := j;
            END IF;
        END LOOP;
        
        result := array_append(result, best_cluster);
    END LOOP;
    
    RETURN result;
END;
$$;

-- Generate data in batches
\echo 'Generating batch 1/3 (500k transactions)...'
INSERT INTO transactions (user_id, amount, transaction_time, merchant_category, location_distance, is_fraud, features)
SELECT 
    (random() * 10000)::int as user_id,
    (random() * 5000 + 10)::numeric(10,2) as amount,
    now() - (random() * interval '365 days') as transaction_time,
    (ARRAY['retail', 'online', 'grocery', 'gas', 'restaurant', 'entertainment'])[(random() * 5 + 1)::int] as merchant_category,
    (random() * 1000)::numeric(10,2) as location_distance,
    (random() > 0.95) as is_fraud,
    ARRAY[
        random()::real,
        random()::real,
        random()::real,
        random()::real,
        random()::real
    ]::vector(5) as features
FROM generate_series(1, 500000);

\echo 'Generating batch 2/3 (500k transactions)...'
INSERT INTO transactions (user_id, amount, transaction_time, merchant_category, location_distance, is_fraud, features)
SELECT 
    (random() * 10000)::int,
    (random() * 5000 + 10)::numeric(10,2),
    now() - (random() * interval '365 days'),
    (ARRAY['retail', 'online', 'grocery', 'gas', 'restaurant', 'entertainment'])[(random() * 5 + 1)::int],
    (random() * 1000)::numeric(10,2),
    (random() > 0.95),
    ARRAY[random()::real, random()::real, random()::real, random()::real, random()::real]::vector(5)
FROM generate_series(1, 500000);

\echo 'Generating batch 3/3 (500k transactions)...'
INSERT INTO transactions (user_id, amount, transaction_time, merchant_category, location_distance, is_fraud, features)
SELECT 
    (random() * 10000)::int,
    (random() * 5000 + 10)::numeric(10,2),
    now() - (random() * interval '365 days'),
    (ARRAY['retail', 'online', 'grocery', 'gas', 'restaurant', 'entertainment'])[(random() * 5 + 1)::int],
    (random() * 1000)::numeric(10,2),
    (random() > 0.95),
    ARRAY[random()::real, random()::real, random()::real, random()::real, random()::real]::vector(5)
FROM generate_series(1, 500000);

-- Create train/test split (80/20)
CREATE VIEW train_data AS 
SELECT * FROM transactions WHERE transaction_id <= 1200000;

CREATE VIEW test_data AS 
SELECT * FROM transactions WHERE transaction_id > 1200000;

-- Show dataset statistics
SELECT 
    COUNT(*) as total_transactions,
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) as fraud_count,
    ROUND(100.0 * SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) / COUNT(*), 2) || '%' as fraud_rate,
    pg_size_pretty(pg_total_relation_size('transactions')) as table_size
FROM transactions;

\echo ''
\echo 'Dataset generation complete!'
\echo '   Ready for ML training. Run 002_first_ML.sql and 003_second_ML.sql'
\echo ''

