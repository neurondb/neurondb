-- ====================================================================
-- NeurondB Regression Tests: ML Quality Metrics
-- ====================================================================
-- Tests for Recall@K, Precision@K, F1@K, MRR, and clustering metrics
-- Uses real data from: ms_marco.data (for retrieval metrics)
-- ====================================================================

\echo '=== Using MS MARCO Dataset for Retrieval Metrics Tests ==='

-- Create ground truth from MS MARCO (simulate relevant documents)
CREATE TEMP TABLE test_ground_truth AS
SELECT 
    (id % 3) + 1 as query_id,
    id as relevant_doc_id
FROM (
    SELECT ROW_NUMBER() OVER() as id
    FROM ms_marco.data
    LIMIT 15
) sub;

-- Create predictions (simulate search results with some overlap)
CREATE TEMP TABLE test_predictions AS
WITH docs AS (
    SELECT ROW_NUMBER() OVER() as id
    FROM ms_marco.data
    LIMIT 30
)
SELECT 
    (id % 3) + 1 as query_id,
    id as predicted_doc_id,
    ROW_NUMBER() OVER(PARTITION BY (id % 3) + 1 ORDER BY id) as rank
FROM docs
WHERE ROW_NUMBER() OVER(PARTITION BY (id % 3) + 1 ORDER BY id) <= 5;

-- Show sample data
SELECT 'Ground Truth' as type, query_id, COUNT(*) as num_relevant
FROM test_ground_truth
GROUP BY query_id
UNION ALL
SELECT 'Predictions' as type, query_id, COUNT(*) as num_predictions
FROM test_predictions
GROUP BY query_id
ORDER BY query_id, type;

\echo '=== Testing Recall@K Metrics ==='

-- Test Recall@5 for each query
SELECT 
    query_id,
    ROUND(neurondb.recall_at_k(
        'test_ground_truth', 'relevant_doc_id', 
        'test_predictions', 'predicted_doc_id',
        'query_id', query_id, 5
    )::numeric, 4) as recall_at_5
FROM (SELECT DISTINCT query_id FROM test_predictions) q
ORDER BY query_id;

-- Test Recall@3
SELECT 
    query_id,
    ROUND(neurondb.recall_at_k(
        'test_ground_truth', 'relevant_doc_id',
        'test_predictions', 'predicted_doc_id',
        'query_id', query_id, 3
    )::numeric, 4) as recall_at_3
FROM (SELECT DISTINCT query_id FROM test_predictions) q
ORDER BY query_id;

-- Test Recall@1
SELECT 
    query_id,
    ROUND(neurondb.recall_at_k(
        'test_ground_truth', 'relevant_doc_id',
        'test_predictions', 'predicted_doc_id',
        'query_id', query_id, 1
    )::numeric, 4) as recall_at_1
FROM (SELECT DISTINCT query_id FROM test_predictions) q
ORDER BY query_id;

\echo '=== Testing Precision@K Metrics ==='

-- Test Precision@5 for each query
SELECT 
    query_id,
    ROUND(neurondb.precision_at_k(
        'test_ground_truth', 'relevant_doc_id',
        'test_predictions', 'predicted_doc_id',
        'query_id', query_id, 5
    )::numeric, 4) as precision_at_5
FROM (SELECT DISTINCT query_id FROM test_predictions) q
ORDER BY query_id;

-- Test Precision@3
SELECT 
    query_id,
    ROUND(neurondb.precision_at_k(
        'test_ground_truth', 'relevant_doc_id',
        'test_predictions', 'predicted_doc_id',
        'query_id', query_id, 3
    )::numeric, 4) as precision_at_3
FROM (SELECT DISTINCT query_id FROM test_predictions) q
ORDER BY query_id;

-- Test Precision@1 (always 0 or 1)
SELECT 
    query_id,
    ROUND(neurondb.precision_at_k(
        'test_ground_truth', 'relevant_doc_id',
        'test_predictions', 'predicted_doc_id',
        'query_id', query_id, 1
    )::numeric, 4) as precision_at_1
FROM (SELECT DISTINCT query_id FROM test_predictions) q
ORDER BY query_id;

\echo '=== Testing F1@K Metrics ==='

-- Test F1@5 (harmonic mean of Precision@5 and Recall@5)
SELECT 
    query_id,
    ROUND(neurondb.f1_at_k(
        'test_ground_truth', 'relevant_doc_id',
        'test_predictions', 'predicted_doc_id',
        'query_id', query_id, 5
    )::numeric, 4) as f1_at_5
FROM (SELECT DISTINCT query_id FROM test_predictions) q
ORDER BY query_id;

-- Test F1@3
SELECT 
    query_id,
    ROUND(neurondb.f1_at_k(
        'test_ground_truth', 'relevant_doc_id',
        'test_predictions', 'predicted_doc_id',
        'query_id', query_id, 3
    )::numeric, 4) as f1_at_3
FROM (SELECT DISTINCT query_id FROM test_predictions) q
ORDER BY query_id;

\echo '=== Testing Mean Reciprocal Rank (MRR) ==='

-- Test MRR (average of reciprocal ranks of first relevant result)
SELECT 
    query_id,
    ROUND(neurondb.mean_reciprocal_rank(
        'test_ground_truth', 'relevant_doc_id',
        'test_predictions', 'predicted_doc_id',
        'query_id', query_id
    )::numeric, 4) as mrr
FROM (SELECT DISTINCT query_id FROM test_predictions) q
ORDER BY query_id;

-- Average MRR across all queries
SELECT 
    ROUND(AVG(neurondb.mean_reciprocal_rank(
        'test_ground_truth', 'relevant_doc_id',
        'test_predictions', 'predicted_doc_id',
        'query_id', query_id
    ))::numeric, 4) as avg_mrr
FROM (SELECT DISTINCT query_id FROM test_predictions) q;

\echo '=== Testing Comprehensive Metrics Report ==='

-- Generate full metrics report for all queries at k=5
WITH metrics AS (
    SELECT 
        query_id,
        neurondb.recall_at_k('test_ground_truth', 'relevant_doc_id', 
                             'test_predictions', 'predicted_doc_id',
                             'query_id', query_id, 5) as recall_5,
        neurondb.precision_at_k('test_ground_truth', 'relevant_doc_id',
                                'test_predictions', 'predicted_doc_id',
                                'query_id', query_id, 5) as precision_5,
        neurondb.f1_at_k('test_ground_truth', 'relevant_doc_id',
                         'test_predictions', 'predicted_doc_id',
                         'query_id', query_id, 5) as f1_5,
        neurondb.mean_reciprocal_rank('test_ground_truth', 'relevant_doc_id',
                                      'test_predictions', 'predicted_doc_id',
                                      'query_id', query_id) as mrr
    FROM (SELECT DISTINCT query_id FROM test_predictions) q
)
SELECT 
    query_id,
    ROUND(recall_5::numeric, 4) as recall_at_5,
    ROUND(precision_5::numeric, 4) as precision_at_5,
    ROUND(f1_5::numeric, 4) as f1_at_5,
    ROUND(mrr::numeric, 4) as mrr
FROM metrics
ORDER BY query_id;

-- Average metrics across all queries
WITH metrics AS (
    SELECT 
        query_id,
        neurondb.recall_at_k('test_ground_truth', 'relevant_doc_id',
                             'test_predictions', 'predicted_doc_id',
                             'query_id', query_id, 5) as recall_5,
        neurondb.precision_at_k('test_ground_truth', 'relevant_doc_id',
                                'test_predictions', 'predicted_doc_id',
                                'query_id', query_id, 5) as precision_5,
        neurondb.f1_at_k('test_ground_truth', 'relevant_doc_id',
                         'test_predictions', 'predicted_doc_id',
                         'query_id', query_id, 5) as f1_5,
        neurondb.mean_reciprocal_rank('test_ground_truth', 'relevant_doc_id',
                                      'test_predictions', 'predicted_doc_id',
                                      'query_id', query_id) as mrr
    FROM (SELECT DISTINCT query_id FROM test_predictions) q
)
SELECT 
    'Average' as metric_type,
    ROUND(AVG(recall_5)::numeric, 4) as recall_at_5,
    ROUND(AVG(precision_5)::numeric, 4) as precision_at_5,
    ROUND(AVG(f1_5)::numeric, 4) as f1_at_5,
    ROUND(AVG(mrr)::numeric, 4) as mrr
FROM metrics;

\echo '=== Edge Cases and Error Handling ==='

-- Test with perfect predictions (all relevant docs in order)
CREATE TABLE test_perfect_pred (
    query_id INT,
    predicted_doc_id INT,
    rank INT
);

INSERT INTO test_perfect_pred (query_id, predicted_doc_id, rank) VALUES
    (1, 101, 1), (1, 102, 2), (1, 103, 3), (1, 104, 4), (1, 105, 5);

-- Perfect Recall@5, Precision@5, F1@5
SELECT 
    'Perfect' as case_type,
    ROUND(neurondb.recall_at_k('test_ground_truth', 'relevant_doc_id',
                                'test_perfect_pred', 'predicted_doc_id',
                                'query_id', 1, 5)::numeric, 4) as recall,
    ROUND(neurondb.precision_at_k('test_ground_truth', 'relevant_doc_id',
                                   'test_perfect_pred', 'predicted_doc_id',
                                   'query_id', 1, 5)::numeric, 4) as precision,
    ROUND(neurondb.f1_at_k('test_ground_truth', 'relevant_doc_id',
                            'test_perfect_pred', 'predicted_doc_id',
                            'query_id', 1, 5)::numeric, 4) as f1;

-- Test with no relevant documents retrieved
CREATE TABLE test_zero_pred (
    query_id INT,
    predicted_doc_id INT,
    rank INT
);

INSERT INTO test_zero_pred (query_id, predicted_doc_id, rank) VALUES
    (1, 999, 1), (1, 998, 2), (1, 997, 3), (1, 996, 4), (1, 995, 5);

-- Zero Recall, Precision, F1
SELECT 
    'Zero Relevant' as case_type,
    ROUND(neurondb.recall_at_k('test_ground_truth', 'relevant_doc_id',
                                'test_zero_pred', 'predicted_doc_id',
                                'query_id', 1, 5)::numeric, 4) as recall,
    ROUND(neurondb.precision_at_k('test_ground_truth', 'relevant_doc_id',
                                   'test_zero_pred', 'predicted_doc_id',
                                   'query_id', 1, 5)::numeric, 4) as precision,
    ROUND(neurondb.f1_at_k('test_ground_truth', 'relevant_doc_id',
                            'test_zero_pred', 'predicted_doc_id',
                            'query_id', 1, 5)::numeric, 4) as f1;

-- Test with k larger than number of predictions
SELECT 
    'Large K' as case_type,
    ROUND(neurondb.recall_at_k('test_ground_truth', 'relevant_doc_id',
                                'test_predictions', 'predicted_doc_id',
                                'query_id', 1, 100)::numeric, 4) as recall_at_100;

-- Cleanup
DROP TABLE test_ground_truth CASCADE;
DROP TABLE test_predictions CASCADE;
DROP TABLE test_perfect_pred CASCADE;
DROP TABLE test_zero_pred CASCADE;

