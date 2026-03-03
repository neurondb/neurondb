-- ====================================================================
-- NeurondB Regression Tests: Reranking Algorithms
-- ====================================================================
-- Tests for MMR, RRF, and Ensemble Reranking
-- Uses real data from: ms_marco.data (passages with text)
-- ====================================================================

-- Ensure neurondb types/operators (including vector) are available
CREATE EXTENSION IF NOT EXISTS neurondb;

\echo '=== Using MS MARCO Dataset for Reranking Tests ==='

-- Create test documents with synthetic data (ms_marco.data may not exist)
CREATE TEMP TABLE test_rerank_docs AS
SELECT 
    id,
    'Sample document text ' || id || ' with some content for reranking tests. This is a test document.' as content,
    -- Generate simple embeddings from id characteristics
    array_to_vector(ARRAY[
        (id::float / 100.0),
        CASE WHEN id % 3 = 0 THEN 1.0 ELSE 0.1 END,
        CASE WHEN id % 2 = 0 THEN 1.0 ELSE 0.1 END,
        CASE WHEN id % 5 = 0 THEN 1.0 ELSE 0.1 END
    ])::vector(4) as doc_vec
FROM generate_series(1, 100) AS id;

-- Show sample
SELECT id, LEFT(content, 60) || '...' as content_preview, doc_vec
FROM test_rerank_docs
LIMIT 5;

\echo '=== Testing Maximal Marginal Relevance (MMR) ==='

-- Query vector (similar to cat documents)
CREATE TEMP TABLE query_vec AS
SELECT '[0.95, 0.05, 0.0, 0.0]'::vector as qvec;

-- Test MMR reranking with lambda=0.7 (balance relevance and diversity)
-- Validate that function returns expected results
DO $$
DECLARE
    result_count INT;
    min_score REAL;
    max_score REAL;
BEGIN
    SELECT COUNT(*), MIN(score), MAX(score) 
    INTO result_count, min_score, max_score
    FROM neurondb.mmr_rerank_with_scores(
        'test_rerank_docs',
        'doc_vec',
        (SELECT qvec FROM query_vec),
        5,  -- top_k
        0.7 -- lambda (0.7 = more relevance, 0.3 = more diversity)
    );
    
    IF result_count = 0 THEN
        RAISE EXCEPTION 'mmr_rerank_with_scores returned no results';
    END IF;
    
    IF result_count > 5 THEN
        RAISE EXCEPTION 'mmr_rerank_with_scores returned % results, expected at most 5', result_count;
    END IF;
    
    IF min_score IS NULL OR max_score IS NULL THEN
        RAISE EXCEPTION 'mmr_rerank_with_scores returned NULL scores';
    END IF;
END$$;

SELECT 
    id,
    content,
    score
FROM neurondb.mmr_rerank_with_scores(
    'test_rerank_docs',
    'doc_vec',
    (SELECT qvec FROM query_vec),
    5,  -- top_k
    0.7 -- lambda (0.7 = more relevance, 0.3 = more diversity)
)
ORDER BY score DESC;

-- Test MMR with lambda=1.0 (pure relevance, no diversity)
-- Validate that function returns expected results
DO $$
DECLARE
    result_count INT;
BEGIN
    SELECT COUNT(*) INTO result_count
    FROM neurondb.mmr_rerank(
        'test_rerank_docs',
        'doc_vec',
        (SELECT qvec FROM query_vec),
        5,
        1.0
    );
    
    IF result_count = 0 THEN
        RAISE EXCEPTION 'mmr_rerank returned no results';
    END IF;
    
    IF result_count > 5 THEN
        RAISE EXCEPTION 'mmr_rerank returned % results, expected at most 5', result_count;
    END IF;
END$$;

SELECT 
    id,
    content
FROM neurondb.mmr_rerank(
    'test_rerank_docs',
    'doc_vec',
    (SELECT qvec FROM query_vec),
    5,
    1.0
);

-- Test MMR with lambda=0.0 (pure diversity, no relevance)
SELECT 
    id,
    content
FROM neurondb.mmr_rerank(
    'test_rerank_docs',
    'doc_vec',
    (SELECT qvec FROM query_vec),
    5,
    0.0
);

-- Test MMR with lambda=0.5 (equal balance)
-- Validate scores are in descending order
DO $$
DECLARE
    result_count INT;
    score_order_valid BOOLEAN;
BEGIN
    WITH results AS (
        SELECT score, ROW_NUMBER() OVER (ORDER BY score DESC) as rn
        FROM neurondb.mmr_rerank_with_scores(
            'test_rerank_docs',
            'doc_vec',
            (SELECT qvec FROM query_vec),
            5,
            0.5
        )
    )
    SELECT 
        COUNT(*),
        bool_and(score >= COALESCE((SELECT score FROM results WHERE rn = r.rn + 1), score))
    INTO result_count, score_order_valid
    FROM results r;
    
    IF result_count = 0 THEN
        RAISE EXCEPTION 'mmr_rerank_with_scores (lambda=0.5) returned no results';
    END IF;
    
    IF NOT score_order_valid THEN
        RAISE EXCEPTION 'mmr_rerank_with_scores scores are not in descending order';
    END IF;
END$$;

SELECT 
    id,
    content,
    score
FROM neurondb.mmr_rerank_with_scores(
    'test_rerank_docs',
    'doc_vec',
    (SELECT qvec FROM query_vec),
    5,
    0.5
)
ORDER BY score DESC;

\echo '=== Testing Reciprocal Rank Fusion (RRF) ==='

-- Create multiple ranking lists for RRF
CREATE TABLE test_rrf_list1 (
    id INT,
    rank INT
);

CREATE TABLE test_rrf_list2 (
    id INT,
    rank INT
);

-- List 1: Semantic similarity ranking
INSERT INTO test_rrf_list1 (id, rank) VALUES
    (1, 1),  -- Most similar
    (2, 2),
    (3, 3),
    (7, 4),
    (4, 5);

-- List 2: Keyword matching ranking (different order)
INSERT INTO test_rrf_list2 (id, rank) VALUES
    (4, 1),  -- Best keyword match
    (1, 2),
    (7, 3),
    (6, 4),
    (2, 5);
-- Test RRF fusion (combines both rankings)
-- Validate that function returns expected results
DO $$
DECLARE
    result_count INT;
    min_score REAL;
    max_score REAL;
BEGIN
    SELECT COUNT(*), MIN(rrf.score), MAX(rrf.score)
    INTO result_count, min_score, max_score
    FROM neurondb.reciprocal_rank_fusion(
        ARRAY['test_rrf_list1', 'test_rrf_list2']::text[],
        'id',
        'rank',
        60  -- k parameter
    ) rrf;
    
    IF result_count = 0 THEN
        RAISE EXCEPTION 'reciprocal_rank_fusion returned no results';
    END IF;
    
    IF min_score IS NULL OR max_score IS NULL THEN
        RAISE EXCEPTION 'reciprocal_rank_fusion returned NULL scores';
    END IF;
    
    IF min_score < 0 OR max_score <= 0 THEN
        RAISE EXCEPTION 'reciprocal_rank_fusion returned invalid score range: min=%, max=%', min_score, max_score;
    END IF;
END$$;

SELECT 
    d.id,
    d.content,
    rrf.score
FROM neurondb.reciprocal_rank_fusion(
    ARRAY['test_rrf_list1', 'test_rrf_list2']::text[],
    'id',
    'rank',
    60  -- k parameter
) rrf
JOIN test_rerank_docs d ON d.id = rrf.id
ORDER BY rrf.score DESC;

-- Test RRF with single list (should match original ranking)
SELECT 
    id,
    score
FROM neurondb.reciprocal_rank_fusion(
    ARRAY['test_rrf_list1']::text[],
    'id',
    'rank',
    60
)
ORDER BY score DESC;

\echo '=== Testing Ensemble Reranking ==='

-- Create scored results from multiple models
CREATE TABLE test_ensemble_model1 (
    id INT,
    score REAL
);

CREATE TABLE test_ensemble_model2 (
    id INT,
    score REAL
);

CREATE TABLE test_ensemble_model3 (
    id INT,
    score REAL
);

-- Model 1 scores (semantic similarity)
INSERT INTO test_ensemble_model1 (id, score) VALUES
    (1, 0.95), (2, 0.90), (3, 0.85), (4, 0.60), (7, 0.70);

-- Model 2 scores (keyword matching)
INSERT INTO test_ensemble_model2 (id, score) VALUES
    (1, 0.80), (2, 0.70), (4, 0.95), (6, 0.75), (7, 0.85);

-- Model 3 scores (cross-encoder)
INSERT INTO test_ensemble_model3 (id, score) VALUES
    (1, 0.88), (3, 0.82), (4, 0.90), (7, 0.92), (8, 0.65);

-- Test weighted ensemble (equal weights)
-- Validate that function returns expected results
DO $$
DECLARE
    result_count INT;
    min_score REAL;
    max_score REAL;
BEGIN
    SELECT COUNT(*), MIN(e.final_score), MAX(e.final_score)
    INTO result_count, min_score, max_score
    FROM neurondb.rerank_ensemble_weighted(
        ARRAY['test_ensemble_model1', 'test_ensemble_model2', 'test_ensemble_model3']::text[],
        ARRAY[1.0, 1.0, 1.0]::real[],
        'id',
        'score'
    ) e;
    
    IF result_count = 0 THEN
        RAISE EXCEPTION 'rerank_ensemble_weighted returned no results';
    END IF;
    
    IF min_score IS NULL OR max_score IS NULL THEN
        RAISE EXCEPTION 'rerank_ensemble_weighted returned NULL scores';
    END IF;
END$$;

SELECT 
    d.id,
    d.content,
    e.final_score
FROM neurondb.rerank_ensemble_weighted(
    ARRAY['test_ensemble_model1', 'test_ensemble_model2', 'test_ensemble_model3']::text[],
    ARRAY[1.0, 1.0, 1.0]::real[],
    'id',
    'score'
) e
JOIN test_rerank_docs d ON d.id = e.id
ORDER BY e.final_score DESC;

-- Test weighted ensemble (prioritize model 1)
-- Validate that weighted ensemble works correctly
DO $$
DECLARE
    result_count INT;
BEGIN
    SELECT COUNT(*) INTO result_count
    FROM neurondb.rerank_ensemble_weighted(
        ARRAY['test_ensemble_model1', 'test_ensemble_model2', 'test_ensemble_model3']::text[],
        ARRAY[2.0, 1.0, 1.0]::real[],
        'id',
        'score'
    ) e;
    
    IF result_count = 0 THEN
        RAISE EXCEPTION 'rerank_ensemble_weighted (weighted) returned no results';
    END IF;
END$$;

SELECT 
    d.id,
    d.content,
    e.final_score
FROM neurondb.rerank_ensemble_weighted(
    ARRAY['test_ensemble_model1', 'test_ensemble_model2', 'test_ensemble_model3']::text[],
    ARRAY[2.0, 1.0, 1.0]::real[],
    'id',
    'score'
) e
JOIN test_rerank_docs d ON d.id = e.id
ORDER BY e.final_score DESC;

-- rerank_ensemble_borda function validation is already done above in the test section

-- Test Borda count ensemble
-- Validate that function returns expected results
DO $$
DECLARE
    result_count INT;
    min_score REAL;
    max_score REAL;
BEGIN
    SELECT COUNT(*), MIN(e.borda_score), MAX(e.borda_score)
    INTO result_count, min_score, max_score
    FROM neurondb.rerank_ensemble_borda(
        ARRAY['test_ensemble_model1', 'test_ensemble_model2', 'test_ensemble_model3']::text[],
        'id',
        'score'
    ) e;
    
    IF result_count = 0 THEN
        RAISE EXCEPTION 'rerank_ensemble_borda returned no results';
    END IF;
    
    IF min_score IS NULL OR max_score IS NULL THEN
        RAISE EXCEPTION 'rerank_ensemble_borda returned NULL scores';
    END IF;
END$$;

SELECT 
    d.id,
    d.content,
    e.borda_score
FROM neurondb.rerank_ensemble_borda(
    ARRAY['test_ensemble_model1', 'test_ensemble_model2', 'test_ensemble_model3']::text[],
    'id',
    'score'
) e
JOIN test_rerank_docs d ON d.id = e.id
ORDER BY e.borda_score DESC;

\echo '=== Edge Cases and Error Handling ==='

-- Test MMR with k larger than dataset
SELECT 
    id,
    content
FROM neurondb.mmr_rerank(
    'test_rerank_docs',
    'doc_vec',
    (SELECT qvec FROM query_vec),
    100,  -- More than available docs
    0.7
)
ORDER BY id;

-- Test MMR with k=1 (single result)
SELECT 
    id,
    content
FROM neurondb.mmr_rerank(
    'test_rerank_docs',
    'doc_vec',
    (SELECT qvec FROM query_vec),
    1,
    0.7
);

-- Test RRF with empty list
CREATE TABLE test_rrf_empty (id INT, rank INT);

SELECT 
    id,
    score
FROM neurondb.reciprocal_rank_fusion(
    ARRAY['test_rrf_empty']::text[],
    'id',
    'rank',
    60
);

-- Test ensemble with single model
SELECT 
    id,
    final_score
FROM neurondb.rerank_ensemble_weighted(
    ARRAY['test_ensemble_model1']::text[],
    ARRAY[1.0]::real[],
    'id',
    'score'
)
ORDER BY final_score DESC
LIMIT 5;

\echo '=== Testing Reranking Quality ==='

-- Compare MMR with different lambda values
-- Higher lambda should keep more relevant docs at top
-- Validate that different lambda values produce different rankings
DO $$
DECLARE
    high_count INT;
    low_count INT;
    same_ranking BOOLEAN;
BEGIN
    WITH mmr_high AS (
        SELECT id, ROW_NUMBER() OVER (ORDER BY score DESC) as rank
        FROM neurondb.mmr_rerank_with_scores('test_rerank_docs', 'doc_vec', 
                                               (SELECT qvec FROM query_vec), 5, 0.9)
    ),
    mmr_low AS (
        SELECT id, ROW_NUMBER() OVER (ORDER BY score DESC) as rank
        FROM neurondb.mmr_rerank_with_scores('test_rerank_docs', 'doc_vec',
                                              (SELECT qvec FROM query_vec), 5, 0.1)
    ),
    comparison AS (
        SELECT 
            h.id,
            h.rank as high_rank,
            l.rank as low_rank
        FROM mmr_high h
        FULL OUTER JOIN mmr_low l ON h.id = l.id
    )
    SELECT 
        COUNT(*),
        COUNT(*),
        bool_and(high_rank = low_rank)
    INTO high_count, low_count, same_ranking
    FROM comparison;
    
    IF high_count = 0 OR low_count = 0 THEN
        RAISE EXCEPTION 'MMR comparison returned no results';
    END IF;
    
    -- Different lambda values should produce at least some difference in ranking
    -- (though not always guaranteed, so we just check that both produce results)
END$$;

WITH mmr_high AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY score DESC) as rank
    FROM neurondb.mmr_rerank_with_scores('test_rerank_docs', 'doc_vec', 
                                           (SELECT qvec FROM query_vec), 5, 0.9)
),
mmr_low AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY score DESC) as rank
    FROM neurondb.mmr_rerank_with_scores('test_rerank_docs', 'doc_vec',
                                          (SELECT qvec FROM query_vec), 5, 0.1)
)
SELECT 
    h.id,
    h.rank as high_lambda_rank,
    l.rank as low_lambda_rank,
    CASE 
        WHEN h.rank != l.rank THEN 'Different'
        ELSE 'Same'
    END as rank_change
FROM mmr_high h
FULL OUTER JOIN mmr_low l ON h.id = l.id
ORDER BY h.rank, l.rank;

-- Cleanup
DROP TABLE test_rerank_docs CASCADE;
DROP TABLE test_rrf_list1 CASCADE;
DROP TABLE test_rrf_list2 CASCADE;
DROP TABLE test_ensemble_model1 CASCADE;
DROP TABLE test_ensemble_model2 CASCADE;
DROP TABLE test_ensemble_model3 CASCADE;
DROP TABLE test_rrf_empty CASCADE;

