-- ====================================================================
-- NeurondB Regression Tests: Reranking Algorithms
-- ====================================================================
-- Tests for MMR, RRF, and Ensemble Reranking
-- Uses real data from: ms_marco.data (passages with text)
-- ====================================================================

\echo '=== Using MS MARCO Dataset for Reranking Tests ==='

-- Create test documents from MS MARCO passages (first 100)
CREATE TEMP TABLE test_rerank_docs AS
SELECT 
    ROW_NUMBER() OVER() as id,
    content,
    -- Generate simple embeddings from text characteristics
    ('[' || 
        (LENGTH(content)::float / 100.0)::text || ',' ||
        (CASE WHEN content ILIKE '%computer%' OR content ILIKE '%technology%' THEN 1.0 ELSE 0.1 END)::text || ',' ||
        (CASE WHEN content ILIKE '%science%' OR content ILIKE '%research%' THEN 1.0 ELSE 0.1 END)::text || ',' ||
        (CASE WHEN content ILIKE '%business%' OR content ILIKE '%market%' THEN 1.0 ELSE 0.1 END)::text ||
    ']')::vector(4) as doc_vec
FROM ms_marco.data
WHERE content IS NOT NULL 
  AND LENGTH(content) > 50
LIMIT 100;

-- Show sample
SELECT id, LEFT(content, 60) || '...' as content_preview, doc_vec
FROM test_rerank_docs
LIMIT 5;

\echo '=== Testing Maximal Marginal Relevance (MMR) ==='

-- Query vector (similar to cat documents)
CREATE TEMP TABLE query_vec AS
SELECT '[0.95, 0.05, 0.0, 0.0]'::vector as qvec;

-- Test MMR reranking with lambda=0.7 (balance relevance and diversity)
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

-- Test Borda count ensemble
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

