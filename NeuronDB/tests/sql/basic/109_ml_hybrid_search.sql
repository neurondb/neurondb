-- ====================================================================
-- NeurondB Regression Tests: Hybrid Search & LTR
-- ====================================================================
-- Tests for Lexical-Semantic Fusion and Learning to Rank
-- Uses real data from: ms_marco.data (passages)
-- ====================================================================

\echo '=== Using MS MARCO Dataset for Hybrid Search Tests ==='

-- Create test documents from MS MARCO with generated embeddings
CREATE TEMP TABLE test_hybrid_docs AS
SELECT 
    ROW_NUMBER() OVER() as id,
    LEFT(content, 50) as title,
    content,
    -- Generate embeddings based on text characteristics
    ('[' || 
        (CASE WHEN content ILIKE '%computer%' OR content ILIKE '%software%' THEN 1.0 ELSE 0.1 END)::text || ',' ||
        (CASE WHEN content ILIKE '%medical%' OR content ILIKE '%health%' THEN 1.0 ELSE 0.1 END)::text || ',' ||
        (CASE WHEN content ILIKE '%business%' OR content ILIKE '%financial%' THEN 1.0 ELSE 0.1 END)::text || ',' ||
        (CASE WHEN content ILIKE '%education%' OR content ILIKE '%learn%' THEN 1.0 ELSE 0.1 END)::text ||
    ']')::vector(4) as embedding
FROM ms_marco.data
WHERE content IS NOT NULL
  AND LENGTH(content) BETWEEN 50 AND 300
LIMIT 50;

-- Show sample
SELECT id, title, embedding
FROM test_hybrid_docs
LIMIT 5;

\echo '=== Testing Lexical-Semantic Hybrid Search ==='

-- Create text search configuration
CREATE INDEX IF NOT EXISTS test_hybrid_docs_fts_idx ON test_hybrid_docs 
    USING gin(to_tsvector('english', title || ' ' || content));

-- Create semantic results (vector similarity)
CREATE TEMP TABLE semantic_results AS
SELECT 
    id,
    1.0 / (1.0 + (embedding <-> '[0.7, 0.7, 0.0, 0.0]'::vector)) as score
FROM test_hybrid_docs
ORDER BY embedding <-> '[0.7, 0.7, 0.0, 0.0]'::vector
LIMIT 10;

-- Create lexical results (text search for "Python")
CREATE TEMP TABLE lexical_results AS
SELECT 
    id,
    ts_rank(to_tsvector('english', title || ' ' || content), 
            plainto_tsquery('english', 'Python')) as score
FROM test_hybrid_docs
WHERE to_tsvector('english', title || ' ' || content) @@ plainto_tsquery('english', 'Python')
ORDER BY score DESC
LIMIT 10;

-- Test hybrid fusion with equal weights (0.5 semantic, 0.5 lexical)
SELECT 
    d.id,
    d.title,
    ROUND(h.combined_score::numeric, 4) as hybrid_score,
    ROUND(h.semantic_score::numeric, 4) as sem_score,
    ROUND(h.lexical_score::numeric, 4) as lex_score
FROM neurondb.hybrid_search_fusion(
    'semantic_results', 'lexical_results',
    'id', 'score', 'score',
    0.5  -- alpha (semantic weight)
) h
JOIN test_hybrid_docs d ON d.id = h.id
ORDER BY h.combined_score DESC;

-- Test with semantic-heavy weighting (0.8 semantic, 0.2 lexical)
SELECT 
    d.id,
    d.title,
    ROUND(h.combined_score::numeric, 4) as hybrid_score
FROM neurondb.hybrid_search_fusion(
    'semantic_results', 'lexical_results',
    'id', 'score', 'score',
    0.8  -- Higher semantic weight
) h
JOIN test_hybrid_docs d ON d.id = h.id
ORDER BY h.combined_score DESC;

-- Test with lexical-heavy weighting (0.2 semantic, 0.8 lexical)
SELECT 
    d.id,
    d.title,
    ROUND(h.combined_score::numeric, 4) as hybrid_score
FROM neurondb.hybrid_search_fusion(
    'semantic_results', 'lexical_results',
    'id', 'score', 'score',
    0.2  -- Higher lexical weight
) h
JOIN test_hybrid_docs d ON d.id = h.id
ORDER BY h.combined_score DESC;

\echo '=== Testing Learning to Rank (LTR) ==='

-- Create candidate documents with multiple features
CREATE TABLE test_ltr_candidates (
    query_id INT,
    doc_id INT,
    semantic_score REAL,
    bm25_score REAL,
    recency_score REAL,
    popularity_score REAL,
    relevance_label INT  -- 0=not relevant, 1=relevant, 2=highly relevant
);

INSERT INTO test_ltr_candidates (query_id, doc_id, semantic_score, bm25_score, recency_score, popularity_score, relevance_label) VALUES
    -- Query 1
    (1, 101, 0.95, 0.80, 0.90, 0.70, 2),  -- Highly relevant
    (1, 102, 0.85, 0.90, 0.70, 0.85, 2),  -- Highly relevant
    (1, 103, 0.70, 0.60, 0.50, 0.60, 1),  -- Somewhat relevant
    (1, 104, 0.50, 0.40, 0.30, 0.45, 0),  -- Not relevant
    (1, 105, 0.65, 0.55, 0.80, 0.50, 1),  -- Somewhat relevant
    -- Query 2
    (2, 201, 0.90, 0.75, 0.85, 0.90, 2),
    (2, 202, 0.80, 0.85, 0.60, 0.75, 1),
    (2, 203, 0.60, 0.50, 0.40, 0.55, 0),
    (2, 204, 0.75, 0.70, 0.75, 0.65, 1),
    (2, 205, 0.95, 0.90, 0.95, 0.95, 2);

-- Train simple LTR model (learned weights)
-- For demo, we'll use pre-defined weights: [0.4, 0.3, 0.2, 0.1]
CREATE TEMP TABLE ltr_weights AS
SELECT 
    ARRAY[0.4, 0.3, 0.2, 0.1]::real[] as weights;

-- Test pointwise LTR reranking for query 1
SELECT 
    doc_id,
    ROUND(score::numeric, 4) as ltr_score,
    relevance_label as actual_relevance
FROM neurondb.ltr_rerank_pointwise(
    'test_ltr_candidates',
    ARRAY['semantic_score', 'bm25_score', 'recency_score', 'popularity_score']::text[],
    (SELECT weights FROM ltr_weights),
    'query_id', 1,
    'doc_id'
)
ORDER BY score DESC;

-- Compare with simple semantic ranking
WITH ltr_ranking AS (
    SELECT 
        doc_id,
        ROW_NUMBER() OVER (ORDER BY score DESC) as ltr_rank
    FROM neurondb.ltr_rerank_pointwise(
        'test_ltr_candidates',
        ARRAY['semantic_score', 'bm25_score', 'recency_score', 'popularity_score']::text[],
        (SELECT weights FROM ltr_weights),
        'query_id', 1,
        'doc_id'
    )
),
semantic_ranking AS (
    SELECT 
        doc_id,
        ROW_NUMBER() OVER (ORDER BY semantic_score DESC) as sem_rank
    FROM test_ltr_candidates
    WHERE query_id = 1
)
SELECT 
    c.doc_id,
    c.relevance_label,
    l.ltr_rank,
    s.sem_rank,
    CASE 
        WHEN l.ltr_rank < s.sem_rank THEN 'LTR Better'
        WHEN l.ltr_rank > s.sem_rank THEN 'Semantic Better'
        ELSE 'Same'
    END as comparison
FROM test_ltr_candidates c
JOIN ltr_ranking l ON c.doc_id = l.doc_id
JOIN semantic_ranking s ON c.doc_id = s.doc_id
WHERE c.query_id = 1
ORDER BY c.relevance_label DESC, l.ltr_rank;

\echo '=== Testing LTR Feature Scoring ==='

-- Score features for a single document
SELECT 
    feature_name,
    ROUND(feature_value::numeric, 4) as value
FROM neurondb.ltr_score_features(
    'test_ltr_candidates',
    ARRAY['semantic_score', 'bm25_score', 'recency_score', 'popularity_score']::text[],
    'query_id', 1,
    'doc_id', 101
)
ORDER BY feature_value DESC;

-- Compare feature distributions for relevant vs non-relevant docs
WITH features AS (
    SELECT 
        c.relevance_label,
        f.feature_name,
        AVG(f.feature_value) as avg_value
    FROM test_ltr_candidates c
    CROSS JOIN LATERAL neurondb.ltr_score_features(
        'test_ltr_candidates',
        ARRAY['semantic_score', 'bm25_score', 'recency_score', 'popularity_score']::text[],
        'query_id', c.query_id,
        'doc_id', c.doc_id
    ) f
    GROUP BY c.relevance_label, f.feature_name
)
SELECT 
    feature_name,
    ROUND(MAX(CASE WHEN relevance_label = 2 THEN avg_value END)::numeric, 4) as highly_relevant_avg,
    ROUND(MAX(CASE WHEN relevance_label = 1 THEN avg_value END)::numeric, 4) as somewhat_relevant_avg,
    ROUND(MAX(CASE WHEN relevance_label = 0 THEN avg_value END)::numeric, 4) as not_relevant_avg
FROM features
GROUP BY feature_name
ORDER BY feature_name;

\echo '=== Edge Cases and Error Handling ==='

-- Test hybrid fusion with only semantic results
CREATE TEMP TABLE only_semantic AS
SELECT * FROM semantic_results;

CREATE TEMP TABLE empty_lexical (id INT, score REAL);

SELECT 
    id,
    ROUND(combined_score::numeric, 4) as score
FROM neurondb.hybrid_search_fusion(
    'only_semantic', 'empty_lexical',
    'id', 'score', 'score',
    0.5
)
ORDER BY combined_score DESC
LIMIT 5;

-- Test LTR with all zero features
CREATE TABLE test_ltr_zeros (
    query_id INT,
    doc_id INT,
    feat1 REAL,
    feat2 REAL
);

INSERT INTO test_ltr_zeros VALUES (1, 1, 0.0, 0.0), (1, 2, 0.0, 0.0);

SELECT 
    doc_id,
    ROUND(score::numeric, 4) as score
FROM neurondb.ltr_rerank_pointwise(
    'test_ltr_zeros',
    ARRAY['feat1', 'feat2']::text[],
    ARRAY[0.5, 0.5]::real[],
    'query_id', 1,
    'doc_id'
);

-- Test with different weight vectors
WITH weight_tests AS (
    SELECT 
        w.name,
        w.weights,
        r.doc_id,
        r.score
    FROM (VALUES 
        ('Balanced', ARRAY[0.25, 0.25, 0.25, 0.25]::real[]),
        ('Semantic Heavy', ARRAY[0.7, 0.1, 0.1, 0.1]::real[]),
        ('BM25 Heavy', ARRAY[0.1, 0.7, 0.1, 0.1]::real[]),
        ('Recency Heavy', ARRAY[0.1, 0.1, 0.7, 0.1]::real[])
    ) w(name, weights)
    CROSS JOIN LATERAL neurondb.ltr_rerank_pointwise(
        'test_ltr_candidates',
        ARRAY['semantic_score', 'bm25_score', 'recency_score', 'popularity_score']::text[],
        w.weights,
        'query_id', 1,
        'doc_id'
    ) r
)
SELECT 
    name as weighting_strategy,
    doc_id,
    ROUND(score::numeric, 4) as ltr_score,
    ROW_NUMBER() OVER (PARTITION BY name ORDER BY score DESC) as rank
FROM weight_tests
ORDER BY name, rank;

-- Cleanup
DROP TABLE test_hybrid_docs CASCADE;
DROP TABLE test_ltr_candidates CASCADE;
DROP TABLE test_ltr_zeros CASCADE;

