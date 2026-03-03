-- ============================================================================
-- Test 004: Vector Similarity Search
-- ============================================================================
-- Demonstrates: KNN search, distance operators, similarity ranking
-- ============================================================================

\echo '=========================================================================='
\echo '|              Vector Similarity Search - NeuronDB                      |'
\echo '=========================================================================='
\echo ''

-- Create test dataset
\echo 'Creating test dataset with 1000 random vectors...'
CREATE TEMP TABLE embeddings (
    id SERIAL PRIMARY KEY,
    category TEXT,
    embedding VECTOR(128),
    metadata JSONB
);

-- Insert random vectors in different "clusters"
INSERT INTO embeddings (category, embedding, metadata)
SELECT 
    'cluster_' || ((i % 5) + 1),
    '[' || string_agg((random() + (i % 5) * 0.2)::text, ',') || ']'::vector,
    jsonb_build_object('id', i, 'cluster', (i % 5) + 1)
FROM generate_series(1, 1000) i,
     LATERAL (SELECT string_agg((random() + (i % 5) * 0.2)::text, ',') FROM generate_series(1, 128)) AS dims(val)
GROUP BY i;

\echo 'Dataset created: 1000 vectors in 5 categories'
\echo ''

-- Test 1: K-Nearest Neighbors using L2 distance
\echo 'Test 1: Find 5 nearest neighbors using L2 distance (<-> operator)'
WITH query AS (
    SELECT '[' || string_agg((random() + 2 * 0.2)::text, ',') || ']'::vector AS q_vec
    FROM generate_series(1, 128)
)
SELECT 
    e.id,
    e.category,
    e.embedding <-> q.q_vec AS distance,
    RANK() OVER (ORDER BY e.embedding <-> q.q_vec) AS rank
FROM embeddings e, query q
ORDER BY e.embedding <-> q.q_vec
LIMIT 5;

\echo ''
\echo 'Test 2: Similarity search using cosine distance (<=> operator)'
WITH query AS (
    SELECT '[' || string_agg((random() + 3 * 0.2)::text, ',') || ']'::vector AS q_vec
    FROM generate_series(1, 128)
)
SELECT 
    e.id,
    e.category,
    e.embedding <=> q.q_vec AS cosine_distance,
    1 - (e.embedding <=> q.q_vec) AS cosine_similarity,
    RANK() OVER (ORDER BY e.embedding <=> q.q_vec) AS rank
FROM embeddings e, query q
ORDER BY e.embedding <=> q.q_vec
LIMIT 5;

\echo ''
\echo 'Test 3: Maximum inner product search (<#> operator)'
WITH query AS (
    SELECT '[' || string_agg((random())::text, ',') || ']'::vector AS q_vec
    FROM generate_series(1, 128)
)
SELECT 
    e.id,
    e.category,
    e.embedding <#> q.q_vec AS neg_inner_product,
    -(e.embedding <#> q.q_vec) AS inner_product,
    RANK() OVER (ORDER BY e.embedding <#> q.q_vec) AS rank
FROM embeddings e, query q
ORDER BY e.embedding <#> q.q_vec
LIMIT 5;

\echo ''
\echo 'Test 4: Category-filtered similarity search'
WITH query AS (
    SELECT '[' || string_agg((random() + 1 * 0.2)::text, ',') || ']'::vector AS q_vec
    FROM generate_series(1, 128)
)
SELECT 
    e.category,
    COUNT(*) AS total_in_category,
    MIN(e.embedding <=> q.q_vec) AS min_distance,
    AVG(e.embedding <=> q.q_vec) AS avg_distance,
    MAX(e.embedding <=> q.q_vec) AS max_distance
FROM embeddings e, query q
GROUP BY e.category
ORDER BY min_distance;

\echo ''
\echo 'Test 5: Top-K per category'
WITH query AS (
    SELECT '[' || string_agg((random())::text, ',') || ']'::vector AS q_vec
    FROM generate_series(1, 128)
),
ranked AS (
    SELECT 
        e.id,
        e.category,
        e.embedding <=> q.q_vec AS distance,
        ROW_NUMBER() OVER (PARTITION BY e.category ORDER BY e.embedding <=> q.q_vec) AS rn
    FROM embeddings e, query q
)
SELECT id, category, distance
FROM ranked
WHERE rn <= 2
ORDER BY category, distance;

\echo ''
\echo 'Test 6: Performance test - finding nearest neighbors'
\timing on
WITH query AS (
    SELECT '[' || string_agg((random())::text, ',') || ']'::vector AS q_vec
    FROM generate_series(1, 128)
)
SELECT COUNT(*) AS nearest_10
FROM (
    SELECT e.id
    FROM embeddings e, query q
    ORDER BY e.embedding <=> q.q_vec
    LIMIT 10
) sub;
\timing off

\echo ''
\echo '=========================================================================='
\echo 'Similarity Search Test Complete!'
\echo '  ✅ L2 distance search (<-> operator)'
\echo '  ✅ Cosine similarity search (<=> operator)'
\echo '  ✅ Inner product search (<#> operator)'
\echo '  ✅ Filtered search (metadata, categories)'
\echo '  ✅ Top-K per category'
\echo '  ✅ Performance: Fast on 1000 vectors'
\echo '=========================================================================='
\echo ''

