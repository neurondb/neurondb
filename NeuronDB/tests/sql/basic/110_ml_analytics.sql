-- ====================================================================
-- NeurondB Regression Tests: ML Analytics & Graph
-- ====================================================================
-- Tests for KNN Graph, Embedding Quality, Topic Discovery, Histograms
-- Uses real data from: deep1b.vectors for analytics
-- ====================================================================

\echo '=== Using Deep1B Dataset for Analytics Tests ==='

-- Create test data from Deep1B vectors (first 500 for speed)
CREATE TEMP TABLE test_graph_data AS
SELECT 
    id,
    array_to_vector(embedding[1:8])::vector(8) as vec,
    -- Assign labels based on vector characteristics for clustering validation
    CASE 
        WHEN id % 3 = 0 THEN 'A'
        WHEN id % 3 = 1 THEN 'B'
        ELSE 'C'
    END as label
FROM deep1b.vectors
WHERE id <= 500
LIMIT 500;

-- Show sample
SELECT COUNT(*) as total_vectors, vector_dims(vec) as dimensions,
       COUNT(DISTINCT label) as num_labels
FROM test_graph_data;

\echo '=== Testing KNN Graph Construction ==='

-- Build KNN graph with k=3
SELECT 
    node_id,
    neighbor_id,
    ROUND(distance::numeric, 4) as dist
FROM neurondb.build_knn_graph('test_graph_data', 'vec', 3)
ORDER BY node_id, distance;

-- Verify each node has k neighbors
SELECT 
    node_id,
    COUNT(*) as num_neighbors
FROM neurondb.build_knn_graph('test_graph_data', 'vec', 3)
GROUP BY node_id
ORDER BY node_id;

-- Build graph with k=2
SELECT 
    node_id,
    COUNT(*) as num_neighbors
FROM neurondb.build_knn_graph('test_graph_data', 'vec', 2)
GROUP BY node_id
ORDER BY node_id;

-- Check if neighbors are from same cluster (graph quality)
WITH graph AS (
    SELECT * FROM neurondb.build_knn_graph('test_graph_data', 'vec', 3)
)
SELECT 
    t1.label as node_label,
    t2.label as neighbor_label,
    COUNT(*) as edge_count,
    CASE 
        WHEN t1.label = t2.label THEN 'Same Cluster'
        ELSE 'Cross Cluster'
    END as edge_type
FROM graph g
JOIN test_graph_data t1 ON g.node_id = t1.id
JOIN test_graph_data t2 ON g.neighbor_id = t2.id
GROUP BY t1.label, t2.label, edge_type
ORDER BY node_label, neighbor_label;

\echo '=== Testing Embedding Quality Metrics ==='

-- Test embedding quality metrics
SELECT 
    metric,
    ROUND(value::numeric, 4) as score
FROM neurondb.compute_embedding_quality('test_graph_data', 'vec')
ORDER BY metric;

-- Create higher quality embeddings (well-separated clusters)
CREATE TABLE test_quality_high (
    id SERIAL PRIMARY KEY,
    vec vector(3)
);

INSERT INTO test_quality_high (vec) VALUES
    ('[10.0, 0.0, 0.0]'::vector),
    ('[11.0, 0.0, 0.0]'::vector),
    ('[0.0, 10.0, 0.0]'::vector),
    ('[0.0, 11.0, 0.0]'::vector),
    ('[0.0, 0.0, 10.0]'::vector),
    ('[0.0, 0.0, 11.0]'::vector);

-- Compare quality scores
WITH high_quality AS (
    SELECT 'High Separation' as dataset, metric, value
    FROM neurondb.compute_embedding_quality('test_quality_high', 'vec')
),
low_quality AS (
    SELECT 'Low Separation' as dataset, metric, value
    FROM neurondb.compute_embedding_quality('test_graph_data', 'vec')
)
SELECT 
    metric,
    ROUND(MAX(CASE WHEN dataset = 'High Separation' THEN value END)::numeric, 4) as high_sep,
    ROUND(MAX(CASE WHEN dataset = 'Low Separation' THEN value END)::numeric, 4) as low_sep
FROM (SELECT * FROM high_quality UNION ALL SELECT * FROM low_quality) all_metrics
GROUP BY metric
ORDER BY metric;

\echo '=== Testing Similarity Histogram ==='

-- Create similarity histogram (distribution of pairwise distances)
SELECT 
    bin,
    ROUND(bin_min::numeric, 2) as min_dist,
    ROUND(bin_max::numeric, 2) as max_dist,
    count as num_pairs,
    ROUND(frequency::numeric, 4) as freq
FROM neurondb.similarity_histogram('test_graph_data', 'vec', 5)
ORDER BY bin;

-- Test with more bins
SELECT 
    bin,
    ROUND(bin_min::numeric, 3) as min_dist,
    ROUND(bin_max::numeric, 3) as max_dist,
    count as pairs
FROM neurondb.similarity_histogram('test_graph_data', 'vec', 10)
ORDER BY bin;

-- Histogram for well-separated data
SELECT 
    bin,
    ROUND(bin_min::numeric, 2) as min_dist,
    ROUND(bin_max::numeric, 2) as max_dist,
    count as pairs
FROM neurondb.similarity_histogram('test_quality_high', 'vec', 5)
ORDER BY bin;

\echo '=== Testing Topic Discovery ==='

-- Create document embeddings for topic discovery
CREATE TABLE test_topic_docs (
    id SERIAL PRIMARY KEY,
    doc_text TEXT,
    embedding vector(5)
);

-- Insert documents from 3 topics
INSERT INTO test_topic_docs (doc_text, embedding) VALUES
    -- Topic 1: Technology
    ('Machine learning algorithms', '[1.0, 0.0, 0.0, 0.0, 0.0]'::vector),
    ('Deep neural networks', '[0.9, 0.1, 0.0, 0.0, 0.0]'::vector),
    ('Artificial intelligence', '[1.0, 0.0, 0.0, 0.0, 0.0]'::vector),
    ('Computer vision', '[0.8, 0.2, 0.0, 0.0, 0.0]'::vector),
    -- Topic 2: Medicine
    ('Clinical trials', '[0.0, 1.0, 0.0, 0.0, 0.0]'::vector),
    ('Medical diagnosis', '[0.0, 0.9, 0.1, 0.0, 0.0]'::vector),
    ('Patient treatment', '[0.0, 1.0, 0.0, 0.0, 0.0]'::vector),
    ('Healthcare systems', '[0.0, 0.8, 0.2, 0.0, 0.0]'::vector),
    -- Topic 3: Finance
    ('Stock market analysis', '[0.0, 0.0, 1.0, 0.0, 0.0]'::vector),
    ('Financial forecasting', '[0.0, 0.0, 0.9, 0.1, 0.0]'::vector),
    ('Investment strategies', '[0.0, 0.0, 1.0, 0.0, 0.0]'::vector),
    ('Economic indicators', '[0.0, 0.0, 0.8, 0.2, 0.0]'::vector);

-- Discover 3 topics
SELECT 
    topic_id,
    COUNT(*) as doc_count,
    ARRAY_AGG(doc_text ORDER BY id) as sample_docs
FROM neurondb.discover_topics_simple('test_topic_docs', 'embedding', 3)
GROUP BY topic_id
ORDER BY topic_id;

-- Discover 2 topics (should merge some)
SELECT 
    topic_id,
    COUNT(*) as doc_count
FROM neurondb.discover_topics_simple('test_topic_docs', 'embedding', 2)
GROUP BY topic_id
ORDER BY topic_id;

-- Get topic assignments for specific documents
WITH topics AS (
    SELECT * FROM neurondb.discover_topics_simple('test_topic_docs', 'embedding', 3)
)
SELECT 
    d.id,
    d.doc_text,
    t.topic_id
FROM test_topic_docs d
JOIN topics t ON d.id = t.id
ORDER BY t.topic_id, d.id;

\echo '=== Testing Graph Analysis ==='

-- Analyze graph connectivity
WITH graph AS (
    SELECT * FROM neurondb.build_knn_graph('test_graph_data', 'vec', 3)
),
node_stats AS (
    SELECT 
        node_id,
        COUNT(*) as out_degree,
        AVG(distance) as avg_distance
    FROM graph
    GROUP BY node_id
)
SELECT 
    t.label as cluster,
    COUNT(*) as nodes,
    ROUND(AVG(ns.out_degree)::numeric, 2) as avg_degree,
    ROUND(AVG(ns.avg_distance)::numeric, 4) as avg_edge_dist
FROM test_graph_data t
JOIN node_stats ns ON t.id = ns.node_id
GROUP BY t.label
ORDER BY cluster;

-- Find most connected nodes
WITH graph AS (
    SELECT * FROM neurondb.build_knn_graph('test_graph_data', 'vec', 3)
)
SELECT 
    t.id,
    t.label,
    COUNT(g.neighbor_id) as degree,
    ROUND(AVG(g.distance)::numeric, 4) as avg_dist
FROM test_graph_data t
LEFT JOIN graph g ON t.id = g.node_id
GROUP BY t.id, t.label
ORDER BY degree DESC, avg_dist
LIMIT 5;

\echo '=== Edge Cases and Error Handling ==='

-- Test with minimal data
CREATE TABLE test_graph_minimal (
    id SERIAL PRIMARY KEY,
    vec vector(2)
);

INSERT INTO test_graph_minimal (vec) VALUES
    ('[1.0, 0.0]'::vector),
    ('[0.0, 1.0]'::vector);

-- Build graph with k larger than available neighbors
SELECT 
    node_id,
    COUNT(*) as neighbors
FROM neurondb.build_knn_graph('test_graph_minimal', 'vec', 5)
GROUP BY node_id
ORDER BY node_id;

-- Test quality metrics with minimal data
SELECT 
    metric,
    ROUND(value::numeric, 4) as score
FROM neurondb.compute_embedding_quality('test_graph_minimal', 'vec')
ORDER BY metric;

-- Test histogram with identical vectors
CREATE TABLE test_identical (
    id SERIAL PRIMARY KEY,
    vec vector(2)
);

INSERT INTO test_identical (vec) VALUES
    ('[1.0, 1.0]'::vector),
    ('[1.0, 1.0]'::vector),
    ('[1.0, 1.0]'::vector);

SELECT 
    bin,
    ROUND(bin_min::numeric, 4) as min_dist,
    ROUND(bin_max::numeric, 4) as max_dist,
    count
FROM neurondb.similarity_histogram('test_identical', 'vec', 3)
ORDER BY bin;

-- Test topic discovery with too few documents
CREATE TABLE test_topic_minimal (
    id SERIAL PRIMARY KEY,
    embedding vector(3)
);

INSERT INTO test_topic_minimal (embedding) VALUES
    ('[1.0, 0.0, 0.0]'::vector),
    ('[0.0, 1.0, 0.0]'::vector);

SELECT 
    topic_id,
    COUNT(*) as docs
FROM neurondb.discover_topics_simple('test_topic_minimal', 'embedding', 3)
GROUP BY topic_id
ORDER BY topic_id;

-- Cleanup
DROP TABLE test_graph_data CASCADE;
DROP TABLE test_quality_high CASCADE;
DROP TABLE test_topic_docs CASCADE;
DROP TABLE test_graph_minimal CASCADE;
DROP TABLE test_identical CASCADE;
DROP TABLE test_topic_minimal CASCADE;

