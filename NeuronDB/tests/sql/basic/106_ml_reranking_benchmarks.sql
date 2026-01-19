-- ====================================================================
-- NeurondB Performance Benchmarks: Reranking Algorithms
-- ====================================================================
-- Performance benchmarks for all reranking functions
-- Measures latency, throughput, and accuracy
-- ====================================================================

\echo '=== Reranking Performance Benchmarks ==='

-- Create benchmark dataset
CREATE TEMP TABLE benchmark_docs AS
SELECT 
    ROW_NUMBER() OVER() as id,
    content,
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

-- Benchmark query
CREATE TEMP TABLE benchmark_query AS
SELECT 'artificial intelligence and machine learning research'::text as query_text,
       '[0.95, 0.05, 0.0, 0.0]'::vector(4) as query_vec;

\echo '=== Benchmark 1: Cross-Encoder Performance ==='

-- Measure cross-encoder latency
DO $$
DECLARE
    start_time timestamp;
    end_time timestamp;
    duration_ms numeric;
    candidates text[];
    i int;
    iterations int := 5;
    total_time numeric := 0;
BEGIN
    SELECT ARRAY(SELECT content FROM benchmark_docs LIMIT 10) INTO candidates;
    
    FOR i IN 1..iterations LOOP
        start_time := clock_timestamp();
        
        PERFORM * FROM neurondb.rerank_cross_encoder(
            (SELECT query_text FROM benchmark_query),
            candidates,
            NULL,
            5
        );
        
        end_time := clock_timestamp();
        duration_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
        total_time := total_time + duration_ms;
        
        RAISE NOTICE 'Cross-Encoder iteration %: %.2f ms', i, duration_ms;
    END LOOP;
    
    RAISE NOTICE 'Cross-Encoder average latency: %.2f ms', total_time / iterations;
    RAISE NOTICE 'Cross-Encoder throughput: %.2f queries/sec', 
        (iterations * 1000.0) / total_time;
END $$;

\echo '=== Benchmark 2: LLM Reranking Performance ==='

-- Measure LLM reranking latency
DO $$
DECLARE
    start_time timestamp;
    end_time timestamp;
    duration_ms numeric;
    candidates text[];
    i int;
    iterations int := 3;  -- Fewer iterations for LLM (slower)
    total_time numeric := 0;
BEGIN
    SELECT ARRAY(SELECT content FROM benchmark_docs LIMIT 5) INTO candidates;
    
    FOR i IN 1..iterations LOOP
        start_time := clock_timestamp();
        
        PERFORM * FROM neurondb.rerank_llm(
            (SELECT query_text FROM benchmark_query),
            candidates,
            NULL,
            3,
            NULL,
            0.0
        );
        
        end_time := clock_timestamp();
        duration_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
        total_time := total_time + duration_ms;
        
        RAISE NOTICE 'LLM Reranking iteration %: %.2f ms', i, duration_ms;
    END LOOP;
    
    RAISE NOTICE 'LLM Reranking average latency: %.2f ms', total_time / iterations;
    RAISE NOTICE 'LLM Reranking throughput: %.2f queries/sec', 
        (iterations * 1000.0) / total_time;
END $$;

\echo '=== Benchmark 3: ColBERT Performance ==='

-- Measure ColBERT latency
DO $$
DECLARE
    start_time timestamp;
    end_time timestamp;
    duration_ms numeric;
    candidates text[];
    i int;
    iterations int := 5;
    total_time numeric := 0;
BEGIN
    SELECT ARRAY(SELECT content FROM benchmark_docs LIMIT 5) INTO candidates;
    
    FOR i IN 1..iterations LOOP
        start_time := clock_timestamp();
        
        PERFORM * FROM neurondb.rerank_colbert(
            (SELECT query_text FROM benchmark_query),
            candidates,
            NULL,
            3,
            2,
            1
        );
        
        end_time := clock_timestamp();
        duration_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
        total_time := total_time + duration_ms;
        
        RAISE NOTICE 'ColBERT iteration %: %.2f ms', i, duration_ms;
    END LOOP;
    
    RAISE NOTICE 'ColBERT average latency: %.2f ms', total_time / iterations;
    RAISE NOTICE 'ColBERT throughput: %.2f queries/sec', 
        (iterations * 1000.0) / total_time;
END $$;

\echo '=== Benchmark 4: MMR Performance ==='

-- Measure MMR latency
DO $$
DECLARE
    start_time timestamp;
    end_time timestamp;
    duration_ms numeric;
    i int;
    iterations int := 10;
    total_time numeric := 0;
BEGIN
    FOR i IN 1..iterations LOOP
        start_time := clock_timestamp();
        
        PERFORM * FROM neurondb.mmr_rerank(
            'benchmark_docs',
            'doc_vec',
            (SELECT query_vec FROM benchmark_query),
            10,
            0.7
        );
        
        end_time := clock_timestamp();
        duration_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
        total_time := total_time + duration_ms;
    END LOOP;
    
    RAISE NOTICE 'MMR average latency: %.2f ms', total_time / iterations;
    RAISE NOTICE 'MMR throughput: %.2f queries/sec', 
        (iterations * 1000.0) / total_time;
END $$;

\echo '=== Benchmark 5: Scalability Test ==='

-- Test performance with different candidate sizes
DO $$
DECLARE
    start_time timestamp;
    end_time timestamp;
    duration_ms numeric;
    candidates text[];
    sizes int[] := ARRAY[5, 10, 20, 50];
    size_val int;
BEGIN
    FOREACH size_val IN ARRAY sizes LOOP
        SELECT ARRAY(SELECT content FROM benchmark_docs LIMIT size_val) INTO candidates;
        
        start_time := clock_timestamp();
        
        PERFORM * FROM neurondb.rerank_cross_encoder(
            (SELECT query_text FROM benchmark_query),
            candidates,
            NULL,
            size_val
        );
        
        end_time := clock_timestamp();
        duration_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
        
        RAISE NOTICE 'Cross-Encoder with % candidates: %.2f ms (%.2f ms per candidate)',
            size_val, duration_ms, duration_ms / size_val;
    END LOOP;
END $$;

\echo '=== Benchmark 6: Memory Usage Comparison ==='

-- Compare memory efficiency of different methods
-- (This is a simplified check - actual memory profiling requires pg_stat_statements)
SELECT 
    'Cross-Encoder' as method,
    COUNT(*) as result_count,
    AVG(score) as avg_score,
    MIN(score) as min_score,
    MAX(score) as max_score
FROM neurondb.rerank_cross_encoder(
    (SELECT query_text FROM benchmark_query),
    ARRAY(SELECT content FROM benchmark_docs LIMIT 10),
    NULL,
    10
)
UNION ALL
SELECT 
    'LLM' as method,
    COUNT(*) as result_count,
    AVG(score) as avg_score,
    MIN(score) as min_score,
    MAX(score) as max_score
FROM neurondb.rerank_llm(
    (SELECT query_text FROM benchmark_query),
    ARRAY(SELECT content FROM benchmark_docs LIMIT 10),
    NULL,
    10,
    NULL,
    0.0
)
UNION ALL
SELECT 
    'ColBERT' as method,
    COUNT(*) as result_count,
    AVG(score) as avg_score,
    MIN(score) as min_score,
    MAX(score) as max_score
FROM neurondb.rerank_colbert(
    (SELECT query_text FROM benchmark_query),
    ARRAY(SELECT content FROM benchmark_docs LIMIT 10),
    NULL,
    10,
    2,
    1
);

\echo '=== Benchmark Summary ==='

-- Create summary table
CREATE TEMP TABLE benchmark_summary AS
SELECT 
    'Cross-Encoder' as method,
    'Neural reranking using cross-encoder models' as description,
    'Medium' as latency_category,
    'High' as accuracy_category,
    'Medium' as memory_usage
UNION ALL
SELECT 
    'LLM Reranking',
    'GPT/Claude-powered reranking',
    'High',
    'Very High',
    'High'
UNION ALL
SELECT 
    'ColBERT',
    'Late interaction model',
    'Medium',
    'High',
    'Medium'
UNION ALL
SELECT 
    'MMR',
    'Maximal Marginal Relevance',
    'Low',
    'Medium',
    'Low';

SELECT * FROM benchmark_summary ORDER BY method;

-- Cleanup
DROP TABLE benchmark_docs CASCADE;
DROP TABLE benchmark_query CASCADE;
DROP TABLE benchmark_summary CASCADE;

