-- Detailed and all possible tests for data management functions
-- Uses real data from: sift1m.vectors for realistic testing

-- 1. Create table with all columns and various settings
CREATE TEMP TABLE test_vectors_dm (
    id serial PRIMARY KEY,
    embedding vector NOT NULL,
    created_at timestamptz DEFAULT now(),
    last_accessed timestamptz DEFAULT now(),
    is_compressed boolean DEFAULT false
);

-- 2. Insert diverse test vectors from real SIFT data with simulated timestamps
INSERT INTO test_vectors_dm (embedding, last_accessed, is_compressed)
SELECT 
    array_to_vector(embedding[1:10])::vector(10) as embedding,
    CASE 
        WHEN id % 6 = 0 THEN now() - INTERVAL '40 days'
        WHEN id % 6 = 1 THEN now() - INTERVAL '20 days'
        WHEN id % 6 = 2 THEN now()
        WHEN id % 6 = 3 THEN now() - INTERVAL '50 days'
        WHEN id % 6 = 4 THEN now() - INTERVAL '70 days'
        ELSE now() - INTERVAL '90 days'
    END as last_accessed,
    CASE WHEN id % 3 = 0 THEN true ELSE false END as is_compressed
FROM sift1m.vectors
WHERE id <= 100
LIMIT 100;

-- Show sample of loaded data
SELECT id, vector_dims(embedding) as dims, last_accessed, is_compressed 
FROM test_vectors_dm 
WHERE id <= 6;

-- 3. Test vacuum_vectors: normal + with dry_run
-- Returns count of rows affected or eligible to vacuum
SELECT vacuum_vectors('test_vectors_dm', false) AS vacuum_executed;
SELECT vacuum_vectors('test_vectors_dm', true) AS vacuum_dry_run_executed;

-- 4. Test compress_cold_tier: multiple thresholds
-- 30 days (only >30d old, uncompressed should compress)
SELECT compress_cold_tier('test_vectors_dm', 30) AS compressed_30d;
-- 60 days (should catch older, uncompressed vectors)
SELECT compress_cold_tier('test_vectors_dm', 60) AS compressed_60d;
-- 0 days (try compress everything possible)
SELECT compress_cold_tier('test_vectors_dm', 0) AS compressed_0d;

-- 5. Test rebalance_index: various thresholds and indexes
-- Typical threshold
SELECT rebalance_index('test_vectors_dm_embedding_idx', 0.8) AS rebalance_default;
-- Lower threshold
SELECT rebalance_index('test_vectors_dm_embedding_idx', 0.5) AS rebalance_lower;
-- Edge-case: threshold at 1
SELECT rebalance_index('test_vectors_dm_embedding_idx', 1.0) AS rebalance_full;
-- Possible when no such index exists (should handle errors)
SELECT rebalance_index('nonexistent_index', 0.8) AS rebalance_nonexistent;

-- 6. Select all data to verify state after transformations
SELECT * FROM test_vectors_dm ORDER BY id;

-- 7. Cleanup
DROP TABLE test_vectors_dm;

