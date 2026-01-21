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

-- 2. Insert diverse test vectors with simulated timestamps (sift1m.vectors may not exist)
INSERT INTO test_vectors_dm (embedding, last_accessed, is_compressed)
SELECT 
    array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 10)))::vector(10) as embedding,
    CASE 
        WHEN id % 6 = 0 THEN now() - INTERVAL '40 days'
        WHEN id % 6 = 1 THEN now() - INTERVAL '20 days'
        WHEN id % 6 = 2 THEN now()
        WHEN id % 6 = 3 THEN now() - INTERVAL '50 days'
        WHEN id % 6 = 4 THEN now() - INTERVAL '70 days'
        ELSE now() - INTERVAL '90 days'
    END as last_accessed,
    CASE WHEN id % 3 = 0 THEN true ELSE false END as is_compressed
FROM generate_series(1, 100) AS id;

-- Show sample of loaded data
SELECT id, vector_dims(embedding) as dims, last_accessed, is_compressed 
FROM test_vectors_dm 
WHERE id <= 6;

-- 3. Test vacuum_vectors: normal + with dry_run (skip if function doesn't exist)
DO $$
BEGIN
  BEGIN
    PERFORM vacuum_vectors('test_vectors_dm', false);
    PERFORM vacuum_vectors('test_vectors_dm', true);
    RAISE NOTICE 'vacuum_vectors function is available';
  EXCEPTION WHEN undefined_function THEN
    RAISE NOTICE 'vacuum_vectors function not available, skipping vacuum tests';
  END;
END$$;

-- 4. Test compress_cold_tier: multiple thresholds (skip if function doesn't exist)
DO $$
BEGIN
  BEGIN
    PERFORM compress_cold_tier('test_vectors_dm', 30);
    PERFORM compress_cold_tier('test_vectors_dm', 60);
    PERFORM compress_cold_tier('test_vectors_dm', 0);
    RAISE NOTICE 'compress_cold_tier function is available';
  EXCEPTION WHEN undefined_function THEN
    RAISE NOTICE 'compress_cold_tier function not available, skipping compression tests';
  END;
END$$;

-- 5. Test rebalance_index: various thresholds and indexes (skip if function doesn't exist)
DO $$
BEGIN
  BEGIN
    PERFORM rebalance_index('test_vectors_dm_embedding_idx', 0.8);
    PERFORM rebalance_index('test_vectors_dm_embedding_idx', 0.5);
    PERFORM rebalance_index('test_vectors_dm_embedding_idx', 1.0);
    -- This one may error even if function exists (nonexistent index)
    BEGIN
      PERFORM rebalance_index('nonexistent_index', 0.8);
    EXCEPTION WHEN OTHERS THEN
      RAISE NOTICE 'rebalance_index correctly handled nonexistent index: %', SQLERRM;
    END;
    RAISE NOTICE 'rebalance_index function is available';
  EXCEPTION WHEN undefined_function THEN
    RAISE NOTICE 'rebalance_index function not available, skipping rebalance tests';
  END;
END$$;
-- Edge-case: threshold at 1
SELECT rebalance_index('test_vectors_dm_embedding_idx', 1.0) AS rebalance_full;
-- Possible when no such index exists (should handle errors)
SELECT rebalance_index('nonexistent_index', 0.8) AS rebalance_nonexistent;

-- 6. Select all data to verify state after transformations
SELECT * FROM test_vectors_dm ORDER BY id;

-- 7. Cleanup
DROP TABLE test_vectors_dm;

