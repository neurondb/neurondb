-- Detailed and exhaustive coverage of vector aggregate functions
-- Uses real data from: sift1m.vectors for realistic testing

-- 1. Basic test table from real SIFT vectors (take first 100)
CREATE TEMP TABLE test_vectors_agg AS
SELECT 
    id::int,
    array_to_vector(embedding[1:2])::vector(2) as v
FROM sift1m.vectors
WHERE id <= 100
LIMIT 100;

-- Show sample of loaded data
SELECT id, v FROM test_vectors_agg WHERE id <= 5;

-- 2. Test: vector_sum, vector_avg, vector_mean (alias of avg), vector_min, vector_max
SELECT vector_sum(v)   AS sum_all
  ,   vector_avg(v)   AS avg_all
  ,   vector_mean(v)  AS mean_all   -- if mean implemented as synonym to avg
  ,   vector_min(v)   AS min_elemwise
  ,   vector_max(v)   AS max_elemwise
FROM test_vectors_agg;

-- 3. Test: aggregate over empty table (should return null or error appropriately)
SELECT 
    vector_sum(v) AS sum_empty, 
    vector_avg(v) AS avg_empty, 
    vector_mean(v) AS mean_empty
FROM (SELECT v FROM test_vectors_agg WHERE false) t;

-- 4. Test: aggregation with WHERE clause / filtered subset
SELECT vector_avg(v) AS avg_12
FROM test_vectors_agg
WHERE id <= 2;

SELECT vector_sum(v) AS sum_34
FROM test_vectors_agg
WHERE id IN (3,4);

SELECT vector_min(v) AS min_13
FROM test_vectors_agg
WHERE id IN (1,3);

SELECT vector_max(v) AS max_24
FROM test_vectors_agg
WHERE id IN (2,4);

-- 5. Test: aggregation with NULLs present
INSERT INTO test_vectors_agg VALUES (6, NULL), (7, NULL);

SELECT vector_sum(v) AS sum_with_nulls,
       vector_avg(v) AS avg_with_nulls,
       vector_min(v) AS min_with_nulls,
       vector_max(v) AS max_with_nulls
FROM test_vectors_agg;

-- 6. Test: Singleton (single row)
SELECT vector_avg(v) AS avg_single,
       vector_sum(v) AS sum_single
FROM test_vectors_agg WHERE id = 3;

-- 7. Test: All values are identical
CREATE TEMP TABLE identical_vectors(v vector);
INSERT INTO identical_vectors VALUES
    ('[5.0, 5.0, 5.0]'),
    ('[5.0, 5.0, 5.0]'),
    ('[5.0, 5.0, 5.0]');

SELECT
    vector_sum(v) AS sum_identical,
    vector_avg(v) AS avg_identical,
    vector_min(v) AS min_identical,
    vector_max(v) AS max_identical
FROM identical_vectors;

-- 8. Test: Negative values, mixed signs
CREATE TEMP TABLE mixed_vectors(v vector);
INSERT INTO mixed_vectors VALUES
    ('[-1.0, 0.0, 2.5]'),
    ('[1.5, -2.0, 3.0]'),
    ('[0.0, 0.0, 0.0]');

SELECT
    vector_sum(v) AS sum_mixed,
    vector_avg(v) AS avg_mixed,
    vector_min(v) AS min_mixed,
    vector_max(v) AS max_mixed
FROM mixed_vectors;

-- 9. Test: Vectors with varying lengths (should error)
DO $$
BEGIN
  BEGIN
    CREATE TEMP TABLE vec_len_mismatch(v vector);
    INSERT INTO vec_len_mismatch VALUES ('[1.0,2.0]'), ('[1.0,2.0,3.0]');
    PERFORM vector_sum(v) FROM vec_len_mismatch;
    RAISE WARNING 'ERROR: vector_sum should not allow dimension mismatch!';
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Correctly rejected varying length input: %', SQLERRM;
  END;
END$$;

-- 10. Test: All NULLs (should return NULL)
SELECT
    vector_sum(v) AS sum_all_nulls,
    vector_avg(v) AS avg_all_nulls,
    vector_min(v) AS min_all_nulls,
    vector_max(v) AS max_all_nulls
FROM (VALUES (NULL),(NULL)) AS t(v);

-- 11. Test: Zero vector(s)
SELECT
    vector_sum(v) AS sum_zero_vectors,
    vector_avg(v) AS avg_zero_vectors
FROM (VALUES ('[0.0, 0.0, 0.0]'), ('[0.0, 0.0, 0.0]')) AS t(v);

-- 12. Test: Extreme float values (Edge: Inf, -Inf, NaN)
CREATE TEMP TABLE extreme_vectors(v vector);
INSERT INTO extreme_vectors VALUES
    ('[1e30, -1e30, 0.0, 1.0]'),
    ('[-1e30, 1e30, 0.0, -1.0]');

SELECT
    vector_sum(v) AS sum_extreme,
    vector_avg(v) AS avg_extreme,
    vector_min(v) AS min_extreme,
    vector_max(v) AS max_extreme
FROM extreme_vectors;

-- 13. Test: Large number of rows for performance/overflow check (short)
CREATE TEMP TABLE many_vecs (v vector);
INSERT INTO many_vecs (v)
SELECT format('[%s,%s]', i::float, (i*2)::float)::vector
FROM generate_series(1, 1000) AS i;

SELECT
    vector_sum(v) AS sum_many,
    vector_avg(v) AS avg_many
FROM many_vecs;

-- 14. Test: Type coercion via array input/array_to_vector
SELECT vector_sum(array_to_vector(ARRAY[1,2,3]::real[])) AS sum_convert;

-- 15. Test: Works for vectorp/packed vector type if aggregate supported (optional)
-- (Comment out if not yet supported)
-- CREATE TEMP TABLE t_packed(vp vectorp);
-- INSERT INTO t_packed VALUES
--   (vectorp_in('[1.2,2.3,3.4]')),
--   (vectorp_in('[2.2,3.3,4.4]'));
-- SELECT 
--   vector_sum(vp::vector) AS sum_packed
-- FROM t_packed;

-- 16. Test: Vector aggregates in GROUP BY aggregation
CREATE TEMP TABLE group_vecs (category TEXT, v vector);
INSERT INTO group_vecs VALUES
  ('A', '[1,2,3]'),
  ('A', '[2,3,4]'),
  ('B', '[3,4,5]'),
  ('B', '[4,5,6]');

SELECT category,
       vector_sum(v) AS sum_group,
       vector_avg(v) AS avg_group
FROM group_vecs
GROUP BY category
ORDER BY category;

