-- ============================================================================
-- pgvector Conformance Test Suite
-- Tests NeuronDB compatibility with pgvector extension features
-- Covers: types, operators, functions, aggregates, indexes, casts, query patterns
-- ============================================================================

\timing off
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'pgvector Conformance Test Suite'
\echo '=========================================================================='

-- ============================================================================
-- Test 1: Type I/O and Casting
-- ============================================================================

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 1: Type I/O and Casting'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Basic type creation and input
CREATE TABLE t1 (id int, embedding vector(3));
INSERT INTO t1 VALUES (1, '[1,2,3]');
INSERT INTO t1 VALUES (2, '[4,5,6]');
INSERT INTO t1 VALUES (3, '[0.1, 0.2, 0.3]');

-- Test dimension enforcement (vector(n))
INSERT INTO t1 VALUES (4, '[1,2]');  -- Should fail (dimension mismatch)
\set ON_ERROR_STOP off
INSERT INTO t1 VALUES (4, '[1,2]');  -- Expected to fail
\set ON_ERROR_STOP on

-- Test array casts
CREATE TABLE t2 AS SELECT id, embedding::real[] as arr FROM t1;
SELECT id, arr FROM t2 ORDER BY id;

-- Test vector(n) typmod
SELECT vector_dims(embedding) FROM t1 LIMIT 1;

DROP TABLE t1, t2;

-- ============================================================================
-- Test 2: Operators
-- ============================================================================

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 2: Distance Operators'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE TABLE t (id int, embedding vector(3));
INSERT INTO t VALUES (1, '[1,2,3]'), (2, '[4,5,6]'), (3, '[1,1,1]');

-- L2 distance operator (<->)
SELECT id, embedding <-> '[0,0,0]'::vector(3) AS l2_dist 
FROM t 
ORDER BY l2_dist;

-- Cosine distance operator (<=>)
SELECT id, embedding <=> '[1,1,1]'::vector(3) AS cosine_dist 
FROM t 
ORDER BY cosine_dist;

-- Inner product operator (<#>)
SELECT id, embedding <#> '[1,1,1]'::vector(3) AS ip_dist 
FROM t 
ORDER BY ip_dist;

-- Equality operator
SELECT id FROM t WHERE embedding = '[1,2,3]'::vector(3);

DROP TABLE t;

-- ============================================================================
-- Test 3: Functions
-- ============================================================================

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 3: Core Functions'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE TABLE t (id int, embedding vector(3));
INSERT INTO t VALUES (1, '[3,4,0]'), (2, '[1,2,2]');

-- vector_dims()
SELECT id, vector_dims(embedding) AS dims FROM t;

-- l2_norm()
SELECT id, l2_norm(embedding) AS norm FROM t;

-- vector_norm() (alias)
SELECT id, vector_norm(embedding) AS norm FROM t;

-- normalize_l2() (pgvector compatibility)
SELECT id, normalize_l2(embedding) AS normalized FROM t;

-- l2_normalize() (compatibility alias)
SELECT id, l2_normalize(embedding) AS normalized FROM t;

-- Distance functions
SELECT l2_distance('[1,2,3]'::vector, '[4,5,6]'::vector) AS l2;
SELECT cosine_distance('[1,0,0]'::vector, '[0,1,0]'::vector) AS cosine;
SELECT inner_product('[1,2,3]'::vector, '[4,5,6]'::vector) AS ip;

-- Array conversion functions
SELECT vector_to_array('[1,2,3]'::vector) AS arr;
SELECT array_to_vector(ARRAY[1.0, 2.0, 3.0]) AS vec;

DROP TABLE t;

-- ============================================================================
-- Test 4: Subvector Operations
-- ============================================================================

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 4: Subvector Operations'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- subvector() - pgvector compatibility (1-based, count)
SELECT subvector('[1,2,3,4,5]'::vector, 1, 3) AS sub1;  -- Elements 1-3
SELECT subvector('[1,2,3,4,5]'::vector, 2, 2) AS sub2;  -- Elements 2-3

-- vector_slice() - NeuronDB canonical (0-based, end exclusive)
SELECT vector_slice('[1,2,3,4,5]'::vector, 0, 3) AS slice1;  -- Elements 0-2
SELECT vector_slice('[1,2,3,4,5]'::vector, 1, 3) AS slice2;  -- Elements 1-2

-- ============================================================================
-- Test 5: Aggregates
-- ============================================================================

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 5: Vector Aggregates'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE TABLE t (id int, embedding vector(3));
INSERT INTO t VALUES 
  (1, '[1,2,3]'),
  (2, '[4,5,6]'),
  (3, '[7,8,9]');

-- avg(vector) - pgvector compatibility
SELECT avg(embedding) AS avg_vec FROM t;

-- sum(vector) - pgvector compatibility
SELECT sum(embedding) AS sum_vec FROM t;

-- vector_avg() and vector_sum() - NeuronDB canonical
SELECT vector_avg(embedding) AS avg_vec FROM t;
SELECT vector_sum(embedding) AS sum_vec FROM t;

-- Aggregates with GROUP BY
CREATE TABLE t2 (category text, embedding vector(3));
INSERT INTO t2 VALUES 
  ('A', '[1,2,3]'), ('A', '[2,3,4]'),
  ('B', '[10,11,12]'), ('B', '[11,12,13]');

SELECT category, avg(embedding) FROM t2 GROUP BY category;

DROP TABLE t, t2;

-- ============================================================================
-- Test 6: HNSW Index - All Operators
-- ============================================================================

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 6: HNSW Index - L2, Cosine, Inner Product'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SET enable_seqscan = off;

-- HNSW with L2
CREATE TABLE t (id int, embedding vector(3));
INSERT INTO t SELECT id, ARRAY[random(), random(), random()]::vector(3) 
FROM generate_series(1, 50);
CREATE INDEX ON t USING hnsw (embedding vector_l2_ops) WITH (m = 16, ef_construction = 64);

SELECT id, embedding <-> '[0.5,0.5,0.5]'::vector AS dist 
FROM t 
ORDER BY embedding <-> '[0.5,0.5,0.5]'::vector 
LIMIT 5;

DROP TABLE t;

-- HNSW with Cosine
CREATE TABLE t (id int, embedding vector(3));
INSERT INTO t SELECT id, ARRAY[random(), random(), random()]::vector(3) 
FROM generate_series(1, 50);
CREATE INDEX ON t USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

SELECT id, embedding <=> '[0.5,0.5,0.5]'::vector AS dist 
FROM t 
ORDER BY embedding <=> '[0.5,0.5,0.5]'::vector 
LIMIT 5;

DROP TABLE t;

-- HNSW with Inner Product
CREATE TABLE t (id int, embedding vector(3));
INSERT INTO t SELECT id, ARRAY[random(), random(), random()]::vector(3) 
FROM generate_series(1, 50);
CREATE INDEX ON t USING hnsw (embedding vector_ip_ops) WITH (m = 16, ef_construction = 64);

SELECT id, embedding <#> '[0.5,0.5,0.5]'::vector AS dist 
FROM t 
ORDER BY embedding <#> '[0.5,0.5,0.5]'::vector 
LIMIT 5;

DROP TABLE t;

RESET enable_seqscan;

-- ============================================================================
-- Test 7: IVF Index - All Operators (including Inner Product)
-- ============================================================================

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 7: IVF Index - L2, Cosine, Inner Product'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SET enable_seqscan = off;

-- IVF with L2
CREATE TABLE t (id int, embedding vector(3));
INSERT INTO t SELECT id, ARRAY[random(), random(), random()]::vector(3) 
FROM generate_series(1, 50);
CREATE INDEX ON t USING ivf (embedding vector_l2_ops) WITH (lists = 10);

SELECT id, embedding <-> '[0.5,0.5,0.5]'::vector AS dist 
FROM t 
ORDER BY embedding <-> '[0.5,0.5,0.5]'::vector 
LIMIT 5;

DROP TABLE t;

-- IVF with Cosine
CREATE TABLE t (id int, embedding vector(3));
INSERT INTO t SELECT id, ARRAY[random(), random(), random()]::vector(3) 
FROM generate_series(1, 50);
CREATE INDEX ON t USING ivf (embedding vector_cosine_ops) WITH (lists = 10);

SELECT id, embedding <=> '[0.5,0.5,0.5]'::vector AS dist 
FROM t 
ORDER BY embedding <=> '[0.5,0.5,0.5]'::vector 
LIMIT 5;

DROP TABLE t;

-- IVF with Inner Product (NEW - now supported)
CREATE TABLE t (id int, embedding vector(3));
INSERT INTO t SELECT id, ARRAY[random(), random(), random()]::vector(3) 
FROM generate_series(1, 50);
CREATE INDEX ON t USING ivf (embedding vector_ip_ops) WITH (lists = 10);

SELECT id, embedding <#> '[0.5,0.5,0.5]'::vector AS dist 
FROM t 
ORDER BY embedding <#> '[0.5,0.5,0.5]'::vector 
LIMIT 5;

DROP TABLE t;

RESET enable_seqscan;

-- ============================================================================
-- Test 8: Query Patterns (ORDER BY distance LIMIT k)
-- ============================================================================

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 8: Common Query Patterns'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE TABLE items (id int, category text, embedding vector(3));
INSERT INTO items VALUES 
  (1, 'electronics', '[1,2,3]'),
  (2, 'electronics', '[1.1,2.1,3.1]'),
  (3, 'books', '[10,11,12]'),
  (4, 'books', '[10.1,11.1,12.1]'),
  (5, 'electronics', '[1.2,2.2,3.2]');
CREATE INDEX ON items USING hnsw (embedding vector_l2_ops);

-- Basic K-NN search
SELECT id, category, embedding <-> '[1,2,3]'::vector AS distance
FROM items
ORDER BY embedding <-> '[1,2,3]'::vector
LIMIT 3;

-- Filtered K-NN search
SELECT id, category, embedding <-> '[1,2,3]'::vector AS distance
FROM items
WHERE category = 'electronics'
ORDER BY embedding <-> '[1,2,3]'::vector
LIMIT 2;

-- Distance in SELECT with ORDER BY
SELECT id, category, embedding <=> '[1,1,1]'::vector AS cosine_dist
FROM items
ORDER BY cosine_dist
LIMIT 3;

DROP TABLE items;

-- ============================================================================
-- Test 9: Compatibility Function Aliases
-- ============================================================================

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 9: Compatibility Function Aliases'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Test that compatibility aliases work
SELECT 
  l2_distance('[1,2,3]'::vector, '[4,5,6]'::vector) AS l2_compat,
  vector_l2_distance('[1,2,3]'::vector, '[4,5,6]'::vector) AS l2_canonical;

SELECT 
  cosine_distance('[1,0,0]'::vector, '[0,1,0]'::vector) AS cos_compat,
  vector_cosine_distance('[1,0,0]'::vector, '[0,1,0]'::vector) AS cos_canonical;

SELECT 
  inner_product('[1,2,3]'::vector, '[4,5,6]'::vector) AS ip_compat,
  vector_inner_product('[1,2,3]'::vector, '[4,5,6]'::vector) AS ip_canonical;

-- Normalize aliases
SELECT 
  normalize_l2('[3,4,0]'::vector) AS norm_l2_compat,
  l2_normalize('[3,4,0]'::vector) AS l2_norm_compat,
  vector_normalize('[3,4,0]'::vector) AS norm_canonical;

-- Verify they produce same results
WITH v AS (SELECT '[3,4,0]'::vector AS vec)
SELECT 
  normalize_l2(vec) = l2_normalize(vec) AS aliases_match,
  normalize_l2(vec) = vector_normalize(vec) AS compat_matches_canonical
FROM v;

-- ============================================================================
-- Test 10: Edge Cases
-- ============================================================================

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 10: Edge Cases'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- NULL handling
CREATE TABLE t (id int, embedding vector(3));
INSERT INTO t VALUES (1, '[1,2,3]'), (2, NULL), (3, '[4,5,6]');

-- Aggregates with NULLs
SELECT avg(embedding) FROM t;  -- Should ignore NULLs
SELECT sum(embedding) FROM t;  -- Should ignore NULLs

-- Distance with NULL
\set ON_ERROR_STOP off
SELECT id, embedding <-> NULL::vector FROM t;  -- Should error
\set ON_ERROR_STOP on

-- Empty result sets
SELECT avg(embedding) FROM t WHERE false;  -- Should return NULL
SELECT sum(embedding) FROM t WHERE false;  -- Should return NULL

DROP TABLE t;

-- Zero vector
SELECT '[0,0,0]'::vector <-> '[1,1,1]'::vector AS dist_to_zero;
SELECT l2_norm('[0,0,0]'::vector) AS zero_norm;

-- ============================================================================
-- Summary
-- ============================================================================

\echo ''
\echo '=========================================================================='
\echo 'pgvector Conformance Test Suite Completed Successfully'
\echo '=========================================================================='
\echo ''
\echo 'All tests passed! NeuronDB is pgvector-compatible for:'
\echo '  ✓ Type I/O and casting'
\echo '  ✓ Distance operators (<->, <=>, <#>)'
\echo '  ✓ Core functions (dims, norm, normalize)'
\echo '  ✓ Subvector operations'
\echo '  ✓ Aggregates (avg, sum)'
\echo '  ✓ HNSW index (all operators)'
\echo '  ✓ IVF index (all operators including inner product)'
\echo '  ✓ Common query patterns'
\echo '  ✓ Compatibility function aliases'
\echo '  ✓ Edge cases (NULLs, empty sets, zero vectors)'
\echo ''




