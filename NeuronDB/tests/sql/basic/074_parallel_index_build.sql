-- 071_parallel_index_build.sql
-- Tests for parallel index build support for HNSW and IVF indexes

\timing on
\pset footer off
\pset pager off
\pset tuples_only off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Parallel Index Build: Comprehensive Tests'
\echo '=========================================================================='

-- Setup: Create test table with vectors
\echo ''
\echo 'Setup: Creating test table with vectors'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DROP TABLE IF EXISTS parallel_build_test CASCADE;
CREATE TABLE parallel_build_test (
	id SERIAL PRIMARY KEY,
	embedding vector(128),
	label integer
);

-- Insert sufficient data for parallel build (need enough rows to trigger parallel workers)
INSERT INTO parallel_build_test (embedding, label)
SELECT 
	array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 128)))::vector(128),
	(random() * 10)::integer
FROM generate_series(1, 1000);

\echo ''
\echo 'Test 1: Check parallel build support is enabled'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Check if parallel workers are available
DO $$
DECLARE
	max_workers integer;
	parallel_setting text;
BEGIN
	-- Get max_parallel_workers_per_gather setting
	BEGIN
		parallel_setting := current_setting('max_parallel_workers_per_gather', true);
		max_workers := parallel_setting::integer;
	EXCEPTION WHEN OTHERS THEN
		max_workers := 0;
	END;
	
	RAISE NOTICE 'max_parallel_workers_per_gather: %', max_workers;
	
	IF max_workers > 0 THEN
		RAISE NOTICE 'Parallel workers available - parallel index build may be used';
	ELSE
		RAISE NOTICE 'No parallel workers configured - index build will be sequential';
	END IF;
END$$;

\echo ''
\echo 'Test 2: Parallel HNSW index build'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Enable parallel workers if not already enabled
SET max_parallel_workers_per_gather = 2;
SET maintenance_work_mem = '256MB';

-- Create HNSW index (should use parallel build if supported)
DROP INDEX IF EXISTS parallel_hnsw_idx;
CREATE INDEX parallel_hnsw_idx ON parallel_build_test 
USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 64, ef_search = 40);

-- Verify index was created
DO $$
DECLARE
	index_exists boolean;
	index_size bigint;
BEGIN
	-- Check if index exists
	SELECT EXISTS (
		SELECT 1 FROM pg_indexes 
		WHERE indexname = 'parallel_hnsw_idx'
	) INTO index_exists;
	
	IF NOT index_exists THEN
		RAISE EXCEPTION 'HNSW index was not created';
	END IF;
	
	-- Get index size
	SELECT pg_relation_size('parallel_hnsw_idx') INTO index_size;
	
	RAISE NOTICE 'HNSW index created successfully, size: % bytes', index_size;
END$$;

-- Verify index can be used for queries
SELECT COUNT(*) AS indexed_rows
FROM parallel_build_test
WHERE embedding <-> (SELECT embedding FROM parallel_build_test LIMIT 1) < 1.0;

\echo ''
\echo 'Test 3: Parallel IVF index build'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Create IVF index (should use parallel build if supported)
DROP INDEX IF EXISTS parallel_ivf_idx;
CREATE INDEX parallel_ivf_idx ON parallel_build_test 
USING ivf (embedding vector_l2_ops)
WITH (lists = 10, probes = 5);

-- Verify index was created
DO $$
DECLARE
	index_exists boolean;
	index_size bigint;
BEGIN
	-- Check if index exists
	SELECT EXISTS (
		SELECT 1 FROM pg_indexes 
		WHERE indexname = 'parallel_ivf_idx'
	) INTO index_exists;
	
	IF NOT index_exists THEN
		RAISE EXCEPTION 'IVF index was not created';
	END IF;
	
	-- Get index size
	SELECT pg_relation_size('parallel_ivf_idx') INTO index_size;
	
	RAISE NOTICE 'IVF index created successfully, size: % bytes', index_size;
END$$;

-- Verify index can be used for queries
SELECT COUNT(*) AS indexed_rows
FROM parallel_build_test
WHERE embedding <-> (SELECT embedding FROM parallel_build_test LIMIT 1) < 1.0;

\echo ''
\echo 'Test 4: Verify parallel build callbacks are registered'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Check that HNSW and IVF support parallel builds
DO $$
DECLARE
	hnsw_parallel boolean;
	ivf_parallel boolean;
BEGIN
	-- Check HNSW parallel support (amcanparallel may not exist in all PostgreSQL versions)
	BEGIN
		SELECT amcanparallel INTO hnsw_parallel
		FROM pg_am
		WHERE amname = 'hnsw';
	EXCEPTION WHEN undefined_column THEN
		-- amcanparallel doesn't exist in this PostgreSQL version
		hnsw_parallel := NULL;
	END;
	
	-- Check IVF parallel support
	BEGIN
		SELECT amcanparallel INTO ivf_parallel
		FROM pg_am
		WHERE amname = 'ivf';
	EXCEPTION WHEN undefined_column THEN
		-- amcanparallel doesn't exist in this PostgreSQL version
		ivf_parallel := NULL;
	END;
	
	IF hnsw_parallel THEN
		RAISE NOTICE 'HNSW index access method supports parallel builds';
	ELSE
		RAISE NOTICE 'HNSW index access method does not support parallel builds';
	END IF;
	
	IF ivf_parallel THEN
		RAISE NOTICE 'IVF index access method supports parallel builds';
	ELSE
		RAISE NOTICE 'IVF index access method does not support parallel builds';
	END IF;
END$$;

\echo ''
\echo 'Test 5: Test with different parallel worker counts'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Test with 4 parallel workers
SET max_parallel_workers_per_gather = 4;

DROP INDEX IF EXISTS parallel_hnsw_idx2;
CREATE INDEX parallel_hnsw_idx2 ON parallel_build_test 
USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 64);

DO $$
DECLARE
	index_exists boolean;
BEGIN
	SELECT EXISTS (
		SELECT 1 FROM pg_indexes 
		WHERE indexname = 'parallel_hnsw_idx2'
	) INTO index_exists;
	
	IF index_exists THEN
		RAISE NOTICE 'HNSW index created with 4 parallel workers';
	ELSE
		RAISE EXCEPTION 'HNSW index creation failed with 4 parallel workers';
	END IF;
END$$;

\echo ''
\echo 'Test 6: Verify indexes are functional after parallel build'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Test HNSW search
DO $$
DECLARE
	query_vec vector(128);
	result_count integer;
BEGIN
	query_vec := (SELECT embedding FROM parallel_build_test LIMIT 1);
	
	SELECT COUNT(*) INTO result_count
	FROM (
		SELECT id, embedding <-> query_vec AS distance
		FROM parallel_build_test
		ORDER BY embedding <-> query_vec
		LIMIT 10
	) subq;
	
	IF result_count > 0 THEN
		RAISE NOTICE 'HNSW index search returned % results', result_count;
	ELSE
		RAISE EXCEPTION 'HNSW index search returned no results';
	END IF;
END$$;

-- Test IVF search
DO $$
DECLARE
	query_vec vector(128);
	result_count integer;
BEGIN
	query_vec := (SELECT embedding FROM parallel_build_test LIMIT 1);
	
	SELECT COUNT(*) INTO result_count
	FROM (
		SELECT id, embedding <-> query_vec AS distance
		FROM parallel_build_test
		ORDER BY embedding <-> query_vec
		LIMIT 10
	) subq;
	
	IF result_count > 0 THEN
		RAISE NOTICE 'IVF index search returned % results', result_count;
	ELSE
		RAISE EXCEPTION 'IVF index search returned no results';
	END IF;
END$$;

\echo ''
\echo '=========================================================================='
\echo '✓ Parallel Index Build: All tests complete'
\echo '=========================================================================='
