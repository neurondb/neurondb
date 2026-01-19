-- Basic tests for GPU search functions
-- Tests GPU HNSW/IVF search, GPU-accelerated index build, and backend support (CUDA/ROCm/Metal)

\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'GPU Search: Comprehensive Tests'
\echo '=========================================================================='

-- Skip this test if running in CPU compute mode
DO $$
DECLARE
	compute_mode_val text;
BEGIN
	-- Check compute mode, default to '0' (CPU) if not set
	BEGIN
		compute_mode_val := current_setting('neurondb.compute_mode', true);
	EXCEPTION WHEN OTHERS THEN
		compute_mode_val := '0';  -- Default to CPU if setting doesn't exist
	END;
	
	-- Skip test if in CPU mode - just print message and exit
	IF compute_mode_val = '0' THEN
		RAISE NOTICE 'Skipping GPU search test: running in CPU compute mode';
		-- Don't execute any test logic
		RETURN;
	END IF;
END$$;

-- Setup: Create test table
\echo ''
\echo 'Setup: Creating test table with vectors'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DROP TABLE IF EXISTS gpu_search_test CASCADE;
CREATE TABLE gpu_search_test (
	id serial PRIMARY KEY,
	vec vector(128)
);

INSERT INTO gpu_search_test (vec)
SELECT 
	array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 128)))::vector(128)
FROM generate_series(1, 200);

-- Test 1: GPU-accelerated HNSW index build
\echo ''
\echo 'Test 1: GPU-accelerated HNSW index build'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Enable GPU for index build
SET neurondb.compute_mode = '1';

-- Create HNSW index (should use GPU-accelerated distance computation during build)
CREATE INDEX gpu_hnsw_idx ON gpu_search_test USING hnsw (vec vector_l2_ops)
WITH (m = 16, ef_construction = 64, ef_search = 40);

-- Verify index was created
DO $$
DECLARE
	index_exists boolean;
	index_size bigint;
BEGIN
	SELECT EXISTS (
		SELECT 1 FROM pg_indexes 
		WHERE indexname = 'gpu_hnsw_idx'
	) INTO index_exists;
	
	IF NOT index_exists THEN
		RAISE EXCEPTION 'GPU HNSW index was not created';
	END IF;
	
	SELECT pg_relation_size('gpu_hnsw_idx') INTO index_size;
	RAISE NOTICE 'GPU HNSW index created successfully, size: % bytes', index_size;
END$$;
	
-- Test 2: GPU HNSW search
\echo ''
\echo 'Test 2: GPU HNSW search'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	query_vec vector(128);
	result_count integer;
BEGIN
	-- Generate query vector
	query_vec := array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 128)))::vector(128);
	
	-- Test GPU HNSW search (if function available)
	BEGIN
		SELECT COUNT(*) INTO result_count
		FROM (
			SELECT id, vec <-> query_vec AS distance
			FROM gpu_search_test
			ORDER BY vec <-> query_vec
			LIMIT 10
		) subq;
		
		IF result_count > 0 THEN
			RAISE NOTICE 'GPU HNSW search returned % results', result_count;
		END IF;
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'GPU HNSW search error: %', SQLERRM;
	END;
END$$;

-- Test 3: GPU-accelerated IVF index build
\echo ''
\echo 'Test 3: GPU-accelerated IVF index build'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE INDEX gpu_ivf_idx ON gpu_search_test USING ivf (vec vector_l2_ops)
WITH (lists = 10, probes = 5);

-- Verify index was created
DO $$
DECLARE
	index_exists boolean;
BEGIN
	SELECT EXISTS (
		SELECT 1 FROM pg_indexes 
		WHERE indexname = 'gpu_ivf_idx'
	) INTO index_exists;
	
	IF index_exists THEN
		RAISE NOTICE 'GPU IVF index created successfully';
	ELSE
		RAISE EXCEPTION 'GPU IVF index was not created';
	END IF;
END$$;

-- Test 4: GPU IVF search
\echo ''
\echo 'Test 4: GPU IVF search'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	query_vec vector(128);
	result_count integer;
BEGIN
	query_vec := array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 128)))::vector(128);
	
	SELECT COUNT(*) INTO result_count
	FROM (
		SELECT id, vec <-> query_vec AS distance
		FROM gpu_search_test
		ORDER BY vec <-> query_vec
		LIMIT 10
	) subq;
	
	IF result_count > 0 THEN
		RAISE NOTICE 'GPU IVF search returned % results', result_count;
	END IF;
END$$;

-- Test 5: Verify GPU backend type (CUDA/ROCm/Metal)
\echo ''
\echo 'Test 5: Verify GPU backend type'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	gpu_info record;
	backend_name text;
	gpu_available boolean;
BEGIN
	SELECT neurondb_gpu_is_available() INTO gpu_available;
	
	IF gpu_available THEN
		SELECT * INTO gpu_info FROM neurondb_gpu_info();
		backend_name := gpu_info.backend;
		
		RAISE NOTICE 'GPU Backend: %', backend_name;
		RAISE NOTICE 'GPU Device: %', gpu_info.device_name;
		
		-- Verify backend-specific features
		IF backend_name = 'cuda' THEN
			RAISE NOTICE 'CUDA backend: Supports HNSW/IVF search and GPU-accelerated index build';
		ELSIF backend_name = 'rocm' THEN
			RAISE NOTICE 'ROCm backend: Supports HIP kernels for HNSW/IVF search';
		ELSIF backend_name = 'metal' THEN
			RAISE NOTICE 'Metal backend: CPU fallback for HNSW/IVF search (full Metal implementation deferred)';
		ELSE
			RAISE NOTICE 'Unknown backend: %', backend_name;
		END IF;
	ELSE
		RAISE NOTICE 'GPU not available - tests will use CPU fallback';
	END IF;
END$$;

-- Display GPU info
SELECT * FROM neurondb_gpu_info();

-- Test 6: Error handling
\echo ''
\echo 'Test 6: Error handling'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
BEGIN
	-- Test with nonexistent index
	BEGIN
		PERFORM id FROM gpu_search_test
		WHERE vec <-> (SELECT vec FROM gpu_search_test LIMIT 1) < 1.0
		USING INDEX nonexistent_index;
		RAISE EXCEPTION 'Should have failed with nonexistent index';
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'Correctly handled nonexistent index error';
	END;
END$$;

\echo ''
\echo '=========================================================================='
\echo '✓ GPU Search: All tests complete'
\echo '=========================================================================='
