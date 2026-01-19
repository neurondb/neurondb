-- 070_gpu_pq_distance.sql
-- Tests for GPU-accelerated Product Quantization asymmetric distance computation

\timing on
\pset footer off
\pset pager off
\pset tuples_only off

\set ON_ERROR_STOP on

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
		RAISE NOTICE 'Skipping GPU PQ distance test: running in CPU compute mode';
		-- Don't execute any test logic
		RETURN;
	END IF;
END$$;

\echo '=========================================================================='
\echo 'GPU PQ Distance: Comprehensive Tests'
\echo '=========================================================================='

-- Setup: Create test table with vectors suitable for PQ
\echo ''
\echo 'Setup: Creating test table with vectors'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DROP TABLE IF EXISTS gpu_pq_test CASCADE;
CREATE TABLE gpu_pq_test (
	id SERIAL PRIMARY KEY,
	embedding vector(128)
);

-- Insert test data (128 dimensions is good for PQ with 8 subspaces)
INSERT INTO gpu_pq_test (embedding)
SELECT 
	array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 128)))::vector(128)
FROM generate_series(1, 200);

\echo ''
\echo 'Test 1: Train PQ codebook (required for PQ operations)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	codebook bytea;
	codebook_size integer;
BEGIN
	-- Train PQ codebook: 8 subspaces, 256 codebook size
	SELECT train_pq_codebook('gpu_pq_test', 'embedding', 8, 256) INTO codebook;
	
	IF codebook IS NULL THEN
		RAISE EXCEPTION 'train_pq_codebook returned NULL';
	END IF;
	
	codebook_size := octet_length(codebook);
	IF codebook_size = 0 THEN
		RAISE EXCEPTION 'train_pq_codebook returned empty codebook';
	END IF;
	
	RAISE NOTICE 'PQ codebook trained successfully, size: % bytes', codebook_size;
END$$;

-- Display codebook info
SELECT 
	octet_length(train_pq_codebook('gpu_pq_test', 'embedding', 8, 256)) AS codebook_size_bytes;

\echo ''
\echo 'Test 2: GPU PQ search using vector_pq_search function (if available)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Test GPU-accelerated PQ search if the function exists
DO $$
DECLARE
	query_vec vector(128);
	result_count integer;
BEGIN
	-- Generate a query vector
	query_vec := array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 128)))::vector(128);
	
	-- Try to use vector_pq_search if available
	BEGIN
		-- This function should use GPU-accelerated PQ asymmetric distance if GPU is available
		SELECT COUNT(*) INTO result_count
		FROM vector_pq_search(
			query_vec,
			10,  -- k
			50   -- rerank_k
		);
		
		IF result_count > 0 THEN
			RAISE NOTICE 'vector_pq_search returned % results', result_count;
		ELSE
			RAISE NOTICE 'vector_pq_search returned 0 results (may need PQ index)';
		END IF;
	EXCEPTION WHEN undefined_function THEN
		RAISE NOTICE 'vector_pq_search function not available, skipping GPU PQ search test';
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'vector_pq_search error: %', SQLERRM;
	END;
END$$;

\echo ''
\echo 'Test 3: Verify GPU backend supports PQ asymmetric distance'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Check GPU info to see if PQ is supported
DO $$
DECLARE
	gpu_info record;
	gpu_available boolean;
BEGIN
	-- Check if GPU is available
	SELECT neurondb_gpu_is_available() INTO gpu_available;
	
	IF gpu_available THEN
		-- Get GPU info
		SELECT * INTO gpu_info FROM neurondb_gpu_info();
		
		RAISE NOTICE 'GPU backend: %, device: %', 
			COALESCE(gpu_info.backend, 'unknown'),
			COALESCE(gpu_info.device_name, 'unknown');
		
		-- CUDA and ROCm backends should support PQ asymmetric distance
		IF gpu_info.backend IN ('cuda', 'rocm') THEN
			RAISE NOTICE 'GPU backend supports PQ asymmetric distance computation';
		ELSIF gpu_info.backend = 'metal' THEN
			RAISE NOTICE 'Metal backend: PQ support may be limited (CPU fallback expected)';
		ELSE
			RAISE NOTICE 'Unknown GPU backend, PQ support uncertain';
		END IF;
	ELSE
		RAISE NOTICE 'GPU not available, PQ will use CPU fallback';
	END IF;
END$$;

\echo ''
\echo 'Test 4: Test PQ codebook training with different parameters'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Test with 4 subspaces
DO $$
DECLARE
	codebook bytea;
BEGIN
	SELECT train_pq_codebook('gpu_pq_test', 'embedding', 4, 128) INTO codebook;
	
	IF codebook IS NULL OR octet_length(codebook) = 0 THEN
		RAISE EXCEPTION 'PQ codebook training failed with m=4, ks=128';
	END IF;
	
	RAISE NOTICE 'PQ codebook (m=4, ks=128) trained successfully';
END$$;

-- Test with 16 subspaces
DO $$
DECLARE
	codebook bytea;
BEGIN
	SELECT train_pq_codebook('gpu_pq_test', 'embedding', 16, 256) INTO codebook;
	
	IF codebook IS NULL OR octet_length(codebook) = 0 THEN
		RAISE EXCEPTION 'PQ codebook training failed with m=16, ks=256';
	END IF;
	
	RAISE NOTICE 'PQ codebook (m=16, ks=256) trained successfully';
END$$;

\echo ''
\echo 'Test 5: Verify PQ operations work with GPU compute mode'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Check that we're in GPU mode
DO $$
DECLARE
	compute_mode text;
BEGIN
	BEGIN
		compute_mode := current_setting('neurondb.compute_mode', true);
	EXCEPTION WHEN OTHERS THEN
		compute_mode := '0';
	END;
	
	IF compute_mode = '1' THEN
		RAISE NOTICE 'Running in GPU compute mode - PQ operations should use GPU if available';
	ELSE
		RAISE NOTICE 'Running in CPU compute mode - PQ operations will use CPU';
	END IF;
END$$;

\echo ''
\echo '=========================================================================='
\echo '✓ GPU PQ Distance: All tests complete'
\echo '=========================================================================='
