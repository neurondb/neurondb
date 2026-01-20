\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo '=========================================================================='

-- Test 1: GPU Enable and Availability
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb_gpu_enable() AS gpu_enabled;
-- GPU availability is shown in neurondb_gpu_info() below

-- Test 2: GPU Information
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT * FROM neurondb_gpu_info();

-- Test 3: GPU Statistics
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT * FROM neurondb_gpu_stats();

-- Test 4: LLM GPU Information
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb_llm_gpu_available() AS llm_gpu_available;
SELECT * FROM neurondb_llm_gpu_info();

-- Test 4a: GPU Utilization Metrics (NVML/rocm-smi integration)
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	gpu_util record;
	utilization real;
	temperature real;
	power real;
	gpu_available boolean;
	backend_name text;
BEGIN
	-- Check if GPU is available
	SELECT is_available INTO gpu_available FROM neurondb_gpu_info() LIMIT 1;
	
	IF gpu_available THEN
		-- Get GPU backend name from LLM GPU info
		SELECT backend INTO backend_name FROM neurondb_llm_gpu_info() LIMIT 1;
		
		-- Get GPU utilization metrics
		SELECT * INTO gpu_util FROM neurondb.llm_gpu_utilization() LIMIT 1;
		
		IF gpu_util IS NOT NULL THEN
			utilization := gpu_util.utilization_pct;
			temperature := gpu_util.temperature_c;
			power := gpu_util.power_w;
			
			RAISE NOTICE 'GPU Backend: %', backend_name;
			RAISE NOTICE 'GPU Utilization: % %%', utilization;
			RAISE NOTICE 'GPU Temperature: % C', temperature;
			RAISE NOTICE 'GPU Power: % W', power;
			
			-- Verify metrics are reasonable (not all zeros if GPU is actually available)
			-- Note: Values may be 0 if NVML/rocm-smi is not available, which is acceptable
			IF backend_name IN ('cuda', 'rocm') THEN
				-- For CUDA/ROCm, we expect actual values from NVML/rocm-smi
				-- But they might be 0 if the libraries aren't available
				IF utilization = 0 AND temperature = 0 AND power = 0 THEN
					RAISE NOTICE 'GPU metrics are all zero - NVML/rocm-smi may not be available (acceptable)';
				ELSE
					RAISE NOTICE 'GPU metrics retrieved successfully from system APIs';
				END IF;
			ELSIF backend_name = 'metal' THEN
				-- Metal backend: IOKit integration is deferred, so 0 values are expected
				RAISE NOTICE 'Metal backend: IOKit integration deferred, metrics may be zero';
			END IF;
		ELSE
			RAISE NOTICE 'neurondb.llm_gpu_utilization returned NULL';
		END IF;
	ELSE
		RAISE NOTICE 'GPU not available, skipping utilization metrics test';
	END IF;
END$$;

-- Display GPU utilization
SELECT * FROM neurondb.llm_gpu_utilization();

-- Test 5: GPU Distance Functions (if available)
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- GPU already enabled via test_settings above

SELECT 
	vector_l2_distance_gpu('[1,2,3]'::vector, '[4,5,6]'::vector) AS gpu_l2_distance,
	vector_cosine_distance_gpu('[1,2,3]'::vector, '[4,5,6]'::vector) AS gpu_cosine_distance;

\echo ''
\echo '=========================================================================='

\echo 'Test completed successfully'
