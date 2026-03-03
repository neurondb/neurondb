-- Template for reading test settings
-- Copy this block to the beginning of each test file after \set ON_ERROR_STOP on
-- Note: GPU settings are configured by test runner via ALTER SYSTEM SET and reload
-- This block only verifies the settings, it does not set them

/* Step 0: Read settings from test_settings table and verify GPU configuration */
DO $$
DECLARE
	gpu_mode TEXT;
	current_compute_mode TEXT;
	current_gpu_kernels TEXT;
	num_rows_val TEXT;
	table_exists BOOLEAN;
BEGIN
	-- Check if test_settings table exists
	SELECT EXISTS (
		SELECT FROM information_schema.tables 
		WHERE table_schema = 'public' 
		AND table_name = 'test_settings'
	) INTO table_exists;
	
	-- Only read settings if table exists
	IF table_exists THEN
		-- Read GPU mode setting from test_settings (may be NULL if not set)
		SELECT setting_value INTO gpu_mode 
		FROM test_settings 
		WHERE setting_key = 'gpu_mode';
		
		-- Verify GPU configuration matches test_settings
		BEGIN
			SELECT current_setting('neurondb.compute_mode', true) INTO current_compute_mode;
		EXCEPTION WHEN OTHERS THEN
			current_compute_mode := NULL;
		END;
		
		-- Only verify if gpu_mode was actually set
		IF gpu_mode IS NOT NULL THEN
			IF gpu_mode = 'gpu' THEN
				-- Verify GPU is enabled (should be set by test runner)
				IF current_compute_mode IS NOT NULL AND current_compute_mode != '1' THEN
					RAISE WARNING 'GPU mode expected but neurondb.compute_mode = % (expected: 1)', current_compute_mode;
				END IF;
			ELSIF gpu_mode = 'cpu' THEN
				-- Verify CPU mode (should be set by test runner)
				IF current_compute_mode IS NOT NULL AND current_compute_mode != '0' THEN
					RAISE WARNING 'CPU mode expected but neurondb.compute_mode = % (expected: 0)', current_compute_mode;
				END IF;
			END IF;
		END IF;
		
		-- Read other settings if needed (may be NULL)
		SELECT setting_value INTO num_rows_val 
		FROM test_settings 
		WHERE setting_key = 'num_rows';
		
		BEGIN
			SELECT current_setting('neurondb.gpu_kernels', true) INTO current_gpu_kernels;
		EXCEPTION WHEN OTHERS THEN
			current_gpu_kernels := NULL;
		END;
	ELSE
		-- Table doesn't exist yet - this is OK, settings will be created by test runner
		RAISE DEBUG 'test_settings table does not exist yet, skipping verification';
	END IF;
	
	-- Additional settings can be read and verified here as needed
END $$;

