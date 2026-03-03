\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Test: FP8 Quantization Negative Tests'
\echo '=========================================================================='

-- Test 1: Invalid compression type
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 1: Invalid compression type'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
BEGIN
	BEGIN
		PERFORM auto_quantize('[1.0,2.0,3.0]'::vector, 'invalid_type');
		RAISE WARNING 'Should have failed!';
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'Correctly rejected invalid type: %', SQLERRM;
	END;
END$$;

-- Test 2: NULL vector
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 2: NULL vector'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
BEGIN
	BEGIN
		PERFORM quantize_fp8_e4m3(NULL);
		RAISE WARNING 'Should have failed!';
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'Correctly rejected NULL: %', SQLERRM;
	END;
END$$;

-- Test 3: Invalid bytea for dequantize
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 3: Invalid bytea for dequantize'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
BEGIN
	BEGIN
		PERFORM dequantize_fp8('\x00'::bytea);
		RAISE WARNING 'Should have failed!';
	EXCEPTION WHEN OTHERS THEN
		RAISE NOTICE 'Correctly rejected invalid bytea: %', SQLERRM;
	END;
END$$;

\echo ''
\echo '✅ Negative FP8 quantization tests completed'

\echo 'Test completed successfully'
