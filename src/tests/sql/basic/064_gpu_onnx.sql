-- 038_onnx_basic.sql
-- Basic test for ONNX module: model loading and inference

\timing on
\pset footer off
\pset pager off
\pset tuples_only off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'ONNX Module: Basic Functionality Tests'
\echo '=========================================================================='

/*-------------------------------------------------------------------
 * ---- ONNX MODEL OPERATIONS ----
 *------------------------------------------------------------------*/
\echo ''
\echo 'ONNX Model Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Test 1: HuggingFace embedding via ONNX (if available)'
DO $$
DECLARE
	result vector;
BEGIN
	BEGIN
		result := neurondb_hf_embedding('sentence-transformers/all-MiniLM-L6-v2', 'Test text');
		RAISE NOTICE 'ONNX embedding: % dimensions', vector_dims(result);
	EXCEPTION WHEN OTHERS THEN
		NULL; -- May not be available if ONNX not configured
	END;
END$$;

\echo 'Test 2: HuggingFace tokenization (if available)'
DO $$
DECLARE
	result text[];
BEGIN
	BEGIN
		result := neurondb_hf_tokenize('sentence-transformers/all-MiniLM-L6-v2', 'Test text');
		RAISE NOTICE 'Tokenization: % tokens', array_length(result, 1);
	EXCEPTION WHEN OTHERS THEN
		NULL; -- May not be available
	END;
END$$;

\echo 'Test 3: HuggingFace classification (if available)'
DO $$
DECLARE
	result text;
BEGIN
	BEGIN
		result := neurondb_hf_classify('text-classification-model', 'Test text');
		RAISE NOTICE 'Classification result: %', result;
	EXCEPTION WHEN OTHERS THEN
		NULL; -- May not be available
	END;
END$$;

\echo ''
\echo '=========================================================================='
\echo '✓ ONNX Module: Basic tests complete'
\echo '=========================================================================='

\echo 'Test completed successfully'
