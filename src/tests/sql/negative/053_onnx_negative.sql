-- 038_onnx_negative.sql
-- Negative test cases for ONNX module: error handling

\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP off

\echo '=========================================================================='
\echo 'ONNX Module: Negative Test Cases (Error Handling)'
\echo '=========================================================================='

/*-------------------------------------------------------------------
 * ---- ONNX MODEL ERRORS ----
 *------------------------------------------------------------------*/
\echo ''
\echo 'ONNX Model Error Tests'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

\echo 'Error Test 1: HuggingFace embedding with NULL model'
SELECT neurondb_hf_embedding(NULL, 'Test text');

\echo 'Error Test 2: HuggingFace embedding with NULL text'
SELECT neurondb_hf_embedding('model-name', NULL);

\echo 'Error Test 3: HuggingFace embedding with invalid model'
SELECT neurondb_hf_embedding('nonexistent_model_xyz', 'Test text');

\echo 'Error Test 4: HuggingFace tokenization with NULL model'
SELECT neurondb_hf_tokenize(NULL, 'Test text');

\echo 'Error Test 5: HuggingFace tokenization with NULL text'
SELECT neurondb_hf_tokenize('model-name', NULL);

\echo 'Error Test 6: HuggingFace classification with NULL model'
SELECT neurondb_hf_classify(NULL, 'Test text');

\echo 'Error Test 7: HuggingFace classification with NULL text'
SELECT neurondb_hf_classify('model-name', NULL);

\echo 'Error Test 8: HuggingFace NER with NULL model'
SELECT neurondb_hf_ner(NULL, 'Test text');

\echo 'Error Test 9: HuggingFace NER with NULL text'
SELECT neurondb_hf_ner('model-name', NULL);

\echo 'Error Test 10: HuggingFace QA with NULL model'
SELECT neurondb_hf_qa(NULL, 'Question?', 'Context');

\echo 'Error Test 11: HuggingFace QA with NULL question'
SELECT neurondb_hf_qa('model-name', NULL, 'Context');

\echo 'Error Test 12: HuggingFace QA with NULL context'
SELECT neurondb_hf_qa('model-name', 'Question?', NULL);

\echo ''
\echo '=========================================================================='
\echo '✓ ONNX Module: Negative test cases complete'
\echo '=========================================================================='

\echo 'Test completed successfully'




