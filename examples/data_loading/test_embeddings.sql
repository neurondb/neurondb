-- Test Embedding Generation in NeuronDB
-- This script tests if embeddings are working correctly
-- Run this to diagnose embedding issues

\echo '========================================'
\echo 'NeuronDB Embedding Test'
\echo '========================================'
\echo ''

-- Step 1: Check Configuration
\echo 'Step 1: Checking Configuration...'
SELECT 
    'API Key' as setting,
    CASE 
        WHEN current_setting('neurondb.llm_api_key', true) IS NULL 
        THEN 'NOT SET ❌'
        WHEN current_setting('neurondb.llm_api_key', true) = ''
        THEN 'EMPTY ❌'
        ELSE 'SET ✓ (' || left(current_setting('neurondb.llm_api_key', true), 8) || '...)'
    END as status
UNION ALL
SELECT 
    'Provider',
    COALESCE(current_setting('neurondb.llm_provider', true), 'huggingface (default)')
UNION ALL
SELECT 
    'Endpoint',
    COALESCE(current_setting('neurondb.llm_endpoint', true), 'https://api-inference.huggingface.co (default)')
UNION ALL
SELECT 
    'Model',
    COALESCE(current_setting('neurondb.llm_model', true), 'sentence-transformers/all-MiniLM-L6-v2 (default)')
UNION ALL
SELECT 
    'Timeout',
    COALESCE(current_setting('neurondb.llm_timeout_ms', true), '15000 (default)') || ' ms';

\echo ''
\echo 'Step 2: Testing Embedding Generation...'

-- Step 2: Test Single Embedding
SELECT 
    'Test Text' as test,
    'The quick brown fox' as input,
    array_length(embed_text('The quick brown fox')::float[], 1) as dimension,
    CASE 
        WHEN embed_text('The quick brown fox') = (SELECT array_agg(0.0::float)::vector(384) FROM generate_series(1, 384))
        THEN 'ZEROS ❌ - API key not working'
        ELSE 'OK ✓ - Embeddings generated'
    END as status;

\echo ''
\echo 'Step 3: Checking Embedding Values...'

-- Step 3: Check if embeddings are zeros
SELECT 
    'First 5 values' as check,
    array_to_string((embed_text('test')::float[])[1:5], ', ') as sample_values,
    CASE 
        WHEN embed_text('test') = (SELECT array_agg(0.0::float)::vector(384) FROM generate_series(1, 384))
        THEN 'ALL ZEROS ❌'
        ELSE 'NON-ZERO ✓'
    END as result;

\echo ''
\echo 'Step 4: Testing Batch Embeddings...'

-- Step 4: Test batch embeddings
SELECT 
    'Batch Test' as test,
    array_length(embed_text_batch(ARRAY['text1', 'text2', 'text3'])::vector[], 1) as count,
    array_length(embed_text_batch(ARRAY['text1', 'text2', 'text3'])[1]::float[], 1) as dimension;

\echo ''
\echo 'Step 5: Full Embedding Sample (first 10 dimensions)...'

-- Step 5: Show sample embedding values
SELECT 
    'Sample Embedding' as description,
    array_to_string((embed_text('Machine learning is fascinating')::float[])[1:10], ', ') as first_10_dimensions,
    array_length(embed_text('Machine learning is fascinating')::float[], 1) as total_dimensions;

\echo ''
\echo '========================================'
\echo 'Diagnostic Summary'
\echo '========================================'

-- Summary
SELECT 
    CASE 
        WHEN current_setting('neurondb.llm_api_key', true) IS NULL 
        THEN '❌ API KEY NOT SET - Run: ALTER SYSTEM SET neurondb.llm_api_key = ''your-key''; SELECT pg_reload_conf();'
        WHEN current_setting('neurondb.llm_api_key', true) = ''
        THEN '❌ API KEY IS EMPTY - Set a valid API key'
        WHEN embed_text('test') = (SELECT array_agg(0.0::float)::vector(384) FROM generate_series(1, 384))
        THEN '❌ EMBEDDINGS ARE ZEROS - API key may be invalid or API is unreachable. Check logs.'
        ELSE '✓ EMBEDDINGS ARE WORKING - Configuration is correct!'
    END as diagnostic_result;

\echo ''
\echo 'If embeddings are zeros, check:'
\echo '  1. API key is set: SELECT current_setting(''neurondb.llm_api_key'', true);'
\echo '  2. Reload config: SELECT pg_reload_conf();'
\echo '  3. Check PostgreSQL logs for errors'
\echo '  4. Test API connectivity'
\echo ''


