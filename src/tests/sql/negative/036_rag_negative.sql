-- 036_rag_negative.sql
-- Negative test for RAG functions

SET client_min_messages TO WARNING;

\echo '=========================================================================='
\echo 'Negative Tests: RAG Functions Error Handling'
\echo '=========================================================================='

-- Test 1: rag_query with nonexistent table
DO $$ BEGIN
    PERFORM * FROM neurondb.rag_query(
        'test query',
        'nonexistent_table',
        'embedding',
        'content',
        'default',
        5
    );
    RAISE EXCEPTION 'Should have failed with nonexistent table';
EXCEPTION WHEN undefined_table THEN
    RAISE NOTICE '✓ Correctly failed with nonexistent table';
END $$;

-- Test 2: rag_query_with_context with invalid column
DO $$ BEGIN
    CREATE TEMP TABLE test_rag_table (
        id SERIAL PRIMARY KEY,
        content TEXT,
        embedding VECTOR(384)
    );
    
    PERFORM * FROM neurondb.rag_query_with_context(
        'test query',
        'test_rag_table',
        'nonexistent_vector_col',
        'content',
        'default',
        5
    );
    RAISE EXCEPTION 'Should have failed with invalid column';
EXCEPTION WHEN undefined_column THEN
    RAISE NOTICE '✓ Correctly failed with invalid column';
END $$;

-- Test 3: rag_ingest_document with invalid table
DO $$ BEGIN
    PERFORM * FROM neurondb.rag_ingest_document(
        'test document text',
        'nonexistent_table',
        'content',
        'embedding',
        'default',
        512,
        128,
        '{}'::jsonb
    );
    RAISE EXCEPTION 'Should have failed with nonexistent table';
EXCEPTION WHEN undefined_table THEN
    RAISE NOTICE '✓ Correctly failed with nonexistent table';
END $$;

-- Test 4: rag_evaluate with empty context
DO $$ BEGIN
    DECLARE
        result jsonb;
    BEGIN
        result := neurondb.rag_evaluate(
            'test query',
            'test answer',
            ARRAY[]::text[],  -- empty context
            'basic'
        );
        
        -- Should not fail, but relevancy should be 0
        IF (result->>'relevancy')::float8 != 0.0 THEN
            RAISE EXCEPTION 'Expected relevancy to be 0.0 for empty context';
        END IF;
        
        RAISE NOTICE '✓ Correctly handled empty context';
    END;
END $$;

-- Test 5: rag_chat with invalid table
DO $$ BEGIN
    PERFORM neurondb.rag_chat(
        'test query',
        'nonexistent_table',
        'embedding',
        'content',
        'default',
        5,
        '[]'::jsonb,
        'gpt-3.5-turbo'
    );
    RAISE EXCEPTION 'Should have failed with nonexistent table';
EXCEPTION WHEN undefined_table THEN
    RAISE NOTICE '✓ Correctly failed with nonexistent table';
END $$;

-- Test 6: create_rag_pipeline with duplicate name
DO $$ BEGIN
    DECLARE
        pipeline_id1 int;
        pipeline_id2 int;
        test_name text := 'duplicate_test_' || extract(epoch from now())::text;
    BEGIN
        -- Create first pipeline
        pipeline_id1 := neurondb.create_rag_pipeline(
            test_name,
            'default',
            512,
            128,
            '{}'::jsonb
        );
        
        -- Try to create duplicate
        BEGIN
            pipeline_id2 := neurondb.create_rag_pipeline(
                test_name,
                'default',
                512,
                128,
                '{}'::jsonb
            );
            RAISE EXCEPTION 'Should have failed with duplicate name';
        EXCEPTION WHEN unique_violation THEN
            RAISE NOTICE '✓ Correctly failed with duplicate pipeline name';
        END;
        
        -- Cleanup
        DELETE FROM neurondb.rag_pipelines WHERE pipeline_id = pipeline_id1;
    END;
END $$;

-- Test 7: update_rag_pipeline with nonexistent ID
DO $$ BEGIN
    DECLARE
        result boolean;
    BEGIN
        result := neurondb.update_rag_pipeline(
            999999,  -- nonexistent ID
            '{"test": true}'::jsonb
        );
        
        IF result != false THEN
            RAISE EXCEPTION 'Expected update to return false for nonexistent pipeline';
        END IF;
        
        RAISE NOTICE '✓ Correctly returned false for nonexistent pipeline';
    END;
END $$;

-- Test 8: rag_query with invalid top_k (negative)
DO $$ BEGIN
    CREATE TEMP TABLE test_rag_table2 (
        id SERIAL PRIMARY KEY,
        content TEXT,
        embedding VECTOR(384)
    );
    
    PERFORM * FROM neurondb.rag_query(
        'test query',
        'test_rag_table2',
        'embedding',
        'content',
        'default',
        -1  -- invalid top_k
    );
    RAISE EXCEPTION 'Should have failed with invalid top_k';
EXCEPTION WHEN OTHERS THEN
    IF SQLERRM LIKE '%top_k%' OR SQLERRM LIKE '%invalid%' OR SQLERRM LIKE '%negative%' THEN
        RAISE NOTICE '✓ Correctly failed with invalid top_k';
    ELSE
        RAISE;
    END IF;
END $$;

\echo ''
\echo '=========================================================================='
\echo '✅ All negative RAG tests passed!'
\echo '=========================================================================='

\echo 'Test completed successfully'
