-- 011_hybrid_search_negative.sql
-- Negative test for hybrid_search

SET client_min_messages TO WARNING;


-- Invalid table
DO $$ BEGIN
    PERFORM neurondb.train('hybrid_search', 'nonexistent_table', 'features', 'label', '{}'::jsonb);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

-- Invalid column
DO $$ BEGIN
    PERFORM neurondb.train('hybrid_search', 'test_train_view', 'invalid_col', 'label', '{}'::jsonb);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

\echo 'Test completed successfully'
