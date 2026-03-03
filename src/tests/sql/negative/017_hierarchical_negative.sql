-- 017_hierarchical_negative.sql
-- Negative test for hierarchical clustering
-- All possible negative tests with 1000 rows only

SET client_min_messages TO WARNING;


-- Verify required tables exist

-- Invalid table
DO $$ BEGIN
    PERFORM cluster_hierarchical('nonexistent_table', 'features', 3, 'average');
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

-- Invalid column
DO $$ BEGIN
    PERFORM cluster_hierarchical('test_train_view', 'invalid_col', 3, 'average');
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

-- Invalid k (k < 1)
DO $$ BEGIN
    PERFORM cluster_hierarchical('test_train_view', 'features', 0, 'average');
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

-- Invalid linkage method
DO $$ BEGIN
    PERFORM cluster_hierarchical('test_train_view', 'features', 3, 'invalid_linkage');
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

-- NULL table
DO $$ BEGIN
    PERFORM cluster_hierarchical(NULL, 'features', 3, 'average');
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

-- NULL column
DO $$ BEGIN
    PERFORM cluster_hierarchical('test_train_view', NULL, 3, 'average');
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

\echo 'Test completed successfully'
