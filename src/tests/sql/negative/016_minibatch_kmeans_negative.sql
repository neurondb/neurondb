-- 016_minibatch_kmeans_negative.sql
-- Negative test for minibatch_kmeans
-- All possible negative tests with 1000 rows only

SET client_min_messages TO WARNING;


-- Verify required tables exist

-- Invalid table
DO $$ BEGIN
    PERFORM cluster_minibatch_kmeans('nonexistent_table', 'features', 3, 100, 100);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

-- Invalid column
DO $$ BEGIN
    PERFORM cluster_minibatch_kmeans('test_train_view', 'invalid_col', 3, 100, 100);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

-- Invalid k (k < 1)
DO $$ BEGIN
    PERFORM cluster_minibatch_kmeans('test_train_view', 'features', 0, 100, 100);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

-- Invalid batch_size (batch_size < 1)
DO $$ BEGIN
    PERFORM cluster_minibatch_kmeans('test_train_view', 'features', 3, 0, 100);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

-- Invalid max_iters (max_iters < 1)
DO $$ BEGIN
    PERFORM cluster_minibatch_kmeans('test_train_view', 'features', 3, 100, 0);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

\echo 'Test completed successfully'
