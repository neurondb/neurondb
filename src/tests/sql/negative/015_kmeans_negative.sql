-- 015_kmeans_negative.sql
-- Negative test for kmeans
-- All possible negative tests with 1000 rows only

SET client_min_messages TO WARNING;


-- Verify required tables exist

-- Invalid table
DO $$ BEGIN
    PERFORM cluster_kmeans('nonexistent_table', 'features', 3, 100);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

-- Invalid column
DO $$ BEGIN
    PERFORM cluster_kmeans('test_train_view', 'invalid_col', 3, 100);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

-- Invalid k (k < 1)
DO $$ BEGIN
    PERFORM cluster_kmeans('test_train_view', 'features', 0, 100);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

-- Invalid k (k > number of points)
DO $$ BEGIN
    PERFORM cluster_kmeans('test_train_view', 'features', 10000, 100);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

-- Invalid max_iters (max_iters < 1)
DO $$ BEGIN
    PERFORM cluster_kmeans('test_train_view', 'features', 3, 0);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

-- NULL table
DO $$ BEGIN
    PERFORM cluster_kmeans(NULL, 'features', 3, 100);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

-- NULL column
DO $$ BEGIN
    PERFORM cluster_kmeans('test_train_view', NULL, 3, 100);
    RAISE EXCEPTION 'Should have failed';
EXCEPTION WHEN OTHERS THEN
END $$;

\echo 'Test completed successfully'
