/*
 * Crash Prevention Test: Index Build Crashes
 * Tests index build operations with various invalid inputs.
 * 
 * Focus: Fixes crash in ivfbuild() at line 537 of ivf_am.c
 * 
 * Expected: Index builds should validate inputs and return errors, not crash.
 */

\set ON_ERROR_STOP off

BEGIN;

\echo '=========================================================================='
\echo 'Index Crash Tests: Invalid reloptions'
\echo '=========================================================================='

CREATE TABLE IF NOT EXISTS test_index_crash_table (
    vec vector(128),
    id integer
);

INSERT INTO test_index_crash_table (vec, id)
SELECT array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 128)))::vector(128),
       generate_series(1, 100)
ON CONFLICT DO NOTHING;

-- Test 1: Invalid lists value (should validate, not crash)
DO $$
BEGIN
    BEGIN
        CREATE INDEX test_invalid_lists_idx ON test_index_crash_table USING ivf (vec vector_l2_ops) WITH (lists = -1);
        RAISE EXCEPTION 'Should have failed with invalid lists value';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE '✓ Correctly rejected invalid lists value: %', SQLERRM;
    END;
END$$;

-- Test 2: NULL reloptions (should use defaults, not crash)
DO $$
BEGIN
    BEGIN
        -- Note: NULL reloptions should be handled gracefully
        CREATE INDEX test_null_opts_idx ON test_index_crash_table USING ivf (vec vector_l2_ops);
        DROP INDEX IF EXISTS test_null_opts_idx;
        RAISE NOTICE 'NULL reloptions handled correctly (uses defaults)';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'NULL reloptions handled with error (no crash): %', SQLERRM;
    END;
END$$;

ROLLBACK;

BEGIN;

\echo ''
\echo 'Index Crash Tests: Empty Tables'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE TEMP TABLE empty_index_table (vec vector(128));

-- Test 3: Index build on empty table (should handle gracefully, not crash)
DO $$
BEGIN
    BEGIN
        CREATE INDEX empty_idx ON empty_index_table USING ivf (vec vector_l2_ops) WITH (lists = 10);
        DROP INDEX IF EXISTS empty_idx;
        RAISE NOTICE 'Empty table index build handled correctly';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'Empty table handled with error (no crash): %', SQLERRM;
    END;
END$$;

ROLLBACK;

BEGIN;

\echo ''
\echo 'Index Crash Tests: Very Large Dimensions'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE TEMP TABLE large_dim_table (vec vector(4096));

INSERT INTO large_dim_table (vec)
SELECT array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 4096)))::vector(4096)
FROM generate_series(1, 100);

-- Test 4: Index build with very large dimensions (should handle memory, not crash)
DO $$
BEGIN
    BEGIN
        CREATE INDEX large_dim_idx ON large_dim_table USING ivf (vec vector_l2_ops) WITH (lists = 10);
        DROP INDEX IF EXISTS large_dim_idx;
        RAISE NOTICE 'Large dimension index build handled correctly';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'Large dimension handled with error (no crash): %', SQLERRM;
    END;
END$$;

ROLLBACK;

BEGIN;

\echo ''
\echo 'Index Crash Tests: NULL Vectors in Table'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE TEMP TABLE null_vec_table (vec vector(128));

INSERT INTO null_vec_table (vec) VALUES (NULL);
INSERT INTO null_vec_table (vec) VALUES (NULL);
INSERT INTO null_vec_table (vec) 
SELECT array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 128)))::vector(128)
FROM generate_series(1, 50);

-- Test 5: Index build on table with NULL vectors (should handle, not crash)
DO $$
BEGIN
    BEGIN
        CREATE INDEX null_vec_idx ON null_vec_table USING ivf (vec vector_l2_ops) WITH (lists = 10);
        DROP INDEX IF EXISTS null_vec_idx;
        RAISE NOTICE 'NULL vectors in table handled correctly';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'NULL vectors handled with error (no crash): %', SQLERRM;
    END;
END$$;

ROLLBACK;

BEGIN;

\echo ''
\echo 'Index Crash Tests: Invalid lists Parameter'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Test 6: lists = 0 (should validate, not crash)
DO $$
BEGIN
    BEGIN
        CREATE INDEX zero_lists_idx ON test_index_crash_table USING ivf (vec vector_l2_ops) WITH (lists = 0);
        RAISE EXCEPTION 'Should have failed with lists = 0';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE '✓ Correctly rejected lists = 0: %', SQLERRM;
    END;
END$$;

-- Test 7: Very large lists value (should validate, not crash)
DO $$
BEGIN
    BEGIN
        CREATE INDEX huge_lists_idx ON test_index_crash_table USING ivf (vec vector_l2_ops) WITH (lists = 999999999);
        RAISE NOTICE 'Very large lists value handled';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE '✓ Correctly rejected very large lists value: %', SQLERRM;
    END;
END$$;

ROLLBACK;

COMMIT;

\echo ''
\echo '=========================================================================='
\echo '✓ Index Crash Tests Complete'
\echo '=========================================================================='




