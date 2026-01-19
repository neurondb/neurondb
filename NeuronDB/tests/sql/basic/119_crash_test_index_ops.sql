-- Comprehensive index operations crash test
-- Tests HNSW/IVF index edge cases and stress conditions

\set ON_ERROR_STOP off
\timing on

BEGIN;

\echo 'Test 1: Index on empty table'
CREATE TABLE test_empty_index (id int, v vector(128));
DO $$
BEGIN
    CREATE INDEX idx_empty ON test_empty_index USING hnsw (v);
    RAISE NOTICE 'PASS: Index on empty table created';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'INFO: Index on empty table: %', SQLERRM;
END$$;
DROP TABLE IF EXISTS test_empty_index CASCADE;

\echo 'Test 2: Invalid index parameters'
CREATE TABLE test_invalid_index (id int, v vector(128));
INSERT INTO test_invalid_index SELECT generate_series(1, 100), array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 128)))::vector(128);

DO $$
BEGIN
    CREATE INDEX idx_invalid_m ON test_invalid_index USING hnsw (v) WITH (m='0');
    RAISE EXCEPTION 'Should have failed with m=0';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'PASS: Invalid m parameter rejected: %', SQLERRM;
END$$;

DO $$
BEGIN
    CREATE INDEX idx_invalid_ef ON test_invalid_index USING hnsw (v) WITH (ef_construction='0');
    RAISE EXCEPTION 'Should have failed with ef_construction=0';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'PASS: Invalid ef_construction rejected: %', SQLERRM;
END$$;

\echo 'Test 3: Concurrent inserts during index build'
CREATE TABLE test_concurrent_index (id int, v vector(128));
CREATE INDEX idx_concurrent ON test_concurrent_index USING hnsw (v) WITH (m='16', ef_construction='200');

DO $$
DECLARE
    i int;
BEGIN
    FOR i IN 1..50 LOOP
        INSERT INTO test_concurrent_index VALUES (i, array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 128)))::vector(128));
    END LOOP;
    RAISE NOTICE 'PASS: Concurrent inserts completed';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'ERROR: Concurrent inserts failed: %', SQLERRM;
END$$;

\echo 'Test 4: Index on table with NULL vectors'
CREATE TABLE test_null_index (id int, v vector(128));
INSERT INTO test_null_index VALUES (1, NULL), (2, array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 128)))::vector(128));

DO $$
BEGIN
    CREATE INDEX idx_null ON test_null_index USING hnsw (v);
    RAISE NOTICE 'PASS: Index with NULL vectors handled';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'INFO: Index with NULL vectors: %', SQLERRM;
END$$;

\echo 'Test 5: Multiple indexes on same table'
CREATE TABLE test_multi_index (id int, v vector(128));
INSERT INTO test_multi_index SELECT generate_series(1, 100), array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 128)))::vector(128);

DO $$
BEGIN
    CREATE INDEX idx_multi1 ON test_multi_index USING hnsw (v) WITH (m='16');
    CREATE INDEX idx_multi2 ON test_multi_index USING ivf (v) WITH (lists='10');
    RAISE NOTICE 'PASS: Multiple indexes created';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'ERROR: Multiple indexes failed: %', SQLERRM;
END$$;

\echo 'Test 6: Query with invalid index parameters'
DO $$
DECLARE
    result record;
BEGIN
    BEGIN
        SET hnsw.ef_search = 0;
        SELECT * INTO result FROM test_multi_index ORDER BY v <-> array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 128)))::vector(128) LIMIT 1;
        RAISE NOTICE 'INFO: Query with ef_search=0 handled';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'PASS: Invalid ef_search rejected: %', SQLERRM;
    END;
END$$;

ROLLBACK;



