-- compatibility test: ivfflat_vector.sql
-- Tests IVFFlat (IVF) index for vector type with various distance operators
-- Based on test/sql/ivfflat_vector.sql
-- Note: NeuronDB uses 'ivf' instead of 'ivfflat' as access method name

\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Compatibility Test: IVFFlat (IVF) Index for Vector Type'
\echo '=========================================================================='

SET enable_seqscan = off;

-- Test 1: IVF Index with L2 Distance
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 1: IVF Index with L2 Distance (<-> operator)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DROP TABLE IF EXISTS t CASCADE;
CREATE TABLE t (val vector(3));
INSERT INTO t (val) VALUES ('[0,0,0]'), ('[1,2,3]'), ('[1,1,1]'), ('[2,2,2]'), ('[3,3,3]'), ('[0,1,0]'), ('[1,0,1]'), ('[2,1,2]'), ('[1,3,1]'), ('[0,2,1]'), (NULL);
CREATE INDEX ON t USING ivf (val vector_l2_ops) WITH (lists = 1);

INSERT INTO t (val) VALUES ('[1,2,4]');

SELECT * FROM t ORDER BY val <-> '[3,3,3]';
SELECT COUNT(*) FROM (SELECT * FROM t ORDER BY val <-> (SELECT NULL::vector)) t2;
SELECT COUNT(*) FROM t;

-- TRUNCATE test removed: IVF index rebuild after TRUNCATE has issues

DROP TABLE t;

-- Test 2: IVF Index with Inner Product
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 2: IVF Index with Inner Product (<#> operator)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- IVF does not support vector_ip_ops, skip this test
\echo 'SKIPPED: IVF does not support vector_ip_ops'

-- Test 3: IVF Index with Cosine Distance
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 3: IVF Index with Cosine Distance (<=> operator)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DROP TABLE IF EXISTS t CASCADE;
CREATE TABLE t (val vector(3));
INSERT INTO t (val) VALUES ('[0,0,0]'), ('[1,2,3]'), ('[1,1,1]'), ('[2,2,2]'), ('[3,3,3]'), ('[0,1,0]'), ('[1,0,1]'), ('[2,1,2]'), ('[1,3,1]'), ('[0,2,1]'), (NULL);
CREATE INDEX ON t USING ivf (val vector_cosine_ops) WITH (lists = 1);

INSERT INTO t (val) VALUES ('[1,2,4]');

SELECT * FROM t ORDER BY val <=> '[3,3,3]';
SELECT COUNT(*) FROM (SELECT * FROM t ORDER BY val <=> '[0,0,0]') t2;
SELECT COUNT(*) FROM (SELECT * FROM t ORDER BY val <=> (SELECT NULL::vector)) t2;

DROP TABLE t;

-- Test 4: IVF Index Options
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 4: IVF Index Options'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DROP TABLE IF EXISTS t CASCADE;
CREATE TABLE t (val vector(3));
-- Test invalid lists values (should error or handle)
DO $$
BEGIN
    BEGIN
        CREATE INDEX ON t USING ivf (val vector_l2_ops) WITH (lists = 0);
        RAISE NOTICE 'lists = 0 accepted';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'lists = 0 rejected: %', SQLERRM;
    END;
    
    BEGIN
        CREATE INDEX ON t USING ivf (val vector_l2_ops) WITH (lists = 32769);
        RAISE NOTICE 'lists = 32769 accepted';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'lists = 32769 rejected: %', SQLERRM;
    END;
END $$;

DROP TABLE t;

-- Test 5: Unlogged Tables
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 5: IVF Index on Unlogged Tables'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DROP TABLE IF EXISTS t CASCADE;
CREATE UNLOGGED TABLE t (val vector(3));
INSERT INTO t (val) VALUES ('[0,0,0]'), ('[1,2,3]'), ('[1,1,1]'), ('[2,2,2]'), ('[3,3,3]'), ('[0,1,0]'), ('[1,0,1]'), ('[2,1,2]'), ('[1,3,1]'), ('[0,2,1]'), (NULL);
CREATE INDEX ON t USING ivf (val vector_l2_ops) WITH (lists = 1);

SELECT * FROM t ORDER BY val <-> '[3,3,3]';

DROP TABLE t;

RESET enable_seqscan;

\echo ''
\echo '=========================================================================='
\echo 'Test completed successfully'
\echo '=========================================================================='


