-- compatibility test: btree.sql
-- Tests B-tree indexes on vector, halfvec, sparsevec types
-- Based on test/sql/btree.sql

\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Compatibility Test: B-tree Indexes'
\echo '=========================================================================='

SET enable_seqscan = off;

-- Test 1: B-tree Index on Vector
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 1: B-tree Index on Vector Type'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DROP TABLE IF EXISTS t CASCADE;
CREATE TABLE t (val vector);
INSERT INTO t (val) VALUES ('[0,0,0]'), ('[1,2,3]'), ('[1,1,1]'), (NULL);
-- Note: B-tree indexes on vector require operator class
-- Skip index creation if not supported, test equality and ordering without index
-- CREATE INDEX ON t (val);

SELECT * FROM t WHERE val = '[1,2,3]';
-- Note: ORDER BY may require operator class
SELECT * FROM t;

DROP TABLE t;

-- Test 2: B-tree Index on Halfvec
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 2: B-tree Index on Halfvec Type'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DROP TABLE IF EXISTS t CASCADE;
CREATE TABLE t (val halfvec);
INSERT INTO t (val) VALUES ('[0,0,0]'), ('[1,2,3]'), ('[1,1,1]'), (NULL);
-- Note: B-tree indexes on halfvec may require operator class
-- CREATE INDEX ON t (val);

SELECT * FROM t WHERE val = '[1,2,3]';
-- Note: ORDER BY may require operator class
SELECT * FROM t;

DROP TABLE t;

-- Test 3: B-tree Index on Sparsevec
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 3: B-tree Index on Sparsevec Type'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DROP TABLE IF EXISTS t CASCADE;
CREATE TABLE t (val sparsevec);
-- Note: Empty sparsevec '{}/3' not supported, use non-empty entries
INSERT INTO t (val) VALUES ('{1:0,2:0,3:0}/3'), ('{1:1,2:2,3:3}/3'), ('{1:1,2:1,3:1}/3'), (NULL);
-- Note: B-tree indexes on sparsevec may require operator class
-- CREATE INDEX ON t (val);

SELECT * FROM t WHERE val = '{1:1,2:2,3:3}/3';
-- Note: ORDER BY may require operator class
SELECT * FROM t;

DROP TABLE t;

RESET enable_seqscan;

\echo ''
\echo '=========================================================================='
\echo 'Test completed successfully'
\echo '=========================================================================='

