-- compatibility test: hnsw_halfvec.sql
-- Tests HNSW index for halfvec type
-- Based on test/sql/hnsw_halfvec.sql

\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Compatibility Test: HNSW Index for Halfvec Type'
\echo '=========================================================================='

SET enable_seqscan = off;

-- Test 1: HNSW Index with L2 Distance
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 1: HNSW Index with L2 Distance (<-> operator)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DROP TABLE IF EXISTS t CASCADE;
-- Note: Using standalone test data to avoid dimension conflicts with dataset
CREATE TABLE t (val halfvec);
INSERT INTO t (val) VALUES ('[0,0,0]'::halfvec), ('[1,2,3]'::halfvec), ('[1,1,1]'::halfvec), (NULL);
CREATE INDEX ON t USING hnsw (val halfvec_l2_ops);

INSERT INTO t (val) VALUES ('[1,2,4]'::halfvec);

SELECT * FROM t WHERE val IS NOT NULL ORDER BY val <-> '[3,3,3]'::halfvec;
SELECT COUNT(*) FROM (SELECT * FROM t ORDER BY val <-> (SELECT NULL::halfvec)) t2;
SELECT COUNT(*) FROM t;

TRUNCATE t;
SELECT * FROM t ORDER BY val <-> '[3,3,3]';

DROP TABLE t;

-- Test 2: HNSW Index with Inner Product
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 2: HNSW Index with Inner Product (<#> operator)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DROP TABLE IF EXISTS t CASCADE;
CREATE TABLE t (val halfvec);
INSERT INTO t (val) VALUES ('[0,0,0]'), ('[1,2,3]'), ('[1,1,1]'), (NULL);
CREATE INDEX ON t USING hnsw (val halfvec_ip_ops);

INSERT INTO t (val) VALUES ('[1,2,4]');

SELECT * FROM t ORDER BY val <#> '[3,3,3]';
SELECT COUNT(*) FROM (SELECT * FROM t ORDER BY val <#> (SELECT NULL::halfvec)) t2;

DROP TABLE t;

-- Test 3: HNSW Index with Cosine Distance
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 3: HNSW Index with Cosine Distance (<=> operator)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DROP TABLE IF EXISTS t CASCADE;
CREATE TABLE t (val halfvec);
INSERT INTO t (val) VALUES ('[0,0,0]'), ('[1,2,3]'), ('[1,1,1]'), (NULL);
CREATE INDEX ON t USING hnsw (val halfvec_cosine_ops);

INSERT INTO t (val) VALUES ('[1,2,4]');

SELECT * FROM t ORDER BY val <=> '[3,3,3]';
SELECT COUNT(*) FROM (SELECT * FROM t ORDER BY val <=> '[0,0,0]') t2;
SELECT COUNT(*) FROM (SELECT * FROM t ORDER BY val <=> (SELECT NULL::halfvec)) t2;

DROP TABLE t;

-- Test 4: HNSW Index with L1 Distance
-- \echo ''
-- \echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
-- \echo 'Test 4: HNSW Index with L1 Distance (<+> operator)'
-- \echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
--
-- DROP TABLE IF EXISTS t CASCADE;
-- CREATE TABLE t (val halfvec);
-- INSERT INTO t (val) VALUES ('[0,0,0]'), ('[1,2,3]'), ('[1,1,1]'), (NULL);
-- CREATE INDEX ON t USING hnsw (val halfvec_l1_ops);
--
-- INSERT INTO t (val) VALUES ('[1,2,4]');
--
-- SELECT * FROM t ORDER BY val <+> '[3,3,3]';
-- SELECT COUNT(*) FROM (SELECT * FROM t ORDER BY val <+> (SELECT NULL::halfvec)) t2;
--
-- DROP TABLE t;

RESET enable_seqscan;

\echo ''
\echo '=========================================================================='
\echo 'Test completed successfully'
\echo '=========================================================================='


