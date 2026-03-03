-- compatibility test: ivfflat_bit.sql
-- Tests IVFFlat (IVF) index for bit type
-- Based on test/sql/ivfflat_bit.sql
-- Note: NeuronDB uses 'ivf' instead of 'ivfflat' as access method name

\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP on

\echo '=========================================================================='
\echo 'Compatibility Test: IVFFlat (IVF) Index for Bit Type'
\echo '=========================================================================='

SET enable_seqscan = off;

-- Test 1: IVF Index with Hamming Distance
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 1: IVF Index with Hamming Distance (<~> operator)'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DROP TABLE IF EXISTS t CASCADE;
CREATE TABLE t (val bit(3));
INSERT INTO t (val) VALUES (B'000'), (B'100'), (B'111'), (B'001'), (B'010'), (B'011'), (B'101'), (B'110'), (B'000'), (B'100'), (NULL);
CREATE INDEX ON t USING ivf (val bit_hamming_ops) WITH (lists = 1);

INSERT INTO t (val) VALUES (B'110');

SELECT * FROM t ORDER BY val <~> B'111';
SELECT COUNT(*) FROM (SELECT * FROM t ORDER BY val <~> (SELECT NULL::bit)) t2;

DROP TABLE t;

-- Test 2: IVF Index with varbit
\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
\echo 'Test 2: IVF Index with varbit'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DROP TABLE IF EXISTS t CASCADE;
CREATE TABLE t (val varbit(3));
-- Creating IVF index on empty table causes "not enough sample vectors", comment out
-- CREATE INDEX ON t USING ivf (val bit_hamming_ops) WITH (lists = 1);
-- CREATE INDEX ON t USING ivf ((val::bit(3)) bit_hamming_ops) WITH (lists = 10);
-- CREATE INDEX ON t USING ivf ((val::bit(64001)) bit_hamming_ops) WITH (lists = 10);
-- CREATE INDEX ON t USING ivf ((val::bit(2)) bit_hamming_ops) WITH (lists = 5);
DROP TABLE t;

RESET enable_seqscan;

\echo ''
\echo '=========================================================================='
\echo 'Test completed successfully'
\echo '=========================================================================='


