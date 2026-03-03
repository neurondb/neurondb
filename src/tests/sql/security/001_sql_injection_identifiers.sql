-- SQL injection resistance tests for ML API (identifiers)
-- Verifies that table_name, column names passed to neurondb.train and similar
-- APIs are safely quoted so that malicious strings cannot execute as SQL.
-- Run with \set ON_ERROR_STOP on

\set ON_ERROR_STOP on

-- Ensure extension and schema exist
CREATE EXTENSION IF NOT EXISTS neurondb CASCADE;
SET search_path TO neurondb, public;

-- Victim table: must still exist after attempts to inject "DROP TABLE victim"
CREATE TABLE IF NOT EXISTS victim (id int, val float);
TRUNCATE victim;
INSERT INTO victim (id, val) VALUES (1, 1.0), (2, 2.0);

-- Training table with safe names and minimal data for regression
CREATE TABLE IF NOT EXISTS train_data_safe (f1 float, f2 float, target float);
TRUNCATE train_data_safe;
INSERT INTO train_data_safe (f1, f2, target) VALUES (1.0, 2.0, 3.0), (2.0, 3.0, 5.0);

-- Test 1: table_name containing SQL fragment must be treated as identifier, not executed
-- A correctly quoted identifier will look for a relation named literally "x; DROP TABLE victim; --"
-- which does not exist, so we expect an error. Victim must still exist.
DO $$
BEGIN
  BEGIN
    PERFORM neurondb.train(
      'proj',
      'linear_regression',
      'x; DROP TABLE victim; --',
      'target',
      ARRAY['f1','f2'],
      '{}'::jsonb
    );
    RAISE NOTICE 'train with malicious table_name did not error (table may not exist, which is ok)';
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'train with malicious table_name raised: %', SQLERRM;
  END;
  IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'victim') THEN
    RAISE EXCEPTION 'SQL injection: victim table was dropped';
  END IF;
END $$;

-- Test 2: again with different payload
DO $$
BEGIN
  BEGIN
    PERFORM neurondb.train(
      'proj2',
      'linear_regression',
      'train_data_safe',
      'target',
      ARRAY['f1; DELETE FROM victim; --'],
      '{}'::jsonb
    );
    RAISE NOTICE 'train with malicious column name did not error';
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'train with malicious column raised: %', SQLERRM;
  END;
  IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'victim') THEN
    RAISE EXCEPTION 'SQL injection: victim table was dropped';
  END IF;
  IF (SELECT count(*) FROM victim) <> 2 THEN
    RAISE EXCEPTION 'SQL injection: victim rows were deleted';
  END IF;
END $$;

-- Test 3: quote_identifier behavior check via simple query (sanity)
-- Creating a table with a weird but legal name and selecting from it
CREATE TABLE IF NOT EXISTS "tab;with;semicolons" (a int);
TRUNCATE "tab;with;semicolons";
INSERT INTO "tab;with;semicolons" (a) VALUES (1);
DO $$
DECLARE r int;
BEGIN
  SELECT a INTO r FROM "tab;with;semicolons" LIMIT 1;
  IF r <> 1 THEN RAISE EXCEPTION 'expected 1'; END IF;
END $$;
DROP TABLE IF EXISTS "tab;with;semicolons";

-- Victim still present
SELECT 1 FROM victim LIMIT 1;

-- Cleanup
DROP TABLE IF EXISTS victim;
DROP TABLE IF EXISTS train_data_safe;
