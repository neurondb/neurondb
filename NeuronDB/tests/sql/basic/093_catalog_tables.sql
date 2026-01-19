-- =============================
-- DETAILED and EXHAUSTIVE TESTS FOR ALL CATALOG TABLES
-- =============================

-- 1. List all NeurondB catalog and related tables, with schema for completeness
SELECT schemaname, tablename
FROM pg_tables
WHERE tablename LIKE 'neurondb_%'
ORDER BY schemaname, tablename;

-- 2. Check table existence using information_schema for all relevant catalog tables
SELECT table_schema, table_name
FROM information_schema.tables
WHERE table_name IN ('neurondb_job_queue', 'neurondb_query_metrics', 'neurondb_embedding_cache')
ORDER BY table_schema, table_name;

-- 3. DDL: Show columns and datatypes for each catalog table
SELECT table_name, column_name, data_type
FROM information_schema.columns
WHERE table_name LIKE 'neurondb_%'
ORDER BY table_name, ordinal_position;

-- 4. Insert detailed, exhaustive combinations into job queue table
-- 4a. Normal job
INSERT INTO neurondb_job_queue (job_type, payload, tenant_id) 
VALUES ('test_job', '{"foo": 1}'::jsonb, 1);

-- 4b. Edge: Empty payload
INSERT INTO neurondb_job_queue (job_type, payload, tenant_id) 
VALUES ('empty_payload', '{}'::jsonb, 42);

-- 4c. Edge: Null payload
INSERT INTO neurondb_job_queue (job_type, payload, tenant_id) 
VALUES ('null_payload', NULL, 99);

-- 4d. Multiple tenants
INSERT INTO neurondb_job_queue (job_type, payload, tenant_id) 
VALUES
  ('multi_tenant', '{"x": 2}'::jsonb, 2),
  ('multi_tenant', '{"y": 3}'::jsonb, 3);

-- 4e. Special characters in job_type
INSERT INTO neurondb_job_queue (job_type, payload, tenant_id) 
VALUES ('specialchars!@#$%^&*', '{"test":true}'::jsonb, 7);

-- 4f. Duplicate job, if allowed
INSERT INTO neurondb_job_queue (job_type, payload, tenant_id) 
VALUES ('test_job', '{"foo": 1}'::jsonb, 1);

-- 4g. All relevant columns with explicit status if the status column is updatable
--     (If not possible, this block can be commented out.)

-- 5. Select full contents and all columns of job queue
SELECT * FROM neurondb_job_queue ORDER BY id;

-- 6. Update: change status of a job (if status column is present and updatable)
UPDATE neurondb_job_queue
   SET status = 'complete'
 WHERE job_type = 'test_job' AND tenant_id = 1;

-- 7. Delete: remove a job (for testing deletions)
DELETE FROM neurondb_job_queue WHERE job_type = 'null_payload';

-- 8. Test query metrics table
-- 8a. Standard row
INSERT INTO neurondb_query_metrics (query_type, latency_ms, recall_at_k, ef_search)
VALUES ('knn_search', 25.5, 0.95, 64);

-- 8b. Edge cases: 0 and NULL values
INSERT INTO neurondb_query_metrics (query_type, latency_ms, recall_at_k, ef_search)
VALUES
  ('brute_force', 0, 0, 0),
  ('null_metrics', NULL, NULL, NULL);

-- 8c. Floating point edge cases
INSERT INTO neurondb_query_metrics (query_type, latency_ms, recall_at_k, ef_search)
VALUES ('float_edge', 1e-5, 1.0, 99999);

-- 9. Select all columns, all rows from query metrics
SELECT * FROM neurondb_query_metrics ORDER BY id;

-- 10. Test embedding cache table
-- 10a. Standard embedding
INSERT INTO neurondb_embedding_cache (cache_key, embedding, model_name)
VALUES ('test_key', '[1.0, 2.0, 3.0]'::vector, 'test_model');

-- 10b. Multiple cache keys and models
INSERT INTO neurondb_embedding_cache (cache_key, embedding, model_name)
VALUES
  ('other_key', '[4.0, 5.0, 6.0]'::vector, 'other_model'),
  ('dup_key', '[1.0, 1.0, 1.0]'::vector, 'dup_model');

-- 10c. Edge: NULL for optional columns if allowed
INSERT INTO neurondb_embedding_cache (cache_key, embedding, model_name)
VALUES ('null_embedding', NULL, NULL);

-- 11. Select all contents of embedding cache
SELECT * FROM neurondb_embedding_cache ORDER BY cache_key;

-- 12. Count all neurondb catalog tables
SELECT COUNT(*) AS catalog_table_count
FROM pg_tables
WHERE tablename LIKE 'neurondb_%';

-- 13. Edge: Try inserting with missing required field (should error)
DO $$
BEGIN
  BEGIN
    INSERT INTO neurondb_embedding_cache (embedding, model_name) VALUES ('[7.0, 8.0, 9.0]'::vector, 'missing_key');
    RAISE WARNING 'ERROR: insert succeeded even though cache_key is missing!';
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Correctly enforced NOT NULL on cache_key: %', SQLERRM;
  END;
END$$;

-- 14. Attempt to select from a non-existent catalog table (should raise error)
DO $$
BEGIN
  BEGIN
    PERFORM * FROM neurondb_nonexistent_table;
    RAISE WARNING 'ERROR: selection from nonexistent table succeeded unexpectedly';
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Correctly rejected query on nonexistent table: %', SQLERRM;
  END;
END$$;

-- 15. Attempt to insert duplicate primary key in embedding_cache (if pk exists)
DO $$
BEGIN
  BEGIN
    -- Try to insert same cache_key twice; should error if primary key exists
    INSERT INTO neurondb_embedding_cache (cache_key, embedding, model_name)
    VALUES ('test_key', '[9.9, 9.9, 9.9]'::vector, 'some_model');
    RAISE WARNING 'ERROR: duplicate cache_key allowed!';
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Correctly rejected duplicate key: %', SQLERRM;
  END;
END$$;

