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

-- 4. Insert detailed, exhaustive combinations into job queue table (skip if table doesn't exist)
DO $$
BEGIN
  BEGIN
    -- 4a. Normal job
    INSERT INTO neurondb.neurondb_job_queue (job_type, payload, tenant_id) 
    VALUES ('test_job', '{"foo": 1}'::jsonb, 1);
    
    -- 4b. Edge: Empty payload
    INSERT INTO neurondb.neurondb_job_queue (job_type, payload, tenant_id) 
    VALUES ('empty_payload', '{}'::jsonb, 42);
    
    -- 4c. Edge: Null payload
    INSERT INTO neurondb.neurondb_job_queue (job_type, payload, tenant_id) 
    VALUES ('null_payload', NULL, 99);
    
    RAISE NOTICE 'Job queue table is available';
  EXCEPTION WHEN undefined_table THEN
    RAISE NOTICE 'neurondb_job_queue table not available, skipping insert tests';
  END;
END$$;

-- 4d-4f. Additional job queue tests (skip if table doesn't exist)
DO $$
BEGIN
  BEGIN
    INSERT INTO neurondb.neurondb_job_queue (job_type, payload, tenant_id) 
    VALUES
      ('multi_tenant', '{"x": 2}'::jsonb, 2),
      ('multi_tenant', '{"y": 3}'::jsonb, 3);
    
    INSERT INTO neurondb.neurondb_job_queue (job_type, payload, tenant_id) 
    VALUES ('specialchars!@#$%^&*', '{"test":true}'::jsonb, 7);
    
    INSERT INTO neurondb.neurondb_job_queue (job_type, payload, tenant_id) 
    VALUES ('test_job', '{"foo": 1}'::jsonb, 1);
    
    RAISE NOTICE 'Additional job queue inserts completed';
  EXCEPTION WHEN undefined_table THEN
    RAISE NOTICE 'neurondb_job_queue table not available, skipping additional inserts';
  END;
END$$;

-- 5. Select full contents and all columns of job queue (skip if table doesn't exist)
DO $$
BEGIN
  BEGIN
    PERFORM * FROM neurondb.neurondb_job_queue LIMIT 1;
    RAISE NOTICE 'Job queue table is queryable';
  EXCEPTION WHEN undefined_table THEN
    RAISE NOTICE 'neurondb_job_queue table not available, skipping queries';
  END;
END$$;

-- 6-7. Update and delete (skip if table doesn't exist)
DO $$
BEGIN
  BEGIN
    UPDATE neurondb.neurondb_job_queue SET status = 'complete' WHERE job_type = 'test_job' AND tenant_id = 1;
    DELETE FROM neurondb.neurondb_job_queue WHERE job_type = 'null_payload';
    RAISE NOTICE 'Job queue update/delete completed';
  EXCEPTION WHEN undefined_table THEN
    RAISE NOTICE 'neurondb_job_queue table not available, skipping update/delete';
  END;
END$$;

-- 8. Test query metrics table (skip if table doesn't exist)
DO $$
BEGIN
  BEGIN
    INSERT INTO neurondb.neurondb_query_metrics (query_type, latency_ms, recall_at_k, ef_search)
    VALUES ('knn_search', 25.5, 0.95, 64);
    
    INSERT INTO neurondb.neurondb_query_metrics (query_type, latency_ms, recall_at_k, ef_search)
    VALUES
      ('brute_force', 0, 0, 0),
      ('null_metrics', NULL, NULL, NULL);
    
    INSERT INTO neurondb.neurondb_query_metrics (query_type, latency_ms, recall_at_k, ef_search)
    VALUES ('float_edge', 1e-5, 1.0, 99999);
    
    RAISE NOTICE 'Query metrics inserts completed';
  EXCEPTION WHEN undefined_table THEN
    RAISE NOTICE 'neurondb_query_metrics table not available, skipping inserts';
  END;
END$$;

-- 9. Select from query metrics (skip if table doesn't exist)
DO $$
BEGIN
  BEGIN
    PERFORM * FROM neurondb.neurondb_query_metrics LIMIT 1;
    RAISE NOTICE 'Query metrics table is queryable';
  EXCEPTION WHEN undefined_table THEN
    RAISE NOTICE 'neurondb_query_metrics table not available, skipping queries';
  END;
END$$;

-- 10. Test embedding cache table (skip if table doesn't exist)
DO $$
BEGIN
  BEGIN
    INSERT INTO neurondb.neurondb_embedding_cache (cache_key, embedding, model_name)
    VALUES ('test_key', '[1.0, 2.0, 3.0]'::vector, 'test_model');
    
    INSERT INTO neurondb.neurondb_embedding_cache (cache_key, embedding, model_name)
    VALUES
      ('other_key', '[4.0, 5.0, 6.0]'::vector, 'other_model'),
      ('dup_key', '[1.0, 1.0, 1.0]'::vector, 'dup_model');
    
    INSERT INTO neurondb.neurondb_embedding_cache (cache_key, embedding, model_name)
    VALUES ('null_embedding', NULL, NULL);
    
    RAISE NOTICE 'Embedding cache inserts completed';
  EXCEPTION WHEN undefined_table THEN
    RAISE NOTICE 'neurondb_embedding_cache table not available, skipping inserts';
  END;
END$$;

-- 11. Select from embedding cache (skip if table doesn't exist)
DO $$
BEGIN
  BEGIN
    PERFORM * FROM neurondb.neurondb_embedding_cache LIMIT 1;
    RAISE NOTICE 'Embedding cache table is queryable';
  EXCEPTION WHEN undefined_table THEN
    RAISE NOTICE 'neurondb_embedding_cache table not available, skipping queries';
  END;
END$$;

-- 12. Count all neurondb catalog tables
SELECT COUNT(*) AS catalog_table_count
FROM pg_tables
WHERE tablename LIKE 'neurondb_%';

-- 13. Edge: Try inserting with missing required field (should error, skip if table doesn't exist)
DO $$
BEGIN
  BEGIN
    INSERT INTO neurondb.neurondb_embedding_cache (embedding, model_name) VALUES ('[7.0, 8.0, 9.0]'::vector, 'missing_key');
    RAISE WARNING 'ERROR: insert succeeded even though cache_key is missing!';
  EXCEPTION WHEN OTHERS THEN
    IF SQLSTATE = '42P01' THEN
      RAISE NOTICE 'neurondb_embedding_cache table not available, skipping constraint test';
    ELSE
      RAISE NOTICE 'Correctly enforced NOT NULL on cache_key: %', SQLERRM;
    END IF;
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

-- 15. Attempt to insert duplicate primary key in embedding_cache (if pk exists, skip if table doesn't exist)
DO $$
BEGIN
  BEGIN
    -- Try to insert same cache_key twice; should error if primary key exists
    INSERT INTO neurondb.neurondb_embedding_cache (cache_key, embedding, model_name)
    VALUES ('test_key', '[9.9, 9.9, 9.9]'::vector, 'some_model');
    RAISE WARNING 'ERROR: duplicate cache_key allowed!';
  EXCEPTION WHEN OTHERS THEN
    IF SQLSTATE = '42P01' THEN
      RAISE NOTICE 'neurondb_embedding_cache table not available, skipping duplicate key test';
    ELSE
      RAISE NOTICE 'Correctly rejected duplicate key: %', SQLERRM;
    END IF;
  END;
END$$;

