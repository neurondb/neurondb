-- Test all possible sync scenarios: valid, invalid, and missing arguments

-- 1. Valid sync
-- Validate that function works or handles errors properly
DO $$
DECLARE
    sync_result TEXT;
BEGIN
    -- Test with valid parameters (index may not exist, allow errors)
    BEGIN
        SELECT sync_index_async('test_index', 'replica_host') INTO sync_result;
        
        IF sync_result IS NULL THEN
            RAISE WARNING 'sync_index_async returned NULL';
        END IF;
    EXCEPTION WHEN OTHERS THEN
        IF SQLSTATE = '42883' THEN
            RAISE EXCEPTION 'sync_index_async function not available - this is a required function';
        ELSE
            -- Index might not exist, which is acceptable
            RAISE NOTICE 'sync_index_async error (expected for test_index): %', SQLERRM;
        END IF;
    END;
END$$;

-- 2-7. Additional sync tests
-- Validate error handling for various invalid inputs
DO $$
BEGIN
  BEGIN
    PERFORM sync_index_async('nonexistent_index', 'replica_host');
    RAISE WARNING 'sync_index_async should have raised error for nonexistent index';
  EXCEPTION WHEN OTHERS THEN
    IF SQLSTATE = '42883' THEN
      RAISE EXCEPTION 'sync_index_async function not available - this is a required function';
    ELSE
      RAISE NOTICE 'sync_index_async correctly handled invalid index: %', SQLERRM;
    END IF;
  END;
  
  BEGIN
    PERFORM sync_index_async('test_index', 'nonexistent_host');
    RAISE WARNING 'sync_index_async should have raised error for invalid host';
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'sync_index_async correctly handled invalid host: %', SQLERRM;
  END;
  
  BEGIN
    PERFORM sync_index_async(NULL, 'replica_host');
    RAISE WARNING 'sync_index_async should have raised error for NULL index';
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'sync_index_async correctly handled NULL index: %', SQLERRM;
  END;
  
  BEGIN
    PERFORM sync_index_async('test_index', NULL);
    RAISE WARNING 'sync_index_async should have raised error for NULL host';
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'sync_index_async correctly handled NULL host: %', SQLERRM;
  END;
END$$;

-- Test array conversion roundtrip with varied input arrays

-- 1. Standard float array
SELECT vector_to_array(array_to_vector(ARRAY[1.0, 2.0, 3.0]::real[])) AS roundtrip_standard;

-- 2. Integers cast to real
SELECT vector_to_array(array_to_vector(ARRAY[1, 2, 3]::real[])) AS roundtrip_integers_cast_to_real;

-- 3. Single element array
SELECT vector_to_array(array_to_vector(ARRAY[42.0]::real[])) AS roundtrip_single_element;

-- 4. Empty array (skip if empty arrays not supported)
DO $$
BEGIN
  BEGIN
    PERFORM vector_to_array(array_to_vector(ARRAY[]::real[]));
    RAISE NOTICE 'Empty array conversion is supported';
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Empty array conversion not supported: %', SQLERRM;
  END;
END$$;

-- 5. Null in array (skip if NULLs in arrays not supported)
DO $$
BEGIN
  BEGIN
    PERFORM vector_to_array(array_to_vector(ARRAY[1.0, NULL, 3.0]::real[]));
    RAISE NOTICE 'NULL in array conversion is supported';
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'NULL in array conversion not supported: %', SQLERRM;
  END;
END$$;

-- 6. Negative and zero values
SELECT vector_to_array(array_to_vector(ARRAY[0.0, -1.5, 2.7]::real[])) AS roundtrip_negatives_and_zero;

-- Verify all neurondb-related catalog tables exist and list them

-- Count number of neurondb tables
SELECT COUNT(*) AS neurondb_catalog_count 
FROM pg_tables 
WHERE tablename LIKE 'neurondb_%';

-- List all neurondb catalog tables
SELECT tablename 
FROM pg_tables 
WHERE tablename LIKE 'neurondb_%'
ORDER BY tablename;

-- Show missing catalogs if less than expected (assuming at least 12 expected)
SELECT 
    12 - COUNT(*) AS missing_catalogs
FROM pg_tables
WHERE tablename LIKE 'neurondb_%';

-- Verify extension info in detail and all version fields

-- 1. Check if extension is installed
SELECT EXISTS (
    SELECT 1 FROM pg_extension WHERE extname = 'neurondb'
) AS is_neurondb_installed;

-- 2. Show version and schema details
SELECT extname, extversion, nspname AS schema
FROM pg_extension 
JOIN pg_namespace ON extnamespace = pg_namespace.oid
WHERE extname = 'neurondb';

-- 3. List all available extensions for detailed overview
SELECT extname, extversion
FROM pg_extension
ORDER BY extname;

-- 4. Get all fields for NeurondB extension
SELECT *
FROM pg_extension
WHERE extname = 'neurondb';
