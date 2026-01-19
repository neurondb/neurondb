-- Test all possible sync scenarios: valid, invalid, and missing arguments

-- 1. Valid sync
SELECT sync_index_async('test_index', 'replica_host') AS sync_initiated_valid;

-- 2. Invalid index
SELECT sync_index_async('nonexistent_index', 'replica_host') AS sync_initiated_invalid_index;

-- 3. Invalid host
SELECT sync_index_async('test_index', 'nonexistent_host') AS sync_initiated_invalid_host;

-- 4. Both arguments invalid
SELECT sync_index_async('nonexistent_index', 'nonexistent_host') AS sync_initiated_both_invalid;

-- 5. Null index
SELECT sync_index_async(NULL, 'replica_host') AS sync_initiated_null_index;

-- 6. Null host
SELECT sync_index_async('test_index', NULL) AS sync_initiated_null_host;

-- 7. Both arguments null
SELECT sync_index_async(NULL, NULL) AS sync_initiated_both_null;

-- Test array conversion roundtrip with varied input arrays

-- 1. Standard float array
SELECT vector_to_array(array_to_vector(ARRAY[1.0, 2.0, 3.0]::real[])) AS roundtrip_standard;

-- 2. Integers cast to real
SELECT vector_to_array(array_to_vector(ARRAY[1, 2, 3]::real[])) AS roundtrip_integers_cast_to_real;

-- 3. Single element array
SELECT vector_to_array(array_to_vector(ARRAY[42.0]::real[])) AS roundtrip_single_element;

-- 4. Empty array
SELECT vector_to_array(array_to_vector(ARRAY[]::real[])) AS roundtrip_empty_array;

-- 5. Null in array
SELECT vector_to_array(array_to_vector(ARRAY[1.0, NULL, 3.0]::real[])) AS roundtrip_with_null;

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
