-- Scan for all vector-related functions possibly used for GPU—cast the net wider to see if more exist

SELECT 
    p.oid AS function_oid,
    p.proname AS function_name,
    pg_catalog.pg_get_function_identity_arguments(p.oid) AS argument_types,
    pg_catalog.pg_get_function_result(p.oid) AS return_type,
    n.nspname AS schema_name,
    r.rolname AS owner,
    p.prosrc AS function_source
FROM pg_proc p
JOIN pg_namespace n ON n.oid = p.pronamespace
JOIN pg_roles r ON r.oid = p.proowner
WHERE 
    (
        p.proname ILIKE '%vector%' -- any function containing 'vector'
        OR p.proname ILIKE '%gpu%' -- or any function containing 'gpu'
        OR p.proname ILIKE 'neurondb_gpu%' -- extension-specific GPU fn
    )
    AND n.nspname ~* '(neurondb|public|pg_temp|ext)' -- likely extension or public/test schema
ORDER BY
    n.nspname, p.proname;
