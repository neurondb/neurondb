-- ============================================================================
-- NeuronDB Model Catalog Demo - Audit & Reporting
-- ============================================================================
\set ON_ERROR_STOP on
\echo '=========================================='
\echo 'STEP 3: Audit & Reporting Checks'
\echo '=========================================='

SET search_path TO neurondb, public;

\echo ''
\echo 'Model status distribution'
SELECT status, COUNT(*) AS versions
FROM neurondb.model_versions
WHERE model_id IN (
    SELECT model_id FROM neurondb.models WHERE model_name LIKE 'demo_%'
)
GROUP BY status
ORDER BY status;

\echo ''
\echo 'Latest model catalog snapshot'
SELECT model_name, tenant_id, provider, task, version_label, status, loaded_at, exported_at
FROM neurondb.model_catalog
WHERE model_name LIKE 'demo_%'
ORDER BY model_name;

\echo ''
\echo 'Recent model events (limit 10)'
SELECT m.model_name,
       e.event_type,
       e.event_at,
       COALESCE(e.event_details::text, '') AS details
FROM neurondb.model_events e
JOIN neurondb.model_versions v ON v.version_id = e.version_id
JOIN neurondb.models m ON m.model_id = v.model_id
WHERE m.model_name LIKE 'demo_%'
ORDER BY e.event_at DESC
LIMIT 10;

\echo ''
\echo 'Event counts by type'
SELECT event_type, COUNT(*) AS occurrences
FROM neurondb.model_events e
JOIN neurondb.model_versions v ON v.version_id = e.version_id
JOIN neurondb.models m ON m.model_id = v.model_id
WHERE m.model_name LIKE 'demo_%'
GROUP BY event_type
ORDER BY event_type;

\echo ''
\echo 'Ensure every model has at least one version and event'
SELECT m.model_name,
       COUNT(DISTINCT v.version_id) AS version_count,
       COUNT(DISTINCT e.event_id) AS event_count
FROM neurondb.models m
LEFT JOIN neurondb.model_versions v ON v.model_id = m.model_id
LEFT JOIN neurondb.model_events e ON e.version_id = v.version_id
WHERE m.model_name LIKE 'demo_%'
GROUP BY m.model_name
ORDER BY m.model_name;

\echo ''
\echo 'STEP 3 complete'
\echo '=========================================='
