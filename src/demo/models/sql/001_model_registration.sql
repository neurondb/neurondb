-- ============================================================================
-- NeuronDB Model Catalog Demo - Registration & Metadata
-- ============================================================================
\set ON_ERROR_STOP on
\echo '=========================================='
\echo 'STEP 1: Model Registration & Catalog Metadata'
\echo '=========================================='

-- Prepare demo workspace on filesystem
\echo ''
\echo 'Creating demo model artifacts under /tmp/neurondb_models_demo'
\! python3 - <<'PY'
import pathlib
base = pathlib.Path('/tmp/neurondb_models_demo')
base.mkdir(parents=True, exist_ok=True)
(base / 'sentiment-mini.onnx').write_bytes(b'SENTIMENT\n' * 16)
(base / 'question-answering.onnx').write_bytes(b'QA\n' * 32)
PY

SET search_path TO neurondb, public;

-- Clean previous demo data (idempotent reruns)
DELETE FROM neurondb.model_events WHERE version_id IN (
    SELECT version_id FROM neurondb.model_versions WHERE model_id IN (
        SELECT model_id FROM neurondb.models WHERE model_name LIKE 'demo_%'
    )
);
DELETE FROM neurondb.model_versions WHERE model_id IN (
    SELECT model_id FROM neurondb.models WHERE model_name LIKE 'demo_%'
);
DELETE FROM neurondb.models WHERE model_name LIKE 'demo_%';

\echo ''
\echo 'Registering demo models via ensure_model and register_model_version'
SELECT neurondb.ensure_model('demo_sentiment', 'public', 'huggingface', 'text-classification', 'onnx', '/tmp/neurondb_models_demo/sentiment-mini.onnx', current_user) AS sentiment_model_id;
SELECT neurondb.register_model_version('demo_sentiment', 'public', 'v1', '/tmp/neurondb_models_demo/sentiment-mini.onnx', 'onnx', 1280, NULL, current_user, '{"dataset": "imdb", "labels": 2}'::jsonb) AS sentiment_version_id;

SELECT neurondb.ensure_model('demo_qa', 'public', 'huggingface', 'question-answering', 'onnx', '/tmp/neurondb_models_demo/question-answering.onnx', current_user) AS qa_model_id;
SELECT neurondb.register_model_version('demo_qa', 'public', NULL, '/tmp/neurondb_models_demo/question-answering.onnx', 'onnx', 2048, NULL, current_user, '{"dataset": "squad"}'::jsonb) AS qa_version_id;

\echo ''
\echo 'Current rows in neurondb.models'
SELECT model_id, model_name, tenant_id, provider, task, default_format, default_path, created_by, created_at
FROM neurondb.models
WHERE model_name LIKE 'demo_%'
ORDER BY model_name;

\echo ''
\echo 'Current rows in neurondb.model_versions'
SELECT version_id, model_id, version_label, storage_uri, format, size_bytes, status, created_at, metadata
FROM neurondb.model_versions
WHERE model_id IN (
    SELECT model_id FROM neurondb.models WHERE model_name LIKE 'demo_%'
)
ORDER BY version_id;

\echo ''
\echo 'Latest entries from neurondb.model_catalog view'
SELECT model_name, version_label, storage_uri, format, status, metadata
FROM neurondb.model_catalog
WHERE model_name LIKE 'demo_%'
ORDER BY model_name;

\echo ''
\echo 'STEP 1 complete'
\echo '=========================================='
