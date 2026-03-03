-- ============================================================================
-- NeuronDB Model Catalog Demo - Lifecycle Events
-- ============================================================================
\set ON_ERROR_STOP on
\echo '=========================================='
\echo 'STEP 2: Model Lifecycle (Load, Predict, Export)'
\echo '=========================================='

SET search_path TO neurondb, public;

-- Fetch version ids for reuse
WITH versions AS (
    SELECT m.model_name, mv.version_id, mv.storage_uri
    FROM neurondb.models m
    JOIN neurondb.model_versions mv ON mv.model_id = m.model_id
    WHERE m.model_name LIKE 'demo_%'
)
SELECT * FROM versions ORDER BY model_name;

\echo ''
\echo 'Loading models into NeuronDB (updates catalog status to loaded)'
SELECT load_model('demo_sentiment', '/tmp/neurondb_models_demo/sentiment-mini.onnx', 'onnx') AS sentiment_loaded;
SELECT load_model('demo_qa', '/tmp/neurondb_models_demo/question-answering.onnx', 'onnx') AS qa_loaded;

\echo ''
\echo 'Catalog view after load_model'
SELECT model_name, version_label, status, loaded_at
FROM neurondb.model_catalog
WHERE model_name LIKE 'demo_%'
ORDER BY model_name;

\echo ''
\echo 'Running predict and predict_batch to generate audit events'
SELECT predict('demo_sentiment', '[0.1,0.2,0.3]'::vector) AS sentiment_prediction;
SELECT predict_batch('demo_sentiment', ARRAY['[0.1,0.2,0.3]'::vector, '[0.4,0.5,0.6]'::vector]) AS sentiment_batch_prediction;

\echo ''
\echo 'Creating training data and invoking finetune_model'
CREATE TEMP TABLE IF NOT EXISTS demo_sentiment_training AS
SELECT generate_series AS id, ('text sample ' || generate_series)::text AS text_data
FROM generate_series(1, 4);
SELECT finetune_model('demo_sentiment', 'demo_sentiment_training', '{"epochs": 1, "lr": 1e-4}');

\echo ''
\echo 'Exporting sentiment model (updates status to exported)'
\! mkdir -p /tmp/neurondb_models_demo/exports
SELECT export_model('demo_sentiment', '/tmp/neurondb_models_demo/exports/sentiment-mini-export.txt', 'onnx');

\echo ''
\echo 'Catalog view after lifecycle operations'
SELECT model_name, version_label, status, loaded_at, exported_at
FROM neurondb.model_catalog
WHERE model_name LIKE 'demo_%'
ORDER BY model_name;

\echo ''
\echo 'Latest model events for demo_sentiment'
SELECT event_type, event_at, event_details
FROM neurondb.model_events me
JOIN neurondb.model_versions mv ON mv.version_id = me.version_id
JOIN neurondb.models m ON m.model_id = mv.model_id
WHERE m.model_name = 'demo_sentiment'
ORDER BY event_at DESC
LIMIT 10;

\echo ''
\echo 'STEP 2 complete'
\echo '=========================================='
