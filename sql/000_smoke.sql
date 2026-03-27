-- PgXS installcheck: cluster must list neurondb in shared_preload_libraries and restart once.
CREATE EXTENSION neurondb;
SELECT extname, extversion FROM pg_extension WHERE extname = 'neurondb';
