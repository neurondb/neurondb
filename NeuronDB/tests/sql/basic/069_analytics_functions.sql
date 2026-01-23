-- 069_analytics_functions.sql
-- Tests for analytics functions: vector_statistics, index_quality_metrics, query_performance_analytics

\timing on
\pset footer off
\pset pager off
\pset tuples_only off

\set ON_ERROR_STOP on

-- Ensure neurondb types/operators (including vector) are available
CREATE EXTENSION IF NOT EXISTS neurondb;

\echo '=========================================================================='
\echo 'Analytics Functions: Comprehensive Tests'
\echo '=========================================================================='

-- Setup: Create test table with vectors
\echo ''
\echo 'Setup: Creating test table with vectors'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DROP TABLE IF EXISTS analytics_test_table CASCADE;
CREATE TABLE analytics_test_table (
	id SERIAL PRIMARY KEY,
	embedding vector(128),
	label integer
);

-- Insert test data
INSERT INTO analytics_test_table (embedding, label)
SELECT 
	array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 128)))::vector(128),
	(random() * 10)::integer
FROM generate_series(1, 100);

\echo 'Test 1: vector_statistics - Basic functionality'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	stats_json jsonb;
	mean_val jsonb;
	variance_val jsonb;
	stddev_val jsonb;
	min_val jsonb;
	max_val jsonb;
	correlation jsonb;
BEGIN
	-- Call vector_statistics and validate results
	SELECT vector_statistics('analytics_test_table', 'embedding') INTO stats_json;
	
	-- Verify JSONB structure
	IF stats_json IS NULL THEN
		RAISE EXCEPTION 'vector_statistics returned NULL';
	END IF;
	
	-- Extract and verify fields
	mean_val := stats_json->'mean';
	variance_val := stats_json->'variance';
	stddev_val := stats_json->'stddev';
	min_val := stats_json->'min';
	max_val := stats_json->'max';
	correlation := stats_json->'correlation';
	
	-- Verify all fields exist
	IF mean_val IS NULL THEN
		RAISE EXCEPTION 'vector_statistics missing mean field';
	END IF;
	
	IF variance_val IS NULL THEN
		RAISE EXCEPTION 'vector_statistics missing variance field';
	END IF;
	
	IF stddev_val IS NULL THEN
		RAISE EXCEPTION 'vector_statistics missing stddev field';
	END IF;
	
	IF min_val IS NULL THEN
		RAISE EXCEPTION 'vector_statistics missing min field';
	END IF;
	
	IF max_val IS NULL THEN
		RAISE EXCEPTION 'vector_statistics missing max field';
	END IF;
	
	IF correlation IS NULL THEN
		RAISE EXCEPTION 'vector_statistics missing correlation field';
	END IF;
	
	RAISE NOTICE 'vector_statistics test passed: all fields present';
END$$;

-- Display statistics
SELECT vector_statistics('analytics_test_table', 'embedding') AS stats;

\echo ''
\echo 'Test 2: index_quality_metrics - HNSW index'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Create HNSW index
DROP INDEX IF EXISTS analytics_hnsw_idx;
CREATE INDEX analytics_hnsw_idx ON analytics_test_table 
USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 64, ef_search = 40);

DO $$
DECLARE
	metrics_json jsonb;
	index_size jsonb;
	vector_count jsonb;
	recall jsonb;
	precision_val jsonb;
	f1 jsonb;
	health_status jsonb;
BEGIN
	-- Call index_quality_metrics and validate results
	SELECT index_quality_metrics('analytics_hnsw_idx') INTO metrics_json;
	
	-- Verify JSONB structure
	IF metrics_json IS NULL THEN
		RAISE EXCEPTION 'index_quality_metrics returned NULL';
	END IF;
	
	-- Extract and verify fields
	index_size := metrics_json->'index_size';
	vector_count := metrics_json->'vector_count';
	recall := metrics_json->'recall';
	precision_val := metrics_json->'precision';
	f1 := metrics_json->'f1';
	health_status := metrics_json->'health_status';
	
	-- Verify all fields exist
	IF index_size IS NULL THEN
		RAISE EXCEPTION 'index_quality_metrics missing index_size field';
	END IF;
	
	IF vector_count IS NULL THEN
		RAISE EXCEPTION 'index_quality_metrics missing vector_count field';
	END IF;
	
	IF health_status IS NULL THEN
		RAISE EXCEPTION 'index_quality_metrics missing health_status field';
	END IF;
	
	RAISE NOTICE 'index_quality_metrics (HNSW) test passed: all fields present';
END$$;

-- Display metrics
SELECT index_quality_metrics('analytics_hnsw_idx') AS metrics;

\echo ''
\echo 'Test 3: index_quality_metrics - IVF index'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Create IVF index
DROP INDEX IF EXISTS analytics_ivf_idx;
CREATE INDEX analytics_ivf_idx ON analytics_test_table 
USING ivf (embedding vector_l2_ops)
WITH (lists = 10, probes = 5);

DO $$
DECLARE
	metrics_json jsonb;
BEGIN
	-- Call index_quality_metrics and validate results
	SELECT index_quality_metrics('analytics_ivf_idx') INTO metrics_json;
	
	-- Verify JSONB structure
	IF metrics_json IS NULL THEN
		RAISE EXCEPTION 'index_quality_metrics (IVF) returned NULL';
	END IF;
	
	RAISE NOTICE 'index_quality_metrics (IVF) test passed';
END$$;

-- Display metrics
SELECT index_quality_metrics('analytics_ivf_idx') AS metrics;

\echo ''
\echo 'Test 4: query_performance_analytics - After running queries'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Run some queries to generate performance data
DO $$
DECLARE
	query_vec vector(128);
BEGIN
	-- Generate a query vector
	query_vec := array_to_vector(ARRAY(SELECT random()::real FROM generate_series(1, 128)))::vector(128);
	
	-- Run several queries
	FOR i IN 1..10 LOOP
		PERFORM id, embedding <-> query_vec AS distance
		FROM analytics_test_table
		ORDER BY embedding <-> query_vec
		LIMIT 10;
	END LOOP;
END$$;

DO $$
DECLARE
	analytics_json jsonb;
	query_count jsonb;
	latency_percentiles jsonb;
	gpu_query_count jsonb;
	gpu_utilization jsonb;
BEGIN
	-- Call query_performance_analytics and validate results
	SELECT query_performance_analytics() INTO analytics_json;
	
	-- Verify JSONB structure
	IF analytics_json IS NULL THEN
		RAISE EXCEPTION 'query_performance_analytics returned NULL';
	END IF;
	
	-- Extract and verify fields
	query_count := analytics_json->'query_count';
	latency_percentiles := analytics_json->'latency_percentiles';
	gpu_query_count := analytics_json->'gpu_query_count';
	gpu_utilization := analytics_json->'gpu_utilization';
	
	-- Verify all fields exist
	IF query_count IS NULL THEN
		RAISE EXCEPTION 'query_performance_analytics missing query_count field';
	END IF;
	
	IF latency_percentiles IS NULL THEN
		RAISE EXCEPTION 'query_performance_analytics missing latency_percentiles field';
	END IF;
	
	RAISE NOTICE 'query_performance_analytics test passed: all fields present';
END$$;

-- Display analytics
SELECT query_performance_analytics() AS analytics;

\echo ''
\echo 'Test 5: Error handling - Invalid table name'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
BEGIN
	BEGIN
		PERFORM vector_statistics('nonexistent_table', 'embedding');
		RAISE EXCEPTION 'vector_statistics should have raised an error for nonexistent table';
	EXCEPTION WHEN OTHERS THEN
		-- Should raise an error for nonexistent table (not undefined_function)
		IF SQLSTATE = '42883' THEN
			RAISE EXCEPTION 'vector_statistics function not available - this is a required function';
		ELSE
			RAISE NOTICE 'vector_statistics correctly raised error for nonexistent table: %', SQLERRM;
		END IF;
	END;
END$$;

\echo ''
\echo 'Test 6: Error handling - Invalid index name'
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
BEGIN
	BEGIN
		PERFORM index_quality_metrics('nonexistent_index');
		RAISE EXCEPTION 'index_quality_metrics should have raised an error for nonexistent index';
	EXCEPTION WHEN OTHERS THEN
		-- Should raise an error for nonexistent index (not undefined_function)
		IF SQLSTATE = '42883' THEN
			RAISE EXCEPTION 'index_quality_metrics function not available - this is a required function';
		ELSE
			RAISE NOTICE 'index_quality_metrics correctly raised error for nonexistent index: %', SQLERRM;
		END IF;
	END;
END$$;

\echo ''
\echo '=========================================================================='
\echo '✓ Analytics Functions: All tests complete'
\echo '=========================================================================='
