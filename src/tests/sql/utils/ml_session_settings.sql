/*-------------------------------------------------------------------------
 *
 * ml_session_settings.sql
 *    Session-level PostgreSQL optimizations for NeurondB ML workloads
 *
 * Run this before executing ML training/prediction queries to optimize
 * performance for large-scale operations.
 *
 * Usage:
 *   \i ml_session_settings.sql
 *   -- Then run your ML queries
 *
 *-------------------------------------------------------------------------*/

/* Memory settings for large ML operations */
SET work_mem = '512MB';  /* Increased for sorting/hashing large datasets */
SET maintenance_work_mem = '2GB';  /* For VACUUM, CREATE INDEX operations */

/* Parallelism settings */
SET max_parallel_workers_per_gather = 8;  /* Use more parallel workers */
SET max_parallel_workers = 15;  /* Maximum parallel workers */

/* Query planner optimizations */
SET random_page_cost = 1.1;  /* Optimized for SSD */
SET effective_io_concurrency = 200;  /* Higher for SSD */

/* Disable timeouts for long-running ML operations */
SET statement_timeout = 0;  /* No timeout for ML training */
SET lock_timeout = 0;  /* No lock timeout */

/* Enable JIT compilation for complex queries */
SET jit = on;
SET jit_above_cost = 100000;  /* Use JIT for expensive queries */
SET jit_optimize_above_cost = 500000;  /* Optimize JIT for very expensive queries */

/* Enable timing for performance monitoring */
\timing on

/* Disable pager for cleaner output */
\pset pager off

/* Show settings */
SELECT 
	'work_mem' AS setting, 
	current_setting('work_mem') AS value
UNION ALL
SELECT 
	'max_parallel_workers_per_gather',
	current_setting('max_parallel_workers_per_gather');

