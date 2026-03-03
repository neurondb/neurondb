\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP off

\echo '=========================================================================='
\echo '=========================================================================='

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT auto_train(
	NULL,
	'features',
	'label',
	'classification',
	'accuracy'
);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT auto_train(
	'test_train_view',
	NULL,
	'label',
	'classification',
	'accuracy'
);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT auto_train(
	'test_train_view',
	'features',
	NULL,
	'classification',
	'accuracy'
);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT auto_train(
	'test_train_view',
	'features',
	'label',
	NULL,
	'accuracy'
);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT auto_train(
	'test_train_view',
	'features',
	'label',
	'classification',
	NULL
);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT auto_train(
	'nonexistent_table_xyz',
	'features',
	'label',
	'classification',
	'accuracy'
);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT auto_train(
	'test_train_view',
	'nonexistent_column',
	'label',
	'classification',
	'accuracy'
);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT auto_train(
	'test_train_view',
	'features',
	'nonexistent_label',
	'classification',
	'accuracy'
);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT auto_train(
	'test_train_view',
	'features',
	'label',
	'invalid_task_type',
	'accuracy'
);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT auto_train(
	'test_train_view',
	'features',
	'label',
	'classification',
	'invalid_metric_name'
);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

-- Using regression metric for classification task
SELECT auto_train(
	'test_train_view',
	'features',
	'label',
	'classification',
	'r2'
);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE TEMP TABLE empty_train (
	features vector,
	label float8
);

SELECT auto_train(
	'empty_train',
	'features',
	'label',
	'classification',
	'accuracy'
);

DROP TABLE IF EXISTS empty_train;

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE TEMP TABLE tiny_train (
	features vector(5),
	label float8
);

-- Insert only 1 row (insufficient for training)
INSERT INTO tiny_train VALUES ('[1,2,3,4,5]'::vector, 1.0);

SELECT auto_train(
	'tiny_train',
	'features',
	'label',
	'classification',
	'accuracy'
);

DROP TABLE IF EXISTS tiny_train;

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE TEMP TABLE invalid_dim_train (
	features vector(3),
	label float8
);

-- Insert rows with different dimensions (if possible)
INSERT INTO invalid_dim_train VALUES 
	('[1,2,3]'::vector, 1.0),
	('[4,5,6]'::vector, 0.0);

SELECT auto_train(
	'invalid_dim_train',
	'features',
	'label',
	'classification',
	'accuracy'
);

DROP TABLE IF EXISTS invalid_dim_train;

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE TEMP TABLE null_target_train (
	features vector(5),
	label float8
);

INSERT INTO null_target_train VALUES 
	('[1,2,3,4,5]'::vector, NULL),
	('[6,7,8,9,10]'::vector, NULL);

SELECT auto_train(
	'null_target_train',
	'features',
	'label',
	'classification',
	'accuracy'
);

DROP TABLE IF EXISTS null_target_train;

\echo ''
\echo '=========================================================================='

\echo 'Test completed successfully'
