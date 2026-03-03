\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP off

\echo '=========================================================================='
\echo '=========================================================================='

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'knn',
	NULL,
	'features',
	'label',
	'{}'::jsonb
);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'knn',
	'test_train_view',
	NULL,
	'label',
	'{}'::jsonb
);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'knn',
	'test_train_view',
	'features',
	NULL,
	'{}'::jsonb
);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'knn',
	'nonexistent_table_xyz',
	'features',
	'label',
	'{}'::jsonb
);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'knn',
	'test_train_view',
	'nonexistent_column',
	'label',
	'{}'::jsonb
);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'knn',
	'test_train_view',
	'features',
	'nonexistent_label',
	'{}'::jsonb
);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.predict(-1, '[1,2,3,4,5]'::vector);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.predict(NULL, '[1,2,3,4,5]'::vector);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	mid integer;
BEGIN
	SELECT neurondb.train(
		'knn',
		'test_train_view',
		'features',
		'label',
		'{"k": 5}'::jsonb
	)::integer INTO mid;
	
	PERFORM neurondb.predict(mid, NULL::vector);
EXCEPTION
	WHEN OTHERS THEN
END
$$;

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

DO $$
DECLARE
	mid integer;
BEGIN
	SELECT neurondb.train(
		'knn',
		'test_train_view',
		'features',
		'label',
		'{"k": 5}'::jsonb
	)::integer INTO mid;
	
	PERFORM neurondb.predict(mid, '[1,2,3]'::vector);
EXCEPTION
	WHEN OTHERS THEN
END
$$;

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'knn',
	'test_train_view',
	'features',
	'label',
	'{"k": 0}'::jsonb
);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'knn',
	'test_train_view',
	'features',
	'label',
	'{"k": -1}'::jsonb
);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'knn',
	'test_train_view',
	'features',
	'label',
	'{"k": 100000}'::jsonb
);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE TEMP TABLE empty_train (
	features vector,
	label float8
);

SELECT neurondb.train(
	'knn',
	'empty_train',
	'features',
	'label',
	'{"k": 5}'::jsonb
);

DROP TABLE IF EXISTS empty_train;

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.evaluate(-1, 'test_test_view', 'features', 'label');

\echo ''
\echo '=========================================================================='

\echo 'Test completed successfully'
