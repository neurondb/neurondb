\timing on
\pset footer off
\pset pager off

\set ON_ERROR_STOP off

\echo '=========================================================================='
\echo '=========================================================================='

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'gmm',
	NULL,
	'features',
	NULL,
	'{}'::jsonb
);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'gmm',
	'test_train_view',
	NULL,
	NULL,
	'{}'::jsonb
);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'gmm',
	'nonexistent_table_xyz',
	'features',
	NULL,
	'{}'::jsonb
);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'gmm',
	'test_train_view',
	'nonexistent_column',
	NULL,
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
		'gmm',
		'test_train_view',
		'features',
		NULL,
		'{"k": 3}'::jsonb
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
		'gmm',
		'test_train_view',
		'features',
		NULL,
		'{"k": 3}'::jsonb
	)::integer INTO mid;
	
	PERFORM neurondb.predict(mid, '[1,2,3]'::vector);
EXCEPTION
	WHEN OTHERS THEN
END
$$;

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'gmm',
	'test_train_view',
	'features',
	NULL,
	'{"k": 0}'::jsonb
);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'gmm',
	'test_train_view',
	'features',
	NULL,
	'{"k": -1}'::jsonb
);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.train(
	'gmm',
	'test_train_view',
	'features',
	NULL,
	'{"k": 10000}'::jsonb
);

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

CREATE TEMP TABLE empty_train (
	features vector
);

SELECT neurondb.train(
	'gmm',
	'empty_train',
	'features',
	NULL,
	'{"k": 3}'::jsonb
);

DROP TABLE IF EXISTS empty_train;

\echo ''
\echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

SELECT neurondb.evaluate(-1, 'test_test_view', 'features', NULL);

\echo ''
\echo '=========================================================================='

\echo 'Test completed successfully'
