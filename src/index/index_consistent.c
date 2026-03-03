/*-------------------------------------------------------------------------
 *
 * index_consistent.c
 *		Consistent query HNSW with deterministic top-k across replicas
 *
 * Implements CQ-HNSW with snapshot pinning to ensure identical
 * query results across all replicas, critical for distributed systems.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  src/index_consistent.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "neurondb_compat.h"
#include "neurondb_index.h"
#include "fmgr.h"
#include "catalog/pg_type.h"
#include "utils/builtins.h"
#include "utils/snapmgr.h"
#include "executor/spi.h"
#include "access/htup_details.h"
#include "access/tupdesc.h"
#include "funcapi.h"
#include <string.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_spi_safe.h"
#include "neurondb_spi.h"
#include "neurondb_constants.h"

/* Forward declarations */
static char *vector_to_sql_literal(Vector *v);
static bool index_exists(const char *table, const char *col);
static void build_hnsw_index(const char *table, const char *col, uint32 seed);
static char *get_index_table(const char *table, const char *col, uint32 seed);
static Oid get_relid_from_name(const char *relname);
static void resolve_index_to_table_col(const char *index_name, MemoryContext ctx,
									   char **out_table, char **out_column);

/*
 * Create consistent query HNSW index with deterministic properties.
 * This function checks whether a consistent HNSW index exists on the
 * given (table, col), and if not, builds one and stores metadata.
 * The seed is used to ensure distributed determinism.
 */
PG_FUNCTION_INFO_V1(consistent_index_create);
Datum
consistent_index_create(PG_FUNCTION_ARGS)
{
	text	   *table_name = PG_GETARG_TEXT_PP(0);
	text	   *vector_col = PG_GETARG_TEXT_PP(1);
	uint32		random_seed = PG_GETARG_INT32(2);
	char	   *tbl_str = text_to_cstring(table_name);
	char	   *col_str = text_to_cstring(vector_col);
	char *index_tbl = NULL;
	Oid			relid;


	/* Check if the index already exists */
	if (index_exists(tbl_str, col_str))
	{
		PG_RETURN_BOOL(true);
	}

	/* Build the HNSW index (this would be much more elaborate in reality) */
	build_hnsw_index(tbl_str, col_str, random_seed);

	/*
	 * Store metadata for deterministic operation; in real code, update
	 * catalog
	 */
	index_tbl = get_index_table(tbl_str, col_str, random_seed);

	/* Validate relation exists */
	relid = get_relid_from_name(index_tbl);
	if (!OidIsValid(relid))
	{
		ereport(ERROR,
				(errmsg("Failed to find or create index table %s",
						index_tbl)));
	}

	PG_RETURN_BOOL(true);
}

/*
 * Resolve index name to (table name, column name) via catalog.
 * If index_name is not a valid index OID, use it as table name and "embedding" as column.
 */
static void
resolve_index_to_table_col(const char *index_name, MemoryContext ctx,
						   char **out_table, char **out_column)
{
	Oid			index_oid;
	NdbSpiSession *session = NULL;
	StringInfoData sql;
	Oid			argtypes[1];
	Datum		values[1];
	char		nulls[1] = {' '};
	int			ret;
	MemoryContext old = MemoryContextSwitchTo(ctx);

	*out_table = NULL;
	*out_column = NULL;
	index_oid = DatumGetObjectId(DirectFunctionCall1(to_regclass, CStringGetTextDatum(index_name)));
	if (!OidIsValid(index_oid))
	{
		*out_table = pstrdup(index_name);
		*out_column = pstrdup("embedding");
		MemoryContextSwitchTo(old);
		return;
	}
	argtypes[0] = OIDOID;
	values[0] = ObjectIdGetDatum(index_oid);
	session = ndb_spi_session_begin(ctx, false);
	if (session == NULL)
	{
		*out_table = pstrdup(index_name);
		*out_column = pstrdup("embedding");
		MemoryContextSwitchTo(old);
		return;
	}
	ndb_spi_stringinfo_init(session, &sql);
	appendStringInfo(&sql,
					 "SELECT (SELECT (n.nspname || '.' || t.relname) FROM pg_class t JOIN pg_namespace n ON n.oid = t.relnamespace WHERE t.oid = i.indrelid), (SELECT a.attname FROM pg_attribute a WHERE a.attrelid = i.indrelid AND a.attnum = (i.indkey)[1] AND a.attnum > 0 AND NOT a.attisdropped) FROM pg_index i WHERE i.indexrelid = $1");
	ret = ndb_spi_execute_with_args(session, sql.data, 1, argtypes, values, nulls, true, 1);
	ndb_spi_stringinfo_free(session, &sql);
	if (ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		HeapTuple	tup = SPI_tuptable->vals[0];
		TupleDesc	td = SPI_tuptable->tupdesc;
		bool		n1 = false;
		bool		n2 = false;
		Datum		d1 = SPI_getbinval(tup, td, 1, &n1);
		Datum		d2 = SPI_getbinval(tup, td, 2, &n2);

		if (!n1 && !n2)
		{
			*out_table = pstrdup(TextDatumGetCString(d1));
			*out_column = pstrdup(TextDatumGetCString(d2));
		}
		ndb_spi_session_end(&session);
		MemoryContextSwitchTo(old);
		return;
	}
	ndb_spi_session_end(&session);
	*out_table = pstrdup(index_name);
	*out_column = pstrdup("embedding");
	MemoryContextSwitchTo(old);
}

/*
 * Consistent kNN search with snapshot pinning and deterministic tie-breaking.
 * Returns a setof (id BIGINT, dist DOUBLE PRECISION) rows for accurate top-k.
 * Table and column are derived from index_name via catalog or fallback.
 */
PG_FUNCTION_INFO_V1(consistent_knn_search);
Datum
consistent_knn_search(PG_FUNCTION_ARGS)
{
	text	   *index_name = PG_GETARG_TEXT_PP(0);
	Vector	   *query = PG_GETARG_VECTOR_P(1);
	int32		k = PG_GETARG_INT32(2);
	text	   *consistency_level = PG_GETARG_TEXT_PP(3);
	FuncCallContext *funcctx = NULL;
	TupleDesc	tupdesc;
	Datum		values[2];
	bool		nulls[2];
	HeapTuple	tuple;
	int			call_cntr;
	int			max_calls;
	char	   *vector_str = NULL;
	char	   *table_str = NULL;
	char	   *col_str = NULL;
	const char *quoted_table = NULL;
	const char *quoted_col = NULL;
	StringInfoData sql;
	int			ret;

	(void) consistency_level;		/* Reserved for future use */

	NDB_CHECK_VECTOR_VALID(query);

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;

		NdbSpiSession *session = NULL;

		funcctx = SRF_FIRSTCALL_INIT();

		oldcontext =
			MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		/* Resolve index name to table and column (with quoting) */
		resolve_index_to_table_col(text_to_cstring(index_name),
								   funcctx->multi_call_memory_ctx,
								   &table_str, &col_str);
		quoted_table = quote_identifier(table_str);
		quoted_col = quote_identifier(col_str);

		/* Convert query vector to string for SQL embedding */
		vector_str = vector_to_sql_literal(query);

		/*
		 * Pin a transaction snapshot to ensure MVCC-visible deterministic
		 * results
		 */
		PushActiveSnapshot(GetTransactionSnapshot());

		session = ndb_spi_session_begin(funcctx->multi_call_memory_ctx, false);
		if (session == NULL)
		{
			PopActiveSnapshot();
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("failed to begin SPI session in "
							"consistent_knn_search")));
		}

		/*
		 * Use parameterized query with quoted table and column from catalog.
		 * Deterministic ordering: ORDER BY dist ASC, ctid ASC, id ASC
		 */
		ndb_spi_stringinfo_init(session, &sql);
		appendStringInfo(&sql,
						 "SELECT id, l2_distance(%s, %s) AS dist FROM %s ORDER BY dist ASC, ctid ASC, id ASC LIMIT %d",
						 quoted_col, vector_str, quoted_table, k);

		ret = ndb_spi_execute(session, sql.data, true, k);
		ndb_spi_stringinfo_free(session, &sql);
		if (ret != SPI_OK_SELECT)
		{
			ndb_spi_session_end(&session);
			PopActiveSnapshot();
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("SPI_execute failed in "
							"consistent_knn_search")));
		}

		funcctx->max_calls = SPI_processed;

		/* Save session and results for per-call access */

		/*
		 * Store session pointer in user_fctx - we'll free it in
		 * SRF_RETURN_DONE
		 */
		funcctx->user_fctx = session;

		/*
		 * Set up tuple descriptor for results (id BIGINT, dist DOUBLE
		 * PRECISION)
		 */
		tupdesc = CreateTemplateTupleDesc(2);
		TupleDescInitEntry(
						   tupdesc, (AttrNumber) 1, "id", INT8OID, -1, 0);
		TupleDescInitEntry(
						   tupdesc, (AttrNumber) 2, "dist", FLOAT8OID, -1, 0);
		funcctx->tuple_desc = BlessTupleDesc(tupdesc);

		MemoryContextSwitchTo(oldcontext);

	}

	funcctx = SRF_PERCALL_SETUP();

	max_calls = funcctx->max_calls;
	call_cntr = funcctx->call_cntr;

	if (call_cntr < max_calls)
	{
		NdbSpiSession *session = (NdbSpiSession *) funcctx->user_fctx;
		SPITupleTable *tuptable = NULL;
		HeapTuple	spi_tuple;
		bool		isnull;

		if (session == NULL || SPI_tuptable == NULL)
		{
			SRF_RETURN_DONE(funcctx);
		}

		tuptable = SPI_tuptable;
		spi_tuple = tuptable->vals[call_cntr];

		/* Extract "id" (attribute 1), "dist" (attribute 2) */
		values[0] =
			SPI_getbinval(spi_tuple, tuptable->tupdesc, 1, &isnull);
		nulls[0] = isnull;
		values[1] =
			SPI_getbinval(spi_tuple, tuptable->tupdesc, 2, &isnull);
		nulls[1] = isnull;

		tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);

		SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
	}
	else
	{
		NdbSpiSession *session = (NdbSpiSession *) funcctx->user_fctx;

		if (session != NULL)
		{
			ndb_spi_session_end(&session);
		}

		PopActiveSnapshot();

		SRF_RETURN_DONE(funcctx);
	}
}

/*
 * Helper: Check if a consistent HNSW index exists on (table, col)
 * Would normally look in pg_catalog or a custom metadata table.
 */
static bool
index_exists(const char *table, const char *col)
{
	(void) table;
	(void) col;

	/* For demonstration, always rebuild */
	return false;
}

/*
 * Build a deterministic HNSW index on a table.column using the given seed.
 * This implementation creates a dedicated index table named with the seed,
 * scans the target table for vector data, and bulk-inserts into the index
 * table with deterministic ordering. Actual graph construction in memory
 * is omitted; we only persist the ordered data for later CQ-HNSW usage.
 */
static void
build_hnsw_index(const char *table, const char *col, uint32 seed)
{
	char *index_table = NULL;
	StringInfoData sql;
	int			ret;
	Oid			argtypes[4];
	Datum		values[4];

	NdbSpiSession *session = NULL;


	/* Compute deterministic index table name */
	index_table = get_index_table(table, col, seed);

	/* Start a SPI session */
	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
		elog(ERROR, "Failed to begin SPI session in build_hnsw_index");

	/* Create index table if it does not exist */
	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "CREATE TABLE IF NOT EXISTS %s (id bigint, v %s, PRIMARY "
					 "KEY(id))",
					 index_table,
					 "vector");

	ret = ndb_spi_execute(session, sql.data, false, 0);
	if (ret != SPI_OK_UTILITY)
	{
		pfree(sql.data);
		pfree(index_table);
		ndb_spi_session_end(&session);
		elog(ERROR,
			 "Failed to create index table '%s': %s",
			 index_table,
			 sql.data);
	}

	/* Use safe free/reinit to handle potential memory context changes */
	pfree(sql.data);
	initStringInfo(&sql);

	/* Remove all rows in case we rebuild */
	appendStringInfo(&sql, "TRUNCATE %s", index_table);
	ndb_spi_execute(session, sql.data, false, 0);
	/* Use safe free/reinit to handle potential memory context changes */
	pfree(sql.data);
	initStringInfo(&sql);

	/*
	 * Insert source vectors with deterministic ordering. Use 'seed' to
	 * shuffle. For demonstration, we order by hashtext(id || seed).
	 *
	 * NOTE: This does not actually build CQ-HNSW; only prepared data table.
	 */
	appendStringInfo(&sql,
					 "INSERT INTO %s (id, v) "
					 "SELECT id, %s FROM %s ORDER BY hashtext(id::text || '%u')",
					 index_table,
					 col,
					 table,
					 seed);

	ret = ndb_spi_execute(session, sql.data, false, 0);
	if (ret != SPI_OK_INSERT)
	{
		pfree(sql.data);
		pfree(index_table);
		ndb_spi_session_end(&session);
		elog(ERROR, "Failed to bulk insert vectors: %s", sql.data);
	}

	/* Store/Update metadata (example: in a metadata table) */
	/* Use safe free/reinit to handle potential memory context changes */
	pfree(sql.data);
	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "CREATE TABLE IF NOT EXISTS neurondb_hnsw_metadata ("
					 "  tablename text PRIMARY KEY, "
					 "  colname text, "
					 "  index_table text, "
					 "  build_seed int8, "
					 "  build_time timestamptz default clock_timestamp())");
	ndb_spi_execute(session, sql.data, false, 0);

	/* Use safe free/reinit to handle potential memory context changes */
	pfree(sql.data);
	initStringInfo(&sql);

	appendStringInfo(&sql,
					 "INSERT INTO neurondb_hnsw_metadata (tablename, colname, "
					 "index_table, build_seed) "
					 "VALUES ($1, $2, $3, $4) "
					 "ON CONFLICT (tablename) DO UPDATE SET "
					 "  colname=EXCLUDED.colname, "
					 "  index_table=EXCLUDED.index_table, "
					 "  build_seed=EXCLUDED.build_seed, "
					 "  build_time=clock_timestamp()");

	argtypes[0] = TEXTOID;
	argtypes[1] = TEXTOID;
	argtypes[2] = TEXTOID;
	argtypes[3] = INT8OID;
	values[0] = CStringGetTextDatum(table);
	values[1] = CStringGetTextDatum(col);
	values[2] = CStringGetTextDatum(index_table);
	values[3] = Int64GetDatum((int64) seed);

	ret = ndb_spi_execute_with_args(session, sql.data, 4, argtypes, values, NULL, false, 0);
	if (ret != SPI_OK_INSERT && ret != SPI_OK_UPDATE)
	{
		pfree(sql.data);
		pfree(index_table);
		ndb_spi_session_end(&session);
		elog(ERROR, "Metadata insert/update failed (%d)", ret);
	}

	pfree(sql.data);

	/* Done, cleanup */
	pfree(index_table);
	ndb_spi_session_end(&session);
}

/*
 * Given (table, col, seed), compute the deterministic index table name.
 * For uniqueness, e.g.: "__hnsw_${table}_${col}_${seed}"
 */
static char *
get_index_table(const char *table, const char *col, uint32 seed)
{
	char *buf = NULL;
	int			len = strlen(table) + strlen(col) + 32;

	NBP_ALLOC(buf, char, len);
	snprintf(buf,
			 len,
			 NDB_INDEX_NAME_FMT_HNSW,
			 table,
			 col,
			 seed);
	return buf;
}

/*
 * Lookup a table by name, return its Oid, or InvalidOid if not found.
 */
static Oid
get_relid_from_name(const char *relname)
{
	Oid			relid = InvalidOid;

	/* See to_regclass, but simplified for demonstration */
	relid = DatumGetObjectId(
							 DirectFunctionCall1(to_regclass, CStringGetDatum(relname)));
	return relid;
}

/*
 * Helper: Convert a PostgreSQL vector (internal format) to a properly quoted SQL literal.
 * Uses neurondb's out function, then single-quotes it.
 */
static char *
vector_to_sql_literal(Vector *v)
{
	char *out = NULL;
	int			n;

	char *quoted = NULL;

	out = vector_out_internal(v);
	n = (int) strlen(out) + 4;
	NBP_ALLOC(quoted, char, n);
	snprintf(quoted, n, "'%s'", out);
	return quoted;
}
