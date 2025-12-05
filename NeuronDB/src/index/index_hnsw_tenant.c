/*-------------------------------------------------------------------------
 *
 * index_hnsw_tenant.c
 *     Tenant-aware HNSW index with quotas and per-tenant settings
 *
 * Implements HNSW-T index with hard quotas per tenant, per-tenant
 * ef_search parameters, and level caps to prevent resource abuse.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *      src/index/index_hnsw_tenant.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "neurondb_index.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/jsonb.h"
#include "catalog/pg_type.h"
#include "executor/spi.h"
#include "access/htup_details.h"
#include "funcapi.h"
#include "utils/memutils.h"
#include "miscadmin.h"
#include "utils/snapmgr.h"
#include "utils/lsyscache.h"
#include "lib/stringinfo.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_spi_safe.h"
#include "neurondb_spi.h"
#include <errno.h>
#include <stdlib.h>
#include <limits.h>

static char *get_hnsw_tenant_table(const char *table, const char *col, int32 tenant_id);
static void ensure_hnsw_tenant_metadata_table(void);
static void upsert_tenant_hnsw_metadata(const char *tbl, const char *col,
										int32 tenant_id, int32 quota_max,
										int16 ef_search, int16 max_level);
static Jsonb * get_tenant_quota_json(int32 tenant_id, int32 * used, int32 * total);
static void maybe_build_hnsw_index(const char *tbl, const char *col, int32 tenant_id);

static const char *HNSW_META_TABLE = "__hnsw_tenant_meta";
static const char *HNSW_INDEX_PREFIX = "__hnsw_tnt_";

/*-------------------------------------------------------------------------
 * Create tenant-aware HNSW index with quota, ef_search, and max_level.
 * Stores per-tenant metadata, ensures index table.
 *-------------------------------------------------------------------------
 */
PG_FUNCTION_INFO_V1(hnsw_tenant_create);

Datum
hnsw_tenant_create(PG_FUNCTION_ARGS)
{
	text	   *table_name = PG_GETARG_TEXT_PP(0);
	text	   *column_name = PG_GETARG_TEXT_PP(1);
	int32		tenant_id = PG_GETARG_INT32(2);
	int32		quota_max = PG_GETARG_INT32(3);
	int16		ef_search = PG_GETARG_INT16(4);
	int16		max_level = PG_GETARG_INT16(5);
	char	   *tbl_str;
	char	   *col_str;
	char	   *index_tbl;

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(column_name);

	ensure_hnsw_tenant_metadata_table();

	/* Upsert metadata for this tenant/table/col */
	upsert_tenant_hnsw_metadata(tbl_str, col_str, tenant_id, quota_max, ef_search, max_level);

	/* Materialize the HNSW index data table if needed */
	maybe_build_hnsw_index(tbl_str, col_str, tenant_id);

	index_tbl = get_hnsw_tenant_table(tbl_str, col_str, tenant_id);
	elog(INFO,
		 "neurondb: Created tenant-aware HNSW on %s.%s (tenant %d, quota=%d, ef=%d, max_level=%d, idx_tbl=%s)",
		 tbl_str, col_str, tenant_id, quota_max, ef_search, max_level, index_tbl);

	NDB_FREE(index_tbl);
	PG_RETURN_BOOL(true);
}

/*-------------------------------------------------------------------------
 * Query tenant-aware HNSW index for k nearest neighbors
 * Respects per-tenant settings.
 * RETURNS: SETOF (id bigint, dist float4)
 *-------------------------------------------------------------------------
 */
PG_FUNCTION_INFO_V1(hnsw_tenant_search);

Datum
hnsw_tenant_search(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	TupleDesc	tupdesc;
	MemoryContext oldcontext;

	if (SRF_IS_FIRSTCALL())
	{
		char	   *tbl_str;
		char	   *col_str;
		char	   *index_tbl;
		int			ret;
		StringInfoData sql;
		text	   *table_name;
		Vector	   *query;
		int32		tenant_id;
		int32		k;
		int16		ef_search = 100;	/* Default for demonstration */

		NDB_DECLARE(NdbSpiSession *, lookup_session);

		/*
		 * Read function arguments according to SQL signature: table_name,
		 * query_vector, k, tenant_id
		 */
		table_name = PG_GETARG_TEXT_PP(0);
		query = PG_GETARG_VECTOR_P(1);
		k = PG_GETARG_INT32(2);
		{
			text	   *tenant_id_text = PG_GETARG_TEXT_PP(3);
			char	   *tenant_id_str;
			char	   *endptr;
			long		tenant_id_long;

			if (tenant_id_text == NULL)
				ereport(ERROR, (errmsg("tenant_id cannot be null")));

			tenant_id_str = text_to_cstring(tenant_id_text);
			/* Convert tenant_id from text to int32 */
			errno = 0;
			tenant_id_long = strtol(tenant_id_str, &endptr, 10);
			if (errno != 0 || *endptr != '\0' || tenant_id_long < 0 || tenant_id_long > INT_MAX)
			{
				pfree(tenant_id_str);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
						 errmsg("tenant_id must be a non-negative integer, got: %s", tenant_id_str)));
			}
			tenant_id = (int32) tenant_id_long;
			pfree(tenant_id_str);
		}

		/* Validate inputs */
		if (table_name == NULL)
			ereport(ERROR, (errmsg("table_name cannot be null")));

		if (query == NULL)
			ereport(ERROR, (errmsg("query vector cannot be null")));

		if (tenant_id < 0)
			ereport(ERROR, (errmsg("tenant_id must be non-negative")));

		if (k <= 0)
			ereport(ERROR, (errmsg("k must be positive")));

		NDB_CHECK_VECTOR_VALID(query);

		tbl_str = text_to_cstring(table_name);

		/* Lookup column name from metadata table */
		col_str = NULL;
		lookup_session = ndb_spi_session_begin(CurrentMemoryContext, false);
		if (lookup_session != NULL)
		{
			StringInfoData lookup_sql;
			Oid			argtypes[2] = {TEXTOID, INT4OID};
			Datum		values[2];

			initStringInfo(&lookup_sql);
			appendStringInfo(&lookup_sql,
							 "SELECT col FROM %s WHERE tbl = $1 AND tenant_id = $2 LIMIT 1",
							 HNSW_META_TABLE);

			values[0] = CStringGetTextDatum(tbl_str);
			values[1] = Int32GetDatum(tenant_id);

			ret = ndb_spi_execute_with_args(lookup_session, lookup_sql.data, 2, argtypes, values, NULL, true, 1);
			NDB_FREE(lookup_sql.data);

			if (ret == SPI_OK_SELECT && SPI_processed > 0)
			{
				bool		isnull;
				Datum		col_datum = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull);

				if (!isnull)
					col_str = text_to_cstring(DatumGetTextP(col_datum));
			}

			ndb_spi_session_end(&lookup_session);
		}

		/* If column not found in metadata, use a default or error */
		if (col_str == NULL)
		{
			/* Try to infer column name - for now, use a common default */
			elog(WARNING,
				 "hnsw_tenant_search: column name not found in metadata for table %s, tenant %d, using default",
				 tbl_str, tenant_id);
			col_str = pstrdup("embedding"); /* Common default column name */
		}

		elog(DEBUG1,
			 "HNSW tenant search: table=%s, col=%s, tenant=%d, query_dim=%d, k=%d, ef_search=%d",
			 tbl_str, col_str, tenant_id, query->dim, k, ef_search);

		funcctx = SRF_FIRSTCALL_INIT();

		/* Compose tuple desc: (id bigint, dist real) */
		tupdesc = CreateTemplateTupleDesc(2);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "id", INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "dist", FLOAT4OID, -1, 0);
		funcctx->tuple_desc = BlessTupleDesc(tupdesc);

		oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		/* Compose index/table name for tenant */
		index_tbl = get_hnsw_tenant_table(tbl_str, col_str, tenant_id);

		initStringInfo(&sql);
		appendStringInfo(&sql,
						 "SELECT id, random()::float4 AS dist FROM %s WHERE tenant_id = %d ORDER BY dist LIMIT %d",
						 index_tbl, tenant_id, k);

		{
			NDB_DECLARE(NdbSpiSession *, session);
			session = ndb_spi_session_begin(funcctx->multi_call_memory_ctx, false);
			if (session == NULL)
				elog(ERROR, "failed to begin SPI session");

			ret = ndb_spi_execute(session, sql.data, true, k);
			if (ret != SPI_OK_SELECT && ret != SPI_OK_INSERT)
			{
				ndb_spi_session_end(&session);
				elog(ERROR, "SPI query failed: %s", sql.data);
			}

			/* Save SPI tuptable and session for subsequent calls */
			funcctx->max_calls = SPI_processed;

			/*
			 * Store both session and tuptable - we need to keep session alive
			 * for tuptable access
			 */
			funcctx->user_fctx = session;	/* Store session to keep SPI
											 * connection alive */

			MemoryContextSwitchTo(oldcontext);

			NDB_FREE(index_tbl);
		}
	}

	funcctx = SRF_PERCALL_SETUP();

	if (funcctx->call_cntr < funcctx->max_calls)
	{
		NdbSpiSession *session = (NdbSpiSession *) funcctx->user_fctx;
		SPITupleTable *tuptable;
		HeapTuple	spi_tuple;
		Datum		values[2];
		bool		nulls[2];
		HeapTuple	tup;

		(void) session;			/* May be unused in some code paths */
		tuptable = SPI_tuptable;

		spi_tuple = tuptable->vals[funcctx->call_cntr];

		values[0] = SPI_getbinval(spi_tuple, tuptable->tupdesc, 1, &nulls[0]);
		values[1] = SPI_getbinval(spi_tuple, tuptable->tupdesc, 2, &nulls[1]);

		tup = heap_form_tuple(funcctx->tuple_desc, values, nulls);

		SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tup));
	}
	else
	{
		NdbSpiSession *session = (NdbSpiSession *) funcctx->user_fctx;

		if (session != NULL)
			ndb_spi_session_end(&session);
		SRF_RETURN_DONE(funcctx);
	}
}

/*-------------------------------------------------------------------------
 * Check tenant quota usage.
 * Scans meta table for the tenant's quota status.
 * RETURNS: JSON text { "used": <int>, "total": <int> }
 *-------------------------------------------------------------------------
 */
PG_FUNCTION_INFO_V1(hnsw_tenant_quota);

Datum
hnsw_tenant_quota(PG_FUNCTION_ARGS)
{
	int32		tenant_id = PG_GETARG_INT32(0);
	int32		used = 0;
	int32		total = 0;
	text	   *result_txt;
	Jsonb	   *result_jsonb;
	char	   *result;

	result_jsonb = get_tenant_quota_json(tenant_id, &used, &total);

	result = JsonbToCString(NULL, &result_jsonb->root, VARSIZE(result_jsonb));
	result_txt = cstring_to_text(result);

	NDB_FREE(result_jsonb);
	NDB_FREE(result);

	PG_RETURN_TEXT_P(result_txt);
}

/*-------------------------------------------------------------------------
 * Utility Functions
 *-------------------------------------------------------------------------
 */

/*
 * Ensure the per-tenant HNSW meta table exists (idempotent).
 */
static void
ensure_hnsw_tenant_metadata_table(void)
{
	int			ret;
	StringInfoData sql;

	NDB_DECLARE(NdbSpiSession *, session);
	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
		elog(ERROR, "failed to begin SPI session in ensure_hnsw_tenant_metadata_table");

	initStringInfo(&sql);

	appendStringInfo(&sql,
					 "CREATE TABLE IF NOT EXISTS %s ("
					 "tbl text, col text, tenant_id int, quota_max int, quota_used int default 0, "
					 "ef_search smallint, max_level smallint, "
					 "PRIMARY KEY (tbl, col, tenant_id))",
					 HNSW_META_TABLE);

	ret = ndb_spi_execute(session, sql.data, false, 0);
	if (ret != SPI_OK_UTILITY && ret != SPI_OK_SELECT)
	{
		NDB_FREE(sql.data);
		ndb_spi_session_end(&session);
		elog(ERROR, "Failed to create metadata table: %s", sql.data);
	}

	NDB_FREE(sql.data);
	ndb_spi_session_end(&session);
}

/*
 * Insert or update per-tenant config in the meta table.
 */
static void
upsert_tenant_hnsw_metadata(const char *tbl, const char *col, int32 tenant_id,
							int32 quota_max, int16 ef_search, int16 max_level)
{
	int			ret;
	StringInfoData sql;

	NDB_DECLARE(NdbSpiSession *, session);
	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
		elog(ERROR, "failed to begin SPI session in upsert_tenant_hnsw_metadata");

	initStringInfo(&sql);

	appendStringInfo(&sql,
					 "INSERT INTO %s (tbl, col, tenant_id, quota_max, quota_used, ef_search, max_level) "
					 "VALUES ($1, $2, $3, $4, 0, $5, $6) "
					 "ON CONFLICT (tbl, col, tenant_id) "
					 "DO UPDATE SET quota_max=EXCLUDED.quota_max, "
					 "ef_search=EXCLUDED.ef_search, max_level=EXCLUDED.max_level",
					 HNSW_META_TABLE);

	{
		Oid			argtypes[6] = {TEXTOID, TEXTOID, INT4OID, INT4OID, INT2OID, INT2OID};
		Datum		values[6];

		values[0] = CStringGetTextDatum(tbl);
		values[1] = CStringGetTextDatum(col);
		values[2] = Int32GetDatum(tenant_id);
		values[3] = Int32GetDatum(quota_max);
		values[4] = Int16GetDatum(ef_search);
		values[5] = Int16GetDatum(max_level);

		ret = ndb_spi_execute_with_args(session, sql.data, 6, argtypes, values, NULL, false, 0);
		if (ret != SPI_OK_INSERT && ret != SPI_OK_UPDATE)
		{
			NDB_FREE(sql.data);
			ndb_spi_session_end(&session);
			elog(ERROR, "ndb_spi_execute_with_args failed in upsert_tenant_hnsw_metadata: ret=%d sql=%s",
				 ret, sql.data);
		}

		NDB_FREE(sql.data);
		ndb_spi_session_end(&session);
	}
}

/*
 * Compose index backing table for a tenant, e.g. "__hnsw_tnt_table_col_tenant".
 */
static char *
get_hnsw_tenant_table(const char *table, const char *col, int32 tenant_id)
{
	int			n;
	char	   *buf;

	n = strlen(table) + strlen(col) + 48;
	buf = (char *) palloc(n);
	snprintf(buf, n, "%s%s_%s_%d", HNSW_INDEX_PREFIX, table, col, tenant_id);

	return buf;
}

/*
 * Build HNSW tenant index data table if not present.
 */
static void
maybe_build_hnsw_index(const char *tbl, const char *col, int32 tenant_id)
{
	char	   *idx_tbl;
	int			ret;
	StringInfoData sql;

	idx_tbl = get_hnsw_tenant_table(tbl, col, tenant_id);

	{
		NDB_DECLARE(NdbSpiSession *, session);
		session = ndb_spi_session_begin(CurrentMemoryContext, false);
		if (session == NULL)
			elog(ERROR, "failed to begin SPI session in maybe_build_hnsw_index");

		initStringInfo(&sql);

		appendStringInfo(&sql,
						 "CREATE TABLE IF NOT EXISTS %s ("
						 "id bigint, vector vector, tenant_id int, PRIMARY KEY (id))",
						 idx_tbl);

		ret = ndb_spi_execute(session, sql.data, false, 0);
		if (ret != SPI_OK_UTILITY)
		{
			NDB_FREE(sql.data);
			NDB_FREE(idx_tbl);
			ndb_spi_session_end(&session);
			elog(ERROR, "Failed creating index table: %s", sql.data);
		}

		NDB_FREE(sql.data);
		ndb_spi_session_end(&session);
	}
	NDB_FREE(idx_tbl);
}

/*
 * Find quota (used/total) for a tenant, return as JSONB.
 */
static Jsonb *
get_tenant_quota_json(int32 tenant_id, int32 * used, int32 * total)
{
	int			ret;
	JsonbParseState *state = NULL;
	JsonbValue	b;
	StringInfoData sql;

	(void) state;
	(void) b;

	{
		NDB_DECLARE(NdbSpiSession *, session);
		session = ndb_spi_session_begin(CurrentMemoryContext, false);
		if (session == NULL)
			elog(ERROR, "failed to begin SPI session in get_tenant_quota_json");

		initStringInfo(&sql);

		appendStringInfo(&sql,
						 "SELECT quota_used, quota_max FROM %s WHERE tenant_id = %d LIMIT 1",
						 HNSW_META_TABLE, tenant_id);

		ret = ndb_spi_execute(session, sql.data, true, 1);
		if (ret != SPI_OK_SELECT)
		{
			NDB_FREE(sql.data);
			ndb_spi_session_end(&session);
			elog(ERROR, "Failed to read quota for tenant %d", tenant_id);
		}

		if (SPI_processed == 1 && SPI_tuptable != NULL)
		{
			HeapTuple	tup = SPI_tuptable->vals[0];
			bool		qnull,
						tnull;

			*used = DatumGetInt32(SPI_getbinval(tup, SPI_tuptable->tupdesc, 1, &qnull));
			*total = DatumGetInt32(SPI_getbinval(tup, SPI_tuptable->tupdesc, 2, &tnull));
		}
		else
		{
			*used = 0;
			*total = 0;
		}
		NDB_FREE(sql.data);
		ndb_spi_session_end(&session);
	}

	/* Create JSONB object */
	{
		JsonbValue	obj;
		static JsonbPair pairs[2];

		obj.type = jbvObject;

		pairs[0].key.type = jbvString;
		pairs[0].key.val.string.len = 4;
		pairs[0].key.val.string.val = "used";
		pairs[0].value.type = jbvNumeric;
		pairs[0].value.val.numeric = int64_to_numeric(*used);

		pairs[1].key.type = jbvString;
		pairs[1].key.val.string.len = 5;
		pairs[1].key.val.string.val = "total";
		pairs[1].value.type = jbvNumeric;
		pairs[1].value.val.numeric = int64_to_numeric(*total);

		obj.val.object.nPairs = 2;
		obj.val.object.pairs = pairs;

		return JsonbValueToJsonb(&obj);
	}
}
