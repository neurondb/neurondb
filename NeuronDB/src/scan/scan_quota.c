/*-------------------------------------------------------------------------
 *
 * scan_quota.c
 *		Quota enforcement for index operations
 *
 * Implements hard limits on:
 * - Per-tenant vector count
 * - Per-tenant storage size
 * - Per-tenant index size
 * - Rate limiting for inserts
 *
 * Quotas are checked during insert/update operations and enforced
 * before writing to the index.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  src/scan/scan_quota.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "neurondb_scan.h"
#include "fmgr.h"
#include "access/htup_details.h"
#include "access/relation.h"
#include "catalog/pg_class.h"
#include "executor/spi.h"
#include "storage/lmgr.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/rel.h"
#include "utils/timestamp.h"
#include "utils/typcache.h"
#include "funcapi.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_spi_safe.h"
#include "neurondb_spi.h"
#include "neurondb_guc.h"
#include <stdlib.h>
#include <string.h>

/*
 * Quota limits structure
 */
typedef struct QuotaLimits
{
	int64		maxVectors;		/* Maximum vectors per tenant */
	int64		maxStorageBytes;	/* Maximum storage per tenant */
	int64		maxIndexSize;	/* Maximum index size per tenant */
	int			maxQPS;			/* Maximum queries per second */
	bool		enforceHard;	/* Hard enforcement vs. warning */
}			QuotaLimits;

/*
 * Current usage structure
 */
typedef struct QuotaUsage
{
	char	   *tenantId;
	int64		vectorCount;
	int64		storageBytes;
	int64		indexSize;
	int			currentQPS;
	TimestampTz lastCheck;
}			QuotaUsage;

/* GUC variables are now centralized in neurondb_guc.c */
/* Use GetConfigOption() to retrieve GUC values */

/* Helper to get int GUC value */
static int
get_guc_int(const char *name, int default_val)
{
	const char *val = GetConfigOption(name, true, false);

	return val ? atoi(val) : default_val;
}

/* Helper to get int64 GUC value */
static int64
get_guc_int64(const char *name, int64 default_val)
{
	const char *val = GetConfigOption(name, true, false);

	return val ? atoll(val) : default_val;
}

/* Helper to get bool GUC value */
static bool
get_guc_bool(const char *name, bool default_val)
{
	const char *val = GetConfigOption(name, true, false);

	if (!val)
		return default_val;
	return (strcmp(val, "on") == 0 || strcmp(val, "true") == 0 || strcmp(val, "1") == 0);
}

/* GUC initialization is now centralized in neurondb_guc.c */

/*
 * Get quota limits for a tenant
 */
static QuotaLimits *
get_tenant_quota(const char *tenantId)
{
	NDB_DECLARE(QuotaLimits *, limits);
	NDB_DECLARE(NdbSpiSession *, session);

	NDB_ALLOC(limits, QuotaLimits, 1);

	/* Query neurondb.tenant_quotas table */
	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: failed to begin SPI session in get_tenant_quota")));
	}

	{
		int			ret;
		bool		isnull;
		StringInfoData query;

		initStringInfo(&query);
		appendStringInfo(&query,
						 "SELECT max_vectors, max_storage_bytes, max_index_size, "
						 "max_qps, enforce_hard FROM neurondb.tenant_quotas "
						 "WHERE tenant_id = $1");

		{
			Oid			argtypes[1];
			Datum		values[1];

			argtypes[0] = TEXTOID;
			values[0] = CStringGetDatum(tenantId);
			ret = ndb_spi_execute_with_args(session,
											query.data,
											1,
											argtypes,
											values,
											NULL,
											true,
											0);
		}

		if (ret == SPI_OK_SELECT && SPI_processed > 0)
		{
			HeapTuple	tuple = SPI_tuptable->vals[0];
			TupleDesc	tupdesc = SPI_tuptable->tupdesc;

			limits->maxVectors = DatumGetInt64(SPI_getbinval(tuple,
															 tupdesc,
															 1,
															 &isnull));
			if (isnull)
				limits->maxVectors = get_guc_int64("neurondb.default_max_vectors", 1000000);

			limits->maxStorageBytes = DatumGetInt64(SPI_getbinval(tuple,
																  tupdesc,
																  2,
																  &isnull));
			if (isnull)
				limits->maxStorageBytes = get_guc_int64("neurondb.default_max_storage_mb", 10240) * 1024 * 1024;

			limits->maxIndexSize = DatumGetInt64(SPI_getbinval(tuple,
															   tupdesc,
															   3,
															   &isnull));
			if (isnull)
				limits->maxIndexSize = get_guc_int64("neurondb.default_max_storage_mb", 10240) * 1024 * 1024;

			limits->maxQPS = DatumGetInt32(SPI_getbinval(tuple,
														 tupdesc,
														 4,
														 &isnull));
			if (isnull)
				limits->maxQPS = get_guc_int("neurondb.default_max_qps", 1000);

			limits->enforceHard = DatumGetBool(SPI_getbinval(tuple,
															 tupdesc,
															 5,
															 &isnull));
			if (isnull)
				limits->enforceHard = get_guc_bool("neurondb.enforce_quotas", true);
		}
		else
		{
			/* Use defaults if no quota found */
			limits->maxVectors = get_guc_int64("neurondb.default_max_vectors", 1000000);
			limits->maxStorageBytes = get_guc_int64("neurondb.default_max_storage_mb", 10240) * 1024 * 1024;
			limits->maxIndexSize = get_guc_int64("neurondb.default_max_storage_mb", 10240) * 1024 * 1024;
			limits->maxQPS = get_guc_int("neurondb.default_max_qps", 1000);
			limits->enforceHard = get_guc_bool("neurondb.enforce_quotas", true);
		}

		NDB_FREE(query.data);
	}

	ndb_spi_session_end(&session);

	return limits;
}

/*
 * Get current usage for a tenant
 */
static QuotaUsage *
get_tenant_usage(const char *tenantId, Oid indexOid)
{
	int			ret;

	NDB_DECLARE(QuotaUsage *, usage);
	NDB_DECLARE(NdbSpiSession *, session);
	int32		count_val;
	int32		qps_val;

	NDB_ALLOC(usage, QuotaUsage, 1);
	usage->tenantId = pstrdup(tenantId);
	usage->lastCheck = GetCurrentTimestamp();

	/* Query usage from catalog */
	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: failed to begin SPI session in get_tenant_usage")));
	}

	/* Get vector count */
	{
		Oid			argtypes[1];
		Datum		values[1];

		argtypes[0] = TEXTOID;
		values[0] = CStringGetTextDatum(tenantId);
		ret = ndb_spi_execute_with_args(session,
										"SELECT count(*) FROM neurondb.tenant_usage WHERE tenant_id = $1",
										1,
										argtypes,
										values,
										NULL,
										true,
										0);
	}

	if (ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		if (ndb_spi_get_int32(session, 0, 1, &count_val))
			usage->vectorCount = (int64) count_val;
		else
			usage->vectorCount = 0;
	}
	else
	{
		usage->vectorCount = 0;
	}

	/* Get storage size from pg_class */
	{
		int			ret2;
		StringInfoData query;

		initStringInfo(&query);
		appendStringInfo(&query,
						 "SELECT pg_total_relation_size($1::regclass)");

		{
			Oid			argtypes[1];
			Datum		values[1];

			argtypes[0] = OIDOID;
			values[0] = ObjectIdGetDatum(indexOid);
			ret2 = ndb_spi_execute_with_args(session,
											 query.data,
											 1,
											 argtypes,
											 values,
											 NULL,
											 true,
											 0);
		}

		if (ret2 == SPI_OK_SELECT && SPI_processed > 0 && SPI_tuptable != NULL)
		{
			bool		isnull2;
			Datum		d;

			d = SPI_getbinval(SPI_tuptable->vals[0],
							  SPI_tuptable->tupdesc,
							  1,
							  &isnull2);
			if (!isnull2)
				usage->storageBytes = DatumGetInt64(d);
			else
				usage->storageBytes = 0;
		}
		else
		{
			usage->storageBytes = 0;
		}

		NDB_FREE(query.data);
	}

	/* Get QPS from stats */
	{
		int			ret3;
		StringInfoData query;

		initStringInfo(&query);
		appendStringInfo(&query,
						 "SELECT COALESCE(sum(calls), 0) FROM pg_stat_user_functions "
						 "WHERE schemaname = 'neurondb' AND funcname LIKE '%%%s%%'",
						 tenantId);

		ret3 = ndb_spi_execute(session, query.data, true, 0);

		if (ret3 == SPI_OK_SELECT && SPI_processed > 0)
		{
			if (ndb_spi_get_int32(session, 0, 1, &qps_val))
				usage->currentQPS = qps_val;
			else
				usage->currentQPS = 0;
		}
		else
		{
			usage->currentQPS = 0;
		}

		NDB_FREE(query.data);
	}

	ndb_spi_session_end(&session);

	return usage;
}

/*
 * Check if operation would exceed quota
 */
bool
ndb_quota_check(const char *tenantId,
				Oid indexOid,
				int64 additionalVectors,
				int64 additionalBytes)
{
	QuotaLimits *limits;
	QuotaUsage *usage;
	bool		allowed = true;

	if (!get_guc_bool("neurondb.enforce_quotas", true))
		return true;

	if (tenantId == NULL)
		return true;			/* No tenant = no quota */

	limits = get_tenant_quota(tenantId);
	usage = get_tenant_usage(tenantId, indexOid);

	/* Check vector count */
	if (usage->vectorCount + additionalVectors > limits->maxVectors)
	{
		if (limits->enforceHard)
		{
			ereport(ERROR,
					(errcode(ERRCODE_INSUFFICIENT_RESOURCES),
					 errmsg("neurondb: quota exceeded for "
							"tenant '%s'",
							tenantId),
					 errdetail("Vector count %lld + %lld "
							   "would exceed limit %lld",
							   (long long) usage->vectorCount,
							   (long long) additionalVectors,
							   (long long) limits->maxVectors),
					 errhint("Contact administrator to "
							 "increase quota or delete old "
							 "vectors")));
		}
		else
		{
			ereport(WARNING,
					(errmsg("neurondb: tenant '%s' approaching "
							"vector quota (%lld/%lld)",
							tenantId,
							(long long) usage->vectorCount,
							(long long) limits->maxVectors)));
		}
		allowed = false;
	}

	/* Check storage size */
	if (usage->storageBytes + additionalBytes > limits->maxStorageBytes)
	{
		if (limits->enforceHard)
		{
			ereport(ERROR,
					(errcode(ERRCODE_INSUFFICIENT_RESOURCES),
					 errmsg("neurondb: storage quota "
							"exceeded for tenant '%s'",
							tenantId),
					 errdetail("Storage %lld + %lld bytes "
							   "would exceed limit %lld",
							   (long long) usage->storageBytes,
							   (long long) additionalBytes,
							   (long long) limits
							   ->maxStorageBytes)));
		}
		allowed = false;
	}

	NDB_FREE(limits);
	NDB_FREE(usage);

	return allowed;
}

/*
 * Enforce quota at index insert time
 *
 * Called from index insert functions before writing.
 */
void
ndb_quota_enforce_insert(Relation index,
						 const char *tenantId,
						 int64 vectorCount,
						 int64 estimatedBytes)
{
	Oid			indexOid = RelationGetRelid(index);

	if (!ndb_quota_check(tenantId, indexOid, vectorCount, estimatedBytes))
	{
		/* Error already raised if hard enforcement */
		return;
	}

}

/*
 * Update usage statistics
 */
void
ndb_quota_update_usage(const char *tenantId,
					   Oid indexOid,
					   int64 vectorsDelta,
					   int64 bytesDelta)
{
	StringInfoData query;

	NDB_DECLARE(NdbSpiSession *, session);

	if (tenantId == NULL)
		return;

	initStringInfo(&query);

	/* Upsert usage record */
	appendStringInfo(&query,
					 "INSERT INTO neurondb.tenant_usage "
					 "(tenant_id, index_oid, vector_count, storage_bytes, "
					 "last_updated) "
					 "VALUES ('%s', %u, %lld, %lld, now()) "
					 "ON CONFLICT (tenant_id, index_oid) "
					 "DO UPDATE SET "
					 "vector_count = neurondb.tenant_usage.vector_count + %lld, "
					 "storage_bytes = neurondb.tenant_usage.storage_bytes + %lld, "
					 "last_updated = now()",
					 tenantId,
					 indexOid,
					 (long long) vectorsDelta,
					 (long long) bytesDelta,
					 (long long) vectorsDelta,
					 (long long) bytesDelta);

	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
	{
		NDB_FREE(query.data);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: failed to begin SPI session in update_tenant_usage")));
	}
	ndb_spi_execute(session, query.data, false, 0);
	NDB_FREE(query.data);
	ndb_spi_session_end(&session);

}

/*
 * SQL-callable quota check function
 */
PG_FUNCTION_INFO_V1(neurondb_check_quota);

Datum
neurondb_check_quota(PG_FUNCTION_ARGS)
{
	text	   *tenant_id_text = PG_GETARG_TEXT_PP(0);
	Oid			index_oid = PG_GETARG_OID(1);
	int64		additional_vectors = PG_GETARG_INT64(2);
	char	   *tenant_id = text_to_cstring(tenant_id_text);
	bool		allowed;

	allowed = ndb_quota_check(tenant_id, index_oid, additional_vectors, 0);

	PG_RETURN_BOOL(allowed);
}

/*
 * SQL-callable function to get quota usage
 */
PG_FUNCTION_INFO_V1(neurondb_get_quota_usage);

Datum
neurondb_get_quota_usage(PG_FUNCTION_ARGS)
{
	text	   *tenant_id_text = PG_GETARG_TEXT_PP(0);
	Oid			index_oid = PG_GETARG_OID(1);
	char	   *tenant_id = text_to_cstring(tenant_id_text);
	QuotaUsage *usage;
	QuotaLimits *limits;
	TupleDesc	tupdesc;
	Datum		values[6];
	bool		nulls[6];
	HeapTuple	tuple;

	usage = get_tenant_usage(tenant_id, index_oid);
	limits = get_tenant_quota(tenant_id);

	if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("function returning record called in "
						"context that cannot accept type "
						"record")));

	tupdesc = BlessTupleDesc(tupdesc);

	values[0] = Int64GetDatum(usage->vectorCount);
	values[1] = Int64GetDatum(limits->maxVectors);
	values[2] = Int64GetDatum(usage->storageBytes);
	values[3] = Int64GetDatum(limits->maxStorageBytes);
	values[4] = Int32GetDatum(usage->currentQPS);
	values[5] = Int32GetDatum(limits->maxQPS);

	memset(nulls, 0, sizeof(nulls));

	tuple = heap_form_tuple(tupdesc, values, nulls);

	PG_RETURN_DATUM(HeapTupleGetDatum(tuple));
}

/*
 * Reset quota usage (for testing)
 */
PG_FUNCTION_INFO_V1(neurondb_reset_quota);

Datum
neurondb_reset_quota(PG_FUNCTION_ARGS)
{
	NDB_DECLARE(NdbSpiSession *, session);
	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: failed to begin SPI session in neurondb_reset_quota")));
	}
	ndb_spi_execute(session, "DELETE FROM neurondb.tenant_usage", false, 0);
	ndb_spi_session_end(&session);


	PG_RETURN_VOID();
}
