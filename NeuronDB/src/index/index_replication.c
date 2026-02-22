/*-------------------------------------------------------------------------
 *
 * index_replication.c
 *    Replication hooks for vector indexes (HNSW and IVF)
 *
 * Implements replication hooks for vector indexes to support
 * streaming replication with consistency guarantees.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/index/index_replication.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "neurondb_compat.h"
#include "neurondb_replication.h"
#include "fmgr.h"
#include "access/amapi.h"
#include "access/rmgr.h"
#include "access/xloginsert.h"
#include "access/xlog_internal.h"
#include "access/xlogreader.h"
#include "storage/bufmgr.h"
#include "storage/bufpage.h"
#include "utils/rel.h"
#include "utils/guc.h"
#include "catalog/pg_type.h"
#include "executor/spi.h"
#include "lib/stringinfo.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_spi_safe.h"
#include "neurondb_spi.h"
#include "neurondb_guc.h"

/* Custom WAL record for index replication: block number, insert/delete flag, index OID */
typedef struct NdbIndexReplWalRecord
{
	Oid			index_oid;
	BlockNumber blkno;
	uint8		is_insert;	/* 1 = insert, 0 = delete */
} NdbIndexReplWalRecord;

#define NDB_XLOG_INDEX_REPL_INFO	0x00

static void ndb_rmgr_redo(XLogReaderState *record);
static void ndb_rmgr_desc(StringInfo buf, XLogReaderState *record);
static const char *ndb_rmgr_identify(uint8 info);

static RmgrData ndb_rmgr = {
	.rm_name = "neurondb_idx_repl",
	.rm_redo = ndb_rmgr_redo,
	.rm_desc = ndb_rmgr_desc,
	.rm_identify = ndb_rmgr_identify,
	.rm_startup = NULL,
	.rm_cleanup = NULL,
	.rm_mask = NULL,
	.rm_decode = NULL
};

/* Use experimental ID for custom WAL; production should reserve a unique RmgrId */
#define NDB_RMGR_ID	RM_EXPERIMENTAL_ID

static void
ndb_rmgr_redo(XLogReaderState *record)
{
	/* No physical redo; record is for replication metadata / logical decoding only */
	(void) record;
}

static void
ndb_rmgr_desc(StringInfo buf, XLogReaderState *record)
{
	char	   *rec = XLogRecGetData(record);
	NdbIndexReplWalRecord *r = (NdbIndexReplWalRecord *) rec;

	appendStringInfo(buf, "index_oid %u blkno %u %s",
					 r->index_oid, r->blkno,
					 r->is_insert ? "insert" : "delete");
}

static const char *
ndb_rmgr_identify(uint8 info)
{
	if (info == NDB_XLOG_INDEX_REPL_INFO)
		return "NEURONDB_IDX_REPL";
	return NULL;
}

/* Emit a WAL record for index change (block, insert/delete, index OID) */
static void
ndb_log_index_repl_wal(Relation index, BlockNumber blkno, bool is_insert)
{
	NdbIndexReplWalRecord payload;

	payload.index_oid = RelationGetRelid(index);
	payload.blkno = blkno;
	payload.is_insert = is_insert ? 1 : 0;

	XLogBeginInsert();
	XLogRegisterData((char *) &payload, sizeof(payload));
	XLogInsert(NDB_RMGR_ID, NDB_XLOG_INDEX_REPL_INFO);
}

void
neurondb_replication_register_rmgr(void)
{
	RegisterCustomRmgr(NDB_RMGR_ID, &ndb_rmgr);
}

/*
 * Replication hook for HNSW index changes
 * Called after index modifications to ensure replication consistency
 */
void
neurondb_hnsw_replication_hook(Relation index, BlockNumber blkno, bool is_insert)
{
	if (!neurondb_replication_enabled())
		return;
	ndb_log_index_repl_wal(index, blkno, is_insert);
}

/*
 * Replication hook for IVF index changes
 */
void
neurondb_ivf_replication_hook(Relation index, BlockNumber blkno, bool is_insert)
{
	if (!neurondb_replication_enabled())
		return;
	ndb_log_index_repl_wal(index, blkno, is_insert);
}

/*
 * Check if replication is enabled
 */
bool
neurondb_replication_enabled(void)
{
	/* Check GUC setting */
	extern bool neurondb_enable_replication;
	
	if (neurondb_enable_replication)
		return true;
	
	/* Check if any replication slots exist for NeuronDB */
	return neurondb_has_replication_slots();
}

/*
 * Check if NeuronDB replication slots exist
 */
bool
neurondb_has_replication_slots(void)
{
	NdbSpiSession *session = NULL;
	int			ret;
	bool		has_slots = false;
	StringInfoData sql;

	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
		return false;

	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT COUNT(*) > 0 FROM pg_replication_slots "
					 "WHERE slot_name LIKE 'neurondb_%%'");

	ret = ndb_spi_execute(session, sql.data, true, 0);
	if (ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		bool		isnull;
		Datum		count_datum = SPI_getbinval(SPI_tuptable->vals[0],
												 SPI_tuptable->tupdesc,
												 1,
												 &isnull);

		if (!isnull)
		{
			int64		count = DatumGetInt64(count_datum);

			has_slots = (count > 0);
		}
	}

	pfree(sql.data);
	ndb_spi_session_end(&session);

	return has_slots;
}

/*
 * Get replication lag for an index
 * Returns lag in bytes, or -1 if not available
 */
int64
neurondb_get_replication_lag(Oid indexOid)
{
	NdbSpiSession *session = NULL;
	int			ret;
	int64		lag = -1;
	StringInfoData sql;
	char	   *index_name = NULL;

	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
		return -1;

	/* Get index name */
	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT relname FROM pg_class WHERE oid = %u",
					 indexOid);
	ret = ndb_spi_execute(session, sql.data, true, 0);
	if (ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		bool		isnull;
		Datum		name_datum = SPI_getbinval(SPI_tuptable->vals[0],
												 SPI_tuptable->tupdesc,
												 1,
												 &isnull);

		if (!isnull)
		{
			index_name = text_to_cstring(DatumGetTextP(name_datum));
		}
	}

	if (index_name == NULL)
	{
		pfree(sql.data);
		ndb_spi_session_end(&session);
		return -1;
	}

	/* Get replication lag */
	resetStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT (pg_current_wal_lsn() - last_lsn)::BIGINT "
					 "FROM neurondb_index_sync_state "
					 "WHERE source_index_name = $1 AND sync_status = 'active' "
					 "LIMIT 1");

	Oid			argtypes[1] = {TEXTOID};
	Datum		values[1];
	char		nulls[1] = {0};

	values[0] = CStringGetTextDatum(index_name);

	ret = ndb_spi_execute_with_args(session, sql.data, 1, argtypes, values, nulls, true, 0);
	if (ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		bool		isnull;
		Datum		lag_datum = SPI_getbinval(SPI_tuptable->vals[0],
											  SPI_tuptable->tupdesc,
											  1,
											  &isnull);

		if (!isnull)
		{
			lag = DatumGetInt64(lag_datum);
		}
	}

	pfree(index_name);
	pfree(sql.data);
	ndb_spi_session_end(&session);

	return lag;
}

/*
 * Verify index consistency between primary and replica index.
 * Compares reltuples and relpages via SPI; returns true if they match.
 */
bool
neurondb_verify_index_consistency(Oid indexOid, Oid replicaOid)
{
	NdbSpiSession *session = NULL;
	StringInfoData sql;
	Oid			argtypes[1] = {OIDOID};
	Datum		values[1];
	char		nulls[1] = {' '};
	int			ret;
	int64		tuples1 = -1;
	int64		tuples2 = -1;
	int32		pages1 = -1;
	int32		pages2 = -1;
	bool		consistent = false;

	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
		return false;

	/* Fetch reltuples and relpages for the primary index */
	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT reltuples::bigint, relpages FROM pg_class WHERE oid = $1");
	values[0] = ObjectIdGetDatum(indexOid);
	ret = ndb_spi_execute_with_args(session, sql.data, 1, argtypes, values, nulls, true, 1);
	if (ret != SPI_OK_SELECT || SPI_processed == 0)
	{
		pfree(sql.data);
		ndb_spi_session_end(&session);
		return false;
	}
	{
		HeapTuple	tup = SPI_tuptable->vals[0];
		TupleDesc	td = SPI_tuptable->tupdesc;
		bool		n1 = false;
		bool		n2 = false;

		tuples1 = DatumGetInt64(SPI_getbinval(tup, td, 1, &n1));
		pages1 = DatumGetInt32(SPI_getbinval(tup, td, 2, &n2));
		if (n1 || n2)
		{
			pfree(sql.data);
			ndb_spi_session_end(&session);
			return false;
		}
	}

	/* Fetch reltuples and relpages for the replica index */
	values[0] = ObjectIdGetDatum(replicaOid);
	ret = ndb_spi_execute_with_args(session, sql.data, 1, argtypes, values, nulls, true, 1);
	pfree(sql.data);
	if (ret != SPI_OK_SELECT || SPI_processed == 0)
	{
		ndb_spi_session_end(&session);
		return false;
	}
	{
		HeapTuple	tup = SPI_tuptable->vals[0];
		TupleDesc	td = SPI_tuptable->tupdesc;
		bool		n1 = false;
		bool		n2 = false;

		tuples2 = DatumGetInt64(SPI_getbinval(tup, td, 1, &n1));
		pages2 = DatumGetInt32(SPI_getbinval(tup, td, 2, &n2));
		if (n1 || n2)
		{
			ndb_spi_session_end(&session);
			return false;
		}
	}

	ndb_spi_session_end(&session);
	consistent = (tuples1 == tuples2 && pages1 == pages2);
	return consistent;
}

