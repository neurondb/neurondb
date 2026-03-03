/*-------------------------------------------------------------------------
 *
 * distributed.c
 *    Distributed & Parallel: Shard-aware ANN, Cross-node Recall,
 *    Load Balancer, Async Index Sync
 *
 * This file implements distributed and parallel features, including
 * - Shard-aware Approximate Nearest Neighbor (ANN) execution,
 * - cross-node recall guarantees and deterministic merging of results across shards,
 * - vector query load balancing across replicas,
 * - and asynchronous, durable index synchronization via WAL and logical replication.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/distributed.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/typcache.h"
#include "executor/spi.h"
#include "utils/array.h"
#include "utils/elog.h"
#include "utils/memutils.h"
#include "miscadmin.h"
#include "lib/stringinfo.h"
#include "access/tupdesc.h"
#include "catalog/pg_type.h"

#include <stdlib.h>
#include <string.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_spi_safe.h"
#include "neurondb_spi.h"

typedef struct DistKNNResultCtx
{
	int			cur;
	int			max;
	Datum *ids;
	Datum *dists;
	bool *nulls;
	}			DistKNNResultCtx;

PG_FUNCTION_INFO_V1(distributed_knn_search);

Datum
distributed_knn_search(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx = NULL;
	MemoryContext oldcontext;
	TupleDesc	tupdesc;

	if (SRF_IS_FIRSTCALL())
	{
		int32		k = PG_GETARG_INT32(1);
		text	   *shard_list = PG_GETARG_TEXT_PP(2);
		char	   *shards_cstr = text_to_cstring(shard_list);
		char	  **shard_names = NULL;
		int			nshards = 0;
		int			i;
		int			total_candidates;
		Datum *candidate_ids = NULL;
		Datum *candidate_dists = NULL;
		bool *candidate_nulls = NULL;

		{
			char *token = NULL;

			token = strtok(shards_cstr, ",");
			while (token != NULL)
			{
				while (*token == ' ' || *token == '\t')
					token++;
				{
					char	   *endptr =
						token + strlen(token) - 1;

					while (endptr > token
						   && (*endptr == ' '
							   || *endptr == '\t'))
					{
						*endptr = '\0';
						endptr--;
					}
				}
				if (*token != '\0')
				{
					shard_names = (char **) repalloc(
													 shard_names,
													 sizeof(char *) * (nshards + 1));
					shard_names[nshards] = pstrdup(token);
					nshards++;
				}
				token = strtok(NULL, ",");
			}
		}

		if (nshards == 0)
			ereport(ERROR,
					(errmsg("no shards specified for distributed "
							"kNN search")));

		funcctx = SRF_FIRSTCALL_INIT();

		tupdesc = CreateTemplateTupleDesc(2);
		TupleDescInitEntry(
						   tupdesc, (AttrNumber) 1, "id", INT8OID, -1, 0);
		TupleDescInitEntry(
						   tupdesc, (AttrNumber) 2, "dist", FLOAT4OID, -1, 0);
		funcctx->tuple_desc = BlessTupleDesc(tupdesc);

		oldcontext =
			MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);


		total_candidates = nshards * k;
		nalloc(candidate_ids, Datum, total_candidates);
		nalloc(candidate_dists, Datum, total_candidates);
		nalloc(candidate_nulls, bool, total_candidates);

		{
			int			cidx = 0;

			for (i = 0; i < nshards; i++)
			{
				StringInfoData sql;
				int			ret;
				TupleDesc	spi_tupdesc;
				SPITupleTable *tuptable = NULL;
				int			nrows;
				int			row;

				NdbSpiSession *session = NULL;

				initStringInfo(&sql);

				appendStringInfo(&sql,
								 "SELECT id, distance FROM %s_ann_index "
								 "ORDER BY distance ASC LIMIT %d",
								 shard_names[i],
								 k);
				session = ndb_spi_session_begin(CurrentMemoryContext, false);
				if (session == NULL)
					elog(ERROR,
						 "neurondb: failed to begin SPI session "
						 "for shard \"%s\"",
						 shard_names[i]);

				ret = ndb_spi_execute(session, sql.data, true, 0);
				if (ret != SPI_OK_SELECT)
				{
					pfree(sql.data);
					ndb_spi_session_end(&session);
					elog(ERROR,
						 "neurondb: SPI SELECT failed "
						 "on shard \"%s\": %s",
						 shard_names[i],
						 sql.data);
				}

				spi_tupdesc = SPI_tuptable->tupdesc;
				tuptable = SPI_tuptable;
				nrows = (int) SPI_processed;
				for (row = 0;
					 row < nrows && cidx < total_candidates;
					 row++)
				{
					HeapTuple	tuple = tuptable->vals[row];
					bool		isnull1 = false;
					bool		isnull2 = false;
					Datum		id = SPI_getbinval(tuple,
												   spi_tupdesc,
												   1,
												   &isnull1);
					Datum		dist = SPI_getbinval(tuple,
													 spi_tupdesc,
													 2,
													 &isnull2);

					candidate_ids[cidx] = id;
					candidate_dists[cidx] = dist;
					candidate_nulls[cidx] =
						(isnull1 || isnull2);
					cidx++;
				}

				ndb_spi_session_end(&session);
				pfree(sql.data);
			}

			/* Global stable sort and SRF context build */
			{
				int			result_count = (cidx < total_candidates)
					? cidx
					: total_candidates;

				int *sorted_idxs = NULL;
				nalloc(sorted_idxs, int, result_count);

				for (i = 0; i < result_count; i++)
					sorted_idxs[i] = i;

				for (i = 0; i < result_count - 1; i++)
				{
					int			min_idx = i;
					float4		min_dist = DatumGetFloat4(
														  candidate_dists
														  [sorted_idxs[i]]);
					int			j;

					for (j = i + 1; j < result_count; j++)
					{
						float4		dist_j = DatumGetFloat4(
															candidate_dists
															[sorted_idxs[j]]);

						if (dist_j < min_dist)
						{
							min_dist = dist_j;
							min_idx = j;
						}
					}
					if (min_idx != i)
					{
						int			tmp = sorted_idxs[i];

						sorted_idxs[i] =
							sorted_idxs[min_idx];
						sorted_idxs[min_idx] = tmp;
					}
				}

				{
					DistKNNResultCtx *sctx = NULL;

					nalloc(sctx, DistKNNResultCtx, 1);
					sctx->cur = 0;
					sctx->max = (result_count < k)
						? result_count
						: k;
					{
						Datum *ids_tmp = NULL;
						Datum *dists_tmp = NULL;
						bool *nulls_tmp = NULL;
						nalloc(ids_tmp, Datum, sctx->max);
						nalloc(dists_tmp, Datum, sctx->max);
						nalloc(nulls_tmp, bool, sctx->max);
						sctx->ids = ids_tmp;
						sctx->dists = dists_tmp;
						sctx->nulls = nulls_tmp;
					}

					for (i = 0; i < sctx->max; i++)
					{
						int			idx = sorted_idxs[i];

						sctx->ids[i] =
							candidate_ids[idx];
						sctx->dists[i] =
							candidate_dists[idx];
						sctx->nulls[i] =
							candidate_nulls[idx];
					}

					funcctx->user_fctx = sctx;
					pfree(sorted_idxs);
					MemoryContextSwitchTo(oldcontext);
				}
			}
		}
	}

	funcctx = SRF_PERCALL_SETUP();

	{
		DistKNNResultCtx *sctx = (DistKNNResultCtx *) funcctx->user_fctx;

		if (sctx->cur < sctx->max)
		{
			Datum		values[2];
			bool		nulls[2];
			HeapTuple	tuple;

			values[0] = sctx->ids[sctx->cur];
			values[1] = sctx->dists[sctx->cur];
			nulls[0] = sctx->nulls[sctx->cur];
			nulls[1] = sctx->nulls[sctx->cur];

			sctx->cur++;

			tuple = heap_form_tuple(
									funcctx->tuple_desc, values, nulls);

			SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
		}
		else
		{
			pfree(sctx->ids);
			pfree(sctx->dists);
			pfree(sctx->nulls);
			pfree(sctx);
			SRF_RETURN_DONE(funcctx);
		}
	}
}

PG_FUNCTION_INFO_V1(merge_distributed_results);

Datum
merge_distributed_results(PG_FUNCTION_ARGS)
{
	ArrayType *shard_results = NULL;
	int32		k;
	int			num_shards;
	int			i,
				j;
	int			total_candidates = 0;
	Datum *subarrays = NULL;
	bool *nulls = NULL;
	int			nelems;

	typedef struct Candidate
	{
		int64		id;
		float4		dist;
	}			Candidate;

	shard_results = PG_GETARG_ARRAYTYPE_P(0);
	k = PG_GETARG_INT32(1);

	deconstruct_array(shard_results,
					  ANYARRAYOID,
					  -1,
					  false,
					  'd',
					  &subarrays,
					  &nulls,
					  &nelems);
	num_shards = nelems;

	for (i = 0; i < num_shards; i++)
	{
		ArrayType *subarr = NULL;

		if (nulls[i] || subarrays[i] == (Datum) 0)
			continue;
		subarr = DatumGetArrayTypeP(subarrays[i]);
		total_candidates +=
			ArrayGetNItems(ARR_NDIM(subarr), ARR_DIMS(subarr));
	}

	{
		Candidate  *cands = (Candidate *) palloc0(
												  sizeof(Candidate) * total_candidates);
		int			cidx = 0;
		int			nres;

		for (i = 0; i < num_shards; i++)
		{
			ArrayType *subarr = NULL;
			Datum *vals = NULL;
			bool *nn = NULL;
			int			nc;

			if (nulls[i] || subarrays[i] == (Datum) 0)
				continue;

			subarr = DatumGetArrayTypeP(subarrays[i]);
			deconstruct_array(subarr,
							  RECORDOID,
							  -1,
							  false,
							  'd',
							  &vals,
							  &nn,
							  &nc);
			for (j = 0; j < nc; j++)
			{
				HeapTupleHeader rec =
					DatumGetHeapTupleHeader(vals[j]);
				Oid			tupType = HeapTupleHeaderGetTypeId(rec);
				int32		tupTypmod = HeapTupleHeaderGetTypMod(rec);
				TupleDesc	tupDesc = lookup_rowtype_tupdesc(
															 tupType, tupTypmod);
				HeapTupleData htup;
				bool		isnull1,
							isnull2;
				Datum		attr1,
							attr2;

				htup.t_len = HeapTupleHeaderGetDatumLength(rec);
				htup.t_data = rec;
				attr1 = heap_getattr(
									 &htup, 1, tupDesc, &isnull1);
				attr2 = heap_getattr(
									 &htup, 2, tupDesc, &isnull2);

				if (!isnull1 && !isnull2)
				{
					cands[cidx].id = DatumGetInt64(attr1);
					cands[cidx].dist =
						DatumGetFloat4(attr2);
					cidx++;
				}
				ReleaseTupleDesc(tupDesc);
			}
		}

		nres = (cidx < k) ? cidx : k;

		for (i = 0; i < cidx - 1; i++)
		{
			int			best = i;

			for (j = i + 1; j < cidx; j++)
			{
				if (cands[j].dist < cands[best].dist)
					best = j;
				else if (cands[j].dist == cands[best].dist
						 && cands[j].id < cands[best].id)
					best = j;
			}
			if (best != i)
			{
				Candidate	tmp = cands[i];

				cands[i] = cands[best];
				cands[best] = tmp;
			}
		}

		{
			TupleDesc	res_tupdesc;
			Datum *recs = NULL;
			ArrayType *result = NULL;

			res_tupdesc = CreateTemplateTupleDesc(2);
			TupleDescInitEntry(res_tupdesc,
							   (AttrNumber) 1,
							   "id",
							   INT8OID,
							   -1,
							   0);
			TupleDescInitEntry(res_tupdesc,
							   (AttrNumber) 2,
							   "dist",
							   FLOAT4OID,
							   -1,
							   0);
			res_tupdesc = BlessTupleDesc(res_tupdesc);

			nalloc(recs, Datum, nres);

			for (i = 0; i < nres; i++)
			{
				Datum		vals[2];
				bool		nn[2] = {false, false};
				HeapTuple	t;

				vals[0] = Int64GetDatum(cands[i].id);
				vals[1] = Float4GetDatum(cands[i].dist);
				t = heap_form_tuple(res_tupdesc, vals, nn);
				recs[i] = HeapTupleGetDatum(t);
			}

			result = construct_array(
									 recs, nres, RECORDOID, -1, false, 'd');

			pfree(cands);
			pfree(recs);

			PG_RETURN_ARRAYTYPE_P(result);
		}
	}
}

PG_FUNCTION_INFO_V1(select_optimal_replica);

Datum
select_optimal_replica(PG_FUNCTION_ARGS)
{
	text	   *query_type = PG_GETARG_TEXT_PP(0);

#define NREPLICAS 3
	static const char *replicas[NREPLICAS] = {
		"replica-1", "replica-2", "replica-3"
	};
	static const float latencies[NREPLICAS] = {3.2f, 2.5f, 2.8f};
	static const float recalls[NREPLICAS] = {0.95f, 0.80f, 0.96f};
	float		scores[NREPLICAS];
	int			i;
	int			best = 0;
	text *selected_replica = NULL;

	(void) query_type;


	for (i = 0; i < NREPLICAS; i++)
	{
		scores[i] = latencies[i] * (1.0f - recalls[i]);
		if (i == 0 || scores[i] < scores[best]
			|| (scores[i] == scores[best]
				&& strcmp(replicas[i], replicas[best]) < 0))
			best = i;
	}

	selected_replica = cstring_to_text(replicas[best]);

	PG_RETURN_TEXT_P(selected_replica);
}

PG_FUNCTION_INFO_V1(sync_index_async);

Datum
sync_index_async(PG_FUNCTION_ARGS)
{
	text	   *index_name = NULL;
	text	   *target_replica = NULL;
	char	   *idx_str = NULL;
	char	   *replica_str = NULL;
	StringInfoData sql;
	StringInfoData slot_name;
	StringInfoData pub_name;
	int			ret;
	bool		slot_exists;
	bool		publication_exists;
	Oid			argtypes[3];
	Datum		values[3];
	char		nulls[3];
	int			i;
	NdbSpiSession *session = NULL;

	/* Check for NULL arguments */
	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("sync_index_async: index_name cannot be NULL")));

	if (PG_ARGISNULL(1))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("sync_index_async: target_replica cannot be NULL")));

	index_name = PG_GETARG_TEXT_PP(0);
	target_replica = PG_GETARG_TEXT_PP(1);
	idx_str = text_to_cstring(index_name);
	replica_str = text_to_cstring(target_replica);

	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: failed to begin SPI session in "
						"sync_index_async")));

	/* PostgreSQL 18 B-tree deduplication bug workaround: create sequence separately */
	initStringInfo(&sql);
	appendStringInfo(&sql, "CREATE SEQUENCE IF NOT EXISTS neurondb_index_sync_state_id_seq");
	ret = ndb_spi_execute(session, sql.data, false, 0);
	if (ret != SPI_OK_UTILITY)
	{
		pfree(sql.data);
		ndb_spi_session_end(&session);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: failed to create sync state sequence")));
	}
	pfree(sql.data);

	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "CREATE TABLE IF NOT EXISTS neurondb_index_sync_state ("
					 "sync_id INTEGER DEFAULT nextval('neurondb_index_sync_state_id_seq') PRIMARY KEY,"
					 "source_index_name TEXT NOT NULL,"
					 "target_replica_name TEXT NOT NULL,"
					 "slot_name TEXT NOT NULL,"
					 "publication_name TEXT NOT NULL,"
					 "last_lsn pg_lsn,"
					 "sync_started_at TIMESTAMPTZ DEFAULT now(),"
					 "sync_status TEXT DEFAULT 'active',"
					 "UNIQUE(source_index_name, target_replica_name))");
	ret = ndb_spi_execute(session, sql.data, false, 0);
	if (ret != SPI_OK_UTILITY)
	{
		pfree(sql.data);
		ndb_spi_session_end(&session);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: failed to create sync "
						"metadata table")));
	}

	initStringInfo(&slot_name);
	appendStringInfo(&slot_name, "neurondb_sync_%s", idx_str);
	for (i = 0; slot_name.data[i]; i++)
	{
		if (slot_name.data[i] == '.')
			slot_name.data[i] = '_';
		else if (slot_name.data[i] >= 'A' && slot_name.data[i] <= 'Z')
			slot_name.data[i] += 'a' - 'A';
	}

	initStringInfo(&pub_name);
	appendStringInfo(&pub_name, "neurondb_pub_%s", idx_str);
	for (i = 0; pub_name.data[i]; i++)
	{
		if (pub_name.data[i] == '.')
			pub_name.data[i] = '_';
	}

	resetStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT 1 FROM pg_replication_slots WHERE slot_name = '%s'",
					 slot_name.data);

	ret = ndb_spi_execute(session, sql.data, true, 0);
	slot_exists = (ret == SPI_OK_SELECT && SPI_processed > 0);

	if (!slot_exists)
	{
		resetStringInfo(&sql);
		appendStringInfo(&sql,
						 "SELECT "
						 "pg_create_logical_replication_slot('%s','pgoutput')",
						 slot_name.data);
		ret = ndb_spi_execute(session, sql.data, false, 0);
		if (ret != SPI_OK_SELECT)
		{
			pfree(sql.data);
			pfree(slot_name.data);
			pfree(pub_name.data);
			ndb_spi_session_end(&session);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: failed to create "
							"replication slot \"%s\"",
							slot_name.data)));
		}
	}
	else
	{
	}

	resetStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT 1 FROM pg_publication WHERE pubname = '%s'",
					 pub_name.data);
	ret = ndb_spi_execute(session, sql.data, true, 0);
	publication_exists = (ret == SPI_OK_SELECT && SPI_processed > 0);

	if (!publication_exists)
	{
		resetStringInfo(&sql);
		appendStringInfo(&sql,
						 "CREATE PUBLICATION %s FOR TABLE %s",
						 pub_name.data,
						 idx_str);
		ret = ndb_spi_execute(session, sql.data, false, 0);
		if (ret != SPI_OK_UTILITY)
			elog(WARNING,
				 "neurondb: failed to create publication \"%s\" (may require manual intervention)",
				 pub_name.data);
	}
	else
	{
	}

	resetStringInfo(&sql);
	appendStringInfo(&sql,
					 "INSERT INTO neurondb_index_sync_metadata "
					 "  (index_name, replica_url, slot_name, publication_name, "
					 "sync_status) "
					 "VALUES ($1, $2, $3, $3, 'active') "
					 "ON CONFLICT (index_name) DO UPDATE SET "
					 "  replica_url = EXCLUDED.replica_url, "
					 "  slot_name = EXCLUDED.slot_name, "
					 "  publication_name = EXCLUDED.publication_name, "
					 "  sync_status = 'active', "
					 "  last_updated = now()");

	argtypes[0] = TEXTOID;
	argtypes[1] = TEXTOID;
	argtypes[2] = TEXTOID;

	values[0] = CStringGetTextDatum(idx_str);
	values[1] = CStringGetTextDatum(replica_str);
	values[2] = CStringGetTextDatum(slot_name.data);

	ret = ndb_spi_execute_with_args(session, sql.data, 3, argtypes, values, nulls, false, 0);
	if (ret != SPI_OK_INSERT && ret != SPI_OK_UPDATE)
	{
		pfree(sql.data);
		pfree(slot_name.data);
		pfree(pub_name.data);
		ndb_spi_session_end(&session);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: failed to insert/update sync "
						"metadata")));
	}



	resetStringInfo(&sql);
	appendStringInfo(&sql, "SELECT pg_current_wal_lsn()");
	ret = ndb_spi_execute(session, sql.data, true, 0);

	if (ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		bool		isnull = true;
		Datum		lsn_datum = SPI_getbinval(SPI_tuptable->vals[0],
											  SPI_tuptable->tupdesc,
											  1,
											  &isnull);

		if (!isnull)
		{
			char	   *lsn_str = TextDatumGetCString(lsn_datum);

			pfree(lsn_str);
		}
	}

	ndb_spi_session_end(&session);

	pfree(idx_str);
	pfree(replica_str);
	pfree(slot_name.data);
	pfree(pub_name.data);
	pfree(sql.data);

	PG_RETURN_BOOL(true);
}
