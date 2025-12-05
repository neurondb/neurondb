/*-------------------------------------------------------------------------
 *
 * neurondb_scan.h
 *		Function prototypes for scan, quota, RLS, and cache modules
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *	  include/neurondb_scan.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_SCAN_H
#define NEURONDB_SCAN_H

#include "postgres.h"
#include "access/relscan.h"
#include "access/genam.h"
#include "nodes/execnodes.h"
#include "utils/timestamp.h"

/* RLS functions */
typedef struct RLSFilterState RLSFilterState;

extern RLSFilterState *ndb_rls_init(Relation rel, EState * estate);
extern bool ndb_rls_check_tuple(RLSFilterState *state, TupleTableSlot * slot);
extern bool ndb_rls_check_item(RLSFilterState *state, ItemPointer tid);
extern void ndb_rls_end(RLSFilterState *state);
extern bool ndb_index_scan_rls_filter(IndexScanDesc scan, ItemPointer tid);
extern int	ndb_rls_filter_results(Relation rel,
								   ItemPointer * items,
								   int count,
								   ItemPointer * *filtered,
								   int *filteredCount);

/* Quota functions */
/* GUC initialization is now centralized in neurondb_guc.c */
extern bool ndb_quota_check(const char *tenantId,
							Oid indexOid,
							int64 additionalVectors,
							int64 additionalBytes);
extern void ndb_quota_enforce_insert(Relation index,
									 const char *tenantId,
									 int64 vectorCount,
									 int64 estimatedBytes);
extern void ndb_quota_update_usage(const char *tenantId,
								   Oid indexOid,
								   int64 vectorsDelta,
								   int64 bytesDelta);

/* Entrypoint cache functions */
extern void entrypoint_cache_init_guc(void);
extern Size entrypoint_cache_shmem_size(void);
extern void entrypoint_cache_shmem_init(void);
extern bool entrypoint_cache_lookup(Oid indexOid,
									BlockNumber * entryPoint,
									int *entryLevel,
									int *maxLevel);
extern void entrypoint_cache_store(Oid indexOid,
								   BlockNumber entryPoint,
								   int entryLevel,
								   int maxLevel);
extern void entrypoint_cache_invalidate(Oid indexOid);

/* Custom scan provider */
extern void register_hybrid_scan_provider(void);

/* Prometheus functions */
extern void prometheus_init_guc(void);
extern void prometheus_register_worker(void);
extern void prometheus_record_query(float8 duration_seconds, bool success);
extern void prometheus_record_cache_hit(void);
extern void prometheus_record_cache_miss(void);

/* HNSW search functions */
extern void hnsw_search_layer(Relation index,
							  BlockNumber entryPoint,
							  int entryLevel,
							  const float4 * query,
							  int dim,
							  int strategy,
							  int efSearch,
							  int k,
							  BlockNumber * *results,
							  float4 * *distances,
							  int *resultCount);

#endif							/* NEURONDB_SCAN_H */
