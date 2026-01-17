/*-------------------------------------------------------------------------
 *
 * neurondb_replication.h
 *    Replication function declarations for NeuronDB
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_REPLICATION_H
#define NEURONDB_REPLICATION_H

#include "postgres.h"
#include "utils/rel.h"

/* Replication hooks for index modifications */
extern void neurondb_hnsw_replication_hook(Relation index, BlockNumber blkno, bool is_insert);
extern void neurondb_ivf_replication_hook(Relation index, BlockNumber blkno, bool is_insert);

/* Replication status functions */
extern bool neurondb_replication_enabled(void);
extern bool neurondb_has_replication_slots(void);
extern int64 neurondb_get_replication_lag(Oid indexOid);
extern bool neurondb_verify_index_consistency(Oid indexOid, Oid replicaOid);

#endif /* NEURONDB_REPLICATION_H */



