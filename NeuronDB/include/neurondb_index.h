/*-------------------------------------------------------------------------
 *
 * neurondb_index.h
 *		Index method definitions for NeuronDB
 *
 * Defines enterprise-grade index structures including tenant-aware HNSW,
 * hybrid ANN+FTS, temporal vector index, consistent query HNSW,
 * and rerank-ready index.
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *	  include/neurondb_index.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_INDEX_H
#define NEURONDB_INDEX_H

#include "postgres.h"
#include "access/amapi.h"

/*
 * HNSW-T: Tenant-aware HNSW index
 * - Hard quotas per tenant
 * - Per-tenant ef_search settings
 * - Level caps to prevent memory abuse
 */
typedef struct HNSWTenantOptions
{
	int32		tenant_id;
	int32		quota_max_vectors;	/* Hard limit per tenant */
	int16		ef_search;		/* Search parameter */
	int16		max_level;		/* Level cap */
} HNSWTenantOptions;

/*
 * HYBRID-F: Fused ANN + GIN full-text index
 * - Single access method
 * - One plan node
 * - Single heap walk
 * - Combines vector and lexical in one scan
 */
typedef struct HybridFusedIndex
{
	BlockNumber vector_root;	/* HNSW graph root */
	BlockNumber fts_root;		/* GIN posting tree root */
	float4		fusion_weight;	/* Default vector weight */
} HybridFusedIndex;

/*
 * TVX: Temporal vector index
 * - Decay based on insert time
 * - Time-gated kNN queries
 * - Automatic aging of old vectors
 */
typedef struct TemporalVectorIndex
{
	BlockNumber index_root;
	float8		decay_rate;		/* Per-day decay factor */
	TimestampTz base_timestamp; /* Reference time */
} TemporalVectorIndex;

/*
 * CQ-HNSW: Consistent Query HNSW
 * - Snapshot pinning for replicas
 * - Deterministic top-k across all nodes
 * - Guarantees identical results
 */
typedef struct ConsistentQueryHNSW
{
	BlockNumber root;
	uint64		snapshot_xmin;	/* Snapshot ID */
	uint32		random_seed;	/* For deterministic tie-breaking */
} ConsistentQueryHNSW;

/*
 * RRI: Rerank Ready Index
 * - Stores pre-computed top-k candidate lists
 * - Zero round trips to heap for reranking
 * - Cached for hot queries
 */
typedef struct RerankReadyIndex
{
	BlockNumber candidates_root;
	int32		cache_size;		/* Number of cached queries */
	int32		k_candidates;	/* Candidates per query */
} RerankReadyIndex;

/* Reloption kinds for HNSW and IVF indexes */
extern int	relopt_kind_hnsw;
extern int	relopt_kind_ivf;

#endif							/* NEURONDB_INDEX_H */
