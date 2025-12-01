/*-------------------------------------------------------------------------
 *
 * neurondb_workers.h
 *		Background worker definitions
 *
 * Defines structures for job queue (neuranq), auto-tuner (neuranmon),
 * and HNSW defragmentation (neurandefrag) background workers.
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *	  include/neurondb_workers.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_WORKERS_H
#define NEURONDB_WORKERS_H

#include "postgres.h"
#include "datatype/timestamp.h"
#include "postmaster/bgworker.h"

/*
 * neuranq: SKIP LOCKED job queue
 * - Rate limits per queue
 * - Exponential backoff
 * - Priority lanes (urgent, normal, low)
 */
typedef struct JobQueueEntry
{
	int64 job_id;
	int32 priority; /* 0=urgent, 1=normal, 2=low */
	TimestampTz created_at;
	TimestampTz scheduled_at;
	int32 retry_count;
	int32 backoff_ms;
	char job_type[32];
	char payload_json[FLEXIBLE_ARRAY_MEMBER];
} JobQueueEntry;

typedef struct QueueRateLimit
{
	int32 max_jobs_per_second;
	int32 max_concurrent;
	int32 current_running;
	TimestampTz last_reset;
} QueueRateLimit;

/*
 * neuranmon: Auto-tuner for query performance
 * - Adjusts ef_search based on SLOs
 * - Tunes beam size
 * - Optimizes hybrid weights
 */
typedef struct AutoTunerConfig
{
	float4 target_latency_ms; /* SLO target */
	float4 target_recall; /* Minimum recall */
	int32 sample_period_sec; /* Sampling window */
	bool enabled;
} AutoTunerConfig;

typedef struct AutoTunerMetrics
{
	TimestampTz window_start;
	float4 avg_latency_ms;
	float4 p95_latency_ms;
	float4 avg_recall;
	int32 queries_sampled;
	int16 current_ef_search;
	float4 current_hybrid_weight;
} AutoTunerMetrics;

/*
 * neurandefrag: Online HNSW compaction
 * - Removes orphan edges
 * - Rebalances levels
 * - Compacts tombstones
 */
typedef struct DefragStats
{
	TimestampTz last_run;
	int64 edges_removed;
	int64 nodes_rebalanced;
	int64 tombstones_cleaned;
	int32 defrag_duration_ms;
} DefragStats;

#endif /* NEURONDB_WORKERS_H */
