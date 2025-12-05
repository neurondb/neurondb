/*-------------------------------------------------------------------------
 *
 * neurondb_embedding.h
 *		Embedding lifecycle management definitions
 *
 * Defines structures for embedding versioning, lineage tracking,
 * and refresh policies with drift detection.
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *	  include/neurondb_embedding.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_EMBEDDING_H
#define NEURONDB_EMBEDDING_H

#include "postgres.h"
#include "datatype/timestamp.h"

/*
 * Embedding version metadata
 * Track changes and enable rollback
 */
typedef struct EmbeddingVersion
{
	int64		row_id;
	int32		version_num;
	TimestampTz created_at;
	char		model_id[64];
	uint64		embedding_hash;
	int32		diff_bytes;		/* Size of diff from previous */
} EmbeddingVersion;

/*
 * Embedding lineage
 * Complete provenance tracking
 */
typedef struct EmbeddingLineage
{
	char		model_id[64];
	char		model_version[32];
	char		prompt_template[512];
	char		parameters_json[256];
	uint64		checksum;
	char		license[64];
	TimestampTz generated_at;
} EmbeddingLineage;

/*
 * Embedding refresh policy
 * Automatic drift detection and re-embedding
 */
typedef struct EmbeddingRefreshPolicy
{
	int32		check_interval_hours;
	float4		centroid_shift_threshold;
	int32		min_rows_changed;
	bool		auto_refresh;
	char		schedule_cron[64];
} EmbeddingRefreshPolicy;

/*
 * Drift detection result
 */
typedef struct EmbeddingDrift
{
	float4		centroid_shift;
	int32		rows_changed;
	int32		rows_missing;
	bool		needs_refresh;
	TimestampTz last_check;
} EmbeddingDrift;

#endif							/* NEURONDB_EMBEDDING_H */
