/*-------------------------------------------------------------------------
 *
 * neurondb_model.h
 *		Model runtime integration definitions
 *
 * Defines structures for HTTP/LLM integration with circuit breaker,
 * retry budget, cost caps, caching tiers, and execution tracing.
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *	  include/neurondb_model.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_MODEL_H
#define NEURONDB_MODEL_H

#include "postgres.h"
#include "datatype/timestamp.h"

/*
 * Model HTTP/LLM request configuration
 * Circuit breaker and rate limiting
 */
typedef struct ModelRequest
{
	char		endpoint[256];
	int32		timeout_ms;
	int32		max_retries;
	int32		cost_cap_cents; /* Per-request cost limit */
	bool		circuit_breaker_open;
	int32		failure_count;
	TimestampTz last_failure;
} ModelRequest;

/*
 * Model cache entry
 * Three-tier caching: exact, semantic, TTL
 */
typedef struct ModelCacheEntry
{
	uint64		query_hash;		/* Exact match hash */
	int64		semantic_id;	/* Semantic match ID */
	TimestampTz created_at;
	TimestampTz expires_at;
	int32		hit_count;
	float4		avg_latency_ms;
	char		response[FLEXIBLE_ARRAY_MEMBER];
} ModelCacheEntry;

/*
 * Model execution trace
 * Observability for all model calls
 */
typedef struct ModelTrace
{
	int64		trace_id;
	TimestampTz timestamp;
	char		provider[64];	/* openai, anthropic, cohere, etc */
	char		model[64];
	int32		latency_ms;
	int32		tokens_in;
	int32		tokens_out;
	int32		cost_cents;
	bool		cache_hit;
	uint64		response_hash;
	char		error_msg[256];
} ModelTrace;

#endif							/* NEURONDB_MODEL_H */
