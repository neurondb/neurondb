/*-------------------------------------------------------------------------
 *
 * neurondb_rag.h
 *		RAG (Retrieval-Augmented Generation) system definitions
 *
 * Defines structures and functions for production RAG including
 * retrieval policies, answer generation, query planning, and
 * security guardrails.
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *	  include/neurondb_rag.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_RAG_H
#define NEURONDB_RAG_H

#include "postgres.h"
#include "datatype/timestamp.h"

/*
 * RAG retrieval policy
 * Controls how documents are retrieved and ranked
 */
typedef struct RAGPolicy
{
	char strategy[32]; /* hybrid, semantic, keyword */
	float4 vector_weight; /* Vector vs FTS weight */
	int32 rerank_k; /* Top-k for reranking */
	char rerank_model[64]; /* Reranker model name */
	bool enable_cache; /* Cache results */
	int32 ttl_seconds; /* Cache TTL */
} RAGPolicy;

/*
 * RAG answer context
 * Metadata for answer generation
 */
typedef struct RAGAnswer
{
	int32 tokens_used;
	int32 context_docs;
	float4 confidence;
	TimestampTz generated_at;
	char answer_text[FLEXIBLE_ARRAY_MEMBER];
} RAGAnswer;

/*
 * RAG query plan
 * Execution plan for RAG queries
 */
typedef struct RAGPlan
{
	int32 num_sources;
	int32 num_tool_calls;
	int32 estimated_cost;
	char plan_json[FLEXIBLE_ARRAY_MEMBER];
} RAGPlan;

/*
 * RAG guardrails result
 * Security and privacy checks
 */
typedef struct RAGGuardrails
{
	int32 num_redactions;
	int32 num_pii_spans;
	int32 num_policy_hits;
	bool allow_response;
	char reason[256];
} RAGGuardrails;

#endif /* NEURONDB_RAG_H */
