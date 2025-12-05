/*-------------------------------------------------------------------------
 *
 * neurondb_sparse.h
 *    Sparse vector type definitions and function declarations
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    include/neurondb_sparse.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_SPARSE_H
#define NEURONDB_SPARSE_H

#include "postgres.h"

/*
 * SparseVector: Learned sparse representation
 * - Stores token IDs (vocabulary indices) and learned weights
 * - Optimized for SPLADE/ColBERTv2 models
 * - Supports BM25-style sparse retrieval
 */
typedef struct SparseVector
{
	int32		vl_len_;		/* varlena header */
	int32		vocab_size;		/* Vocabulary size */
	int32		nnz;			/* Number of non-zero entries */
	uint16		model_type;		/* 0=BM25, 1=SPLADE, 2=ColBERTv2 */
	uint16		flags;			/* Reserved */
	/* Followed by: int32 token_ids[nnz], float4 weights[nnz] */
} SparseVector;

#define SPARSE_VEC_TOKEN_IDS(sv) \
	((int32 *)(((char *)(sv)) + sizeof(SparseVector)))
#define SPARSE_VEC_WEIGHTS(sv) \
	((float4 *)(SPARSE_VEC_TOKEN_IDS(sv) + (sv)->nnz))

#define SPARSE_VEC_SIZE(nnz) \
	(offsetof(SparseVector, flags) + sizeof(uint16) + \
	 sizeof(int32) * (nnz) + sizeof(float4) * (nnz))

/* Macros for getting sparse_vector arguments with proper detoasting */
#define DatumGetSparseVector(x) ((SparseVector *)PG_DETOAST_DATUM(x))
#define PG_GETARG_SPARSE_VECTOR_P(x) DatumGetSparseVector(PG_GETARG_DATUM(x))
#define PG_RETURN_SPARSE_VECTOR_P(x) PG_RETURN_POINTER(x)

#endif							/* NEURONDB_SPARSE_H */
