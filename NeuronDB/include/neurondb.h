/*-------------------------------------------------------------------------
 *
 * neurondb.h
 *		Type definitions and function declarations for NeurondB
 *
 * This header file contains all vector type definitions, macros,
 * and function declarations for the NeurondB extension including
 * vector types, distance metrics, indexing, and ML operations.
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *	  contrib/neurondb/neurondb.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_H
#define NEURONDB_H

#include "postgres.h"
#include "fmgr.h"
#include "access/generic_xlog.h"
#include "utils/array.h"
#include "utils/builtins.h"

/* PostgreSQL version support: 16/17/18 only */
#if PG_VERSION_NUM < 160000 || PG_VERSION_NUM >= 190000
#error "NeurondB supports only PostgreSQL 16, 17, and 18"
#endif

/* ========== Vector Type Definitions ========== */

/* Main vector type (float32) */
typedef struct Vector
{
	int32 vl_len_; /* varlena header (required) */
	int16 dim; /* number of dimensions */
	int16 unused; /* padding for alignment */
	float4 data[FLEXIBLE_ARRAY_MEMBER];
} Vector;

/* Float16 quantized vector (2x compression) */
typedef struct VectorF16
{
	int32 vl_len_;
	int16 dim;
	int16 data[FLEXIBLE_ARRAY_MEMBER];
} VectorF16;

/* INT8 quantized vector (8x compression) */
typedef struct VectorI8
{
	int32 vl_len_;
	int16 dim;
	int8 data[FLEXIBLE_ARRAY_MEMBER];
} VectorI8;

/* Binary vector (32x compression) */
typedef struct VectorBinary
{
	int32 vl_len_;
	int16 dim; /* number of bits */
	uint8 data[FLEXIBLE_ARRAY_MEMBER];
} VectorBinary;

/* UINT8 quantized vector (8x compression, unsigned) */
typedef struct VectorU8
{
	int32 vl_len_;
	int16 dim;
	uint8 data[FLEXIBLE_ARRAY_MEMBER];
} VectorU8;

/* Ternary vector (16x compression, 2 bits per dimension) */
typedef struct VectorTernary
{
	int32 vl_len_;
	int16 dim; /* number of dimensions */
	uint8 data[FLEXIBLE_ARRAY_MEMBER]; /* Packed: 4 values per byte */
} VectorTernary;

/* INT4 quantized vector (16x compression, 4 bits per dimension) */
typedef struct VectorI4
{
	int32 vl_len_;
	int16 dim; /* number of dimensions */
	uint8 data[FLEXIBLE_ARRAY_MEMBER]; /* Packed: 2 values per byte */
} VectorI4;

/* ========== Function Declarations ========== */

/* Temporal integration */
extern void temporal_integrate_hnsw_search(Relation heapRel,
											ItemPointer *items,
											float4 *distances,
											int resultCount,
											float4 decayRate,
											float4 recencyWeight,
											const char *timestampColumnName);

/* ========== Macros ========== */

#define DatumGetVector(x) ((Vector *)PG_DETOAST_DATUM(x))
#define PG_GETARG_VECTOR_P(x) DatumGetVector(PG_GETARG_DATUM(x))
#define PG_RETURN_VECTOR_P(x) PG_RETURN_POINTER(x)

#define VECTOR_SIZE(dim) (offsetof(Vector, data) + sizeof(float4) * (dim))
#define VECTOR_DIM(v) ((v)->dim)
#define VECTOR_DATA(v) ((v)->data)

/* Maximum vector dimensions */
#define VECTOR_MAX_DIM 16000

/* ========== Function Declarations ========== */

/* neurondb.c - Core functions */
Vector *new_vector(int dim);
Vector *copy_vector(Vector *vector);
Vector *vector_in_internal(char *str, int *out_dim, bool check);
char *vector_out_internal(Vector *vector);
void normalize_vector(Vector *v);
Vector *normalize_vector_new(Vector *v);

/* distance.c - Distance metrics */
float4 l2_distance(Vector *a, Vector *b);
float4 inner_product_distance(Vector *a, Vector *b);
float4 cosine_distance(Vector *a, Vector *b);
float4 l1_distance(Vector *a, Vector *b);

/* vector_distance_simd.c - SIMD-optimized distance functions */
float4 l2_distance_simd(Vector *a, Vector *b);
float4 inner_product_simd(Vector *a, Vector *b);
float4 cosine_distance_simd(Vector *a, Vector *b);
float4 l1_distance_simd(Vector *a, Vector *b);

/* quantization.c - Vector quantization */
VectorI8 *quantize_vector_i8(Vector *v);
VectorF16 *quantize_vector_f16(Vector *v);
VectorBinary *quantize_vector_binary(Vector *v);
VectorU8 *quantize_vector_uint8(Vector *v);
VectorTernary *quantize_vector_ternary(Vector *v);
VectorI4 *quantize_vector_int4(Vector *v);
Vector *dequantize_vector(void *qv, int type);

/* vector_distance_simd.c - SIMD-optimized distance functions */
int detect_simd_capabilities(void);
float4 l2_distance_simd(Vector *a, Vector *b);
float4 inner_product_simd(Vector *a, Vector *b);
float4 cosine_distance_simd(Vector *a, Vector *b);
float4 l1_distance_simd(Vector *a, Vector *b);

/* index_hnsw.c - HNSW index */
void hnsw_build(Relation index, Relation heap);

/* index_ivf.c - IVF index */
void ivf_build(Relation index, Relation heap);

/* GUC variables are now in neurondb_guc.h */
#include "neurondb_guc.h"

#endif /* NEURONDB_H */
