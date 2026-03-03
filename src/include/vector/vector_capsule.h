/*-------------------------------------------------------------------------
 *
 * vector_capsule.h
 *	  VectorCapsule: Multi-representation vector with metadata
 *
 * VectorCapsule is a best-in-class vector type that can store multiple
 * representations (dense fp32, fp16, int8, binary, sparse) and metadata
 * (norms, checksums, provenance) for adaptive execution and indexing.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  include/vector/vector_capsule.h
 *
 *-------------------------------------------------------------------------
 */
#ifndef VECTOR_CAPSULE_H
#define VECTOR_CAPSULE_H

#include "postgres.h"
#include "fmgr.h"
#include "neurondb.h"
#include "neurondb_types.h"

/* VectorCapsule flags */
#define VC_FLAG_NORMALIZED		(1U << 0)	/* Vector is L2-normalized */
#define VC_FLAG_QUANTIZED		(1U << 1)	/* Has quantized representations */
#define VC_FLAG_ENCRYPTED		(1U << 2)	/* Vector data is encrypted */
#define VC_FLAG_PROVENANCE		(1U << 3)	/* Has provenance metadata */
#define VC_FLAG_CACHED_NORM		(1U << 4)	/* L2 norm is cached */
#define VC_FLAG_CACHED_MINMAX	(1U << 5)	/* Min/max values cached for quant */

/* VectorCapsule structure */
typedef struct VectorCapsule
{
	int32		vl_len_;		/* varlena header */
	uint16		version;		/* Schema version */
	uint16		flags;			/* Feature flags */
	int16		dim;			/* Primary dimension */
	uint16		unused;			/* Alignment padding */
	
	/* Cached values (optional, based on flags) */
	float4		cached_norm;	/* L2 norm if VC_FLAG_CACHED_NORM */
	float4		cached_min;		/* Min value if VC_FLAG_CACHED_MINMAX */
	float4		cached_max;		/* Max value if VC_FLAG_CACHED_MINMAX */
	
	/* Integrity */
	uint64		checksum;		/* xxhash64 or crc32c checksum */
	
	/* Provenance (optional, if VC_FLAG_PROVENANCE) */
	uint32		model_id;		/* Embedding model identifier */
	uint16		embedding_version;	/* Model version */
	uint16		unused2;		/* Padding */
	TimestampTz created_at;		/* Creation timestamp */
	
	/* Representations follow (variable layout based on flags):
	 * - Primary: float4 data[dim] (always present)
	 * - FP16: uint16 fp16_data[dim] (if VC_FLAG_QUANTIZED)
	 * - INT8: int8 int8_data[dim] (if VC_FLAG_QUANTIZED)
	 * - Binary: uint8 binary_data[(dim+7)/8] (if VC_FLAG_QUANTIZED)
	 * - Sparse: VectorMap (if sparse representation exists)
	 */
} VectorCapsule;

/* Macros for accessing representations */
#define VC_PRIMARY_DATA(vc) ((float4 *)(((char *)(vc)) + sizeof(VectorCapsule)))
#define VC_FP16_DATA(vc) ((uint16 *)(VC_PRIMARY_DATA(vc) + (vc)->dim))
#define VC_INT8_DATA(vc) ((int8 *)(VC_FP16_DATA(vc) + (vc)->dim))
#define VC_BINARY_DATA(vc) ((uint8 *)(VC_INT8_DATA(vc) + (vc)->dim))

/* Size calculation macros */
#define VC_SIZE_BASE(dim) (offsetof(VectorCapsule, checksum) + sizeof(uint64))
#define VC_SIZE_WITH_FP16(dim) (VC_SIZE_BASE(dim) + sizeof(uint16) * (dim))
#define VC_SIZE_WITH_INT8(dim) (VC_SIZE_WITH_FP16(dim) + sizeof(int8) * (dim))
#define VC_SIZE_WITH_BINARY(dim) (VC_SIZE_WITH_INT8(dim) + ((dim + 7) / 8))

/* Function declarations */
extern Datum vector_capsule_in(PG_FUNCTION_ARGS);
extern Datum vector_capsule_out(PG_FUNCTION_ARGS);
extern Datum vector_capsule_from_vector(PG_FUNCTION_ARGS);
extern Datum vector_capsule_to_vector(PG_FUNCTION_ARGS);
extern Datum vector_capsule_get_representation(PG_FUNCTION_ARGS);
extern Datum vector_capsule_get_provenance(PG_FUNCTION_ARGS);
extern Datum vector_capsule_validate_integrity(PG_FUNCTION_ARGS);

/* Internal functions */
VectorCapsule *vector_capsule_create(int dim, uint16 flags);
void vector_capsule_compute_checksum(VectorCapsule *vc);
bool vector_capsule_verify_checksum(VectorCapsule *vc);
float4 vector_capsule_get_norm(VectorCapsule *vc, bool recompute);

#endif							/* VECTOR_CAPSULE_H */



