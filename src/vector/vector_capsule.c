/*-------------------------------------------------------------------------
 *
 * vector_capsule.c
 *	  VectorCapsule implementation: Multi-representation vector with metadata
 *
 * Implements VectorCapsule type with adaptive representation selection,
 * integrity checking, and provenance tracking.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  src/vector/vector_capsule.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/timestamp.h"
#include "lib/stringinfo.h"
#include "neurondb.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "vector/vector_capsule.h"
#include <string.h>
#include <xxhash.h>
#include <ctype.h>
#include <stdlib.h>

/* Forward declaration for FP16 conversion */
static inline uint16 float_to_fp16_local(float f);

/* GUC for enabling VectorCapsule features */
bool		neurondb_vector_capsule_enabled = false;

/*
 * float_to_fp16_local - Convert float32 to FP16
 * Based on implementation from vector_cast.c
 */
static inline uint16
float_to_fp16_local(float f)
{
	uint32		u;
	uint16		sign;
	uint32		mantissa;
	int16		exp;

	memcpy(&u, &f, sizeof(uint32));
	sign = (u >> 16) & 0x8000;
	mantissa = u & 0x7fffff;
	exp = ((u >> 23) & 0xff) - 127 + 15;

	if (exp <= 0)
		return sign;
	else if (exp >= 31)
		return sign | 0x7c00;
	else
		return sign | (exp << 10) | (mantissa >> 13);
}

PG_FUNCTION_INFO_V1(vector_capsule_in);
Datum
vector_capsule_in(PG_FUNCTION_ARGS)
{
	char	   *str = NULL;
	char	   *endptr = NULL;
	char	   *ptr = NULL;
	float4	   *temp_data = NULL;
	int			capacity = 16;
	int			dim = 0;
	int			size;
	VectorCapsule *result = NULL;
	float4	   *primary_data = NULL;
	uint16		flags = 0;

	if (!neurondb_vector_capsule_enabled)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("VectorCapsule feature is disabled"),
				 errhint("Set neurondb.vector_capsule_enabled = true to enable")));

	str = PG_GETARG_CSTRING(0);
	if (str == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("VectorCapsule input string cannot be NULL")));

	/* Parse vector data from string like "[1,2,3]" */
	ptr = str;
	while (isspace((unsigned char) *ptr))
		ptr++;

	if (*ptr == '[' || *ptr == '{')
		ptr++;

	nalloc(temp_data, float4, capacity);

	while (*ptr && *ptr != ']' && *ptr != '}')
	{
		while (isspace((unsigned char) *ptr) || *ptr == ',')
			ptr++;

		if (*ptr == ']' || *ptr == '}' || *ptr == '\0')
			break;

		if (dim >= capacity)
		{
			capacity *= 2;
			temp_data = (float4 *) repalloc(temp_data, sizeof(float4) * capacity);
		}

		temp_data[dim] = strtof(ptr, &endptr);
		if (ptr == endptr)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("invalid input syntax for type vector_capsule: \"%s\"", str)));

		if (isinf(temp_data[dim]) || isnan(temp_data[dim]))
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("vector_capsule values cannot be NaN or Infinity")));

		ptr = endptr;
		dim++;
	}

	if (dim == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("vector_capsule must have at least 1 dimension")));

	if (dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector_capsule dimension %d exceeds maximum %d", dim, VECTOR_MAX_DIM)));

	/* Calculate size - for now, just primary representation */
	size = sizeof(VectorCapsule) + sizeof(float4) * dim;
	flags = VC_FLAG_CACHED_NORM; /* Cache norm by default */

	/* Allocate VectorCapsule */
	{
		char	   *tmp = NULL;

		nalloc(tmp, char, size);
		result = (VectorCapsule *) tmp;
		MemSet(result, 0, size);
	}
	SET_VARSIZE(result, size);

	/* Initialize header */
	result->version = 1;
	result->flags = flags;
	result->dim = dim;
	result->created_at = GetCurrentTimestamp();

	/* Copy primary data */
	primary_data = VC_PRIMARY_DATA(result);
	memcpy(primary_data, temp_data, sizeof(float4) * dim);
	pfree(temp_data);

	/* Compute and cache norm */
	{
		double		sum = 0.0;
		int			i;

		for (i = 0; i < dim; i++)
			sum += (double) primary_data[i] * (double) primary_data[i];
		result->cached_norm = (float4) sqrt(sum);
	}

	/* Compute checksum */
	vector_capsule_compute_checksum(result);

	PG_RETURN_POINTER(result);
}

PG_FUNCTION_INFO_V1(vector_capsule_out);
Datum
vector_capsule_out(PG_FUNCTION_ARGS)
{
	VectorCapsule *vc = NULL;
	StringInfoData buf;
	float4	   *primary_data = NULL;
	int			i;

	if (!neurondb_vector_capsule_enabled)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("VectorCapsule feature is disabled")));

	vc = (VectorCapsule *) PG_GETARG_POINTER(0);
	if (vc == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("VectorCapsule cannot be NULL")));

	/* Validate checksum */
	if (!vector_capsule_verify_checksum(vc))
		elog(WARNING, "vector_capsule_out: checksum verification failed");

	/* Output primary representation as [1,2,3] format */
	initStringInfo(&buf);
	appendStringInfoChar(&buf, '[');

	primary_data = VC_PRIMARY_DATA(vc);
	for (i = 0; i < vc->dim; i++)
	{
		if (i > 0)
			appendStringInfoChar(&buf, ',');
		appendStringInfo(&buf, "%g", primary_data[i]);
	}

	appendStringInfoChar(&buf, ']');
	PG_RETURN_CSTRING(buf.data);
}

/*
 * vector_capsule_from_vector
 *    Convert a standard Vector to VectorCapsule with optional quantized representations
 */
PG_FUNCTION_INFO_V1(vector_capsule_from_vector);
Datum
vector_capsule_from_vector(PG_FUNCTION_ARGS)
{
	Vector	   *vec = NULL;
	bool		include_fp16 = false;
	bool		include_int8 = false;
	bool		include_binary = false;
	bool		cache_norm = false;
	VectorCapsule *result = NULL;
	int			size;
	uint16		flags = 0;
	float4	   *primary_data = NULL;
	uint16	   *fp16_data = NULL;
	int8	   *int8_data = NULL;
	uint8	   *binary_data = NULL;
	int			i;

	if (!neurondb_vector_capsule_enabled)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("VectorCapsule feature is disabled"),
				 errhint("Set neurondb.vector_capsule_enabled = true to enable")));

	vec = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(vec);

	if (PG_NARGS() >= 2)
		include_fp16 = PG_GETARG_BOOL(1);
	if (PG_NARGS() >= 3)
		include_int8 = PG_GETARG_BOOL(2);
	if (PG_NARGS() >= 4)
		include_binary = PG_GETARG_BOOL(3);
	if (PG_NARGS() >= 5)
		cache_norm = PG_GETARG_BOOL(4);

	if (vec->dim <= 0 || vec->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid vector dimension: %d", vec->dim)));

	/* Calculate size */
	size = sizeof(VectorCapsule);
	if (include_fp16)
	{
		size += sizeof(uint16) * vec->dim;
		flags |= VC_FLAG_QUANTIZED;
	}
	if (include_int8)
	{
		size += sizeof(int8) * vec->dim;
		flags |= VC_FLAG_QUANTIZED;
	}
	if (include_binary)
	{
		size += (vec->dim + 7) / 8;
		flags |= VC_FLAG_QUANTIZED;
	}
	if (cache_norm)
	{
		flags |= VC_FLAG_CACHED_NORM;
	}

	/* Allocate */
	{
		char	   *tmp = NULL;

		nalloc(tmp, char, size);
		result = (VectorCapsule *) tmp;
		MemSet(result, 0, size);
	}
	SET_VARSIZE(result, size);

	/* Initialize header */
	result->version = 1;
	result->flags = flags;
	result->dim = vec->dim;
	result->created_at = GetCurrentTimestamp();

	/* Copy primary representation */
	primary_data = VC_PRIMARY_DATA(result);
	memcpy(primary_data, vec->data, sizeof(float4) * vec->dim);

	/* Compute and cache norm if requested */
	if (cache_norm)
	{
		double		sum = 0.0;
		int			j;

		for (j = 0; j < vec->dim; j++)
			sum += (double) vec->data[j] * (double) vec->data[j];
		result->cached_norm = (float4) sqrt(sum);
	}

	/* Generate quantized representations */
	if (include_fp16)
	{
		fp16_data = VC_FP16_DATA(result);
		/* Use FP16 conversion */
		for (i = 0; i < vec->dim; i++)
		{
			fp16_data[i] = float_to_fp16_local(vec->data[i]);
		}
	}

	if (include_int8)
	{
		int8_data = VC_INT8_DATA(result);
		/* Compute min/max for quantization */
		float4		min_val = vec->data[0];
		float4		max_val = vec->data[0];

		for (i = 1; i < vec->dim; i++)
		{
			if (vec->data[i] < min_val)
				min_val = vec->data[i];
			if (vec->data[i] > max_val)
				max_val = vec->data[i];
		}
		result->cached_min = min_val;
		result->cached_max = max_val;
		flags |= VC_FLAG_CACHED_MINMAX;

		/* Quantize */
		for (i = 0; i < vec->dim; i++)
		{
			float		range = max_val - min_val;
			float		normalized;

			if (range > 0.0f)
			{
				normalized = (vec->data[i] - min_val) / range;
				int8_data[i] = (int8) (normalized * 127.0f - 128.0f);
			}
			else
			{
				int8_data[i] = 0;
			}
		}
	}

	if (include_binary)
	{
		binary_data = VC_BINARY_DATA(result);
		for (i = 0; i < vec->dim; i++)
		{
			int			byte_idx = i / 8;
			int			bit_idx = i % 8;

			if (vec->data[i] > 0.0f)
				binary_data[byte_idx] |= (1 << bit_idx);
		}
	}

	/* Compute checksum */
	vector_capsule_compute_checksum(result);

	PG_RETURN_POINTER(result);
}

/*
 * vector_capsule_compute_checksum
 *    Compute xxhash64 checksum over vector data and metadata
 */
void
vector_capsule_compute_checksum(VectorCapsule *vc)
{
	XXH64_state_t *state = NULL;
	float4	   *primary_data = NULL;
	size_t		data_size;

	if (vc == NULL)
		return;

	state = XXH64_createState();
	if (state == NULL)
		return;

	XXH64_reset(state, 0);

	/* Hash header (excluding checksum field itself) */
	XXH64_update(state, vc, offsetof(VectorCapsule, checksum));

	/* Hash primary data */
	primary_data = VC_PRIMARY_DATA(vc);
	data_size = sizeof(float4) * vc->dim;
	XXH64_update(state, primary_data, data_size);

	/* Hash quantized representations if present */
	if (vc->flags & VC_FLAG_QUANTIZED)
	{
		if (vc->flags & VC_FLAG_QUANTIZED)	/* Has FP16 */
		{
			uint16	   *fp16_data = VC_FP16_DATA(vc);

			XXH64_update(state, fp16_data, sizeof(uint16) * vc->dim);
		}
		/* Add INT8 and binary if present */
	}

	vc->checksum = XXH64_digest(state);
	XXH64_freeState(state);
}

/*
 * vector_capsule_verify_checksum
 *    Verify integrity of VectorCapsule
 */
bool
vector_capsule_verify_checksum(VectorCapsule *vc)
{
	uint64		stored_checksum;
	uint64		computed_checksum;

	if (vc == NULL)
		return false;

	stored_checksum = vc->checksum;
	vector_capsule_compute_checksum(vc);
	computed_checksum = vc->checksum;
	vc->checksum = stored_checksum;	/* Restore original */

	return (stored_checksum == computed_checksum);
}

PG_FUNCTION_INFO_V1(vector_capsule_validate_integrity);
Datum
vector_capsule_validate_integrity(PG_FUNCTION_ARGS)
{
	VectorCapsule *vc = NULL;

	if (!neurondb_vector_capsule_enabled)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("VectorCapsule feature is disabled")));

	vc = (VectorCapsule *) PG_GETARG_POINTER(0);

	if (vc == NULL)
		PG_RETURN_BOOL(false);

	PG_RETURN_BOOL(vector_capsule_verify_checksum(vc));
}



