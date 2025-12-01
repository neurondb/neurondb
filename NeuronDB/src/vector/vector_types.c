/*-------------------------------------------------------------------------
 *
 * types_core.c
 *		Core enterprise data types implementation
 *
 * Implements vectorp (packed SIMD), vecmap (sparse), rtext (retrievable
 * text), and vgraph (compact graph) data types with I/O functions.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  src/types_core.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "lib/stringinfo.h"
#include "libpq/pqformat.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "utils/varlena.h"
#include "access/htup_details.h"
#include "access/tupdesc.h"
#include "access/heapam.h"
#include "commands/vacuum.h"
#include "neurondb.h"
#include "neurondb_types.h"
#include <zlib.h>
#include <ctype.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

/* Forward declarations for vecmap distance functions */
extern Datum vecmap_l2_distance(PG_FUNCTION_ARGS);
extern Datum vecmap_cosine_distance(PG_FUNCTION_ARGS);
extern Datum vecmap_inner_product(PG_FUNCTION_ARGS);

/*
 * vectorp_in: Parse vectorp from text
 * Format: "[1.0,2.0,3.0]"
 */
PG_FUNCTION_INFO_V1(vectorp_in);
Datum
vectorp_in(PG_FUNCTION_ARGS)
{
	char	   *str = PG_GETARG_CSTRING(0);
	VectorPacked *result;
	float4	   *temp_data = NULL;
	int			dim;
	int			capacity;
	char	   *ptr;
	char	   *endptr;
	uint32		fingerprint;
	int			size;

	dim = 0;
	capacity = 16;

	ptr = str;
	while (isspace((unsigned char) *ptr))
		ptr++;

	if (*ptr == '[')
		ptr++;

	NDB_ALLOC(temp_data, float4, capacity);

	while (*ptr && *ptr != ']')
	{
		while (isspace((unsigned char) *ptr) || *ptr == ',')
			ptr++;

		if (*ptr == ']' || *ptr == '\0')
			break;

		if (dim >= capacity)
		{
			capacity *= 2;
			/* For reallocation, we need to manually handle since NDB_ALLOC doesn't support repalloc */
			/* This is acceptable for dynamic growth patterns */
			temp_data = (float4 *) repalloc(
											temp_data, sizeof(float4) * capacity);
		}

		temp_data[dim] = strtof(ptr, &endptr);
		if (ptr == endptr)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("invalid input for vectorp")));

		ptr = endptr;
		dim++;
	}

	if (dim == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("vectorp must have at least 1 "
						"dimension")));

	size = VECTORP_SIZE(dim);
	/* Variable-length type requires custom size - use palloc0 but ensure proper cleanup */
	/* Note: For variable-length PostgreSQL types, palloc0 is standard */
	result = (VectorPacked *) palloc0(size);
	SET_VARSIZE(result, size);

	/* Compute fingerprint (CRC32 of dimension count) */
	fingerprint = crc32(0L, Z_NULL, 0);
	fingerprint = crc32(fingerprint, (unsigned char *) &dim, sizeof(dim));

	result->fingerprint = fingerprint;
	result->version = 1;
	result->dim = dim;
	result->endian_guard = 0x01;	/* Little endian */
	result->flags = 0;

	memcpy(result->data, temp_data, sizeof(float4) * dim);
	NDB_FREE(temp_data);

	PG_RETURN_POINTER(result);
}

/*
 * vectorp_out: Convert vectorp to text
 */
PG_FUNCTION_INFO_V1(vectorp_out);
Datum
vectorp_out(PG_FUNCTION_ARGS)
{
	VectorPacked *vec = (VectorPacked *) PG_GETARG_POINTER(0);
	StringInfoData buf;
	int			i;

	initStringInfo(&buf);
	appendStringInfoChar(&buf, '[');

	for (i = 0; i < vec->dim; i++)
	{
		if (i > 0)
			appendStringInfoChar(&buf, ',');
		appendStringInfo(&buf, "%g", vec->data[i]);
	}

	appendStringInfoChar(&buf, ']');
	PG_RETURN_CSTRING(buf.data);
}

/*
 * vecmap_in: Parse sparse vector map
 * Format: "{dim:1000,nnz:5,indices:[0,10,20],values:[1.0,2.0,3.0]}"
 */
PG_FUNCTION_INFO_V1(vecmap_in);
Datum
vecmap_in(PG_FUNCTION_ARGS)
{
	char	   *str = PG_GETARG_CSTRING(0);
	VectorMap  *result;
	int32		dim;
	int32		nnz;
	int32	   *indices;
	float4	   *values;
	char	   *ptr;
	char	   *endptr;
	int			i;
	int			size;

	dim = 0;
	nnz = 0;

	(void) i;					/* Suppress unused warning */

	/* Parse JSON-like format */
	ptr = str;
	while (isspace((unsigned char) *ptr))
		ptr++;

	if (*ptr != '{')
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("vecmap must start with '{'")));
	ptr++;

	/* Parse dim */
	while (isspace((unsigned char) *ptr))
		ptr++;

	if (strncmp(ptr, "dim:", 4) == 0)
	{
		ptr += 4;
		dim = strtol(ptr, &endptr, 10);
		if (ptr == endptr || dim <= 0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("invalid dim value in vecmap")));
		ptr = endptr;
	}

	/* Parse nnz */
	while (isspace((unsigned char) *ptr) || *ptr == ',')
		ptr++;

	if (strncmp(ptr, "nnz:", 4) == 0)
	{
		ptr += 4;
		nnz = strtol(ptr, &endptr, 10);
		if (ptr == endptr || nnz < 0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("invalid nnz value in vecmap")));
		ptr = endptr;
	}

	if (dim == 0 || nnz == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("vecmap must specify dim and nnz")));

	if (nnz > dim)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("nnz cannot exceed dim")));

	indices = NULL;
	values = NULL;
	NDB_ALLOC(indices, int32, nnz);
	NDB_ALLOC(values, float4, nnz);

	/* Parse indices array */
	while (isspace((unsigned char) *ptr) || *ptr == ',')
		ptr++;

	if (strncmp(ptr, "indices:", 8) == 0)
	{
		ptr += 8;
		while (isspace((unsigned char) *ptr))
			ptr++;

		if (*ptr != '[')
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("indices must be an array")));
		ptr++;

		for (i = 0; i < nnz; i++)
		{
			while (isspace((unsigned char) *ptr) || *ptr == ',')
				ptr++;

			indices[i] = strtol(ptr, &endptr, 10);
			if (ptr == endptr)
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
						 errmsg("invalid index value")));

			if (indices[i] < 0 || indices[i] >= dim)
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("index %d out of range "
								"[0, %d)",
								indices[i],
								dim)));

			ptr = endptr;
		}

		while (isspace((unsigned char) *ptr))
			ptr++;
		if (*ptr != ']')
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("expected ']' after indices")));
		ptr++;
	}

	/* Parse values array */
	while (isspace((unsigned char) *ptr) || *ptr == ',')
		ptr++;

	if (strncmp(ptr, "values:", 7) == 0)
	{
		ptr += 7;
		while (isspace((unsigned char) *ptr))
			ptr++;

		if (*ptr != '[')
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("values must be an array")));
		ptr++;

		for (i = 0; i < nnz; i++)
		{
			while (isspace((unsigned char) *ptr) || *ptr == ',')
				ptr++;

			values[i] = strtof(ptr, &endptr);
			if (ptr == endptr)
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
						 errmsg("invalid value")));

			ptr = endptr;
		}

		while (isspace((unsigned char) *ptr))
			ptr++;
		if (*ptr != ']')
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("expected ']' after values")));
		ptr++;
	}

	/* Build result */
	size = sizeof(VectorMap) + sizeof(int32) * nnz + sizeof(float4) * nnz;
	result = (VectorMap *) palloc0(size);
	SET_VARSIZE(result, size);

	result->total_dim = dim;
	result->nnz = nnz;

	memcpy(VECMAP_INDICES(result), indices, sizeof(int32) * nnz);
	memcpy(VECMAP_VALUES(result), values, sizeof(float4) * nnz);

	NDB_FREE(indices);
	NDB_FREE(values);

	PG_RETURN_POINTER(result);
}

/*
 * vecmap_out: Convert sparse vector map to text
 */
PG_FUNCTION_INFO_V1(vecmap_out);
Datum
vecmap_out(PG_FUNCTION_ARGS)
{
	VectorMap  *vec = (VectorMap *) PG_GETARG_POINTER(0);
	StringInfoData buf;
	int32	   *indices;
	float4	   *values;
	int			i;

	indices = VECMAP_INDICES(vec);
	values = VECMAP_VALUES(vec);

	initStringInfo(&buf);

	appendStringInfo(
					 &buf, "{dim:%d,nnz:%d,indices:[", vec->total_dim, vec->nnz);

	for (i = 0; i < vec->nnz; i++)
	{
		if (i > 0)
			appendStringInfoChar(&buf, ',');
		appendStringInfo(&buf, "%d", indices[i]);
	}

	appendStringInfoString(&buf, "],values:[");

	for (i = 0; i < vec->nnz; i++)
	{
		if (i > 0)
			appendStringInfoChar(&buf, ',');
		appendStringInfo(&buf, "%g", values[i]);
	}

	appendStringInfoString(&buf, "]}");

	PG_RETURN_CSTRING(buf.data);
}

/*-------------------------------------------------------------------------
 * sparsevec type I/O functions (pgvector-compatible sparse vector type)
 * Format: "{1:0.5, 5:0.3, 10:0.8}" or "{dim:1000,1:0.5,5:0.3,10:0.8}"
 *-------------------------------------------------------------------------
 */

/*
 * sparsevec_in: Parse pgvector-compatible sparse vector format
 * Format: "{index:value, index:value, ...}" or "{dim:N, index:value, ...}"
 */
PG_FUNCTION_INFO_V1(sparsevec_in);
Datum
sparsevec_in(PG_FUNCTION_ARGS)
{
	char	   *str = PG_GETARG_CSTRING(0);
	VectorMap  *result;
	int32		dim = 0;
	int32		nnz = 0;
	int32	   *indices;
	float4	   *values;
	char	   *ptr;
	char	   *endptr;
	int			i;
	int			size;
	int			capacity;
	int			max_index = -1;

	ptr = str;
	while (isspace((unsigned char) *ptr))
		ptr++;

	if (*ptr != '{')
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("sparsevec must start with '{'")));
	ptr++;

	capacity = 16;
	indices = NULL;
	values = NULL;
	NDB_ALLOC(indices, int32, capacity);
	NDB_ALLOC(values, float4, capacity);

	/* Parse entries */
	while (*ptr && *ptr != '}')
	{
		while (isspace((unsigned char) *ptr) || *ptr == ',')
			ptr++;

		if (*ptr == '}' || *ptr == '\0')
			break;

		/* Check for dim: prefix */
		if (strncmp(ptr, "dim:", 4) == 0)
		{
			ptr += 4;
			dim = strtol(ptr, &endptr, 10);
			if (ptr == endptr || dim <= 0)
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
						 errmsg("invalid dim value in sparsevec")));
			ptr = endptr;
			continue;
		}

		/* Parse index:value pair */
		if (nnz >= capacity)
		{
			capacity *= 2;
			indices = (int32 *) repalloc(indices, sizeof(int32) * capacity);
			values = (float4 *) repalloc(values, sizeof(float4) * capacity);
		}

		indices[nnz] = strtol(ptr, &endptr, 10);
		if (ptr == endptr)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("invalid index in sparsevec")));

		if (indices[nnz] < 0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("sparsevec indices must be non-negative")));

		if (indices[nnz] > max_index)
			max_index = indices[nnz];

		ptr = endptr;
		if (*ptr != ':')
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("expected ':' after index in sparsevec")));
		ptr++;

		values[nnz] = strtof(ptr, &endptr);
		if (ptr == endptr)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("invalid value in sparsevec")));

		ptr = endptr;
		nnz++;
	}

	if (nnz == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("sparsevec must have at least one entry")));

	if (nnz > 1000)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("sparsevec cannot have more than 1000 nonzero entries")));

	/* Set dimension if not specified */
	if (dim == 0)
		dim = max_index + 1;

	if (dim > 1000000)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("sparsevec dimension %d exceeds maximum of 1000000",
						dim)));

	/* Validate indices are within dimension */
	for (i = 0; i < nnz; i++)
	{
		if (indices[i] >= dim)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("sparsevec index %d out of range [0, %d)",
							indices[i],
							dim)));
	}

	/* Build result */
	size = sizeof(VectorMap) + sizeof(int32) * nnz + sizeof(float4) * nnz;
	result = (VectorMap *) palloc0(size);
	SET_VARSIZE(result, size);

	result->total_dim = dim;
	result->nnz = nnz;

	memcpy(VECMAP_INDICES(result), indices, sizeof(int32) * nnz);
	memcpy(VECMAP_VALUES(result), values, sizeof(float4) * nnz);

	NDB_FREE(indices);
	NDB_FREE(values);

	PG_RETURN_POINTER(result);
}

/*
 * sparsevec_out: Convert sparsevec to pgvector-compatible text format
 */
PG_FUNCTION_INFO_V1(sparsevec_out);
Datum
sparsevec_out(PG_FUNCTION_ARGS)
{
	VectorMap  *vec = (VectorMap *) PG_GETARG_POINTER(0);
	StringInfoData buf;
	int32	   *indices;
	float4	   *values;
	int			i;

	if (vec == NULL)
		PG_RETURN_CSTRING(pstrdup("NULL"));

	indices = VECMAP_INDICES(vec);
	values = VECMAP_VALUES(vec);

	initStringInfo(&buf);
	appendStringInfoChar(&buf, '{');

	/* Output dim: if needed */
	if (vec->total_dim > 0)
		appendStringInfo(&buf, "dim:%d,", vec->total_dim);

	/* Output index:value pairs */
	for (i = 0; i < vec->nnz; i++)
	{
		if (i > 0)
			appendStringInfoChar(&buf, ',');
		appendStringInfo(&buf, "%d:%g", indices[i], values[i]);
	}

	appendStringInfoChar(&buf, '}');
	PG_RETURN_CSTRING(buf.data);
}

/*
 * sparsevec_recv: Binary receive function
 */
PG_FUNCTION_INFO_V1(sparsevec_recv);
Datum
sparsevec_recv(PG_FUNCTION_ARGS)
{
	StringInfo	buf = (StringInfo) PG_GETARG_POINTER(0);
	VectorMap  *result;
	int32		dim;
	int32		nnz;
	int			size;
	int			i;

	dim = pq_getmsgint(buf, sizeof(int32));
	nnz = pq_getmsgint(buf, sizeof(int32));

	if (dim <= 0 || dim > 1000000)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_BINARY_REPRESENTATION),
				 errmsg("invalid sparsevec dimension: %d", dim)));

	if (nnz < 0 || nnz > 1000)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_BINARY_REPRESENTATION),
				 errmsg("invalid sparsevec nnz: %d", nnz)));

	size = sizeof(VectorMap) + sizeof(int32) * nnz + sizeof(float4) * nnz;
	result = (VectorMap *) palloc0(size);
	SET_VARSIZE(result, size);
	result->total_dim = dim;
	result->nnz = nnz;

	for (i = 0; i < nnz; i++)
		VECMAP_INDICES(result)[i] = pq_getmsgint(buf, sizeof(int32));

	for (i = 0; i < nnz; i++)
		VECMAP_VALUES(result)[i] = pq_getmsgfloat4(buf);

	PG_RETURN_POINTER(result);
}

/*
 * sparsevec_send: Binary send function
 */
PG_FUNCTION_INFO_V1(sparsevec_send);
Datum
sparsevec_send(PG_FUNCTION_ARGS)
{
	VectorMap  *vec = (VectorMap *) PG_GETARG_POINTER(0);
	StringInfoData buf;
	int32	   *indices;
	float4	   *values;
	int			i;

	indices = VECMAP_INDICES(vec);
	values = VECMAP_VALUES(vec);

	pq_begintypsend(&buf);
	pq_sendint(&buf, vec->total_dim, sizeof(int32));
	pq_sendint(&buf, vec->nnz, sizeof(int32));

	for (i = 0; i < vec->nnz; i++)
		pq_sendint(&buf, indices[i], sizeof(int32));

	for (i = 0; i < vec->nnz; i++)
		pq_sendfloat4(&buf, values[i]);

	PG_RETURN_BYTEA_P(pq_endtypsend(&buf));
}

/*-------------------------------------------------------------------------
 * Comparison operators for sparsevec type
 *-------------------------------------------------------------------------
 */

/*
 * sparsevec_eq: Equality comparison for sparsevec
 */
PG_FUNCTION_INFO_V1(sparsevec_eq);
Datum
sparsevec_eq(PG_FUNCTION_ARGS)
{
	VectorMap  *a = (VectorMap *) PG_GETARG_POINTER(0);
	VectorMap  *b = (VectorMap *) PG_GETARG_POINTER(1);
	int32	   *a_indices,
			   *b_indices;
	float4	   *a_values,
			   *b_values;
	int			i;

	/* Handle NULL vectors */
	if (a == NULL && b == NULL)
		PG_RETURN_BOOL(true);
	if (a == NULL || b == NULL)
		PG_RETURN_BOOL(false);

	if (a->total_dim != b->total_dim || a->nnz != b->nnz)
		PG_RETURN_BOOL(false);

	a_indices = VECMAP_INDICES(a);
	b_indices = VECMAP_INDICES(b);
	a_values = VECMAP_VALUES(a);
	b_values = VECMAP_VALUES(b);

	/* Compare indices and values */
	for (i = 0; i < a->nnz; i++)
	{
		if (a_indices[i] != b_indices[i])
			PG_RETURN_BOOL(false);
		if (fabs(a_values[i] - b_values[i]) > 1e-6)
			PG_RETURN_BOOL(false);
	}

	PG_RETURN_BOOL(true);
}

/*
 * sparsevec_ne: Inequality comparison for sparsevec
 */
PG_FUNCTION_INFO_V1(sparsevec_ne);
Datum
sparsevec_ne(PG_FUNCTION_ARGS)
{
	return DirectFunctionCall2(
							   sparsevec_eq, PG_GETARG_DATUM(0), PG_GETARG_DATUM(1))
		? BoolGetDatum(false)
		: BoolGetDatum(true);
}

/*
 * sparsevec_hash: Hash function for sparsevec
 */
PG_FUNCTION_INFO_V1(sparsevec_hash);
Datum
sparsevec_hash(PG_FUNCTION_ARGS)
{
	VectorMap  *v = (VectorMap *) PG_GETARG_POINTER(0);
	int32	   *indices;
	float4	   *values;
	uint32		hash = 5381;
	int			i;

	if (v == NULL)
		PG_RETURN_UINT32(0);

	indices = VECMAP_INDICES(v);
	values = VECMAP_VALUES(v);

	hash = ((hash << 5) + hash) + (uint32) v->total_dim;
	hash = ((hash << 5) + hash) + (uint32) v->nnz;

	for (i = 0; i < v->nnz && i < 16; i++)
	{
		int32		tmp;

		hash = ((hash << 5) + hash) + (uint32) indices[i];
		tmp = (int32) (values[i] * 1000000.0f);
		hash = ((hash << 5) + hash) + (uint32) tmp;
	}

	PG_RETURN_UINT32(hash);
}

/*-------------------------------------------------------------------------
 * Distance functions for sparsevec type (reuse vecmap functions)
 *-------------------------------------------------------------------------
 */

/*
 * sparsevec_l2_distance: L2 distance for sparsevec (uses vecmap_l2_distance)
 */
PG_FUNCTION_INFO_V1(sparsevec_l2_distance);
Datum
sparsevec_l2_distance(PG_FUNCTION_ARGS)
{
	/* Reuse vecmap_l2_distance since sparsevec uses VectorMap structure */
	return vecmap_l2_distance(fcinfo);
}

/*
 * sparsevec_cosine_distance: Cosine distance for sparsevec
 */
PG_FUNCTION_INFO_V1(sparsevec_cosine_distance);
Datum
sparsevec_cosine_distance(PG_FUNCTION_ARGS)
{
	/* Reuse vecmap_cosine_distance since sparsevec uses VectorMap structure */
	return vecmap_cosine_distance(fcinfo);
}

/*
 * sparsevec_inner_product: Inner product for sparsevec
 */
PG_FUNCTION_INFO_V1(sparsevec_inner_product);
Datum
sparsevec_inner_product(PG_FUNCTION_ARGS)
{
	/* Reuse vecmap_inner_product since sparsevec uses VectorMap structure */
	return vecmap_inner_product(fcinfo);
}

/*
 * sparsevec_l2_norm: L2 norm for sparsevec (pgvector compatibility)
 */
PG_FUNCTION_INFO_V1(sparsevec_l2_norm);
Datum
sparsevec_l2_norm(PG_FUNCTION_ARGS)
{
	VectorMap  *v = (VectorMap *) PG_GETARG_POINTER(0);
	float4	   *values;
	double		sum = 0.0;
	int			i;

	if (v == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot compute norm of NULL sparsevec")));

	values = VECMAP_VALUES(v);

	for (i = 0; i < v->nnz; i++)
		sum += (double) values[i] * (double) values[i];

	PG_RETURN_FLOAT8(sqrt(sum));
}

/*
 * sparsevec_l2_normalize: Normalize sparsevec with L2 norm (pgvector compatibility)
 */
PG_FUNCTION_INFO_V1(sparsevec_l2_normalize);
Datum
sparsevec_l2_normalize(PG_FUNCTION_ARGS)
{
	VectorMap  *v = (VectorMap *) PG_GETARG_POINTER(0);
	VectorMap  *result;
	float4	   *values;
	float4	   *result_values;
	int32	   *result_indices;
	double		norm = 0.0;
	int			i;
	int			size;

	if (v == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot normalize NULL sparsevec")));

	values = VECMAP_VALUES(v);

	/* Compute L2 norm */
	for (i = 0; i < v->nnz; i++)
		norm += (double) values[i] * (double) values[i];
	norm = sqrt(norm);

	/* If norm is zero, return zero vector */
	if (norm == 0.0 || v->nnz == 0)
	{
		size = sizeof(VectorMap) + sizeof(int32) * 0 + sizeof(float4) * 0;
		result = (VectorMap *) palloc0(size);
		SET_VARSIZE(result, size);
		result->total_dim = v->total_dim;
		result->nnz = 0;
		PG_RETURN_POINTER(result);
	}

	/* Allocate result with same structure */
	size = sizeof(VectorMap) + sizeof(int32) * v->nnz + sizeof(float4) * v->nnz;
	result = (VectorMap *) palloc0(size);
	SET_VARSIZE(result, size);
	result->total_dim = v->total_dim;
	result->nnz = v->nnz;

	result_indices = VECMAP_INDICES(result);
	result_values = VECMAP_VALUES(result);

	/* Copy indices and normalize values */
	memcpy(result_indices, VECMAP_INDICES(v), sizeof(int32) * v->nnz);
	for (i = 0; i < v->nnz; i++)
		result_values[i] = (float4) ((double) values[i] / norm);

	PG_RETURN_POINTER(result);
}

/*
 * rtext_in: Parse retrievable text
 */
PG_FUNCTION_INFO_V1(rtext_in);
Datum
rtext_in(PG_FUNCTION_ARGS)
{
	char	   *str = PG_GETARG_CSTRING(0);
	RetrievableText *result;
	int			text_len;
	int			size;

	text_len = strlen(str);

	/* Basic implementation: store text, tokenize later */
	size = sizeof(RetrievableText) + text_len + 1;
	result = (RetrievableText *) palloc0(size);
	SET_VARSIZE(result, size);

	result->text_len = text_len;
	result->num_tokens = 0;		/* Will be computed on first access */
	result->lang_tag = 0;		/* Auto-detect */
	result->flags = 0;

	memcpy(RTEXT_DATA(result), str, text_len);

	PG_RETURN_POINTER(result);
}

/*
 * rtext_out: Convert retrievable text to string
 */
PG_FUNCTION_INFO_V1(rtext_out);
Datum
rtext_out(PG_FUNCTION_ARGS)
{
	RetrievableText *rt = (RetrievableText *) PG_GETARG_POINTER(0);
	char	   *result = NULL;

	NDB_ALLOC(result, char, rt->text_len + 1);
	memcpy(result, RTEXT_DATA(rt), rt->text_len);
	result[rt->text_len] = '\0';

	PG_RETURN_CSTRING(result);
}

/*
 * vgraph_in: Parse graph structure
 * Format: "{nodes:5,edges:[[0,1],[1,2],[2,3]]}"
 */
PG_FUNCTION_INFO_V1(vgraph_in);
Datum
vgraph_in(PG_FUNCTION_ARGS)
{
	char	   *str = PG_GETARG_CSTRING(0);
	VectorGraph *result;
	int32		num_nodes;
	int32		num_edges;
	GraphEdge  *edges;
	char	   *ptr;
	char	   *endptr;
	int			size;
	int			edge_capacity;

	num_nodes = 0;
	num_edges = 0;
	edge_capacity = 32;

	/* Parse JSON-like format */
	ptr = str;
	while (isspace((unsigned char) *ptr))
		ptr++;

	if (*ptr != '{')
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("vgraph must start with '{'")));
	ptr++;

	/* Parse nodes */
	while (isspace((unsigned char) *ptr))
		ptr++;

	if (strncmp(ptr, "nodes:", 6) == 0)
	{
		ptr += 6;
		num_nodes = strtol(ptr, &endptr, 10);
		if (ptr == endptr || num_nodes <= 0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("invalid nodes value in "
							"vgraph")));
		ptr = endptr;
	}

	if (num_nodes == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("vgraph must specify nodes")));

	edges = NULL;
	NDB_ALLOC(edges, GraphEdge, edge_capacity);

	/* Parse edges array */
	while (isspace((unsigned char) *ptr) || *ptr == ',')
		ptr++;

	if (strncmp(ptr, "edges:", 6) == 0)
	{
		ptr += 6;
		while (isspace((unsigned char) *ptr))
			ptr++;

		if (*ptr != '[')
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("edges must be an array")));
		ptr++;

		/* Parse edge pairs */
		while (*ptr && *ptr != ']')
		{
			int32		from_node,
						to_node;

			while (isspace((unsigned char) *ptr) || *ptr == ',')
				ptr++;

			if (*ptr == ']')
				break;

			if (*ptr != '[')
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
						 errmsg("each edge must be an "
								"array [from,to]")));
			ptr++;

			/* Parse from node */
			while (isspace((unsigned char) *ptr))
				ptr++;

			from_node = strtol(ptr, &endptr, 10);
			if (ptr == endptr)
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
						 errmsg("invalid from node")));

			if (from_node < 0 || from_node >= num_nodes)
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("from node %d out of "
								"range [0, %d)",
								from_node,
								num_nodes)));

			ptr = endptr;

			/* Parse comma */
			while (isspace((unsigned char) *ptr))
				ptr++;
			if (*ptr != ',')
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
						 errmsg("expected ',' between "
								"edge nodes")));
			ptr++;

			/* Parse to node */
			while (isspace((unsigned char) *ptr))
				ptr++;

			to_node = strtol(ptr, &endptr, 10);
			if (ptr == endptr)
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
						 errmsg("invalid to node")));

			if (to_node < 0 || to_node >= num_nodes)
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("to node %d out of "
								"range [0, %d)",
								to_node,
								num_nodes)));

			ptr = endptr;

			/* Close edge array */
			while (isspace((unsigned char) *ptr))
				ptr++;
			if (*ptr != ']')
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
						 errmsg("expected ']' after "
								"edge pair")));
			ptr++;

			/* Store edge */
			if (num_edges >= edge_capacity)
			{
				edge_capacity *= 2;
				edges = (GraphEdge *) repalloc(edges,
											   sizeof(GraphEdge) * edge_capacity);
			}

			edges[num_edges].src_idx = from_node;
			edges[num_edges].dst_idx = to_node;
			edges[num_edges].edge_type = 0; /* Default edge type */
			edges[num_edges].weight = 1.0;	/* Default weight */
			num_edges++;
		}

		if (*ptr == ']')
			ptr++;
	}

	/* Build result - simplified: no node IDs, just edges */
	size = sizeof(VectorGraph) + sizeof(GraphEdge) * num_edges;
	result = (VectorGraph *) palloc0(size);
	SET_VARSIZE(result, size);

	result->num_nodes = num_nodes;
	result->num_edges = num_edges;
	result->edge_types = 0;		/* No labeled edge types in simple format */

	memcpy(VGRAPH_EDGES(result), edges, sizeof(GraphEdge) * num_edges);

	NDB_FREE(edges);

	PG_RETURN_POINTER(result);
}

/*
 * vgraph_out: Convert graph to text
 */
PG_FUNCTION_INFO_V1(vgraph_out);
Datum
vgraph_out(PG_FUNCTION_ARGS)
{
	VectorGraph *graph = (VectorGraph *) PG_GETARG_POINTER(0);
	GraphEdge  *edges;
	StringInfoData buf;
	int			i;

	edges = VGRAPH_EDGES(graph);

	initStringInfo(&buf);

	appendStringInfo(&buf, "{nodes:%d,edges:[", graph->num_nodes);

	for (i = 0; i < graph->num_edges; i++)
	{
		if (i > 0)
			appendStringInfoChar(&buf, ',');
		appendStringInfo(
						 &buf, "[%d,%d]", edges[i].src_idx, edges[i].dst_idx);
	}

	appendStringInfoString(&buf, "]}");

	PG_RETURN_CSTRING(buf.data);
}

/*
 * vectorp_dims: Get dimensions of packed vector
 */
PG_FUNCTION_INFO_V1(vectorp_dims);
Datum
vectorp_dims(PG_FUNCTION_ARGS)
{
	VectorPacked *vec = (VectorPacked *) PG_GETARG_POINTER(0);

	PG_RETURN_INT32(vec->dim);
}

/*
 * vectorp_validate: Validate fingerprint and endianness
 */
PG_FUNCTION_INFO_V1(vectorp_validate);
Datum
vectorp_validate(PG_FUNCTION_ARGS)
{
	VectorPacked *vec = (VectorPacked *) PG_GETARG_POINTER(0);
	uint32		expected_fp;
	uint32		dim;

	dim = vec->dim;

	/* Recompute fingerprint */
	expected_fp = crc32(0L, Z_NULL, 0);
	expected_fp = crc32(expected_fp, (unsigned char *) &dim, sizeof(dim));

	if (vec->fingerprint != expected_fp)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("vectorp fingerprint mismatch: "
						"corrupted data")));

	if (vec->endian_guard != 0x01 && vec->endian_guard != 0x10)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("vectorp endianness guard invalid")));

	PG_RETURN_BOOL(true);
}

/* Helper function for compute_stats callback (PostgreSQL 16+ API) */
static void
vector_compute_stats(VacAttrStats * stats, AnalyzeAttrFetchFunc fetchfunc, int samplerows, double totalrows)
{
	int			i;
	int			null_cnt = 0;
	int			nonnull_cnt = 0;
	int			dim = 0;
	float	   *norms = NULL;
	float	   *dim_means = NULL;
	float	   *dim_mins = NULL;
	float	   *dim_maxs = NULL;
	int			max_sample_dims = 10;
	Vector	   *vec;
	float		vec_norm;
	float		min_norm = FLT_MAX;
	float		max_norm = 0.0f;
	Datum		value;
	bool		isnull;
	int			sample_size;

	if (samplerows <= 0)
	{
		stats->stats_valid = true;
		stats->stanullfrac = 0.0;
		stats->stawidth = 0;
		return;
	}

	sample_size = samplerows;

	NDB_ALLOC(norms, float, sample_size);
	if (max_sample_dims > 0)
	{
		NDB_DECLARE(float *, dim_means);
		NDB_DECLARE(float *, dim_mins);
		NDB_DECLARE(float *, dim_maxs);
		NDB_ALLOC(dim_means, float, max_sample_dims);
		NDB_ALLOC(dim_mins, float, max_sample_dims);
		NDB_ALLOC(dim_maxs, float, max_sample_dims);
		/* Initialize dim_means to zero */
		memset(dim_means, 0, sizeof(float) * max_sample_dims);
		for (i = 0; i < max_sample_dims; i++)
		{
			dim_mins[i] = FLT_MAX;
			dim_maxs[i] = -FLT_MAX;
		}
	}

	for (i = 0; i < sample_size; i++)
	{
		value = fetchfunc(stats, i, &isnull);
		if (isnull)
		{
			null_cnt++;
			continue;
		}

		nonnull_cnt++;
		vec = (Vector *) DatumGetPointer(value);
		if (vec == NULL)
			continue;

		vec = (Vector *) PG_DETOAST_DATUM(value);

		if (dim == 0)
			dim = VECTOR_DIM(vec);

		vec_norm = 0.0f;
		{
			const float *data = VECTOR_DATA(vec);
			int			j;

			for (j = 0; j < dim; j++)
			{
				vec_norm += data[j] * data[j];
			}
			vec_norm = sqrtf(vec_norm);
		}

		if (nonnull_cnt <= sample_size)
		{
			norms[nonnull_cnt - 1] = vec_norm;
			if (vec_norm < min_norm)
				min_norm = vec_norm;
			if (vec_norm > max_norm)
				max_norm = vec_norm;
		}

		if (dim_means && dim_mins && dim_maxs && dim > 0)
		{
			const float *data = VECTOR_DATA(vec);
			int			sample_dims = Min(dim, max_sample_dims);
			int			j;

			for (j = 0; j < sample_dims; j++)
			{
				float		val = data[j];

				dim_means[j] = (dim_means[j] * (nonnull_cnt - 1) + val) / nonnull_cnt;
				if (val < dim_mins[j])
					dim_mins[j] = val;
				if (val > dim_maxs[j])
					dim_maxs[j] = val;
			}
		}
	}

	stats->stats_valid = true;
	if (null_cnt + nonnull_cnt > 0)
		stats->stanullfrac = (double) null_cnt / (null_cnt + nonnull_cnt);
	else
		stats->stanullfrac = 0.0;

	if (dim > 0)
		stats->stawidth = sizeof(Vector) + (dim * sizeof(float));
	else
		stats->stawidth = sizeof(Vector);

	if (nonnull_cnt > 0)
	{
		stats->stadistinct = -1.0;
	}

	if (norms)
		NDB_FREE(norms);
	if (dim_means)
		NDB_FREE(dim_means);
	if (dim_mins)
		NDB_FREE(dim_mins);
	if (dim_maxs)
		NDB_FREE(dim_maxs);
}

/*
 * vector_analyze: ANALYZE hook for collecting vector column statistics
 * This function is called during ANALYZE to collect statistics about vector columns
 * that can help the query planner make better decisions.
 *
 * Collects:
 * - Dimensionality (all vectors should have same dimension)
 * - Vector norm statistics (min, max, mean, stddev)
 * - Per-dimension statistics (min, max, mean for first few dimensions)
 * - Sample vectors for distance estimation
 */
PG_FUNCTION_INFO_V1(vector_analyze);
Datum
vector_analyze(PG_FUNCTION_ARGS)
{
	VacAttrStats *stats = (VacAttrStats *) PG_GETARG_POINTER(0);

	if (!stats)
		PG_RETURN_BOOL(false);

	stats->compute_stats = vector_compute_stats;
	stats->minrows = 300;

	PG_RETURN_BOOL(true);
}

/* ===================== BINARY VECTOR TYPE ===================== */

/*
 * BinaryVec: Packed binary vector for efficient quantization
 * Stores bits in a compact format where each bit represents a binary value
 */
typedef struct BinaryVec
{
	int32		vl_len_;		/* varlena header */
	int32		dim;			/* dimension (number of bits) */
	uint8		data[FLEXIBLE_ARRAY_MEMBER];	/* packed bits */
}			BinaryVec;

#define BINARYVEC_SIZE(dim) (offsetof(BinaryVec, data) + ((dim + 7) / 8))
#define BINARYVEC_DIMENSION(bv) ((bv)->dim)

/*
 * binaryvec_in: Parse binaryvec from text
 * Format: "[1,0,1,0]" or "1010" (binary string)
 */
PG_FUNCTION_INFO_V1(binaryvec_in);
Datum
binaryvec_in(PG_FUNCTION_ARGS)
{
	char	   *str = PG_GETARG_CSTRING(0);
	BinaryVec  *result;
	int			dim = 0;
	char	   *ptr = str;
	char	   *count_ptr;
	int			bit_index;
	int			bit_value;

	/* Skip whitespace */
	while (*ptr && isspace(*ptr))
		ptr++;

	/* Check for array format [1,0,1,0] */
	if (*ptr == '[')
	{
		ptr++;					/* skip '[' */

		/* Count bits and validate */
		count_ptr = ptr;
		while (*count_ptr)
		{
			if (*count_ptr == '0' || *count_ptr == '1')
			{
				dim++;
			}
			else if (*count_ptr == ',' || *count_ptr == ']' || isspace(*count_ptr))
			{
				/* valid separators */
			}
			else
			{
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
						 errmsg("binaryvec: invalid character '%c' in array format", *count_ptr)));
			}
			count_ptr++;
		}

		if (dim == 0)
		{
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("binaryvec: empty array not allowed")));
		}

		/* Allocate result */
		result = (BinaryVec *) palloc0(BINARYVEC_SIZE(dim));
		SET_VARSIZE(result, BINARYVEC_SIZE(dim));
		result->dim = dim;

		/* Parse bits */
		ptr = str + 1;			/* skip '[' */
		bit_index = 0;
		while (*ptr && *ptr != ']')
		{
			if (*ptr == '0' || *ptr == '1')
			{
				bit_value = *ptr - '0';
				if (bit_value)
				{
					int			byte_idx = bit_index / 8;

					if (byte_idx < 0 || byte_idx >= result->dim)
					{
						ereport(ERROR,
								(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
								 errmsg("neurondb: bit index %d out of bounds (dim=%d)",
										bit_index, result->dim)));
					}
					result->data[byte_idx] |= (1 << (bit_index % 8));
				}
				bit_index++;
			}
			ptr++;
		}
	}
	else
	{
		/* Binary string format "1010" */
		while (*ptr)
		{
			if (*ptr == '0' || *ptr == '1')
			{
				dim++;
			}
			else if (!isspace(*ptr))
			{
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
						 errmsg("binaryvec: invalid character '%c' in binary string format", *ptr)));
			}
			ptr++;
		}

		if (dim == 0)
		{
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("binaryvec: empty binary string not allowed")));
		}

		/* Allocate result */
		result = (BinaryVec *) palloc0(BINARYVEC_SIZE(dim));
		SET_VARSIZE(result, BINARYVEC_SIZE(dim));
		result->dim = dim;

		/* Parse bits */
		ptr = str;
		bit_index = 0;
		while (*ptr)
		{
			if (*ptr == '0' || *ptr == '1')
			{
				bit_value = *ptr - '0';
				if (bit_value)
				{
					int			byte_idx = bit_index / 8;

					if (byte_idx < 0 || byte_idx >= result->dim)
					{
						ereport(ERROR,
								(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
								 errmsg("neurondb: bit index %d out of bounds (dim=%d)",
										bit_index, result->dim)));
					}
					result->data[byte_idx] |= (1 << (bit_index % 8));
				}
				bit_index++;
			}
			ptr++;
		}
	}

	PG_RETURN_POINTER(result);
}

/*
 * binaryvec_out: Convert binaryvec to text
 * Output format: "[1,0,1,0]"
 */
PG_FUNCTION_INFO_V1(binaryvec_out);
Datum
binaryvec_out(PG_FUNCTION_ARGS)
{
	BinaryVec  *bv = (BinaryVec *) PG_GETARG_VARLENA_P(0);
	StringInfoData buf;
	int			i;
	int			byte_index;
	int			bit_index;
	int			bit_value;

	initStringInfo(&buf);
	appendStringInfoChar(&buf, '[');

	for (i = 0; i < bv->dim; i++)
	{
		if (i > 0)
		{
			appendStringInfoString(&buf, ",");
		}

		byte_index = i / 8;
		bit_index = i % 8;
		bit_value = (bv->data[byte_index] >> bit_index) & 1;

		appendStringInfoChar(&buf, bit_value ? '1' : '0');
	}

	appendStringInfoChar(&buf, ']');

	PG_RETURN_CSTRING(buf.data);
}

/*
 * binaryvec_hamming_distance: Compute Hamming distance between two binary vectors
 * Returns the number of positions where the bits differ
 */
PG_FUNCTION_INFO_V1(binaryvec_hamming_distance);
Datum
binaryvec_hamming_distance(PG_FUNCTION_ARGS)
{
	BinaryVec  *bv1 = (BinaryVec *) PG_GETARG_VARLENA_P(0);
	BinaryVec  *bv2 = (BinaryVec *) PG_GETARG_VARLENA_P(1);
	int			distance = 0;
	int			i;

	if (bv1->dim != bv2->dim)
	{
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("binaryvec: dimensions must match: %d vs %d", bv1->dim, bv2->dim)));
	}

	/* Count differing bits */
	for (i = 0; i < bv1->dim; i++)
	{
		int			byte_index = i / 8;
		int			bit_index = i % 8;
		int			bit1 = (bv1->data[byte_index] >> bit_index) & 1;
		int			bit2 = (bv2->data[byte_index] >> bit_index) & 1;

		if (bit1 != bit2)
		{
			distance++;
		}
	}

	PG_RETURN_INT32(distance);
}
