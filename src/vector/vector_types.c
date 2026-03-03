/*-------------------------------------------------------------------------
 *
 * types_core.c
 *		Core enterprise data types implementation
 *
 * Implements vectorp (packed SIMD), vecmap (sparse), rtext (retrievable
 * text), and vgraph (compact graph) data types with I/O functions.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
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

extern Datum vecmap_l2_distance(PG_FUNCTION_ARGS);
extern Datum vecmap_cosine_distance(PG_FUNCTION_ARGS);
extern Datum vecmap_inner_product(PG_FUNCTION_ARGS);

PG_FUNCTION_INFO_V1(vectorp_in);
Datum
vectorp_in(PG_FUNCTION_ARGS)
{
	char	   *endptr = NULL;
	char	   *ptr = NULL;
	char	   *str = NULL;
	float4	   *temp_data = NULL;
	int			capacity;
	int			dim;
	int			size;
	uint32		fingerprint;
	VectorPacked *result = NULL;

	/* Validate minimum argument count */
	if (PG_NARGS() < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vectorp_in requires at least 1 argument")));

	str = PG_GETARG_CSTRING(0);

	dim = 0;
	capacity = 16;

	ptr = str;
	while (isspace((unsigned char) *ptr))
		ptr++;

	if (*ptr == '[')
		ptr++;

	nalloc(temp_data, float4, capacity);

	while (*ptr && *ptr != ']')
	{
		while (isspace((unsigned char) *ptr) || *ptr == ',')
			ptr++;

		if (*ptr == ']' || *ptr == '\0')
			break;

		if (dim >= capacity)
		{
			capacity *= 2;
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
	result = (VectorPacked *) palloc0(size);
	SET_VARSIZE(result, size);

	fingerprint = crc32(0L, Z_NULL, 0);
	fingerprint = crc32(fingerprint, (unsigned char *) &dim, sizeof(dim));

	result->fingerprint = fingerprint;
	result->version = 1;
	result->dim = dim;
	result->endian_guard = 0x01;
	result->flags = 0;

	memcpy(result->data, temp_data, sizeof(float4) * dim);
	pfree(temp_data);

	PG_RETURN_POINTER(result);
}

PG_FUNCTION_INFO_V1(vectorp_out);
Datum
vectorp_out(PG_FUNCTION_ARGS)
{
	VectorPacked *vec = NULL;
	StringInfoData buf;
	int			i;

	/* Validate argument count */
	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vectorp_out requires 1 argument")));

	vec = (VectorPacked *) PG_GETARG_POINTER(0);

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

PG_FUNCTION_INFO_V1(vecmap_in);
Datum
vecmap_in(PG_FUNCTION_ARGS)
{
	char	   *endptr = NULL;
	char	   *ptr = NULL;
	char	   *str = PG_GETARG_CSTRING(0);
	float4	   *values = NULL;
	int			i;
	int			size;
	int32		dim;
	int32		nnz;
	int32	   *indices = NULL;
	VectorMap  *result = NULL;

	dim = 0;
	nnz = 0;

	(void) i;

	ptr = str;
	while (isspace((unsigned char) *ptr))
		ptr++;

	if (*ptr != '{')
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("vecmap must start with '{'")));
	ptr++;

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

	nalloc(indices, int32, nnz);
	nalloc(values, float4, nnz);

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

	size = sizeof(VectorMap) + sizeof(int32) * nnz + sizeof(float4) * nnz;
	result = (VectorMap *) palloc0(size);
	SET_VARSIZE(result, size);

	result->total_dim = dim;
	result->nnz = nnz;

	/* Suppress nonnull warning - result is guaranteed non-null after palloc0 */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnonnull"
	memcpy(VECMAP_INDICES(result), indices, sizeof(int32) * nnz);
	memcpy(VECMAP_VALUES(result), values, sizeof(float4) * nnz);
#pragma GCC diagnostic pop

	pfree(indices);
	pfree(values);

	PG_RETURN_POINTER(result);
}

PG_FUNCTION_INFO_V1(vecmap_out);
Datum
vecmap_out(PG_FUNCTION_ARGS)
{
	VectorMap  *vec = NULL;
	StringInfoData buf;
	int32	   *indices = NULL;
	float4	   *values = NULL;
	int			i;

	/* Validate argument count */
	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vecmap_out requires 1 argument")));

	vec = (VectorMap *) PG_GETARG_POINTER(0);

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

PG_FUNCTION_INFO_V1(sparsevec_in);
Datum
sparsevec_in(PG_FUNCTION_ARGS)
{
	char	   *str = NULL;
	VectorMap  *result = NULL;
	int32		dim = 0;
	int32		nnz = 0;
	int32	   *indices = NULL;
	float4	   *values = NULL;
	char	   *ptr = NULL;
	char	   *endptr = NULL;
	int			i;
	int			size;
	int			capacity;
	int			max_index = -1;

	/* Validate minimum argument count */
	if (PG_NARGS() < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: sparsevec_in requires at least 1 argument")));

	str = PG_GETARG_CSTRING(0);

	ptr = str;
	while (isspace((unsigned char) *ptr))
		ptr++;

	if (*ptr != '{')
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("sparsevec must start with '{'")));
	ptr++;

	capacity = 16;
	nalloc(indices, int32, capacity);
	nalloc(values, float4, capacity);

	while (*ptr && *ptr != '}')
	{
		while (isspace((unsigned char) *ptr) || *ptr == ',')
			ptr++;

		if (*ptr == '}' || *ptr == '\0')
			break;

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

		/* sparsevec uses 1-based indexing, convert to 0-based */
		if (indices[nnz] < 1)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("sparsevec indices must be positive (1-based)")));
		
		indices[nnz]--;  /* Convert from 1-based to 0-based */

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

	/* Handle empty sparsevec format: {}/dim */
	if (nnz == 0)
	{
		/* Check if dim was specified */
		if (dim == 0)
		{
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("sparsevec must have at least one entry or specify dim")));
		}
		/* Allow empty sparsevec if dim is specified */
	}

	/* Check for /N dimension specifier after closing brace */
	if (*ptr == '}')
		ptr++;
	
	while (isspace((unsigned char) *ptr))
		ptr++;
	
	if (*ptr == '/')
	{
		int32		specified_dim;

		ptr++;
		specified_dim = strtol(ptr, &endptr, 10);
		if (ptr != endptr && specified_dim > 0)
		{
			/* Use explicitly specified dimension */
			dim = specified_dim;
		}
	}

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

	for (i = 0; i < nnz; i++)
	{
		if (indices[i] >= dim)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("sparsevec index %d out of range [0, %d)",
							indices[i],
							dim)));
	}

	size = sizeof(VectorMap) + sizeof(int32) * nnz + sizeof(float4) * nnz;
	result = (VectorMap *) palloc0(size);
	SET_VARSIZE(result, size);

	result->total_dim = dim;
	result->nnz = nnz;

	/* Suppress nonnull warning - result is guaranteed non-null after palloc0 */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnonnull"
	memcpy(VECMAP_INDICES(result), indices, sizeof(int32) * nnz);
	memcpy(VECMAP_VALUES(result), values, sizeof(float4) * nnz);
#pragma GCC diagnostic pop

	pfree(indices);
	pfree(values);

	PG_RETURN_POINTER(result);
}

PG_FUNCTION_INFO_V1(sparsevec_out);
Datum
sparsevec_out(PG_FUNCTION_ARGS)
{
	VectorMap  *vec = NULL;
	StringInfoData buf;
	int32	   *indices = NULL;
	float4	   *values = NULL;
	int			i;

	/* Validate argument count */
	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: sparsevec_out requires 1 argument")));

	vec = (VectorMap *) PG_DETOAST_DATUM(PG_GETARG_DATUM(0));

	if (vec == NULL)
		PG_RETURN_CSTRING(pstrdup("NULL"));

	indices = VECMAP_INDICES(vec);
	values = VECMAP_VALUES(vec);

	initStringInfo(&buf);
	appendStringInfoChar(&buf, '{');

	if (vec->total_dim > 0)
		appendStringInfo(&buf, "dim:%d,", vec->total_dim);

	for (i = 0; i < vec->nnz; i++)
	{
		if (i > 0)
			appendStringInfoChar(&buf, ',');
		/* sparsevec uses 1-based indexing */
		appendStringInfo(&buf, "%d:%g", indices[i] + 1, values[i]);
	}

	appendStringInfoChar(&buf, '}');
	PG_RETURN_CSTRING(buf.data);
}

PG_FUNCTION_INFO_V1(sparsevec_recv);
Datum
sparsevec_recv(PG_FUNCTION_ARGS)
{
	StringInfo	buf;
	VectorMap  *result = NULL;
	int32		dim;
	int32		nnz;
	int			size;
	int			i;

	/* Validate minimum argument count */
	if (PG_NARGS() < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: sparsevec_recv requires at least 1 argument")));

	buf = (StringInfo) PG_GETARG_POINTER(0);

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

PG_FUNCTION_INFO_V1(sparsevec_send);
Datum
sparsevec_send(PG_FUNCTION_ARGS)
{
	VectorMap  *vec = NULL;
	StringInfoData buf;
	int32	   *indices = NULL;
	float4	   *values = NULL;
	int			i;

	/* Validate argument count */
	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: sparsevec_send requires 1 argument")));

	vec = (VectorMap *) PG_GETARG_POINTER(0);

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

PG_FUNCTION_INFO_V1(sparsevec_eq);
Datum
sparsevec_eq(PG_FUNCTION_ARGS)
{
	VectorMap  *a = NULL;
	VectorMap  *b = NULL;
	int32	   *a_indices = NULL,
			   *b_indices = NULL;
	float4	   *a_values = NULL,
			   *b_values = NULL;
	int			i;

	/* Validate argument count */
	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: sparsevec_eq requires 2 arguments")));

	a = (VectorMap *) PG_GETARG_POINTER(0);
	b = (VectorMap *) PG_GETARG_POINTER(1);

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

	for (i = 0; i < a->nnz; i++)
	{
		if (a_indices[i] != b_indices[i])
			PG_RETURN_BOOL(false);
		if (fabs(a_values[i] - b_values[i]) > 1e-6)
			PG_RETURN_BOOL(false);
	}

	PG_RETURN_BOOL(true);
}

PG_FUNCTION_INFO_V1(sparsevec_ne);
Datum
sparsevec_ne(PG_FUNCTION_ARGS)
{
	return DirectFunctionCall2(
							   sparsevec_eq, PG_GETARG_DATUM(0), PG_GETARG_DATUM(1))
		? BoolGetDatum(false)
		: BoolGetDatum(true);
}

PG_FUNCTION_INFO_V1(sparsevec_hash);
Datum
sparsevec_hash(PG_FUNCTION_ARGS)
{
	VectorMap  *v = NULL;
	int32	   *indices = NULL;
	float4	   *values = NULL;
	uint32		hash = 5381;
	int			i;

	/* Validate argument count */
	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: sparsevec_hash requires 1 argument")));

	v = (VectorMap *) PG_GETARG_POINTER(0);

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

PG_FUNCTION_INFO_V1(sparsevec_l2_distance);
Datum
sparsevec_l2_distance(PG_FUNCTION_ARGS)
{
	return vecmap_l2_distance(fcinfo);
}

PG_FUNCTION_INFO_V1(sparsevec_cosine_distance);
Datum
sparsevec_cosine_distance(PG_FUNCTION_ARGS)
{
	return vecmap_cosine_distance(fcinfo);
}

PG_FUNCTION_INFO_V1(sparsevec_inner_product);
Datum
sparsevec_inner_product(PG_FUNCTION_ARGS)
{
	return vecmap_inner_product(fcinfo);
}

PG_FUNCTION_INFO_V1(sparsevec_l2_norm);
Datum
sparsevec_l2_norm(PG_FUNCTION_ARGS)
{
	VectorMap  *v = NULL;
	float4	   *values = NULL;
	double		sum = 0.0;
	int			i;

	/* Validate argument count */
	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: sparsevec_l2_norm requires 1 argument")));

	v = (VectorMap *) PG_GETARG_POINTER(0);

	if (v == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot compute norm of NULL sparsevec")));

	values = VECMAP_VALUES(v);

	for (i = 0; i < v->nnz; i++)
		sum += (double) values[i] * (double) values[i];

	PG_RETURN_FLOAT8(sqrt(sum));
}

PG_FUNCTION_INFO_V1(sparsevec_l2_normalize);
Datum
sparsevec_l2_normalize(PG_FUNCTION_ARGS)
{
	VectorMap  *v = NULL;
	VectorMap  *result = NULL;
	float4	   *values = NULL;
	float4	   *result_values = NULL;
	int32	   *result_indices = NULL;
	double		norm = 0.0;
	int			i;
	int			size;

	/* Validate argument count */
	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: sparsevec_l2_normalize requires 1 argument")));

	v = (VectorMap *) PG_GETARG_POINTER(0);

	if (v == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot normalize NULL sparsevec")));

	values = VECMAP_VALUES(v);

	for (i = 0; i < v->nnz; i++)
		norm += (double) values[i] * (double) values[i];
	norm = sqrt(norm);

	if (norm == 0.0 || v->nnz == 0)
	{
		size = sizeof(VectorMap) + sizeof(int32) * 0 + sizeof(float4) * 0;
		result = (VectorMap *) palloc0(size);
		SET_VARSIZE(result, size);
		result->total_dim = v->total_dim;
		result->nnz = 0;
		PG_RETURN_POINTER(result);
	}

	size = sizeof(VectorMap) + sizeof(int32) * v->nnz + sizeof(float4) * v->nnz;
	result = (VectorMap *) palloc0(size);
	SET_VARSIZE(result, size);
	result->total_dim = v->total_dim;
	result->nnz = v->nnz;

	result_indices = VECMAP_INDICES(result);
	result_values = VECMAP_VALUES(result);

	/* Suppress nonnull warning - result and v are guaranteed non-null */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnonnull"
	memcpy(result_indices, VECMAP_INDICES(v), sizeof(int32) * v->nnz);
#pragma GCC diagnostic pop
	for (i = 0; i < v->nnz; i++)
		result_values[i] = (float4) ((double) values[i] / norm);

	PG_RETURN_POINTER(result);
}

PG_FUNCTION_INFO_V1(rtext_in);
Datum
rtext_in(PG_FUNCTION_ARGS)
{
	char	   *str = NULL;
	RetrievableText *result = NULL;
	int			text_len;
	int			size;

	/* Validate minimum argument count */
	if (PG_NARGS() < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: rtext_in requires at least 1 argument")));

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb: rtext_in input must not be null")));

	str = PG_GETARG_CSTRING(0);
	if (str == NULL)
		str = "";

	text_len = strlen(str);

	size = sizeof(RetrievableText) + text_len + 1;
	result = (RetrievableText *) palloc0(size);
	SET_VARSIZE(result, size);

	result->text_len = text_len;
	result->num_tokens = 0;
	result->lang_tag = 0;
	result->flags = 0;

	memcpy(RTEXT_DATA(result), str, text_len);

	PG_RETURN_POINTER(result);
}

PG_FUNCTION_INFO_V1(rtext_out);
Datum
rtext_out(PG_FUNCTION_ARGS)
{
	RetrievableText *rt = NULL;
	char *result = NULL;

	/* Validate argument count */
	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: rtext_out requires 1 argument")));

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb: rtext_out input must not be null")));

	rt = (RetrievableText *) PG_GETARG_POINTER(0);
	if (rt == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb: rtext_out invalid pointer")));

	if (rt->text_len < 0 || rt->text_len > 1048576)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("neurondb: rtext_out invalid text_len %d", rt->text_len)));

	nalloc(result, char, rt->text_len + 1);
	memcpy(result, RTEXT_DATA(rt), rt->text_len);
	result[rt->text_len] = '\0';

	PG_RETURN_CSTRING(result);
}

PG_FUNCTION_INFO_V1(vgraph_in);
Datum
vgraph_in(PG_FUNCTION_ARGS)
{
	char	   *str = NULL;
	VectorGraph *result = NULL;
	int32		num_nodes;
	int32		num_edges;
	GraphEdge  *edges = NULL;
	char	   *ptr = NULL;
	char	   *endptr = NULL;
	int			size;
	int			edge_capacity;

	/* Validate minimum argument count */
	if (PG_NARGS() < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vgraph_in requires at least 1 argument")));

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb: vgraph_in input must not be null")));

	str = PG_GETARG_CSTRING(0);
	if (str == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("neurondb: vgraph_in invalid input")));

	num_nodes = 0;
	num_edges = 0;
	edge_capacity = 32;

	ptr = str;
	while (isspace((unsigned char) *ptr))
		ptr++;

	if (*ptr != '{')
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("vgraph must start with '{'")));
	ptr++;

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

	nalloc(edges, GraphEdge, edge_capacity);

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

			while (isspace((unsigned char) *ptr))
				ptr++;
			if (*ptr != ',')
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
						 errmsg("expected ',' between "
								"edge nodes")));
			ptr++;

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

			while (isspace((unsigned char) *ptr))
				ptr++;
			if (*ptr != ']')
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
						 errmsg("expected ']' after "
								"edge pair")));
			ptr++;

			if (num_edges >= edge_capacity)
			{
				edge_capacity *= 2;
				edges = (GraphEdge *) repalloc(edges,
											   sizeof(GraphEdge) * edge_capacity);
			}

			edges[num_edges].src_idx = from_node;
			edges[num_edges].dst_idx = to_node;
			edges[num_edges].edge_type = 0;
			edges[num_edges].weight = 1.0;
			num_edges++;
		}

		if (*ptr == ']')
			ptr++;
	}

	size = sizeof(VectorGraph) + sizeof(GraphEdge) * num_edges;
	result = (VectorGraph *) palloc0(size);
	SET_VARSIZE(result, size);

	result->num_nodes = num_nodes;
	result->num_edges = num_edges;
	result->edge_types = 0;

	memcpy(VGRAPH_EDGES(result), edges, sizeof(GraphEdge) * num_edges);

	pfree(edges);

	PG_RETURN_POINTER(result);
}

PG_FUNCTION_INFO_V1(vgraph_out);
Datum
vgraph_out(PG_FUNCTION_ARGS)
{
	VectorGraph *graph = (VectorGraph *) PG_GETARG_POINTER(0);
	GraphEdge  *edges = NULL;
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

PG_FUNCTION_INFO_V1(vectorp_dims);
Datum
vectorp_dims(PG_FUNCTION_ARGS)
{
	VectorPacked *vec = NULL;

	/* Validate argument count */
	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vectorp_dims requires 1 argument")));

	vec = (VectorPacked *) PG_GETARG_POINTER(0);

	PG_RETURN_INT32(vec->dim);
}

PG_FUNCTION_INFO_V1(vectorp_validate);
Datum
vectorp_validate(PG_FUNCTION_ARGS)
{
	VectorPacked *vec = NULL;
	uint32		expected_fp;
	uint32		dim;

	/* Validate argument count */
	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vectorp_validate requires 1 argument")));

	vec = (VectorPacked *) PG_GETARG_POINTER(0);

	dim = vec->dim;

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

static void
vector_compute_stats(VacAttrStats * stats, AnalyzeAttrFetchFunc fetchfunc, int samplerows, double totalrows)
{
	int			i;
	int			null_cnt = 0;
	int			nonnull_cnt = 0;
	int			dim = 0;
	float *norms = NULL;
	float *dim_means = NULL;
	float *dim_mins = NULL;
	float *dim_maxs = NULL;
	int			max_sample_dims = 10;
	Vector *vec = NULL;
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

	nalloc(norms, float, sample_size);
	if (max_sample_dims > 0)
	{
		nalloc(dim_means, float, max_sample_dims);
		nalloc(dim_mins, float, max_sample_dims);
		nalloc(dim_maxs, float, max_sample_dims);
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
		pfree(norms);
	if (dim_means)
		pfree(dim_means);
	if (dim_mins)
		pfree(dim_mins);
	if (dim_maxs)
		pfree(dim_maxs);
}

PG_FUNCTION_INFO_V1(vector_analyze);
Datum
vector_analyze(PG_FUNCTION_ARGS)
{
	VacAttrStats *stats = NULL;

	/* Validate argument count */
	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: vector_analyze requires 1 argument")));

	stats = (VacAttrStats *) PG_GETARG_POINTER(0);

	if (!stats)
		PG_RETURN_BOOL(false);

	stats->compute_stats = vector_compute_stats;
	stats->minrows = 300;

	PG_RETURN_BOOL(true);
}

typedef struct BinaryVec
{
	int32		vl_len_;		/* varlena header */
	int32		dim;			/* dimension (number of bits) */
	uint8		data[FLEXIBLE_ARRAY_MEMBER];	/* packed bits */
}			BinaryVec;

#define BINARYVEC_SIZE(dim) (offsetof(BinaryVec, data) + ((dim + 7) / 8))
#define BINARYVEC_DIMENSION(bv) ((bv)->dim)

PG_FUNCTION_INFO_V1(binaryvec_in);
Datum
binaryvec_in(PG_FUNCTION_ARGS)
{
	char	   *str = NULL;
	BinaryVec  *result = NULL;
	int			dim = 0;
	char	   *ptr = NULL;
	char	   *count_ptr = NULL;
	int			bit_index;
	int			bit_value;

	/* Validate minimum argument count */
	if (PG_NARGS() < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: binaryvec_in requires at least 1 argument")));

	str = PG_GETARG_CSTRING(0);
	ptr = str;

	while (*ptr && isspace(*ptr))
		ptr++;

	if (*ptr == '[')
	{
		ptr++;

		count_ptr = ptr;
		while (*count_ptr)
		{
			if (*count_ptr == '0' || *count_ptr == '1')
			{
				dim++;
			}
			else if (*count_ptr == ',' || *count_ptr == ']' || isspace(*count_ptr))
			{
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

		result = (BinaryVec *) palloc0(BINARYVEC_SIZE(dim));
		SET_VARSIZE(result, BINARYVEC_SIZE(dim));
		result->dim = dim;

		ptr = str + 1;
		bit_index = 0;
		while (*ptr && *ptr != ']')
		{
			if (*ptr == '0' || *ptr == '1')
			{
				bit_value = *ptr - '0';
				if (bit_value)
				{
					int			byte_idx = bit_index / 8;
					int			max_bytes = (result->dim + 7) / 8;

					if (byte_idx < 0 || byte_idx >= max_bytes)
					{
						ereport(ERROR,
								(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
								 errmsg("neurondb: bit index %d out of bounds (dim=%d bits, max_byte=%d)",
										bit_index, result->dim, max_bytes)));
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

		result = (BinaryVec *) palloc0(BINARYVEC_SIZE(dim));
		SET_VARSIZE(result, BINARYVEC_SIZE(dim));
		result->dim = dim;

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
					int			max_bytes = (result->dim + 7) / 8;

					if (byte_idx < 0 || byte_idx >= max_bytes)
					{
						ereport(ERROR,
								(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
								 errmsg("neurondb: bit index %d out of bounds (dim=%d bits, max_byte=%d)",
										bit_index, result->dim, max_bytes)));
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

PG_FUNCTION_INFO_V1(binaryvec_out);
Datum
binaryvec_out(PG_FUNCTION_ARGS)
{
	BinaryVec  *bv = NULL;
	StringInfoData buf;
	int			i;
	int			byte_index;
	int			bit_index;
	int			bit_value;

	/* Validate argument count */
	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: binaryvec_out requires 1 argument")));

	bv = (BinaryVec *) PG_GETARG_VARLENA_P(0);

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

PG_FUNCTION_INFO_V1(binaryvec_hamming_distance);
Datum
binaryvec_hamming_distance(PG_FUNCTION_ARGS)
{
	BinaryVec  *bv1 = NULL;
	BinaryVec  *bv2 = NULL;
	int			distance = 0;
	int			i;

	/* Validate argument count */
	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: binaryvec_hamming_distance requires 2 arguments")));

	bv1 = (BinaryVec *) PG_GETARG_VARLENA_P(0);
	bv2 = (BinaryVec *) PG_GETARG_VARLENA_P(1);

	if (bv1->dim != bv2->dim)
	{
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("binaryvec: dimensions must match: %d vs %d", bv1->dim, bv2->dim)));
	}

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
