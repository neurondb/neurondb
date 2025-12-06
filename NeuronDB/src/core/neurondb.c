/*-------------------------------------------------------------------------
 *
 * neurondb.c
 *	  Core implementation of NeurondB vector type and operations
 *
 * This file contains the main entry point and shared vector utilities
 * including type I/O functions, vector construction, normalization,
 * arithmetic operations, and array conversions.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  src/core/neurondb.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "fmgr.h"
#include "lib/stringinfo.h"
#include "libpq/pqformat.h"
#include "utils/array.h"
#include "utils/lsyscache.h"
#include "utils/guc.h"
#include "catalog/pg_type.h"
#include <math.h>
#include <float.h>
#include <ctype.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

PG_MODULE_MAGIC;

/* GUC variables are now defined in neurondb_guc.c */

extern void neurondb_worker_fini(void);

/*
 * new_vector - Allocate a new vector structure
 *
 * Allocates and initializes a new Vector structure with the specified
 * dimension. Validates dimension bounds and allocates memory for the
 * vector data.
 *
 * Parameters:
 *   dim - Dimension of the vector (must be between 1 and VECTOR_MAX_DIM)
 *
 * Returns:
 *   Pointer to newly allocated Vector structure
 *
 * Notes:
 *   Memory is allocated in CurrentMemoryContext. The vector is zero-initialized.
 *   Errors are reported if dimension is out of valid range.
 */
Vector *
new_vector(int dim)
{
	Vector	   *result;
	int			size;

	if (dim < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector dimension must be at least 1")));

	if (dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector dimension cannot exceed %d",
						VECTOR_MAX_DIM)));

	size = VECTOR_SIZE(dim);
	result = (Vector *) palloc0(size);
	SET_VARSIZE(result, size);
	result->dim = dim;

	return result;
}

/*
 * copy_vector - Create a copy of a vector
 *
 * Creates a deep copy of the input vector by allocating new memory and
 * copying all vector data. Validates the input vector structure before copying.
 *
 * Parameters:
 *   vector - Vector to copy (must not be NULL)
 *
 * Returns:
 *   Pointer to newly allocated copy of the vector
 *
 * Notes:
 *   Memory is allocated in CurrentMemoryContext. The function validates
 *   the vector size to ensure it's within valid bounds before copying.
 */
Vector *
copy_vector(Vector *vector)
{
	Vector	   *result;
	int			size;

	if (vector == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot copy NULL vector")));

	size = VARSIZE_ANY(vector);

	/*
	 * Validate vector size before copying. The size must be at least
	 * the size of the Vector structure up to the data array (offsetof
	 * gives the byte offset of the data field). The upper bound ensures
	 * the vector does not exceed VECTOR_MAX_DIM elements, preventing
	 * buffer overflows from corrupted or malicious input. This check
	 * protects against reading beyond the allocated vector memory.
	 */
	if (size < (int) offsetof(Vector, data) || size > (int) (offsetof(Vector, data) + sizeof(float4) * VECTOR_MAX_DIM))
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("invalid vector size: %d", size)));

	result = (Vector *) palloc(size);
	memcpy(result, vector, size);
	return result;
}

/*
 * vector_in_internal - Parse vector from string representation
 *
 * Parses a vector from its string representation (e.g., "[1,2,3]") and
 * optionally validates the result. Handles whitespace and various formats.
 *
 * Parameters:
 *   str - String representation of the vector
 *   out_dim - Output parameter to receive the vector dimension
 *   check - If true, perform validation checks on the parsed vector
 *
 * Returns:
 *   Pointer to newly allocated Vector structure
 *
 * Notes:
 *   Memory is allocated in CurrentMemoryContext. The function parses
 *   comma-separated numeric values enclosed in square brackets.
 */
Vector *
vector_in_internal(char *str, int *out_dim, bool check)
{
	char	   *ptr = str;
	float4	   *data;
	int			dim = 0;
	int			capacity = 16;
	Vector	   *result;
	char	   *endptr;

	while (isspace((unsigned char) *ptr))
		ptr++;

	if (*ptr == '[' || *ptr == '{')
		ptr++;

	data = (float4 *) palloc(sizeof(float4) * capacity);

	while (*ptr && *ptr != ']' && *ptr != '}')
	{
		while (isspace((unsigned char) *ptr) || *ptr == ',')
			ptr++;

		if (*ptr == ']' || *ptr == '}' || *ptr == '\0')
			break;

		if (dim >= capacity)
		{
			capacity *= 2;
			data = (float4 *) repalloc(
									   data, sizeof(float4) * capacity);
		}

		data[dim] = strtof(ptr, &endptr);
		if (ptr == endptr)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("invalid input syntax for type "
							"vector: \"%s\"",
							str)));

		if (check && (isinf(data[dim]) || isnan(data[dim])))
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("vector values cannot be NaN or "
							"Infinity")));

		ptr = endptr;
		dim++;
	}

	if (dim == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("vector must have at least 1 "
						"dimension")));

	result = new_vector(dim);
	memcpy(result->data, data, sizeof(float4) * dim);
	NDB_FREE(data);

	if (out_dim)
		*out_dim = dim;

	return result;
}

char *
vector_out_internal(Vector *vector)
{
	StringInfoData buf;
	int			i;

	initStringInfo(&buf);
	appendStringInfoChar(&buf, '[');

	for (i = 0; i < vector->dim; i++)
	{
		if (i > 0)
			appendStringInfoChar(&buf, ',');
		appendStringInfo(&buf, "%g", vector->data[i]);
	}

	appendStringInfoChar(&buf, ']');
	return buf.data;
}

PG_FUNCTION_INFO_V1(vector_in);
Datum
vector_in(PG_FUNCTION_ARGS)
{
	char	   *str = PG_GETARG_CSTRING(0);
	Vector	   *result = vector_in_internal(str, NULL, true);

	if (PG_NARGS() >= 3)
	{
		int32		typmod = PG_GETARG_INT32(2);

		if (typmod >= 0 && result->dim != typmod)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("vector dimension %d does not "
							"match type modifier %d",
							result->dim,
							typmod)));
	}

	PG_RETURN_VECTOR_P(result);
}

PG_FUNCTION_INFO_V1(vector_out);
Datum
vector_out(PG_FUNCTION_ARGS)
{
	Vector	   *vector;
	char	   *result;

	vector = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(vector);
	result = vector_out_internal(vector);

	PG_RETURN_CSTRING(result);
}

PG_FUNCTION_INFO_V1(vector_recv);
Datum
vector_recv(PG_FUNCTION_ARGS)
{
	StringInfo	buf = (StringInfo) PG_GETARG_POINTER(0);
	Vector	   *result;
	int16		dim;
	int			i;

	dim = pq_getmsgint(buf, sizeof(int16));
	result = new_vector(dim);

	for (i = 0; i < dim; i++)
		result->data[i] = pq_getmsgfloat4(buf);

	if (PG_NARGS() >= 3)
	{
		int32		typmod = PG_GETARG_INT32(2);

		if (typmod >= 0 && result->dim != typmod)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("vector dimension %d does not "
							"match type modifier %d",
							result->dim,
							typmod)));
	}

	PG_RETURN_VECTOR_P(result);
}

PG_FUNCTION_INFO_V1(vector_send);
Datum
vector_send(PG_FUNCTION_ARGS)
{
	Vector	   *vec;
	StringInfoData buf;
	int			i;

	vec = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(vec);

	pq_begintypsend(&buf);
	pq_sendint(&buf, vec->dim, sizeof(int16));

	for (i = 0; i < vec->dim; i++)
		pq_sendfloat4(&buf, vec->data[i]);

	PG_RETURN_BYTEA_P(pq_endtypsend(&buf));
}

PG_FUNCTION_INFO_V1(vector_dims);
Datum
vector_dims(PG_FUNCTION_ARGS)
{
	Vector	   *vector = PG_GETARG_VECTOR_P(0);

	NDB_CHECK_VECTOR_VALID(vector);

	PG_RETURN_INT32(vector->dim);
}

PG_FUNCTION_INFO_V1(vector_norm);
Datum
vector_norm(PG_FUNCTION_ARGS)
{
	Vector	   *vector;
	double		sum = 0.0;
	int			i;

	vector = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(vector);

	if (vector == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot compute norm of NULL vector")));

	if (vector->dim <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("cannot compute norm of vector with dimension %d",
						vector->dim)));

	for (i = 0; i < vector->dim; i++)
	{
		double		val = (double) vector->data[i];

		if (isnan(val) || isinf(val))
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("cannot compute norm of vector containing NaN or Infinity")));
		sum += val * val;
	}

	PG_RETURN_FLOAT8(sqrt(sum));
}

void
normalize_vector(Vector *v)
{
	double		norm = 0.0;
	int			i;

	if (v == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot normalize NULL vector")));

	if (v->dim <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("cannot normalize vector with dimension %d",
						v->dim)));

	for (i = 0; i < v->dim; i++)
	{
		double		val = (double) v->data[i];

		if (isnan(val) || isinf(val))
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("cannot normalize vector containing NaN or Infinity")));
		norm += val * val;
	}

	if (norm > 0.0)
	{
		norm = sqrt(norm);
		if (norm > 0.0)
		{
			for (i = 0; i < v->dim; i++)
				v->data[i] /= norm;
		}
	}
}

Vector *
normalize_vector_new(Vector *v)
{
	Vector	   *result;

	if (v == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot normalize NULL vector")));

	result = copy_vector(v);
	normalize_vector(result);
	return result;
}

PG_FUNCTION_INFO_V1(vector_normalize);
Datum
vector_normalize(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	Vector	   *result;

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);
	result = normalize_vector_new(v);

	PG_RETURN_VECTOR_P(result);
}

PG_FUNCTION_INFO_V1(vector_concat);
Datum
vector_concat(PG_FUNCTION_ARGS)
{
	Vector	   *a;
	Vector	   *b;
	Vector	   *result;
	int			new_dim;

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);

	if (a == NULL || b == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot concatenate NULL vectors")));

	if (a->dim > VECTOR_MAX_DIM - b->dim)
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("concatenated vector dimension %d would exceed maximum %d",
						a->dim + b->dim,
						VECTOR_MAX_DIM)));

	new_dim = a->dim + b->dim;
	result = new_vector(new_dim);
	memcpy(result->data, a->data, sizeof(float4) * a->dim);
	memcpy(result->data + a->dim, b->data, sizeof(float4) * b->dim);

	PG_RETURN_VECTOR_P(result);
}

PG_FUNCTION_INFO_V1(vector_add);
Datum
vector_add(PG_FUNCTION_ARGS)
{
	Vector	   *a;
	Vector	   *b;
	Vector	   *result;
	int			i;

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);

	if (a == NULL || b == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot add NULL vectors")));

	if (a->dim != b->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("vector dimensions must match: %d vs %d",
						a->dim,
						b->dim)));

	result = new_vector(a->dim);
	for (i = 0; i < a->dim; i++)
	{
		double		sum = (double) a->data[i] + (double) b->data[i];

		if (isinf(sum))
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("vector addition resulted in infinity at index %d", i)));
		result->data[i] = (float4) sum;
	}

	PG_RETURN_VECTOR_P(result);
}

PG_FUNCTION_INFO_V1(vector_sub);
Datum
vector_sub(PG_FUNCTION_ARGS)
{
	Vector	   *a;
	Vector	   *b;
	Vector	   *result;
	int			i;

	a = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(a);
	b = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(b);

	if (a == NULL || b == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot subtract NULL vectors")));

	if (a->dim != b->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("vector dimensions must match: %d vs %d",
						a->dim,
						b->dim)));

	result = new_vector(a->dim);
	for (i = 0; i < a->dim; i++)
	{
		double		diff = (double) a->data[i] - (double) b->data[i];

		if (isinf(diff))
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("vector subtraction resulted in infinity at index %d", i)));
		result->data[i] = (float4) diff;
	}

	PG_RETURN_VECTOR_P(result);
}

PG_FUNCTION_INFO_V1(vector_mul);
Datum
vector_mul(PG_FUNCTION_ARGS)
{
	Vector	   *v;
	float8		scalar;
	Vector	   *result;
	int			i;

	v = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(v);
	scalar = PG_GETARG_FLOAT8(1);

	if (v == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cannot multiply NULL vector")));

	if (isnan(scalar) || isinf(scalar))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("scalar multiplier cannot be NaN or Infinity")));

	result = new_vector(v->dim);
	for (i = 0; i < v->dim; i++)
	{
		double		product = (double) v->data[i] * scalar;

		if (isinf(product))
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("vector multiplication resulted in infinity at index %d", i)));
		result->data[i] = (float4) product;
	}

	PG_RETURN_VECTOR_P(result);
}

PG_FUNCTION_INFO_V1(array_to_vector);
Datum
array_to_vector(PG_FUNCTION_ARGS)
{
	ArrayType  *array = PG_GETARG_ARRAYTYPE_P(0);
	Vector	   *result;
	int16		typlen;
	bool		typbyval;
	char		typalign;
	Datum	   *elems;
	bool	   *nulls;
	int			nelems;
	int			i;
	Oid			elem_type;

	if (array == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("array_to_vector: input array cannot be NULL")));

	if (ARR_NDIM(array) != 1)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("array must be one-dimensional"),
				 errdetail("array_to_vector received array with %d dimensions", ARR_NDIM(array))));

	elem_type = ARR_ELEMTYPE(array);

	/* Validate element type - must be a numeric type, not an array type */
	if (get_element_type(elem_type) != InvalidOid)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("array_to_vector: input array contains nested arrays"),
				 errdetail("array_to_vector expects a flat array of numeric values, but received an array of arrays (element type: %u)", elem_type),
				 errhint("Extract the nested array element first, or use array_to_vector on each nested array separately")));

	get_typlenbyvalalign(elem_type, &typlen, &typbyval, &typalign);

	/* Validate type information before deconstructing */
	if (typlen <= 0 && elem_type != FLOAT4OID && elem_type != FLOAT8OID)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("array_to_vector: invalid element type"),
				 errdetail("Expected numeric type (float4 or float8), got type %u with length %d", elem_type, typlen)));

	deconstruct_array(array,
					  elem_type,
					  typlen,
					  typbyval,
					  typalign,
					  &elems,
					  &nulls,
					  &nelems);

	result = new_vector(nelems);

	for (i = 0; i < nelems; i++)
	{
		if (nulls[i])
			ereport(ERROR,
					(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					 errmsg("array must not contain "
							"nulls")));

		result->data[i] = DatumGetFloat4(elems[i]);
	}

	PG_RETURN_VECTOR_P(result);
}

PG_FUNCTION_INFO_V1(vector_to_array);
/* ------------------------
 *  Typmod in/out for vector(dim)
 * ------------------------
 */
PG_FUNCTION_INFO_V1(vector_typmod_in);
Datum
vector_typmod_in(PG_FUNCTION_ARGS)
{
	ArrayType  *ta = (ArrayType *) PG_GETARG_POINTER(0);
	Datum	   *elem_values;
	int			nelems;
	int16		typlen;
	bool		typbyval;
	char		typalign;
	char	   *s;
	long		dim;

	get_typlenbyvalalign(CSTRINGOID, &typlen, &typbyval, &typalign);
	deconstruct_array(ta,
					  CSTRINGOID,
					  typlen,
					  typbyval,
					  typalign,
					  &elem_values,
					  NULL,
					  &nelems);

	if (nelems != 1)
		ereport(ERROR,
				(errcode(ERRCODE_SYNTAX_ERROR),
				 errmsg("vector typmod requires a single "
						"dimension argument")));

	s = DatumGetCString(elem_values[0]);
	dim = strtol(s, NULL, 10);
	if (dim <= 0 || dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid vector dimension %ld", dim)));

	PG_RETURN_INT32((int32) dim);
}

PG_FUNCTION_INFO_V1(vector_typmod_out);
Datum
vector_typmod_out(PG_FUNCTION_ARGS)
{
	int32		typmod = PG_GETARG_INT32(0);
	StringInfoData buf;

	if (typmod < 0)
		PG_RETURN_CSTRING(pstrdup(""));

	initStringInfo(&buf);
	appendStringInfo(&buf, "(%d)", typmod);
	PG_RETURN_CSTRING(buf.data);
}

Datum
vector_to_array(PG_FUNCTION_ARGS)
{
	Vector	   *vec;
	Datum	   *elems;
	ArrayType  *result;
	int			i;

	vec = PG_GETARG_VECTOR_P(0);
	NDB_CHECK_VECTOR_VALID(vec);

	elems = (Datum *) palloc(sizeof(Datum) * vec->dim);

	for (i = 0; i < vec->dim; i++)
		elems[i] = Float4GetDatum(vec->data[i]);

	result = construct_array(
							 elems, vec->dim, FLOAT4OID, sizeof(float4), true, 'i');

	NDB_FREE(elems);

	PG_RETURN_ARRAYTYPE_P(result);
}
