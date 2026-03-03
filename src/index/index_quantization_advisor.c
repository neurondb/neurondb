/*-------------------------------------------------------------------------
 *
 * index_quantization_advisor.c
 *    Adaptive quantization selection advisor
 *
 * Analyzes vector distribution and query patterns to recommend optimal
 * quantization strategy (FP16, INT8, PQ, etc.).
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/index/index_quantization_advisor.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "neurondb_types.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "catalog/pg_type.h"
#include "parser/parse_type.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include <math.h>
#include <float.h>

/*
 * Quantization strategy recommendation
 */
typedef enum QuantizationStrategy
{
	QUANT_NONE,					/* Full precision */
	QUANT_FP16,					/* Half precision */
	QUANT_INT8,					/* 8-bit integer */
	QUANT_PQ,					/* Product Quantization */
	QUANT_BINARY				/* Binary quantization */
}			QuantizationStrategy;

/*
 * Analyze vector distribution and recommend quantization
 */
static QuantizationStrategy
analyze_vector_distribution(const float *vectors, int num_vectors, int dim)
{
	float		min_val = FLT_MAX;
	float		max_val = -FLT_MAX;
	float		mean = 0.0f;
	float		variance = 0.0f;
	int			i,
				d;

	/* Compute statistics */
	for (i = 0; i < num_vectors; i++)
	{
		for (d = 0; d < dim; d++)
		{
			float		val = vectors[i * dim + d];

			if (val < min_val)
				min_val = val;
			if (val > max_val)
				max_val = val;
			mean += val;
		}
	}

	mean /= (num_vectors * dim);

	/* Compute variance */
	for (i = 0; i < num_vectors; i++)
	{
		for (d = 0; d < dim; d++)
		{
			float		diff = vectors[i * dim + d] - mean;

			variance += diff * diff;
		}
	}

	variance /= (num_vectors * dim);
	float		stddev = sqrtf(variance);

	/* Recommend strategy based on distribution */
	if (stddev < 0.1f)
	{
		/* Low variance - use binary quantization */
		return QUANT_BINARY;
	}
	else if (stddev < 1.0f && fabsf(min_val) < 10.0f && fabsf(max_val) < 10.0f)
	{
		/* Moderate variance, bounded range - use INT8 */
		return QUANT_INT8;
	}
	else if (stddev < 5.0f)
	{
		/* Medium variance - use FP16 */
		return QUANT_FP16;
	}
	else
	{
		/* High variance - use PQ for better compression */
		return QUANT_PQ;
	}
}

/*
 * SQL function: Recommend quantization strategy
 */
PG_FUNCTION_INFO_V1(vector_quantization_advise);
Datum
vector_quantization_advise(PG_FUNCTION_ARGS)
{
	ArrayType *vectors_array = PG_GETARG_ARRAYTYPE_P(0);
	Datum	   *elems;
	bool	   *nulls;
	int			nvec;
	Vector	   **vectors = NULL;
	float	   *flat_vectors = NULL;
	int			dim = 0;
	QuantizationStrategy strategy;
	const char *strategy_name;

	/* Extract vectors from array */
	/* Note: This assumes vector type OID - in production, look up from catalog */
	Oid			vectorOid = InvalidOid;
	
	/* Try to get vector type OID */
	{
		List	   *names = list_make2(makeString("public"), makeString("vector"));
		vectorOid = LookupTypeNameOid(NULL, makeTypeNameFromNameList(names), true);
		list_free(names);
	}

	if (!OidIsValid(vectorOid))
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_OBJECT),
				 errmsg("vector type not found")));

	deconstruct_array(vectors_array, vectorOid, -1, false, 'i',
					  &elems, &nulls, &nvec);

	if (nvec == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector array must not be empty")));

	/* Get dimension from first vector */
	vectors = (Vector **) palloc(nvec * sizeof(Vector *));
	for (int i = 0; i < nvec; i++)
	{
		if (nulls[i])
			continue;
		vectors[i] = DatumGetVectorP(elems[i]);
		NDB_CHECK_VECTOR_VALID(vectors[i]);
		if (dim == 0)
			dim = VECTOR_SIZE(vectors[i]);
		else if (VECTOR_SIZE(vectors[i]) != dim)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("all vectors must have the same dimension")));
	}

	/* Flatten vectors */
	flat_vectors = (float *) palloc(nvec * dim * sizeof(float));
	for (int i = 0; i < nvec; i++)
	{
		if (nulls[i])
			continue;
		float	   *vec_data = VECTOR_DATA(vectors[i]);
		for (int d = 0; d < dim; d++)
			flat_vectors[i * dim + d] = vec_data[d];
	}

	/* Analyze and recommend */
	strategy = analyze_vector_distribution(flat_vectors, nvec, dim);

	switch (strategy)
	{
		case QUANT_NONE:
			strategy_name = "none";
			break;
		case QUANT_FP16:
			strategy_name = "fp16";
			break;
		case QUANT_INT8:
			strategy_name = "int8";
			break;
		case QUANT_PQ:
			strategy_name = "pq";
			break;
		case QUANT_BINARY:
			strategy_name = "binary";
			break;
		default:
			strategy_name = "unknown";
	}

	pfree(flat_vectors);
	pfree(vectors);

	PG_RETURN_TEXT_P(cstring_to_text(strategy_name));
}

