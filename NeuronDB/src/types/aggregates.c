/*-------------------------------------------------------------------------
 *
 * aggregates.c
 *		Vector aggregate functions for SQL GROUP BY operations
 *
 * This file implements aggregate functions for vectors including AVG,
 * SUM, and centroid calculations. Uses transition and final functions
 * following PostgreSQL aggregate function conventions.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  src/types/aggregates.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "neurondb_macros.h"
#include "neurondb_spi.h"
#include "fmgr.h"
#include "utils/array.h"
#include <string.h>
#include "neurondb_safe_memory.h"
#include "neurondb_validation.h"

/*
 * State for vector aggregate functions
 */
typedef struct VectorAggState
{
	int32		dim;
	int64		count;
	double	   *sum;			/* Running sum for each dimension */
}			VectorAggState;

/*
 * Vector AVG aggregate - transition function
 */
PG_FUNCTION_INFO_V1(vector_avg_transfn);
Datum
vector_avg_transfn(PG_FUNCTION_ARGS)
{
	MemoryContext aggcontext;
	VectorAggState *state;
	Vector	   *vec;
	int			i;

	if (!AggCheckCallContext(fcinfo, &aggcontext))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector_avg_transfn called in "
						"non-aggregate context")));

	vec = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(vec);

	if (PG_ARGISNULL(0))
	{
		/* First call - initialize state */
		MemoryContext oldcontext = MemoryContextSwitchTo(aggcontext);

		state = (VectorAggState *) palloc0(sizeof(VectorAggState));
		state->dim = vec->dim;
		state->count = 0;
		state->sum = (double *) palloc0(sizeof(double) * vec->dim);

		MemoryContextSwitchTo(oldcontext);
	}
	else
	{
		state = (VectorAggState *) PG_GETARG_POINTER(0);

		if (state->dim != vec->dim)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("vector dimensions must be "
							"consistent")));
	}

	/* Accumulate */
	for (i = 0; i < vec->dim; i++)
		state->sum[i] += vec->data[i];

	state->count++;

	PG_RETURN_POINTER(state);
}

/*
 * Vector AVG aggregate - final function
 */
PG_FUNCTION_INFO_V1(vector_avg_finalfn);
Datum
vector_avg_finalfn(PG_FUNCTION_ARGS)
{
	VectorAggState *state;
	Vector	   *result;
	int			i;

	if (PG_ARGISNULL(0))
		PG_RETURN_NULL();

	state = (VectorAggState *) PG_GETARG_POINTER(0);

	if (state->count == 0)
		PG_RETURN_NULL();

	result = new_vector(state->dim);

	for (i = 0; i < state->dim; i++)
		result->data[i] = state->sum[i] / state->count;

	PG_RETURN_VECTOR_P(result);
}

/*
 * Vector SUM aggregate - final function
 */
PG_FUNCTION_INFO_V1(vector_sum_finalfn);
Datum
vector_sum_finalfn(PG_FUNCTION_ARGS)
{
	VectorAggState *state;
	Vector	   *result;
	int			i;

	if (PG_ARGISNULL(0))
		PG_RETURN_NULL();

	state = (VectorAggState *) PG_GETARG_POINTER(0);

	if (state->count == 0)
		PG_RETURN_NULL();

	result = new_vector(state->dim);

	for (i = 0; i < state->dim; i++)
		result->data[i] = state->sum[i];

	PG_RETURN_VECTOR_P(result);
}

/*
 * Vector centroid (same as AVG)
 */
PG_FUNCTION_INFO_V1(vector_centroid);
Datum
vector_centroid(PG_FUNCTION_ARGS)
{
	return vector_avg_finalfn(fcinfo);
}
