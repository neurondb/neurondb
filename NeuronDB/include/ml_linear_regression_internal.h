/*-------------------------------------------------------------------------
 *
 * ml_linear_regression_internal.h
 *    Internal structures for Linear Regression
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    include/ml_linear_regression_internal.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef ML_LINEAR_REGRESSION_INTERNAL_H
#define ML_LINEAR_REGRESSION_INTERNAL_H

#include "postgres.h"

typedef struct LinRegModel
{
	int32		model_id;
	int			n_features;
	int			n_samples;
	double		intercept;
	double	   *coefficients;
	double		r_squared;
	double		mse;
	double		mae;
} LinRegModel;

#endif							/* ML_LINEAR_REGRESSION_INTERNAL_H */
