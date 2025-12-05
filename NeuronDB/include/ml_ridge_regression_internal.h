/*-------------------------------------------------------------------------
 *
 * ml_ridge_regression_internal.h
 *    Internal structures for Ridge Regression
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    include/ml_ridge_regression_internal.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef ML_RIDGE_REGRESSION_INTERNAL_H
#define ML_RIDGE_REGRESSION_INTERNAL_H

#include "postgres.h"

typedef struct RidgeModel
{
	int32		model_id;
	int			n_features;
	int			n_samples;
	double		intercept;
	double	   *coefficients;
	double		lambda;
	double		r_squared;
	double		mse;
	double		mae;
} RidgeModel;

#endif							/* ML_RIDGE_REGRESSION_INTERNAL_H */
