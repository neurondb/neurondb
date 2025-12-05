/*-------------------------------------------------------------------------
 *
 * ml_lasso_regression_internal.h
 *    Internal structures for Lasso Regression
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    include/ml_lasso_regression_internal.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef ML_LASSO_REGRESSION_INTERNAL_H
#define ML_LASSO_REGRESSION_INTERNAL_H

#include "postgres.h"

typedef struct LassoModel
{
	int32		model_id;
	int			n_features;
	int			n_samples;
	double		intercept;
	double	   *coefficients;
	double		lambda;
	int			max_iters;
	double		r_squared;
	double		mse;
	double		mae;
} LassoModel;

#endif							/* ML_LASSO_REGRESSION_INTERNAL_H */
