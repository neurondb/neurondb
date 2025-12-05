/*-------------------------------------------------------------------------
 *
 * ml_logistic_regression_internal.h
 *    Internal structures for Logistic Regression
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    include/ml_logistic_regression_internal.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef ML_LOGISTIC_REGRESSION_INTERNAL_H
#define ML_LOGISTIC_REGRESSION_INTERNAL_H

#include "postgres.h"

typedef struct LRModel
{
	int32		model_id;
	int			n_features;
	int			n_samples;
	double		bias;
	double	   *weights;
	double		learning_rate;
	double		lambda;
	int			max_iters;
	double		final_loss;
	double		accuracy;
} LRModel;

#endif							/* ML_LOGISTIC_REGRESSION_INTERNAL_H */
