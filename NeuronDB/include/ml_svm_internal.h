/*-------------------------------------------------------------------------
 *
 * ml_svm_internal.h
 *    Internal structures for Support Vector Machine
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    include/ml_svm_internal.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef ML_SVM_INTERNAL_H
#define ML_SVM_INTERNAL_H

#include "postgres.h"

typedef struct SVMModel
{
	int32		model_id;
	int			n_features;
	int			n_samples;
	int			n_support_vectors;
	double		bias;
	double	   *alphas;
	float	   *support_vectors;
	int		   *support_vector_indices;
	double	   *support_labels; /* y_i for each SV, in {-1, 1} */
	double		C;
	int			max_iters;
}			SVMModel;

#endif							/* ML_SVM_INTERNAL_H */
