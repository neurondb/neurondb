/*-------------------------------------------------------------------------
 *
 * ml_decision_tree_internal.h
 *    Internal structures for Decision Tree
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    include/ml_decision_tree_internal.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef ML_DECISION_TREE_INTERNAL_H
#define ML_DECISION_TREE_INTERNAL_H

#include "postgres.h"

/* Tree node structure */
typedef struct DTNode
{
	bool		is_leaf;
	double		leaf_value;		/* For leaves: class (classification) or value
								 * (regression) */
	int			feature_idx;	/* For internal nodes: feature to split on */
	float		threshold;		/* For internal nodes: split threshold */
	struct DTNode *left;		/* Samples <= threshold */
	struct DTNode *right;		/* Samples > threshold */
} DTNode;

/* Decision Tree model structure */
typedef struct DTModel
{
	int32		model_id;
	int			n_features;
	int			n_samples;
	int			max_depth;
	int			min_samples_split;
	DTNode	   *root;
} DTModel;

#endif							/* ML_DECISION_TREE_INTERNAL_H */
