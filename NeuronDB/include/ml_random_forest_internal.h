/*-------------------------------------------------------------------------
 *
 * ml_random_forest_internal.h
 *	  Shared internal structures for Random Forest support.
 *
 *-------------------------------------------------------------------------
 */

#ifndef ML_RANDOM_FOREST_INTERNAL_H
#define ML_RANDOM_FOREST_INTERNAL_H

#include "postgres.h"

#include "gtree.h"

typedef struct RFModel
{
	int32		model_id;
	int			n_features;
	int			n_samples;
	int			n_classes;
	double		majority_value;
	double		majority_fraction;
	double		gini_impurity;
	int		   *class_counts;
	double	   *feature_means;
	double	   *feature_variances;
	GTree	   *tree;
	int			split_feature;
	double		split_threshold;
	double		second_value;
	double		second_fraction;
	double	   *feature_importance;
	double		label_entropy;
	double		max_deviation;
	double		max_split_deviation;
	double		left_branch_value;
	double		left_branch_fraction;
	double		right_branch_value;
	double		right_branch_fraction;
	int			feature_limit;
	double	   *left_branch_means;
	double	   *right_branch_means;
	int			tree_count;
	GTree	  **trees;
	double	   *tree_majority;
	double	   *tree_majority_fraction;
	double	   *tree_second;
	double	   *tree_second_fraction;
	double	   *tree_oob_accuracy;
	double		oob_accuracy;
} RFModel;

#endif							/* ML_RANDOM_FOREST_INTERNAL_H */
