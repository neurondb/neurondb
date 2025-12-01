/*-------------------------------------------------------------------------
 *
 * ml_random_forest_shared.h
 *    Shared Random Forest utilities for CPU and GPU implementations
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    include/ml_random_forest_shared.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef ML_RANDOM_FOREST_SHARED_H
#define ML_RANDOM_FOREST_SHARED_H

#include "postgres.h"
#include "utils/jsonb.h"
#include "gtree.h"

/* Node iteration callback */
typedef void (*rf_node_iter_fn)(void *arg, const GTreeNode *node, int index);

/* Metrics specification for JSON building */
typedef struct RFMetricsSpec
{
	const char *storage;
	const char *algorithm;
	int tree_count;
	int majority_class;
	double majority_fraction;
	double gini;
	double oob_accuracy;
} RFMetricsSpec;

/* Gini impurity computation */
extern double rf_gini_impurity(const int *counts, int n_classes, int total);
extern double rf_split_gini(const int *left_counts,
	const int *right_counts,
	int class_count,
	int *left_total_out,
	int *right_total_out,
	int *left_majority_out,
	int *right_majority_out);

/* Tree utilities */
extern void
rf_tree_iterate_nodes(const GTree *tree, rf_node_iter_fn iter, void *arg);

/* Metrics JSON building */
extern Jsonb *rf_build_metrics_json(const RFMetricsSpec *spec);

#endif /* ML_RANDOM_FOREST_SHARED_H */
