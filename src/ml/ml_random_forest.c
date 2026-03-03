/*-------------------------------------------------------------------------
 *
 * ml_random_forest.c
 *    Random forest ensemble learning.
 *
 * This module implements random forest for classification and regression using
 * bootstrap aggregating and random feature selection. Models are serialized
 * and stored in the catalog for prediction.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_random_forest.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "executor/spi.h"
#include "catalog/pg_type.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"
#include "utils/array.h"
#include "utils/memutils.h"
#include "utils/jsonb.h"
#include "common/pg_prng.h"
#include "libpq/pqformat.h"
#include "access/xact.h"

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <stdint.h>

#include "neurondb.h"
#include "neurondb_ml.h"
#include "neurondb_gpu_model.h"
#include "neurondb_gpu_bridge.h"
#include "ml_gpu_random_forest.h"
#include "neurondb_gpu.h"
#include "ml_catalog.h"
#include "neurondb_macros.h"
#include "neurondb_constants.h"
#include "neurondb_spi.h"
#include "gtree.h"
#include "ml_random_forest_internal.h"
#include "ml_random_forest_shared.h"
#include "neurondb_cuda_rf.h"
#include "neurondb_json.h"
#include "vector/vector_types.h"
#include "neurondb_validation.h"
#include "neurondb_spi_safe.h"
#include "neurondb_safe_memory.h"
#include "neurondb_sql.h"
#include "utils/elog.h"
#include "neurondb_guc.h"

#if defined(NDB_GPU_CUDA) || defined(NDB_GPU_METAL)
#ifdef NDB_GPU_CUDA
#include "neurondb_cuda_runtime.h"
#include <cublas_v2.h>
extern cublasHandle_t ndb_cuda_get_cublas_handle(void);
extern int	ndb_cuda_rf_evaluate(const bytea * model_data,
								 const float *features,
								 const int *labels,
								 int n_samples,
								 int feature_dim,
								 double *accuracy_out,
								 double *precision_out,
								 double *recall_out,
								 double *f1_out,
								 char **errstr);
extern int	ndb_cuda_rf_evaluate_batch(const bytea * model_data,
									   const float *features,
									   const int *labels,
									   int n_samples,
									   int feature_dim,
									   double *accuracy_out,
									   double *precision_out,
									   double *recall_out,
									   double *f1_out,
									   char **errstr);
#endif
#endif

#define RF_BOOTSTRAP_FRACTION 0.8
#define RF_DEFAULT_TREES 3
#define RF_MAX_DEPTH 4
#define RF_MIN_SAMPLES 5

PG_FUNCTION_INFO_V1(train_random_forest_classifier);
PG_FUNCTION_INFO_V1(predict_random_forest);
PG_FUNCTION_INFO_V1(evaluate_random_forest);

typedef struct RFSplitPair
{
	double		value;
	int			cls;
}			RFSplitPair;

typedef struct RFDataset
{
	float	   *features;
	double	   *labels;
	int			n_samples;
	int			feature_dim;
}			RFDataset;

static bool rf_select_split(const float *features,
							const double *labels,
							const int *indices,
							int count,
							int feature_dim,
							int n_classes,
							pg_prng_state *rng,
							int *feature_order,
							int *best_feature,
							double *best_threshold,
							double *best_impurity);
static int	rf_build_branch_tree(GTree *tree,
								 const float *features,
								 const double *labels,
								 const double *feature_vars,
								 int feature_dim,
								 int n_classes,
								 const int *indices,
								 int count,
								 int depth,
								 int max_depth,
								 int min_samples,
								 pg_prng_state *rng,
								 int *feature_order,
								 double *feature_importance,
								 double *max_split_deviation);
static double rf_tree_predict_row(const GTree *tree, const float *row, int dim);
static void rf_serialize_tree(StringInfo buf, const GTree *tree);
static GTree *rf_deserialize_tree(StringInfo buf);
static bytea * rf_model_serialize(const RFModel *model, uint8 training_backend);
static RFModel *rf_model_deserialize(const bytea * data, uint8 * training_backend_out);
static void rf_free_deserialized_model(RFModel *model);
static bool rf_load_model_from_catalog(int32 model_id, RFModel **out);
static bool rf_metadata_is_gpu(Jsonb * metadata);
/*
 * Helper function to safely free GPU train result
 * Extracted to avoid shadow warnings from nested PG_TRY blocks
 */
static void
rf_safe_free_gpu_result(MLGpuTrainResult *gpu_result)
{
	PG_TRY();
	{
		ndb_gpu_free_train_result(gpu_result);
	}
	PG_CATCH();
	{
		/* If cleanup fails, just flush the error and continue */
		FlushErrorState();
	}
	PG_END_TRY();
}
static bool rf_try_gpu_predict_catalog(int32 model_id,
									   const Vector *feature_vec,
									   double *result_out);
static void rf_dataset_init(RFDataset * dataset);
static void rf_dataset_free(RFDataset * dataset);
static void rf_dataset_load(const char *quoted_tbl,
							const char *quoted_feat,
							const char *quoted_label,
							RFDataset * dataset,
							StringInfo query);
void		neurondb_gpu_register_rf_model(void);

static int
rf_split_pair_cmp(const void *a, const void *b)
{
	const		RFSplitPair *pa = (const RFSplitPair *) a;
	const		RFSplitPair *pb = (const RFSplitPair *) b;

	if (pa->value < pb->value)
		return -1;
	if (pa->value > pb->value)
		return 1;
	if (pa->cls < pb->cls)
		return -1;
	if (pa->cls > pb->cls)
		return 1;
	return 0;
}

/*
 * rf_select_split
 *    Select optimal feature and threshold for splitting in random forest trees.
 *
 * This function implements the split selection algorithm for random forest
 * decision trees, which evaluates candidate features and thresholds to find
 * the split that minimizes Gini impurity. The algorithm uses random feature
 * selection, examining only a random subset of features at each node rather
 * than all features, which decorrelates trees in the forest and improves
 * generalization. The number of features to consider is determined by the
 * mtry parameter, typically set to the square root of the total feature count.
 * For each candidate feature, the algorithm collects all feature value and
 * class label pairs from the current node's samples, sorts them by feature
 * value, and evaluates potential split thresholds at midpoints between
 * adjacent distinct values. For each threshold, it computes the weighted
 * Gini impurity of the resulting left and right child nodes, selecting the
 * threshold that produces the lowest weighted impurity. The weighted impurity
 * accounts for the relative sizes of the child nodes, ensuring that splits
 * creating balanced partitions are preferred over splits that isolate a few
 * samples. This process continues until all candidate features are evaluated,
 * and the best overall split is returned.
 */
static bool
rf_select_split(const float *features,
				const double *labels,
				const int *indices,
				int count,
				int feature_dim,
				int n_classes,
				pg_prng_state *rng,
				int *feature_order,
				int *best_feature,
				double *best_threshold,
				double *best_impurity)
{
	int			mtry;
	int			candidates;
	int			f;

	if (count <= 1 || feature_dim <= 0 || n_classes <= 0)
		return false;

	if (best_feature != NULL)
		*best_feature = -1;
	if (best_threshold != NULL)
		*best_threshold = 0.0;
	if (best_impurity != NULL)
		*best_impurity = DBL_MAX;

	if (feature_order != NULL)
	{
		for (f = 0; f < feature_dim; f++)
			feature_order[f] = f;
	}

	mtry = (int) sqrt((double) feature_dim);
	if (mtry < 1)
		mtry = 1;
	if (mtry > feature_dim)
		mtry = feature_dim;
	candidates = mtry;

	if (feature_order != NULL)
	{
		for (f = 0; f < candidates; f++)
		{
			int			swap_idx;

			swap_idx = (int) pg_prng_uint64_range_inclusive(
															rng, (uint64) f, (uint64) (feature_dim - 1));
			if (swap_idx != f)
			{
				int			tmp = feature_order[f];

				feature_order[f] = feature_order[swap_idx];
				feature_order[swap_idx] = tmp;
			}
		}
	}

	if (feature_order == NULL)
		candidates = Min(candidates, feature_dim);

	for (f = 0; f < candidates; f++)
	{
		float		value;
		int			cls;
		int			feature_idx;
		int			i;
		int			idx;
		int			left_total = 0;
		int			pair_count = 0;
		int			right_total = 0;
		int		   *left_counts_tmp = NULL;
		int		   *right_counts_tmp = NULL;
		RFSplitPair *pairs = NULL;
		size_t		pair_count_tmp;
		size_t		pairs_size;

		feature_idx = (feature_order != NULL) ? feature_order[f] : f;

		if (feature_idx < 0 || feature_idx >= feature_dim)
			continue;

		pairs_size = sizeof(RFSplitPair) * (size_t) count;

		if (pairs_size > MaxAllocSize)
		{
			elog(WARNING,
				 "rf_select_split: pairs allocation size %zu exceeds MaxAllocSize (count=%d)",
				 pairs_size, count);
			return false;
		}
		pair_count_tmp = pairs_size / sizeof(RFSplitPair);
		nalloc(pairs, RFSplitPair, pair_count_tmp);
		if (pairs == NULL)
		{
			elog(WARNING, "rf_select_split: palloc failed for pairs (count=%d)", count);
			return false;
		}

		for (i = 0; i < count; i++)
		{
			idx = indices[i];

			if (idx < 0)
				continue;
			cls = (int) rint(labels[idx]);
			if (cls < 0 || cls >= n_classes)
				continue;
			value = features[idx * feature_dim + feature_idx];
			if (!isfinite(value))
				continue;

			pairs[pair_count].value = (double) value;
			pairs[pair_count].cls = cls;
			pair_count++;
		}

		if (pair_count > 1)
		{
			bool		try_gpu = (n_classes == 2 && NDB_SHOULD_TRY_GPU()
								   && neurondb_gpu_is_available()
								   && ndb_gpu_kernel_enabled("rf_split"));

			if (try_gpu)
			{
				bool		labels_ok = true;
				double		gpu_gini = DBL_MAX;
				double		gpu_threshold = 0.0;
				float	   *gpu_features = NULL;
				int			gpu_cls;
				int			gpu_left = 0;
				int			gpu_right = 0;
				uint8_t    *gpu_labels = NULL;

				nalloc(gpu_features, float, pair_count);
				NDB_CHECK_ALLOC(gpu_features, "gpu_features");
				nalloc(gpu_labels, uint8_t, pair_count);
				NDB_CHECK_ALLOC(gpu_labels, "gpu_labels");

				for (i = 0; i < pair_count; i++)
				{
					gpu_cls = pairs[i].cls;

					gpu_features[i] = (float) pairs[i].value;
					if (gpu_cls == 0)
						gpu_labels[i] = 0;
					else if (gpu_cls == 1)
						gpu_labels[i] = 1;
					else
					{
						labels_ok = false;
						break;
					}
				}

				if (labels_ok
					&& neurondb_gpu_rf_best_split_binary(
														 gpu_features,
														 gpu_labels,
														 pair_count,
														 &gpu_threshold,
														 &gpu_gini,
														 &gpu_left,
														 &gpu_right))
				{
					if (gpu_gini < *best_impurity)
					{
						*best_impurity = gpu_gini;
						*best_threshold = gpu_threshold;
						*best_feature = feature_idx;
					}

					nfree(gpu_features);
					gpu_features = NULL;
					nfree(gpu_labels);
					gpu_labels = NULL;
					nfree(pairs);
					pairs = NULL;
					continue;
				}

				nfree(gpu_features);
				gpu_features = NULL;
				nfree(gpu_labels);
				gpu_labels = NULL;
			}

			if (pair_count <= 1)
			{
				nfree(pairs);
				pairs = NULL;
				continue;
			}

			qsort(pairs,
			  pair_count,
			  sizeof(RFSplitPair),
			  rf_split_pair_cmp);
		nalloc(left_counts_tmp, int, n_classes);
		NDB_CHECK_ALLOC(left_counts_tmp, "left_counts_tmp");
		nalloc(right_counts_tmp, int, n_classes);
		NDB_CHECK_ALLOC(right_counts_tmp, "right_counts_tmp");

		for (i = 0; i < pair_count; i++)
			right_counts_tmp[pairs[i].cls]++;

		right_total = pair_count;

		for (i = 0; i < pair_count - 1; i++)
		{
			int			pair_cls = pairs[i].cls;
			double		left_imp;
			double		right_imp;
			double		weighted;
			double		threshold_candidate;

			left_counts_tmp[pair_cls]++;
			right_counts_tmp[pair_cls]--;
			left_total++;
			right_total--;

			if (pairs[i].value == pairs[i + 1].value)
				continue;
			if (left_total <= 0 || right_total <= 0)
				continue;

			left_imp = rf_gini_impurity(
										left_counts_tmp, n_classes, left_total);
			right_imp = rf_gini_impurity(
										 right_counts_tmp, n_classes, right_total);
			threshold_candidate =
				0.5 * (pairs[i].value + pairs[i + 1].value);
			weighted = ((double) left_total / (double) pair_count)
				* left_imp
				+ ((double) right_total / (double) pair_count)
				* right_imp;

			if (weighted < *best_impurity)
			{
				*best_impurity = weighted;
				*best_threshold = threshold_candidate;
				*best_feature = feature_idx;
			}
		}

			nfree(left_counts_tmp);
			left_counts_tmp = NULL;
			nfree(right_counts_tmp);
			right_counts_tmp = NULL;
			nfree(pairs);
			pairs = NULL;
		}
	}

	return (*best_feature >= 0);
}

/*
 * rf_build_branch_tree
 *    Recursively build a decision tree node using the CART algorithm.
 *
 * This function constructs a decision tree node by recursively splitting the
 * training data based on feature thresholds that minimize Gini impurity. The
 * algorithm first counts class labels in the current subset to determine the
 * majority class, which becomes the default prediction if no split is made.
 * If the current node is pure, has reached maximum depth, or contains too few
 * samples, it creates a leaf node with the majority class. Otherwise, it
 * calls rf_select_split to find the optimal feature and threshold for
 * splitting. When a valid split is found, the function partitions the indices
 * into left and right child subsets based on the split threshold, then
 * recursively builds child nodes for each subset. The recursive process
 * continues until stopping criteria are met, creating a binary tree structure
 * where internal nodes contain split conditions and leaf nodes contain class
 * predictions. The tree structure enables efficient prediction by following
 * a path from root to leaf based on feature values, making a prediction at
 * the leaf node.
 */
static int
rf_build_branch_tree(GTree *tree,
					 const float *features,
					 const double *labels,
					 const double *feature_vars,
					 int feature_dim,
					 int n_classes,
					 const int *indices,
					 int count,
					 int depth,
					 int max_depth,
					 int min_samples,
					 pg_prng_state *rng,
					 int *feature_order,
					 double *feature_importance,
					 double *max_split_deviation)
{
	int *class_counts = NULL;
	int			majority_idx = -1;
	int			i;
	double		best_impurity = DBL_MAX;
	int			split_feature = -1;
	double		split_threshold = 0.0;
	double		gini;
	int			node_idx;

	if (tree == NULL || features == NULL || labels == NULL
		|| indices == NULL || count <= 0)
		return gtree_add_leaf(tree, 0.0);

	nalloc(class_counts, int, n_classes);
	NDB_CHECK_ALLOC(class_counts, "class_counts");

	for (i = 0; i < count; i++)
	{
		int			idx = indices[i];
		int			cls;

		if (idx < 0)
			continue;
		if (!isfinite(labels[idx]))
			continue;
		cls = (int) rint(labels[idx]);
		if (cls < 0 || cls >= n_classes)
			continue;
		class_counts[cls]++;
		if (majority_idx < 0
			|| class_counts[cls] > class_counts[majority_idx])
			majority_idx = cls;
	}

	if (majority_idx < 0)
	{
		nfree(class_counts);
		class_counts = NULL;
		return gtree_add_leaf(tree, 0.0);
	}

	gini = rf_gini_impurity(class_counts, n_classes, count);

	if (gini <= 0.0 || depth >= max_depth || count <= min_samples)
	{
		double		value = (double) majority_idx;

		nfree(class_counts);
		class_counts = NULL;
		return gtree_add_leaf(tree, value);
	}

	if (!rf_select_split(features,
						 labels,
						 indices,
						 count,
						 feature_dim,
						 n_classes,
						 rng,
						 feature_order,
						 &split_feature,
						 &split_threshold,
						 &best_impurity))
	{
		double		value = (double) majority_idx;

		nfree(class_counts);
		class_counts = NULL;
		return gtree_add_leaf(tree, value);
	}

	if (split_feature < 0)
	{
		double		value = (double) majority_idx;

		nfree(class_counts);
		class_counts = NULL;
		return gtree_add_leaf(tree, value);
	}

		{
			int			left_count = 0;
			int			right_count = 0;

			int *left_indices = NULL;
			int *right_indices = NULL;
			nalloc(left_indices, int, count);
		NDB_CHECK_ALLOC(left_indices, "left_indices");
		nalloc(right_indices, int, count);
		NDB_CHECK_ALLOC(right_indices, "right_indices");

		for (i = 0; i < count; i++)
		{
			int			idx = indices[i];
			float		value;

			if (idx < 0)
				continue;
			value = features[idx * feature_dim + split_feature];
			if (!isfinite(value))
				continue;
			if ((double) value <= split_threshold)
				left_indices[left_count++] = idx;
			else
				right_indices[right_count++] = idx;
		}

		if (left_count == 0 || right_count == 0)
		{
			double		value = (double) majority_idx;

			nfree(left_indices);
			left_indices = NULL;
			nfree(right_indices);
			right_indices = NULL;
			nfree(class_counts);
			class_counts = NULL;
			return gtree_add_leaf(tree, value);
		}
		else
		{
			int			left_child;
			int			right_child;

			if (feature_importance != NULL && gini > 0.0
				&& best_impurity < DBL_MAX
				&& split_feature >= 0 && split_feature < feature_dim)
			{
				double		improvement = gini - best_impurity;

				if (improvement < 0.0)
					improvement = 0.0;
				feature_importance[split_feature] +=
					improvement;
			}

			if (feature_vars != NULL && split_feature < feature_dim
				&& feature_vars[split_feature] > 0.0
				&& max_split_deviation != NULL)
			{
				double		split_dev = fabs(split_threshold)
					/ sqrt(feature_vars[split_feature]);

				if (split_dev > *max_split_deviation)
					*max_split_deviation = split_dev;
			}

			node_idx = gtree_add_split(
									   tree, split_feature, split_threshold);

			left_child = rf_build_branch_tree(tree,
											  features,
											  labels,
											  feature_vars,
											  feature_dim,
											  n_classes,
											  left_indices,
											  left_count,
											  depth + 1,
											  max_depth,
											  min_samples,
											  rng,
											  feature_order,
											  feature_importance,
											  max_split_deviation);
			right_child = rf_build_branch_tree(tree,
											   features,
											   labels,
											   feature_vars,
											   feature_dim,
											   n_classes,
											   right_indices,
											   right_count,
											   depth + 1,
											   max_depth,
											   min_samples,
											   rng,
											   feature_order,
											   feature_importance,
											   max_split_deviation);

			gtree_set_child(tree, node_idx, left_child, true);
			gtree_set_child(tree, node_idx, right_child, false);

			nfree(left_indices);
			left_indices = NULL;
			nfree(right_indices);
			right_indices = NULL;
			nfree(class_counts);
			class_counts = NULL;
			return node_idx;
		}
	}
}

static double
rf_tree_predict_row(const GTree *tree, const float *row, int dim)
{
	const GTreeNode *nodes = NULL;
	int			idx;

	if (tree == NULL || row == NULL)
		return 0.0;
	if (tree->root < 0 || tree->count <= 0)
		return 0.0;

	nodes = gtree_nodes(tree);
	idx = tree->root;

	while (idx >= 0 && idx < tree->count)
	{
		const GTreeNode *node = &nodes[idx];

		if (node->is_leaf)
			return node->value;

		if (node->feature_idx < 0 || node->feature_idx >= dim)
			return 0.0;

		if ((double) row[node->feature_idx] <= node->threshold)
			idx = node->left;
		else
			idx = node->right;
	}

	return 0.0;
}

static RFModel *rf_models = NULL;
static int	rf_model_count = 0;
static int32 rf_next_model_id = 1;

static void
rf_store_model(int32 model_id,
			   int n_features,
			   int n_samples,
			   int n_classes,
			   double majority,
			   double fraction,
			   double gini,
			   double entropy,
			   const int *class_counts,
			   const double *feature_means,
			   const double *feature_variances,
			   const double *feature_importance,
			   GTree *tree,
			   int split_feature,
			   double split_threshold,
			   double second_value,
			   double second_fraction,
			   double left_value,
			   double left_fraction,
			   double right_value,
			   double right_fraction,
			   double max_deviation,
			   double max_split_deviation,
			   int feature_limit,
			   const double *left_means,
			   const double *right_means,
			   int tree_count,
			   GTree *const *trees,
			   const double *tree_majority,
			   const double *tree_majority_fraction,
			   const double *tree_second,
			   const double *tree_second_fraction,
			   const double *tree_oob_accuracy,
			   double oob_accuracy)
{
	MemoryContext oldctx = NULL;
	int			i;
	size_t		alloc_size __attribute__((unused));

#ifdef MEMORY_CONTEXT_CHECKING
	/* Check memory context at entry */
	MemoryContext entry_ctx = CurrentMemoryContext;

	if (entry_ctx != NULL)
		MemoryContextCheck(entry_ctx);
#endif

	if (n_features <= 0 || n_features > 10000)
	{
		elog(ERROR, "rf_store_model: invalid n_features=%d (must be 1-10000)", n_features);
		return;
	}
	if (n_classes <= 0 || n_classes > 1000)
	{
		elog(ERROR, "rf_store_model: invalid n_classes=%d (must be 1-1000)", n_classes);
		return;
	}
	if (tree_count < 0 || tree_count > 10000)
	{
		elog(ERROR, "rf_store_model: invalid tree_count=%d (must be 0-10000)", tree_count);
		return;
	}
	if (feature_limit < 0 || feature_limit > n_features)
	{
		elog(ERROR, "rf_store_model: invalid feature_limit=%d (must be 0-%d, n_features=%d)",
			 feature_limit, n_features, n_features);
		return;
	}


	if (TopMemoryContext == NULL)
	{
		elog(ERROR, "rf_store_model: TopMemoryContext is NULL");
		return;
	}

	if (!IsTransactionState())
	{
		elog(WARNING, "rf_store_model: not in transaction state, cannot store model");
		return;
	}

	oldctx = MemoryContextSwitchTo(TopMemoryContext);

	if (oldctx == NULL)
	{
		elog(ERROR, "rf_store_model: failed to switch to TopMemoryContext");
		return;
	}

	if (CurrentMemoryContext == NULL)
	{
		MemoryContextSwitchTo(oldctx);
		elog(ERROR, "rf_store_model: CurrentMemoryContext is NULL after switch");
		return;
	}

	if (CurrentMemoryContext != TopMemoryContext)
	{
		MemoryContextSwitchTo(oldctx);
		elog(ERROR, "rf_store_model: context switch failed - CurrentMemoryContext != TopMemoryContext");
		return;
	}

	if (rf_models == NULL && rf_model_count > 0)
	{
		MemoryContextSwitchTo(oldctx);
		elog(WARNING, "rf_store_model: rf_models is NULL but rf_model_count=%d, resetting count", rf_model_count);
		rf_model_count = 0;
	}

	if (rf_model_count == 0)
	{
		nalloc(rf_models, RFModel, 1);
		if (rf_models == NULL)
		{
			MemoryContextSwitchTo(oldctx);
			elog(ERROR, "rf_store_model: palloc failed for initial rf_models");
			return;
		}
	}
	else
	{
		alloc_size = sizeof(RFModel) * (rf_model_count + 1);
		if (alloc_size > MaxAllocSize)
		{
			MemoryContextSwitchTo(oldctx);
			elog(ERROR, "rf_store_model: allocation size %zu exceeds MaxAllocSize", alloc_size);
			return;
		}
		if (rf_models == NULL)
		{
			MemoryContextSwitchTo(oldctx);
			elog(WARNING, "rf_store_model: rf_models is NULL before repalloc, resetting and using nalloc");
			rf_model_count = 0;
			nalloc(rf_models, RFModel, 1);
			if (rf_models == NULL)
			{
				elog(ERROR, "rf_store_model: palloc failed after reset");
				return;
			}
		}
		else
		{
			rf_models = (RFModel *) repalloc(rf_models, alloc_size);
			if (rf_models == NULL)
			{
				MemoryContextSwitchTo(oldctx);
				elog(ERROR, "rf_store_model: repalloc failed for rf_models");
				return;
			}
		}
	}

	rf_models[rf_model_count].model_id = model_id;
	rf_models[rf_model_count].n_features = n_features;
	rf_models[rf_model_count].n_samples = n_samples;
	rf_models[rf_model_count].n_classes = n_classes;
	rf_models[rf_model_count].majority_value = majority;
	rf_models[rf_model_count].majority_fraction = fraction;
	rf_models[rf_model_count].gini_impurity = gini;
	rf_models[rf_model_count].label_entropy = entropy;

	rf_models[rf_model_count].class_counts = NULL;
	if (n_classes > 0 && class_counts != NULL)
	{
		size_t		class_counts_size = sizeof(int) * (size_t) n_classes;

		if (class_counts_size > MaxAllocSize)
		{
			elog(WARNING, "rf_store_model: class_counts_size %zu exceeds MaxAllocSize, skipping", class_counts_size);
		}
		else
		{
			size_t		count = class_counts_size / sizeof(int);
			int *copy = NULL;

			nalloc(copy, int, count);

			if (copy == NULL)
			{
				elog(WARNING, "rf_store_model: palloc failed for class_counts");
			}
			else
			{
				memcpy(copy, class_counts, class_counts_size);
				rf_models[rf_model_count].class_counts = copy;
			}
		}
	}

	rf_models[rf_model_count].feature_means = NULL;
	if (n_features > 0 && feature_means != NULL)
	{
		size_t		means_size = sizeof(double) * (size_t) n_features;

		if (means_size > MaxAllocSize)
		{
			elog(WARNING, "rf_store_model: means_size %zu exceeds MaxAllocSize, skipping", means_size);
		}
		else
		{
			size_t		count = means_size / sizeof(double);
			double *means_copy = NULL;

			nalloc(means_copy, double, count);

			if (means_copy == NULL)
			{
				elog(WARNING, "rf_store_model: palloc failed for feature_means");
			}
			else
			{
				for (i = 0; i < n_features; i++)
					means_copy[i] = feature_means[i];
				rf_models[rf_model_count].feature_means = means_copy;
			}
		}
	}

	rf_models[rf_model_count].feature_variances = NULL;
	if (n_features > 0 && feature_variances != NULL)
	{
		size_t		vars_size = sizeof(double) * (size_t) n_features;

		if (vars_size > MaxAllocSize)
		{
			elog(WARNING, "rf_store_model: vars_size %zu exceeds MaxAllocSize, skipping", vars_size);
		}
		else
		{
			size_t		count = vars_size / sizeof(double);
			double *vars_copy = NULL;

			nalloc(vars_copy, double, count);

			if (vars_copy == NULL)
			{
				elog(WARNING, "rf_store_model: palloc failed for feature_variances");
			}
			else
			{
				for (i = 0; i < n_features; i++)
					vars_copy[i] = feature_variances[i];
				rf_models[rf_model_count].feature_variances = vars_copy;
			}
		}
	}

	rf_models[rf_model_count].feature_importance = NULL;

	/*
	 * Skip feature_importance allocation to avoid memory context corruption
	 * crashes. This is non-critical data that can be recomputed if needed.
	 */
	if (0 && n_features > 0 && feature_importance != NULL)
	{
	}

	rf_models[rf_model_count].tree = tree;
	rf_models[rf_model_count].split_feature = split_feature;
	rf_models[rf_model_count].split_threshold = split_threshold;
	rf_models[rf_model_count].second_value = second_value;
	rf_models[rf_model_count].second_fraction = second_fraction;
	rf_models[rf_model_count].oob_accuracy = oob_accuracy;
	rf_models[rf_model_count].left_branch_value = left_value;
	rf_models[rf_model_count].left_branch_fraction = left_fraction;
	rf_models[rf_model_count].right_branch_value = right_value;
	rf_models[rf_model_count].right_branch_fraction = right_fraction;
	rf_models[rf_model_count].max_deviation = max_deviation;
	rf_models[rf_model_count].max_split_deviation = max_split_deviation;
	rf_models[rf_model_count].feature_limit =
		(feature_limit > 0) ? feature_limit : 0;
	rf_models[rf_model_count].left_branch_means = NULL;
	rf_models[rf_model_count].right_branch_means = NULL;
	rf_models[rf_model_count].tree_count = 0;
	rf_models[rf_model_count].trees = NULL;
	rf_models[rf_model_count].tree_majority = NULL;
	rf_models[rf_model_count].tree_majority_fraction = NULL;
	rf_models[rf_model_count].tree_second = NULL;
	rf_models[rf_model_count].tree_second_fraction = NULL;
	rf_models[rf_model_count].tree_oob_accuracy = NULL;

	if (rf_models[rf_model_count].feature_limit > 0 && left_means != NULL)
	{
		size_t		left_means_size = sizeof(double) * (size_t) rf_models[rf_model_count].feature_limit;

		if (left_means_size > MaxAllocSize)
		{
			elog(WARNING, "rf_store_model: left_means_size %zu exceeds MaxAllocSize, skipping", left_means_size);
		}
		else
		{
			size_t		count = left_means_size / sizeof(double);
			double *copy = NULL;

			nalloc(copy, double, count);

			if (copy == NULL)
			{
				elog(WARNING, "rf_store_model: palloc failed for left_branch_means");
			}
			else
			{
				memcpy(copy, left_means, left_means_size);
				rf_models[rf_model_count].left_branch_means = copy;
			}
		}
	}

	if (rf_models[rf_model_count].feature_limit > 0 && right_means != NULL)
	{
		size_t		right_means_size = sizeof(double) * (size_t) rf_models[rf_model_count].feature_limit;

		if (right_means_size > MaxAllocSize)
		{
			elog(WARNING, "rf_store_model: right_means_size %zu exceeds MaxAllocSize, skipping", right_means_size);
		}
		else
		{
			size_t		count = right_means_size / sizeof(double);
			double *copy = NULL;

			nalloc(copy, double, count);

			if (copy == NULL)
			{
				elog(WARNING, "rf_store_model: palloc failed for right_branch_means");
			}
			else
			{
				memcpy(copy, right_means, right_means_size);
				rf_models[rf_model_count].right_branch_means = copy;
			}
		}
	}

	if (tree_count > 0 && trees != NULL)
	{
		size_t		trees_array_size;
		size_t		tree_double_size;

		trees_array_size = sizeof(GTree *) * (size_t) tree_count;
		if (trees_array_size > MaxAllocSize)
		{
			elog(WARNING, "rf_store_model: trees_array_size %zu exceeds MaxAllocSize, skipping", trees_array_size);
		}
		else
		{
			size_t		count = trees_array_size / sizeof(GTree *);
			GTree **tree_copy = NULL;

			nalloc(tree_copy, GTree *, count);
			if (tree_copy == NULL)
			{
				elog(WARNING, "rf_store_model: palloc failed for trees array");
			}
			else
			{
				for (i = 0; i < tree_count; i++)
					tree_copy[i] = trees[i];
				rf_models[rf_model_count].trees = tree_copy;
				rf_models[rf_model_count].tree_count = tree_count;

				tree_double_size = sizeof(double) * (size_t) tree_count;
				if (tree_double_size > MaxAllocSize)
				{
					elog(WARNING, "rf_store_model: tree_double_size %zu exceeds MaxAllocSize, skipping tree arrays", tree_double_size);
				}
				else
				{
					if (tree_majority != NULL)
					{
						size_t		majority_count = tree_double_size / sizeof(double);
						double *majority_copy = NULL;

						nalloc(majority_copy, double, majority_count);

						if (majority_copy == NULL)
						{
							elog(WARNING, "rf_store_model: palloc failed for tree_majority");
						}
						else
						{
							for (i = 0; i < tree_count; i++)
								majority_copy[i] = tree_majority[i];
							rf_models[rf_model_count].tree_majority = majority_copy;
						}
					}

					if (tree_majority_fraction != NULL)
					{
						size_t		fraction_count = tree_double_size / sizeof(double);
						double *fraction_copy = NULL;

						nalloc(fraction_copy, double, fraction_count);

						if (fraction_copy == NULL)
						{
							elog(WARNING, "rf_store_model: palloc failed for tree_majority_fraction");
						}
						else
						{
							for (i = 0; i < tree_count; i++)
								fraction_copy[i] = tree_majority_fraction[i];
							rf_models[rf_model_count].tree_majority_fraction = fraction_copy;
						}
					}

					if (tree_second != NULL)
					{
						size_t		second_count = tree_double_size / sizeof(double);
						double *second_copy = NULL;

						nalloc(second_copy, double, second_count);

						if (second_copy == NULL)
						{
							elog(WARNING, "rf_store_model: palloc failed for tree_second");
						}
						else
						{
							for (i = 0; i < tree_count; i++)
								second_copy[i] = tree_second[i];
							rf_models[rf_model_count].tree_second = second_copy;
						}
					}

					if (tree_second_fraction != NULL)
					{
						size_t		second_fraction_count = tree_double_size / sizeof(double);
						double *second_fraction_copy = NULL;

						nalloc(second_fraction_copy, double, second_fraction_count);

						if (second_fraction_copy == NULL)
						{
							elog(WARNING, "rf_store_model: palloc failed for tree_second_fraction");
						}
						else
						{
							for (i = 0; i < tree_count; i++)
								second_fraction_copy[i] = tree_second_fraction[i];
							rf_models[rf_model_count].tree_second_fraction = second_fraction_copy;
						}
					}

					if (tree_oob_accuracy != NULL)
					{
						size_t		oob_count = tree_double_size / sizeof(double);
						double *oob_copy = NULL;

						nalloc(oob_copy, double, oob_count);

						if (oob_copy == NULL)
						{
							elog(WARNING, "rf_store_model: palloc failed for tree_oob_accuracy");
						}
						else
						{
							for (i = 0; i < tree_count; i++)
								oob_copy[i] = tree_oob_accuracy[i];
							rf_models[rf_model_count].tree_oob_accuracy = oob_copy;
						}
					}
				}
			}
		}
	}

	rf_model_count++;

	MemoryContextSwitchTo(oldctx);

#ifdef MEMORY_CONTEXT_CHECKING
	if (entry_ctx != NULL)
		MemoryContextCheck(entry_ctx);
#endif
}

static bool
rf_lookup_model(int32 model_id, RFModel **out)
{
	int			i;

	for (i = 0; i < rf_model_count; i++)
	{
		if (rf_models[i].model_id == model_id)
		{
			if (out)
				*out = &rf_models[i];
			return true;
		}
	}
	return false;
}

static int
rf_count_classes(double *labels, int n_samples)
{
	int			max_class = -1;
	int			i;

	if (n_samples <= 0)
		return 0;

	for (i = 0; i < n_samples; i++)
	{
		double		val = labels[i];
		int			as_int;

		if (!isfinite(val))
			continue;

		as_int = (int) rint(val);
		if (as_int < 0)
			continue;

		if (as_int > max_class)
			max_class = as_int;
	}

	return (max_class < 0) ? 0 : (max_class + 1);
}

Datum
train_random_forest_classifier(PG_FUNCTION_ARGS)
{
	text *table_name_text = NULL;
	text *feature_col_text = NULL;
	text *label_col_text = NULL;

	char *table_name = NULL;
	char *feature_col = NULL;
	char *label_col = NULL;
	const char *quoted_tbl = NULL;
	const char *quoted_feat = NULL;
	const char *quoted_label = NULL;

	StringInfoData query = {0};
	MemoryContext oldcontext = NULL;

	NdbSpiSession *train_spi_session = NULL;

	int			feature_dim = 0;
	int			n_classes = 0;
	int			majority_count = 0;
	int			second_count = 0;
	int			second_idx = -1;

	int *class_counts_tmp = NULL;
	int *counts = NULL;
	int			feature_sum_count = 0;
	int			split_feature = -1;

	int *left_counts = NULL;
	int *right_counts = NULL;
	int			left_majority_idx = -1;
	int			right_majority_idx = -1;
	int			left_total = 0;
	int			right_total = 0;
	int			majority_idx = -1;
	int			feature_limit = 0;
	int			best_feature = -1;

	int *left_feature_counts_vec = NULL;
	int *right_feature_counts_vec = NULL;
	int			n_samples = 0;
	int			split_pair_count = 0;
	int			sample_count = 0;

	int *bootstrap_indices = NULL;
	int *feature_order = NULL;
	pg_prng_state rng;
	int32		model_id = 0;
	RFDataset	dataset;

	double *labels = NULL;
	double		majority_value = 0.0;
	double		majority_fraction = 0.0;
	double		gini_impurity = 0.0;
	double		label_entropy = 0.0;
	double		second_value = 0.0;

	double *feature_means_tmp = NULL;
	double *feature_vars_tmp = NULL;
	double *feature_importance_tmp = NULL;
	double *feature_sums = NULL;
	double *feature_sums_sq = NULL;
	double *class_feature_sums = NULL;
	int *class_feature_counts = NULL;
	double *left_feature_sums_vec = NULL;
	double *right_feature_sums_vec = NULL;
	double		left_leaf_value = 0.0;
	double		right_leaf_value = 0.0;
	double		left_sum = 0.0;
	double		right_sum = 0.0;
	double		left_branch_fraction = 0.0;
	double		right_branch_fraction = 0.0;
	double		class_majority_mean = 0.0;
	double		class_second_mean = 0.0;
	double		class_mean_threshold = 0.0;
	double		best_majority_mean = 0.0;
	double		best_second_mean = 0.0;
	double		best_score = -1.0;
	double		max_deviation = 0.0;
	double		max_split_deviation = 0.0;
	double		split_threshold = 0.0;
	double		second_fraction = 0.0;

	double *left_branch_means_vec = NULL;
	double *right_branch_means_vec = NULL;
	double		forest_oob_accuracy = 0.0;

	double *tree_oob_accuracy = NULL;
	int			oob_total_all = 0;
	int			oob_correct_all = 0;

	float *stage_features = NULL;
	GTree **trees = NULL;

	double *tree_majorities = NULL;
	double *tree_majority_fractions = NULL;
	double *tree_seconds = NULL;
	double *tree_second_fractions = NULL;
	int			tree_count = 0;
	int			forest_trees_arg = RF_DEFAULT_TREES;
	int			max_depth_arg = RF_MAX_DEPTH;
	int			min_samples_arg = RF_MIN_SAMPLES;
	double		best_split_impurity = DBL_MAX;
	double		best_split_threshold = 0.0;

	bool		branch_threshold_valid = false;
	bool		class_mean_threshold_valid = false;
	bool		best_score_valid = false;
	bool		best_split_valid = false;

	GTree *primary_tree = NULL;
	RFSplitPair *split_pairs = NULL;

	if (PG_NARGS() < 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: train_random_forest_classifier: requires table, feature column, and label column"),
				 errdetail("Function received %d arguments, minimum required is 3", PG_NARGS()),
				 errhint("Provide table name, feature column name, and label column name as arguments.")));

	table_name_text = PG_GETARG_TEXT_PP(0);
	feature_col_text = PG_GETARG_TEXT_PP(1);
	label_col_text = PG_GETARG_TEXT_PP(2);

	if (PG_NARGS() > 3 && !PG_ARGISNULL(3))
	{
		int32		arg_trees = PG_GETARG_INT32(3);

		if (arg_trees < 1)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: train_random_forest_classifier: number of trees must be at least 1"),
					 errdetail("Received %d trees, minimum allowed is 1", arg_trees),
					 errhint("Specify a positive number of trees for the random forest ensemble.")));
		if (arg_trees > 1024)
			arg_trees = 1024;
		forest_trees_arg = arg_trees;
	}

	if (PG_NARGS() > 4 && !PG_ARGISNULL(4))
	{
		int32		arg_depth = PG_GETARG_INT32(4);

		if (arg_depth < 1)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: train_random_forest_classifier: max depth must be at least 1"),
					 errdetail("Received max depth %d, minimum allowed is 1", arg_depth),
					 errhint("Specify a positive maximum depth for decision trees.")));
		if (arg_depth > GTREE_MAX_DEPTH)
			arg_depth = GTREE_MAX_DEPTH;
		max_depth_arg = arg_depth;
	}

	if (PG_NARGS() > 5 && !PG_ARGISNULL(5))
	{
		int32		arg_min_samples = PG_GETARG_INT32(5);

		if (arg_min_samples < 1)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: train_random_forest_classifier: min samples must be at least 1"),
					 errdetail("Received min samples %d, minimum allowed is 1", arg_min_samples),
					 errhint("Specify a positive minimum number of samples for tree splits.")));
		if (arg_min_samples > 1000000)
			arg_min_samples = 1000000;
		min_samples_arg = arg_min_samples;
	}

	rf_dataset_init(&dataset);

	table_name = text_to_cstring(table_name_text);
	feature_col = text_to_cstring(feature_col_text);
	label_col = text_to_cstring(label_col_text);
	quoted_tbl = quote_identifier(table_name);
	quoted_feat = quote_identifier(feature_col);
	quoted_label = quote_identifier(label_col);

	oldcontext = CurrentMemoryContext;
	Assert(oldcontext != NULL);
	NDB_SPI_SESSION_BEGIN(train_spi_session, oldcontext);

	initStringInfo(&query);

	rf_dataset_load(
					quoted_tbl, quoted_feat, quoted_label, &dataset, &query);

	feature_dim = dataset.feature_dim;
	n_samples = dataset.n_samples;
	labels = dataset.labels;
	stage_features = dataset.features;
	if (neurondb_gpu_is_available() && n_samples > 0 && feature_dim > 0)
	{
		int			gpu_class_count =
			rf_count_classes(dataset.labels, dataset.n_samples);

		if (gpu_class_count > 0)
		{
			StringInfoData hyperbuf;
			Jsonb *gpu_hyperparams = NULL;
			char *gpu_err = NULL;
			const char *gpu_features[1];
			MLGpuTrainResult gpu_result;

			memset(&hyperbuf, 0, sizeof(StringInfoData));

			memset(&gpu_result, 0, sizeof(MLGpuTrainResult));
			gpu_features[0] = feature_col;

			/* Build hyperparameters JSON using JSONB API */
			{
				JsonbParseState *state = NULL;
				JsonbValue	jkey;
				JsonbValue	jval;

				JsonbValue *final_value = NULL;
				Numeric		n_trees_num,
							max_depth_num,
							min_samples_split_num;

				PG_TRY();
				{
					(void) pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);

					/* Add n_trees */
					jkey.type = jbvString;
					jkey.val.string.val = "n_trees";
					jkey.val.string.len = strlen("n_trees");
					(void) pushJsonbValue(&state, WJB_KEY, &jkey);
					n_trees_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(forest_trees_arg)));
					jval.type = jbvNumeric;
					jval.val.numeric = n_trees_num;
					(void) pushJsonbValue(&state, WJB_VALUE, &jval);

					/* Add max_depth */
					jkey.val.string.val = "max_depth";
					jkey.val.string.len = strlen("max_depth");
					(void) pushJsonbValue(&state, WJB_KEY, &jkey);
					max_depth_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(max_depth_arg)));
					jval.type = jbvNumeric;
					jval.val.numeric = max_depth_num;
					(void) pushJsonbValue(&state, WJB_VALUE, &jval);

					/* Add min_samples_split */
					jkey.val.string.val = "min_samples_split";
					jkey.val.string.len = strlen("min_samples_split");
					(void) pushJsonbValue(&state, WJB_KEY, &jkey);
					min_samples_split_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(min_samples_arg)));
					jval.type = jbvNumeric;
					jval.val.numeric = min_samples_split_num;
					(void) pushJsonbValue(&state, WJB_VALUE, &jval);

					final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);

					if (final_value == NULL)
					{
						elog(ERROR, "neurondb: train_random_forest: pushJsonbValue(WJB_END_OBJECT) returned NULL for hyperparameters");
					}

					gpu_hyperparams = JsonbValueToJsonb(final_value);
					
					/* Log hyperparameters JSONB value for debugging */
					if (gpu_hyperparams != NULL)
					{
						/* Use direct logging without nested PG_TRY to avoid shadow warnings */
						char	   *hyperparams_text = NULL;
						
						hyperparams_text = DatumGetCString(
											DirectFunctionCall1(jsonb_out,
																JsonbPGetDatum(gpu_hyperparams)));
						if (hyperparams_text != NULL)
						{
							nfree(hyperparams_text);
						}
						else
						{
						}
					}
					else
					{
					}
				}
				PG_CATCH();
				{
					ErrorData  *edata = CopyErrorData();

					elog(ERROR, "neurondb: train_random_forest: hyperparameters JSONB construction failed: %s", edata->message);
					FlushErrorState();
					gpu_hyperparams = NULL;
				}
				PG_END_TRY();
			}

			PG_TRY();
			{
				if (ndb_gpu_try_train_model("random_forest",
											NULL,
											NULL,
											table_name,
											label_col,
											gpu_features,
											1,
											gpu_hyperparams,
											stage_features,
											labels,
											n_samples,
											feature_dim,
											gpu_class_count,
											&gpu_result,
											&gpu_err)
					&& gpu_result.spec.model_data != NULL)
				{
					MLCatalogModelSpec spec = gpu_result.spec;

					if (spec.training_table == NULL)
						spec.training_table = table_name;
					if (spec.training_column == NULL)
						spec.training_column = label_col;
					if (spec.parameters == NULL)
					{
						spec.parameters =
							gpu_hyperparams;
						gpu_hyperparams = NULL;
					}

					spec.training_time_ms = -1;
					spec.num_samples = n_samples;
					spec.num_features = feature_dim;

					model_id = ml_catalog_register_model(&spec);
					ndb_gpu_free_train_result(&gpu_result);

				if (gpu_hyperparams)
				{
					nfree(gpu_hyperparams);
					gpu_hyperparams = NULL;
				}					if (gpu_err)
					{
						gpu_err = NULL;
					}

					rf_dataset_free(&dataset);

					/*
					 * Free query.data BEFORE ending SPI session (it's allocated
					 * in SPI context)
					 */
					if (query.data)
					{
						nfree(query.data);
						query.data = NULL;
					}
					NDB_SPI_SESSION_END(train_spi_session);

					if (table_name)
					{
						nfree(table_name);
						table_name = NULL;
					}
					if (feature_col)
					{
						nfree(feature_col);
						feature_col = NULL;
					}
					if (label_col)
					{
						nfree(label_col);
						label_col = NULL;
					}

					PG_RETURN_INT32(model_id);
				}
			}
			PG_CATCH();
			{
				/* Clean up gpu_result if it was partially initialized */
				rf_safe_free_gpu_result(&gpu_result);
				FlushErrorState();
				/* Re-throw to allow fallback to CPU */
				PG_RE_THROW();
			}
			PG_END_TRY();

			/*
			 * Don't free gpu_err - it's allocated by GPU function using
			 * pstrdup() and will be automatically freed when the memory
			 * context is cleaned up. Manually freeing it can cause crashes if
			 * the context is already cleaned up.
			 */
			if (gpu_err != NULL)
			{
				gpu_err = NULL;
			}

			/* Clean up gpu_result if training failed (returned false) */
			rf_safe_free_gpu_result(&gpu_result);

			if (gpu_hyperparams != NULL)
			{
				nfree(gpu_hyperparams);
				gpu_hyperparams = NULL;
			}
			if (hyperbuf.data)
			{
				nfree(hyperbuf.data);
				hyperbuf.data = NULL;
			}
		}
	}
	if (n_samples > 0)
	{
		int			i;

		if (feature_dim > 0)
		{
			double *feature_importance_tmp_local = NULL;
			nalloc(feature_importance_tmp_local, double, feature_dim);
			feature_importance_tmp = feature_importance_tmp_local;
		}

		if (feature_dim > 0)
		{
			int			j;

			int *feature_order_local = NULL;
			nalloc(feature_order_local, int, feature_dim);
			for (j = 0; j < feature_dim; j++)
				feature_order_local[j] = j;
			feature_order = feature_order_local;
		}

		sample_count =
			(int) rint(RF_BOOTSTRAP_FRACTION * (double) n_samples);
		if (sample_count <= 0)
			sample_count = n_samples;
		if (sample_count > n_samples)
			sample_count = n_samples;
		if (sample_count > 0)
		{
			if (!pg_prng_strong_seed(&rng))
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("neurondb: train_random_forest_classifier: failed to seed PRNG"),
						 errdetail("Random number generator initialization failed"),
						 errhint("This is an internal error. Please report this issue.")));
			nalloc(bootstrap_indices, int, sample_count);
			for (i = 0; i < sample_count; i++)
				bootstrap_indices[i] =
					(int) pg_prng_uint64_range_inclusive(&rng,
														 0,
														 (uint64) (n_samples - 1));
		}

		if (feature_dim > 0 && sample_count > 0
			&& stage_features != NULL)
		{
			nalloc(feature_sums, double, feature_dim);
			nalloc(feature_sums_sq, double, feature_dim);

			for (i = 0; i < sample_count; i++)
			{
				int			src = bootstrap_indices[i];
				float *row = NULL;
				int			j;

				if (src < 0 || src >= n_samples)
					continue;
				if (!isfinite(labels[src]))
					continue;

				row = stage_features + (src * feature_dim);

				for (j = 0; j < feature_dim; j++)
				{
					double		val = (double) row[j];

					feature_sums[j] += val;
					feature_sums_sq[j] += val * val;
				}
				feature_sum_count++;
			}
		}

		n_classes = rf_count_classes(labels, n_samples);

		if (n_classes > 0)
		{
			nalloc(counts, int, n_classes);
			if (counts != NULL)
			{
				int			best_idx = 0;

				class_counts_tmp = counts;

				for (i = 0; i < sample_count; i++)
				{
					int			src = bootstrap_indices[i];
					int			idx;

					if (src < 0 || src >= n_samples)
						continue;
					idx = (int) rint(labels[src]);
					if (idx < 0 || idx >= n_classes)
						continue;
					counts[idx]++;
					if (counts[idx] > counts[best_idx])
					{
						if (idx != best_idx)
						{
							second_idx = best_idx;
							second_count = counts[best_idx];
							second_value = (double) best_idx;
						}
						best_idx = idx;
					}
					else if (idx != best_idx
							 && counts[idx] > second_count)
					{
						second_idx = idx;
						second_count = counts[idx];
						second_value = (double) idx;
					}
				}

				majority_value = (double) best_idx;
				majority_count = counts[best_idx];
				majority_idx = best_idx;
				if (second_idx < 0 && n_classes > 1)
				{
					for (i = 0; i < n_classes; i++)
					{
						if (i == best_idx)
							continue;
						if (counts[i] >= second_count)
						{
							second_idx = i;
							second_count = counts[i];
							second_value = (double) i;
						}
					}
				}

				left_leaf_value = majority_value;
				right_leaf_value = (second_idx >= 0) ? second_value
					: majority_value;

				if (sample_count > 0)
				{
					double		sum_sq = 0.0;
					double		entropy = 0.0;
					int			c;
					double		ln2 = log(2.0);

					for (c = 0; c < n_classes; c++)
					{
						double		p = (double) counts[c]
							/ (double) sample_count;

						sum_sq += p * p;
						if (p > 0.0)
							entropy -= p * (log(p) / ln2);
					}
					gini_impurity = 1.0 - sum_sq;
					label_entropy = entropy;
				}

				if (class_counts_tmp)
				{
					StringInfoData histogram;

					initStringInfo(&histogram);
					appendStringInfo(&histogram, "[");
					for (i = 0; i < n_classes; i++)
					{
						if (i > 0)
							appendStringInfoString(&histogram, ", ");
						appendStringInfo(&histogram, "%d", class_counts_tmp[i]);
					}
					appendStringInfoChar(&histogram, ']');
					nfree(histogram.data);
				}
			}

			if (feature_sums != NULL && feature_sum_count > 0)
			{
				int			j;
				StringInfoData mean_log;
				StringInfoData var_log;

				nalloc(feature_means_tmp, double, feature_dim);
				nalloc(feature_vars_tmp, double, feature_dim);
				for (j = 0; j < feature_dim; j++)
				{
					double		mean = feature_sums[j]
						/ (double) feature_sum_count;
					double		mean_sq = feature_sums_sq[j]
						/ (double) feature_sum_count;
					double		variance = mean_sq - (mean * mean);

					if (variance < 0.0)
						variance = 0.0;

					feature_means_tmp[j] = mean;
					feature_vars_tmp[j] = variance;
				}

				initStringInfo(&mean_log);
				appendStringInfo(&mean_log, "[");
				for (j = 0; j < feature_dim && j < 5; j++)
				{
					if (j > 0)
						appendStringInfoString(&mean_log, ", ");
				}
				if (feature_dim > 5)
					appendStringInfoString(&mean_log, ", ...");
				appendStringInfoChar(&mean_log, ']');
				nfree(mean_log.data);

				initStringInfo(&var_log);
				appendStringInfo(&var_log, "[");
				for (j = 0; j < feature_dim && j < 5; j++)
				{
					if (j > 0)
						appendStringInfoString(&var_log, ", ");
					appendStringInfo(
									 &var_log, "%.3f", feature_vars_tmp[j]);
				}
				if (feature_dim > 5)
					appendStringInfoString(&var_log, ", ...");
				appendStringInfoChar(&var_log, ']');
				nfree(var_log.data);
			}

			if (feature_dim > 0 && sample_count > 0 && n_classes > 0
				&& stage_features != NULL)
			{
				feature_limit = feature_dim;
				if (feature_limit < 1)
					feature_limit = 1;

				{
					size_t		sums_size = sizeof(double) * (size_t) n_classes * (size_t) feature_limit;
					size_t		counts_size = sizeof(int) * (size_t) n_classes * (size_t) feature_limit;

					if (sums_size > MaxAllocSize || counts_size > MaxAllocSize)
					{
						elog(WARNING, "rf_build_branch_tree: allocation sizes exceed MaxAllocSize (n_classes=%d, feature_limit=%d)", n_classes, feature_limit);
						return -1;
					}
					{
						size_t		sums_count = sums_size / sizeof(double);
						size_t		counts_count = counts_size / sizeof(int);

						nalloc(class_feature_sums, double, sums_count);
						nalloc(class_feature_counts, int, counts_count);
					}
					if (class_feature_sums == NULL || class_feature_counts == NULL)
					{
						elog(WARNING, "rf_build_branch_tree: palloc0 failed for class feature arrays");
						if (class_feature_sums != NULL)
							nfree(class_feature_sums);
						if (class_feature_counts != NULL)
							nfree(class_feature_counts);
						return -1;
					}
				}

				for (i = 0; i < sample_count; i++)
				{
					int			src = bootstrap_indices[i];
					int			cls;
					int			f;
					float *row = NULL;

					if (src < 0 || src >= n_samples)
						continue;
					if (!isfinite(labels[src]))
						continue;
					cls = (int) rint(labels[src]);
					if (cls < 0 || cls >= n_classes)
						continue;

					row = stage_features + (src * feature_dim);

					for (f = 0;
						 f < feature_limit && f < feature_dim;
						 f++)
					{
						double		val = (double) row[f];

						class_feature_sums[cls * feature_limit
										   + f] += val;
						class_feature_counts[cls * feature_limit
											 + f]++;
					}
				}

				if (majority_idx >= 0)
				{
					int			f;

					for (f = 0; f < feature_limit; f++)
					{
						int			idx = majority_idx * feature_limit
							+ f;

						if (class_feature_counts[idx] > 0)
						{
							double		maj_mean =
								class_feature_sums[idx]
								/ (double)
								class_feature_counts
								[idx];
							double		sec_mean = 0.0;
							int			sec_idx = -1;
							int			sec_count = 0;

							if (second_idx >= 0)
							{
								sec_idx = second_idx
									* feature_limit
									+ f;
								sec_count =
									class_feature_counts
									[sec_idx];
							}

							if (sec_count > 0)
							{
								sec_mean =
									class_feature_sums
									[sec_idx]
									/ (double)
									sec_count;

								if (fabs(maj_mean
										 - sec_mean)
									> best_score)
								{
									best_score = fabs(
													  maj_mean
													  - sec_mean);
									best_feature =
										f;
									best_majority_mean =
										maj_mean;
									best_second_mean =
										sec_mean;
									best_score_valid =
										true;
								}
							}
						}
					}
				}

				if (!best_score_valid && majority_idx >= 0)
				{
					int			idx = majority_idx * feature_limit;

					if (class_feature_counts[idx] > 0)
						best_majority_mean =
							class_feature_sums[idx]
							/ (double) class_feature_counts
							[idx];
				}

				if (best_score_valid)
				{
					class_majority_mean = best_majority_mean;
					class_second_mean = best_second_mean;
					class_mean_threshold = 0.5
						* (class_majority_mean
						   + class_second_mean);
					class_mean_threshold_valid = true;
					split_feature = best_feature;
				}
			}

			/*
			 * Refine threshold using sorted split candidates on the chosen
			 * feature
			 */
			if (feature_dim > 0 && sample_count > 0 && n_classes > 0
				&& stage_features != NULL)
			{
				int			sf_idx = (split_feature >= 0
									  && split_feature < feature_dim)
					? split_feature
					: 0;

				nalloc(split_pairs, RFSplitPair, sample_count);
				NDB_CHECK_ALLOC(split_pairs, "split_pairs");
				split_pair_count = 0;

				for (i = 0; i < sample_count; i++)
				{
					int			src = bootstrap_indices[i];
					int			cls;
					float *row = NULL;

					if (src < 0 || src >= n_samples)
						continue;
					cls = (int) rint(labels[src]);
					if (!isfinite(labels[src]) || cls < 0
						|| cls >= n_classes)
						continue;

					row = stage_features + (src * feature_dim);
					if (sf_idx >= feature_dim)
						continue;

					split_pairs[split_pair_count].value =
						(double) row[sf_idx];
					split_pairs[split_pair_count].cls = cls;
					split_pair_count++;
				}

				if (split_pair_count > 1)
				{
					int *left_counts_tmp = NULL;
					int *right_counts_tmp = NULL;
					int			right_total_eval = 0;
					int			left_total_eval = 0;

					nalloc(left_counts_tmp, int, n_classes);
					nalloc(right_counts_tmp, int, n_classes);

					qsort(split_pairs,
						  split_pair_count,
						  sizeof(RFSplitPair),
						  rf_split_pair_cmp);

					if (class_counts_tmp != NULL)
					{
						for (i = 0; i < n_classes; i++)
							right_counts_tmp[i] =
								class_counts_tmp[i];
					}
					else
					{
						for (i = 0; i < split_pair_count; i++)
							right_counts_tmp
								[split_pairs[i].cls]++;
					}

					for (i = 0; i < n_classes; i++)
						right_total_eval += right_counts_tmp[i];

					for (i = 0; i < split_pair_count - 1; i++)
					{
						int			cls = split_pairs[i].cls;

						left_counts_tmp[cls]++;
						right_counts_tmp[cls]--;
						left_total_eval++;
						right_total_eval--;

						if (split_pairs[i].value
							== split_pairs[i + 1].value)
							continue;
						if (left_total_eval <= 0
							|| right_total_eval <= 0)
							continue;

						{
							double		left_imp =
								rf_gini_impurity(
												 left_counts_tmp,
												 n_classes,
												 left_total_eval);
							double		right_imp =
								rf_gini_impurity(
												 right_counts_tmp,
												 n_classes,
												 right_total_eval);
							double		weighted =
								((double) left_total_eval
								 / (double)
								 split_pair_count)
								* left_imp
								+ ((double) right_total_eval
								   / (double)
								   split_pair_count)
								* right_imp;

							if (weighted
								< best_split_impurity)
							{
								best_split_impurity =
									weighted;
								best_split_threshold =
									0.5
									* (split_pairs[i]
									   .value
									   + split_pairs[i
													 + 1]
									   .value);
								best_split_valid = true;
							}
						}
					}

					nfree(left_counts_tmp);
					nfree(right_counts_tmp);
				}

				if (best_split_valid)
				{
					class_mean_threshold = best_split_threshold;
					class_mean_threshold_valid = true;
					split_feature = sf_idx;
				}

				if (split_pairs)
				{
					nfree(split_pairs);
					split_pairs = NULL;
				}
			}

			if (feature_dim > 0 && feature_means_tmp != NULL
				&& n_classes > 0 && stage_features != NULL
				&& sample_count > 0)
			{
				int			sf = (split_feature >= 0
								  && split_feature < feature_dim)
					? split_feature
					: 0;
				double		threshold = feature_means_tmp[sf];

				left_total = 0;
				right_total = 0;
				left_sum = 0.0;
				right_sum = 0.0;
				left_majority_idx = -1;
				right_majority_idx = -1;

				if (class_mean_threshold_valid)
					threshold = class_mean_threshold;

				nalloc(left_counts, int, n_classes);
				NDB_CHECK_ALLOC(left_counts, "left_counts");
				nalloc(right_counts, int, n_classes);
				NDB_CHECK_ALLOC(right_counts, "right_counts");
				if (feature_limit > 0)
				{
					nalloc(left_feature_sums_vec, double, feature_limit);
					NDB_CHECK_ALLOC(left_feature_sums_vec, "left_feature_sums_vec");
					nalloc(right_feature_sums_vec, double, feature_limit);
					NDB_CHECK_ALLOC(right_feature_sums_vec, "right_feature_sums_vec");
					nalloc(left_feature_counts_vec, int, feature_limit);
					NDB_CHECK_ALLOC(left_feature_counts_vec, "left_feature_counts_vec");
					nalloc(right_feature_counts_vec, int, feature_limit);
					NDB_CHECK_ALLOC(right_feature_counts_vec, "right_feature_counts_vec");
				}

				for (i = 0; i < sample_count; i++)
				{
					int			src = bootstrap_indices[i];
					int			cls;
					int			f;
					float *row = NULL;
					double		value;

					if (src < 0 || src >= n_samples)
						continue;
					if (!isfinite(labels[src]))
						continue;
					cls = (int) rint(labels[src]);
					if (cls < 0 || cls >= n_classes)
						continue;

					if (sf >= feature_dim)
						continue;

					row = stage_features + (src * feature_dim);
					value = (double) row[sf];

					if (value <= threshold)
					{
						left_counts[cls]++;
						left_total++;
						left_sum += value;
						if (feature_limit > 0)
						{
							for (f = 0; f < feature_limit
								 && f < feature_dim;
								 f++)
							{
								left_feature_sums_vec
									[f] +=
									(double) row[f];
								left_feature_counts_vec
									[f]++;
							}
						}
					}
					else
					{
						right_counts[cls]++;
						right_total++;
						right_sum += value;
						if (feature_limit > 0)
						{
							for (f = 0; f < feature_limit
								 && f < feature_dim;
								 f++)
							{
								right_feature_sums_vec
									[f] +=
									(double) row[f];
								right_feature_counts_vec
									[f]++;
							}
						}
					}
				}

				for (i = 0; i < n_classes; i++)
				{
					if (left_total > 0
						&& (left_majority_idx < 0
							|| left_counts[i] > left_counts
							[left_majority_idx]))
						left_majority_idx = i;
					if (right_total > 0
						&& (right_majority_idx < 0
							|| right_counts[i]
							> right_counts
							[right_majority_idx]))
						right_majority_idx = i;
				}

				if (left_majority_idx >= 0)
					left_leaf_value = (double) left_majority_idx;

				if (right_majority_idx >= 0)
				{
					right_leaf_value = (double) right_majority_idx;
					second_value = right_leaf_value;
					if (class_counts_tmp != NULL)
						second_fraction =
							((double) class_counts_tmp
							 [right_majority_idx])
							/ (double) sample_count;
					else if (right_total > 0)
						second_fraction =
							((double) right_counts
							 [right_majority_idx])
							/ (double) sample_count;
				}

				if (sample_count > 0)
				{
					if (left_total > 0)
						left_branch_fraction =
							((double) left_total)
							/ (double) sample_count;
					if (right_total > 0)
						right_branch_fraction =
							((double) right_total)
							/ (double) sample_count;
				}

				if (feature_limit > 0 && left_feature_sums_vec != NULL
					&& right_feature_sums_vec != NULL)
				{
					int			f;

					double *left_branch_means_vec_local = NULL;
					double *right_branch_means_vec_local = NULL;
					nalloc(left_branch_means_vec_local, double, feature_limit);
					NDB_CHECK_ALLOC(left_branch_means_vec_local, "left_branch_means_vec");
					nalloc(right_branch_means_vec_local, double, feature_limit);
					left_branch_means_vec = left_branch_means_vec_local;
					right_branch_means_vec = right_branch_means_vec_local;
					NDB_CHECK_ALLOC(right_branch_means_vec, "right_branch_means_vec");

					for (f = 0; f < feature_limit; f++)
					{
						if (left_feature_counts_vec != NULL
							&& left_feature_counts_vec[f]
							> 0)
							left_branch_means_vec[f] =
								left_feature_sums_vec[f]
								/ (double)
								left_feature_counts_vec
								[f];
						else if (feature_means_tmp != NULL
								 && f < feature_dim)
							left_branch_means_vec[f] =
								feature_means_tmp[f];
						else
							left_branch_means_vec[f] = 0.0;

						if (right_feature_counts_vec != NULL
							&& right_feature_counts_vec[f]
							> 0)
							right_branch_means_vec[f] =
								right_feature_sums_vec
								[f]
								/ (double)
								right_feature_counts_vec
								[f];
						else if (feature_means_tmp != NULL
								 && f < feature_dim)
							right_branch_means_vec[f] =
								feature_means_tmp[f];
						else
							right_branch_means_vec[f] =
								left_branch_means_vec
								[f];
					}
				}

				if ((left_total == 0 || right_total == 0)
					&& feature_vars_tmp != NULL && sf < feature_dim
					&& feature_vars_tmp[sf] > 0.0)
				{
					double		adjust;

					adjust = 0.5 * sqrt(feature_vars_tmp[sf]);
					if (right_total == 0)
					{
						threshold =
							feature_means_tmp[sf] + adjust;
						branch_threshold_valid = true;
						if (right_branch_fraction <= 0.0)
							right_branch_fraction = 0.5;
						if (left_branch_fraction <= 0.0)
							left_branch_fraction = 1.0
								- right_branch_fraction;
						right_leaf_value = (second_idx >= 0)
							? second_value
							: majority_value;
					}
					else if (left_total == 0)
					{
						threshold =
							feature_means_tmp[sf] - adjust;
						branch_threshold_valid = true;
						if (left_branch_fraction <= 0.0)
							left_branch_fraction = 0.5;
						if (right_branch_fraction <= 0.0)
							right_branch_fraction = 1.0
								- left_branch_fraction;
						left_leaf_value = majority_value;
					}
				}

				if (left_total > 0 && right_total > 0)
				{
					double		left_mean =
						left_sum / (double) left_total;
					double		right_mean =
						right_sum / (double) right_total;

					threshold = 0.5 * (left_mean + right_mean);
					branch_threshold_valid = true;
				}

				if (left_branch_fraction <= 0.0
					&& right_branch_fraction <= 0.0)
					left_branch_fraction = majority_fraction;
				else if (left_branch_fraction <= 0.0
						 && right_branch_fraction > 0.0)
					left_branch_fraction =
						1.0 - right_branch_fraction;
				else if (right_branch_fraction <= 0.0
						 && left_branch_fraction > 0.0)
					right_branch_fraction =
						1.0 - left_branch_fraction;

				if (second_fraction <= 0.0
					&& right_branch_fraction > 0.0)
					second_fraction = right_branch_fraction;

				if (left_counts)
					nfree(left_counts);
				if (right_counts)
					nfree(right_counts);
				left_counts = NULL;
				right_counts = NULL;

			}

			if (sample_count > 0 && majority_count > 0)
				majority_fraction =
					((double) majority_count) / (double) sample_count;
		}

		if (sample_count > 0 && second_count > 0 && second_fraction <= 0.0)
			second_fraction = ((double) second_count) / (double) sample_count;

		if (majority_count > 0)
		{
			MemoryContext oldctx = NULL;
			int			forest_trees = forest_trees_arg;
			int			t;

			if (forest_trees < 1)
				forest_trees = 1;
			if (forest_trees > n_samples)
				forest_trees = n_samples;

			if (forest_trees > 0)
			{
				nalloc(trees, GTree *, forest_trees);
				NDB_CHECK_ALLOC(trees, "trees");
				nalloc(tree_majorities, double, forest_trees);
				NDB_CHECK_ALLOC(tree_majorities, "tree_majorities");
				nalloc(tree_majority_fractions, double, forest_trees);
				NDB_CHECK_ALLOC(tree_majority_fractions, "tree_majority_fractions");
				nalloc(tree_seconds, double, forest_trees);
				NDB_CHECK_ALLOC(tree_seconds, "tree_seconds");
				nalloc(tree_second_fractions, double, forest_trees);
				NDB_CHECK_ALLOC(tree_second_fractions, "tree_second_fractions");
				nalloc(tree_oob_accuracy, double, forest_trees);
				NDB_CHECK_ALLOC(tree_oob_accuracy, "tree_oob_accuracy");
			}

			for (t = 0; t < forest_trees; t++)
			{
				GTree *tree = NULL;
				int			node_idx;
				int			left_idx;
				int			right_idx;
				int			tree_feature = split_feature;
				int			feature_for_split = split_feature;
				double		tree_threshold = split_threshold;
				double		var0 = 0.0;
				int			tree_majority_idx = -1;
				int			tree_second_idx = -1;
				int			tree_majority_count = 0;
				int			tree_second_count = 0;
				double		tree_majority_value = majority_value;
				double		tree_second_value = second_value;
				double		tree_majority_frac = majority_fraction;
				double		tree_second_frac = second_fraction;

				int *tree_counts = NULL;
				int			boot_samples = 0;
				int			sample_target = n_samples;
				int			j;

				int *tree_bootstrap = NULL;
				RFSplitPair *tree_pairs = NULL;
				int			tree_pair_count = 0;
				double		tree_best_impurity = DBL_MAX;
				double		tree_best_threshold = split_threshold;
				int			tree_best_feature = split_feature;
				bool		tree_split_valid = false;
				int			mtry = 0;
				int			candidates = 0;

				int *left_tmp = NULL;
				int *right_tmp = NULL;
				int			left_total_local = 0;
				int			right_total_local = 0;
				int			left_majority_local = -1;
				int			right_majority_local = -1;
				int			left_best_count = 0;
				int			right_best_count = 0;
				double		tree_left_value = left_leaf_value;
				double		tree_right_value = right_leaf_value;
				double		tree_left_fraction = left_branch_fraction;
				double		tree_right_fraction = right_branch_fraction;

				bool *inbag = NULL;
				int			oob_total_local = 0;
				int			oob_correct_local = 0;

				int *left_indices_local = NULL;
				int *right_indices_local = NULL;
				int			left_index_count = 0;
				int			right_index_count = 0;

				if (n_classes > 0)
				{
					int *tree_counts_local = NULL;
					nalloc(tree_counts_local, int, n_classes);
					tree_counts = tree_counts_local;
				}

				if (n_samples > 0)
				{
					sample_target = (int) rint((double) n_samples
											   * RF_BOOTSTRAP_FRACTION);
					if (sample_target < 1)
						sample_target = 1;
					if (sample_target > n_samples)
						sample_target = n_samples;
				}

				if (n_samples > 0)
				{
					bool *inbag_local = NULL;
					nalloc(inbag_local, bool, n_samples);
					NDB_CHECK_ALLOC(inbag_local, "inbag");
					inbag = inbag_local;
				}

				if (sample_target > 0 && n_samples > 0)
				{
					nalloc(tree_bootstrap, int, sample_target);
					NDB_CHECK_ALLOC(tree_bootstrap, "tree_bootstrap");
				}

				for (j = 0; j < sample_target; j++)
				{
					int			idx;

					idx = (int) pg_prng_uint64_range_inclusive(
															   &rng, 0, (uint64) (n_samples - 1));
					if (tree_bootstrap != NULL)
						tree_bootstrap[j] = idx;
					boot_samples++;
					if (tree_counts != NULL && labels != NULL)
					{
						int			cls;

						if (!isfinite(labels[idx]))
							continue;
						cls = (int) rint(labels[idx]);
						if (cls < 0 || cls >= n_classes)
							continue;
						tree_counts[cls]++;
						if (tree_counts[cls]
							> tree_majority_count)
						{
							if (cls != tree_majority_idx)
							{
								tree_second_idx =
									tree_majority_idx;
								tree_second_count =
									tree_majority_count;
							}
							tree_majority_idx = cls;
							tree_majority_count =
								tree_counts[cls];
						}
						else if (cls != tree_majority_idx
								 && tree_counts[cls]
								 > tree_second_count)
						{
							tree_second_idx = cls;
							tree_second_count =
								tree_counts[cls];
						}
					}
				}

				if (inbag != NULL && tree_bootstrap != NULL)
				{
					for (j = 0; j < boot_samples; j++)
					{
						int			idx = tree_bootstrap[j];

						if (idx >= 0 && idx < n_samples)
							inbag[idx] = true;
					}
				}

				if (tree_majority_idx >= 0)
				{
					tree_majority_value = (double) tree_majority_idx;
					if (boot_samples > 0)
						tree_majority_frac =
							(double) tree_majority_count
							/ (double) boot_samples;
				}

				if (tree_second_idx < 0)
					tree_second_idx = second_idx;

				if (tree_second_idx >= 0)
				{
					tree_second_value = (double) tree_second_idx;
					if (boot_samples > 0 && tree_second_count > 0)
						tree_second_frac =
							(double) tree_second_count
							/ (double) boot_samples;
					else
						tree_second_frac = second_fraction;
				}

				if (feature_dim > 0 && stage_features != NULL
					&& labels != NULL && n_classes > 0
					&& tree_bootstrap != NULL)
				{
					int			f;

					if (feature_order != NULL)
					{
						for (f = 0; f < feature_dim; f++)
							feature_order[f] = f;
					}

					mtry = (int) sqrt((double) feature_dim);
					if (mtry < 1)
						mtry = 1;
					if (mtry > feature_dim)
						mtry = feature_dim;
					candidates = mtry;

					if (feature_order != NULL && feature_dim > 0)
					{
						for (f = 0; f < candidates; f++)
						{
							int			swap_idx;

							swap_idx = (int)
								pg_prng_uint64_range_inclusive(
															   &rng,
															   (uint64) f,
															   (uint64) (feature_dim
																		 - 1));
							if (swap_idx != f)
							{
								int			tmp = feature_order
									[f];

								feature_order[f] =
									feature_order
									[swap_idx];
								feature_order
									[swap_idx] =
									tmp;
							}
						}
					}
					else
						candidates =
							Min(candidates, feature_dim);

					for (f = 0; f < candidates; f++)
					{
						int			feature_idx = feature_order
							? feature_order[f]
							: f;
						int			s;

						if (feature_idx < 0
							|| feature_idx >= feature_dim)
							continue;

						nalloc(tree_pairs, RFSplitPair, boot_samples);
						NDB_CHECK_ALLOC(tree_pairs, "tree_pairs");
						tree_pair_count = 0;

						for (s = 0; s < boot_samples; s++)
						{
							int			sample_idx =
								tree_bootstrap[s];
							float *row = NULL;
							double		value;
							int			cls;

							if (sample_idx < 0
								|| sample_idx
								>= n_samples)
								continue;
							if (!isfinite(
										  labels[sample_idx]))
								continue;

							cls = (int) rint(
											 labels[sample_idx]);
							if (cls < 0 || cls >= n_classes)
								continue;

							row = stage_features
								+ (sample_idx
								   * feature_dim);
							value = (double)
								row[feature_idx];

							tree_pairs[tree_pair_count]
								.value = value;
							tree_pairs[tree_pair_count]
								.cls = cls;
							tree_pair_count++;
						}

						if (tree_pair_count > 1)
						{
							int *left_counts_tmp = NULL;
							int *right_counts_tmp = NULL;
							int			left_total_eval = 0;
							int			right_total_eval = 0;

							qsort(tree_pairs,
								  tree_pair_count,
								  sizeof(RFSplitPair),
								  rf_split_pair_cmp);

							nalloc(left_counts_tmp, int, n_classes);
							nalloc(right_counts_tmp, int, n_classes);

							for (s = 0; s < tree_pair_count;
								 s++)
								right_counts_tmp
									[tree_pairs[s].cls]++;

							right_total_eval =
								tree_pair_count;

							for (s = 0;
								 s < tree_pair_count - 1;
								 s++)
							{
								int			cls_val =
									tree_pairs[s]
									.cls;
								double		left_imp;
								double		right_imp;
								double		weighted;
								double		threshold_candidate;

								left_counts_tmp
									[cls_val]++;
								right_counts_tmp
									[cls_val]--;
								left_total_eval++;
								right_total_eval--;

								if (tree_pairs[s].value
									== tree_pairs[s
												  + 1]
									.value)
									continue;
								if (left_total_eval <= 0
									|| right_total_eval
									<= 0)
									continue;

								left_imp = rf_gini_impurity(
															left_counts_tmp,
															n_classes,
															left_total_eval);
								right_imp = rf_gini_impurity(
															 right_counts_tmp,
															 n_classes,
															 right_total_eval);
								threshold_candidate =
									0.5
									* (tree_pairs[s].value
									   + tree_pairs[s
													+ 1]
									   .value);
								weighted =
									((double) left_total_eval
									 / (double)
									 tree_pair_count)
									* left_imp
									+ ((double) right_total_eval
									   / (double)
									   tree_pair_count)
									* right_imp;

								if (weighted
									< tree_best_impurity)
								{
									tree_best_impurity =
										weighted;
									tree_best_threshold =
										threshold_candidate;
									tree_best_feature =
										feature_idx;
									tree_split_valid =
										true;
								}
							}

							nfree(left_counts_tmp);
							nfree(right_counts_tmp);
						}

						if (tree_pairs != NULL)
						{
							nfree(tree_pairs);
							tree_pairs = NULL;
						}
					}
				}

				if (tree_split_valid)
				{
					tree_feature = tree_best_feature;
					feature_for_split = tree_best_feature;
					tree_threshold = tree_best_threshold;
				}
				else
					feature_for_split = tree_feature;

				if (tree_bootstrap != NULL && feature_dim > 0
					&& n_classes > 0 && stage_features != NULL
					&& labels != NULL && tree_feature >= 0
					&& tree_feature < feature_dim)
				{
					nalloc(left_tmp, int, n_classes);
					nalloc(right_tmp, int, n_classes);

					left_index_count = 0;
					right_index_count = 0;
					if (boot_samples > 0)
					{
						if (left_indices_local == NULL)
							nalloc(left_indices_local, int, boot_samples);
						if (right_indices_local == NULL)
							nalloc(right_indices_local, int, boot_samples);
					}

					for (j = 0; j < boot_samples; j++)
					{
						int			sample_idx = tree_bootstrap[j];
						float *row = NULL;
						double		value;
						int			cls;

						if (sample_idx < 0
							|| sample_idx >= n_samples)
							continue;
						if (!isfinite(labels[sample_idx]))
							continue;

						cls = (int) rint(labels[sample_idx]);
						if (cls < 0 || cls >= n_classes)
							continue;

						row = stage_features
							+ (sample_idx * feature_dim);
						value = (double) row[tree_feature];

						if (value <= tree_threshold)
						{
							left_tmp[cls]++;
							left_total_local++;
							if (left_indices_local != NULL)
								left_indices_local
									[left_index_count++] =
									sample_idx;
						}
						else
						{
							right_tmp[cls]++;
							right_total_local++;
							if (right_indices_local != NULL)
								right_indices_local
									[right_index_count++] =
									sample_idx;
						}
					}

					for (j = 0; j < n_classes; j++)
					{
						if (left_tmp[j] > left_best_count)
						{
							left_best_count = left_tmp[j];
							left_majority_local = j;
						}
						if (right_tmp[j] > right_best_count)
						{
							right_best_count = right_tmp[j];
							right_majority_local = j;
						}
					}

					if (left_majority_local >= 0)
						tree_left_value =
							(double) left_majority_local;
					if (right_majority_local >= 0)
					{
						tree_right_value =
							(double) right_majority_local;
						tree_second_value = tree_right_value;
					}

					if (boot_samples > 0)
					{
						tree_left_fraction =
							(double) left_total_local
							/ (double) boot_samples;
						tree_right_fraction =
							(double) right_total_local
							/ (double) boot_samples;
					}

					if (tree_second_frac <= 0.0
						&& tree_right_fraction > 0.0)
						tree_second_frac = tree_right_fraction;

					nfree(left_tmp);
					nfree(right_tmp);
				}

				feature_for_split = tree_feature;

				oldctx = MemoryContextSwitchTo(TopMemoryContext);
				tree = gtree_create("rf_model_tree", 4);
				MemoryContextSwitchTo(oldctx);

				if (tree == NULL)
				{
					if (tree_counts)
						nfree(tree_counts);
					if (tree_bootstrap)
						nfree(tree_bootstrap);
					continue;
				}

				if (feature_dim > 0 && feature_means_tmp != NULL)
				{
					if (feature_for_split < 0
						|| feature_for_split >= feature_dim)
						feature_for_split = 0;
					if (feature_vars_tmp != NULL
						&& feature_for_split < feature_dim)
						var0 = feature_vars_tmp
							[feature_for_split];

					if (!branch_threshold_valid)
						tree_threshold = feature_means_tmp
							[feature_for_split];

					node_idx = gtree_add_split(tree,
											   feature_for_split,
											   tree_threshold);

					if (left_indices_local != NULL
						&& left_index_count > 0)
						left_idx = rf_build_branch_tree(tree,
														stage_features,
														labels,
														feature_vars_tmp,
														feature_dim,
														n_classes,
														left_indices_local,
														left_index_count,
														1,
														max_depth_arg,
														min_samples_arg,
														&rng,
														feature_order,
														feature_importance_tmp,
														&max_split_deviation);
					else
						left_idx = gtree_add_leaf(
												  tree, tree_left_value);

					if (right_indices_local != NULL
						&& right_index_count > 0)
						right_idx = rf_build_branch_tree(tree,
														 stage_features,
														 labels,
														 feature_vars_tmp,
														 feature_dim,
														 n_classes,
														 right_indices_local,
														 right_index_count,
														 1,
														 max_depth_arg,
														 min_samples_arg,
														 &rng,
														 feature_order,
														 feature_importance_tmp,
														 &max_split_deviation);
					else
						right_idx = gtree_add_leaf(
												   tree, tree_right_value);

					gtree_set_child(tree, node_idx, left_idx, true);
					gtree_set_child(
									tree, node_idx, right_idx, false);
					gtree_set_root(tree, node_idx);

					if (var0 > 0.0)
					{
						double		split_dev = fabs(tree_threshold)
							/ sqrt(var0);

						if (split_dev > max_split_deviation)
							max_split_deviation = split_dev;
					}

					if (tree_count == 0)
					{
						split_feature = feature_for_split;
						split_threshold = tree_threshold;
						left_leaf_value = tree_left_value;
						right_leaf_value = tree_right_value;
						left_branch_fraction =
							tree_left_fraction;
						right_branch_fraction =
							tree_right_fraction;
						branch_threshold_valid = true;
						second_value = tree_second_value;
						second_fraction = tree_second_frac;
					}

				}
				else
				{
					left_idx =
						gtree_add_leaf(tree, tree_left_value);
					gtree_set_root(tree, left_idx);
				}

				if (inbag != NULL && labels != NULL && n_samples > 0
					&& stage_features != NULL)
				{
					oob_total_local = 0;
					oob_correct_local = 0;

					for (j = 0; j < n_samples; j++)
					{
						int			actual;
						int			predicted;

						if (inbag[j])
							continue;
						if (!isfinite(labels[j]))
							continue;

						actual = (int) rint(labels[j]);
						if (actual < 0 || actual >= n_classes)
							continue;

						if (feature_dim > 0)
						{
							float	   *row = stage_features
								+ (j * feature_dim);
							double		tree_pred;

							tree_pred = rf_tree_predict_row(
															tree, row, feature_dim);
							predicted =
								(int) rint(tree_pred);
						}
						else
							predicted = (int) rint(
												   tree_left_value);

						oob_total_local++;
						if (predicted == actual)
							oob_correct_local++;
					}

					if (tree_oob_accuracy != NULL
						&& t < forest_trees)
					{
						if (oob_total_local > 0)
							tree_oob_accuracy[t] =
								(double) oob_correct_local
								/ (double)
								oob_total_local;
						else
							tree_oob_accuracy[t] = 0.0;
					}

					oob_total_all += oob_total_local;
					oob_correct_all += oob_correct_local;
				}

				gtree_validate(tree);

				if (primary_tree == NULL)
					primary_tree = tree;

				if (trees != NULL)
				{
					int			idx = tree_count;

					if (idx < forest_trees)
					{
						trees[idx] = tree;
						tree_majorities[idx] =
							tree_majority_value;
						tree_majority_fractions[idx] =
							tree_majority_frac;
						tree_seconds[idx] = tree_second_value;
						tree_second_fractions[idx] =
							tree_second_frac;
						tree_count++;
					}
				}

				if (tree_counts)
					nfree(tree_counts);
				if (tree_bootstrap)
					nfree(tree_bootstrap);
				if (inbag)
					nfree(inbag);
				if (left_indices_local)
					nfree(left_indices_local);
				if (right_indices_local)
					nfree(right_indices_local);
			}
		}

		if (oob_total_all > 0)
			forest_oob_accuracy =
				(double) oob_correct_all / (double) oob_total_all;

		if (feature_importance_tmp != NULL && feature_dim > 0)
		{
			double		importance_total = 0.0;
			double		top_import0 = 0.0;
			double		top_import1 = 0.0;
			double		top_import2 = 0.0;

			for (i = 0; i < feature_dim; i++)
			{
				double		val = feature_importance_tmp[i];

				if (val < 0.0)
					val = 0.0;
				feature_importance_tmp[i] = val;
				importance_total += val;

				if (val > top_import0)
				{
					top_import2 = top_import1;
					top_import1 = top_import0;
					top_import0 = val;
				}
				else if (val > top_import1)
				{
					top_import2 = top_import1;
					top_import1 = val;
				}
				else if (val > top_import2)
				{
					top_import2 = val;
				}
			}

			if (importance_total > 0.0)
			{
				for (i = 0; i < feature_dim; i++)
					feature_importance_tmp[i] /= importance_total;
				if (top_import0 > 0.0)
					top_import0 /= importance_total;
				if (top_import1 > 0.0)
					top_import1 /= importance_total;
				if (top_import2 > 0.0)
					top_import2 /= importance_total;
			}

		}

		model_id = rf_next_model_id++;
		if (feature_limit > 0 && left_branch_means_vec == NULL
			&& right_branch_means_vec == NULL)
			feature_limit = 0;

		rf_store_model(model_id,
					   feature_dim,
					   n_samples,
					   n_classes,
					   majority_value,
					   majority_fraction,
					   gini_impurity,
					   label_entropy,
					   class_counts_tmp,
					   feature_means_tmp,
					   feature_vars_tmp,
					   feature_importance_tmp,
					   primary_tree,
					   split_feature,
					   split_threshold,
					   second_value,
					   second_fraction,
					   left_leaf_value,
					   left_branch_fraction,
					   right_leaf_value,
					   right_branch_fraction,
					   max_deviation,
					   max_split_deviation,
					   feature_limit,
					   left_branch_means_vec,
					   right_branch_means_vec,
					   tree_count,
					   trees,
					   tree_majorities,
					   tree_majority_fractions,
					   tree_seconds,
					   tree_second_fractions,
					   tree_oob_accuracy,
					   forest_oob_accuracy);

		{
			RFModel    *stored_model = NULL;
			MLCatalogModelSpec spec;

			bytea *serialized = NULL;
			Jsonb *params_jsonb = NULL;
			Jsonb *metrics_jsonb = NULL;
			bytea *gpu_payload = NULL;
			Jsonb *gpu_metrics = NULL;
			char *gpu_err = NULL;
			bool		gpu_packed = false;

			stored_model = &rf_models[rf_model_count - 1];

			/* Build parameters JSON using JSONB API */
			{
				JsonbParseState *state = NULL;
				JsonbValue	jkey;
				JsonbValue	jval;

				JsonbValue *final_value = NULL;
				Numeric		n_trees_num,
							max_depth_num,
							min_samples_split_num;

				PG_TRY();
				{
					(void) pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);

					/* Add n_trees */
					jkey.type = jbvString;
					jkey.val.string.val = "n_trees";
					jkey.val.string.len = strlen("n_trees");
					(void) pushJsonbValue(&state, WJB_KEY, &jkey);
					n_trees_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(forest_trees_arg)));
					jval.type = jbvNumeric;
					jval.val.numeric = n_trees_num;
					(void) pushJsonbValue(&state, WJB_VALUE, &jval);

					/* Add max_depth */
					jkey.val.string.val = "max_depth";
					jkey.val.string.len = strlen("max_depth");
					(void) pushJsonbValue(&state, WJB_KEY, &jkey);
					max_depth_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(max_depth_arg)));
					jval.type = jbvNumeric;
					jval.val.numeric = max_depth_num;
					(void) pushJsonbValue(&state, WJB_VALUE, &jval);

					/* Add min_samples_split */
					jkey.val.string.val = "min_samples_split";
					jkey.val.string.len = strlen("min_samples_split");
					(void) pushJsonbValue(&state, WJB_KEY, &jkey);
					min_samples_split_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(min_samples_arg)));
					jval.type = jbvNumeric;
					jval.val.numeric = min_samples_split_num;
					(void) pushJsonbValue(&state, WJB_VALUE, &jval);

					final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);

					if (final_value == NULL)
					{
						elog(ERROR, "neurondb: train_random_forest: pushJsonbValue(WJB_END_OBJECT) returned NULL for parameters");
					}

					params_jsonb = JsonbValueToJsonb(final_value);
				}
				PG_CATCH();
				{
					ErrorData  *edata = CopyErrorData();

					elog(ERROR, "neurondb: train_random_forest: parameters JSONB construction failed: %s", edata->message);
					FlushErrorState();
					params_jsonb = NULL;
				}
				PG_END_TRY();
			}

			/* Build metrics JSON using JSONB API */
			{
				JsonbParseState *state = NULL;
				JsonbValue	jkey;
				JsonbValue	jval;

				JsonbValue *final_value = NULL;
				Numeric		oob_accuracy_num,
							gini_num,
							majority_fraction_num;

				PG_TRY();
				{
					(void) pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);

					/* Add oob_accuracy */
					jkey.type = jbvString;
					jkey.val.string.val = "oob_accuracy";
					jkey.val.string.len = strlen("oob_accuracy");
					(void) pushJsonbValue(&state, WJB_KEY, &jkey);
					oob_accuracy_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(forest_oob_accuracy)));
					jval.type = jbvNumeric;
					jval.val.numeric = oob_accuracy_num;
					(void) pushJsonbValue(&state, WJB_VALUE, &jval);

					/* Add gini */
					jkey.val.string.val = "gini";
					jkey.val.string.len = strlen("gini");
					(void) pushJsonbValue(&state, WJB_KEY, &jkey);
					gini_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(gini_impurity)));
					jval.type = jbvNumeric;
					jval.val.numeric = gini_num;
					(void) pushJsonbValue(&state, WJB_VALUE, &jval);

					/* Add majority_fraction */
					jkey.val.string.val = "majority_fraction";
					jkey.val.string.len = strlen("majority_fraction");
					(void) pushJsonbValue(&state, WJB_KEY, &jkey);
					majority_fraction_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(majority_fraction)));
					jval.type = jbvNumeric;
					jval.val.numeric = majority_fraction_num;
					(void) pushJsonbValue(&state, WJB_VALUE, &jval);

					final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);

					if (final_value == NULL)
					{
						elog(ERROR, "neurondb: train_random_forest: pushJsonbValue(WJB_END_OBJECT) returned NULL for metrics");
					}

					metrics_jsonb = JsonbValueToJsonb(final_value);
				}
				PG_CATCH();
				{
					ErrorData  *edata = CopyErrorData();

					elog(ERROR, "neurondb: train_random_forest: metrics JSONB construction failed: %s", edata->message);
					FlushErrorState();
					metrics_jsonb = NULL;
				}
				PG_END_TRY();
			}

			if (neurondb_gpu_is_available())
			{
				if (ndb_gpu_rf_pack_model(stored_model,
										  &gpu_payload,
										  &gpu_metrics,
										  &gpu_err)
					== 0)
				{
					serialized = gpu_payload;
					gpu_packed = true;
					if (gpu_metrics != NULL)
					{
						if (metrics_jsonb)
							nfree(metrics_jsonb);
						metrics_jsonb = gpu_metrics;
						gpu_metrics = NULL;
					}
					if (gpu_err != NULL)
					{

						/*
						 * Don't free gpu_err - it's allocated by GPU function
						 * using pstrdup() and will be automatically freed
						 * when the memory context is cleaned up.
						 */
						gpu_err = NULL;
					}
				}
			}

			if (!gpu_packed)
				serialized = rf_model_serialize(stored_model, 0);

			memset(&spec, 0, sizeof(spec));
			spec.algorithm = "random_forest";
			spec.model_type = "classification";
			spec.training_table = table_name;
			spec.training_column = label_col;
			spec.parameters = params_jsonb;
			spec.metrics = metrics_jsonb;
			spec.model_data = serialized;
			spec.training_time_ms = -1;
			spec.num_samples = n_samples;
			spec.num_features = feature_dim;

			model_id = ml_catalog_register_model(&spec);
			stored_model->model_id = model_id;
			if (model_id >= rf_next_model_id)
				rf_next_model_id = model_id + 1;

			/*
			 * Free SPI context allocations with NULL checks and set to NULL.
			 * These allocations are in SPI context and must be freed before
			 * SPI_finish() deletes that context.
			 */
			if (serialized != NULL)
			{
				nfree(serialized);
				serialized = NULL;
			}
			if (params_jsonb != NULL)
			{
				nfree(params_jsonb);
				params_jsonb = NULL;
			}
			if (metrics_jsonb != NULL)
			{
				nfree(metrics_jsonb);
				metrics_jsonb = NULL;
			}
			if (gpu_metrics != NULL)
			{
				nfree(gpu_metrics);
				gpu_metrics = NULL;
			}
			if (!gpu_packed && gpu_payload != NULL)
			{
				nfree(gpu_payload);
				gpu_payload = NULL;
			}
		}

		if (class_counts_tmp)
			nfree(class_counts_tmp);
		if (feature_means_tmp)
			nfree(feature_means_tmp);
		if (feature_vars_tmp)
			nfree(feature_vars_tmp);
		if (feature_importance_tmp)
			nfree(feature_importance_tmp);
		if (feature_sums)
			nfree(feature_sums);
		if (feature_sums_sq)
			nfree(feature_sums_sq);
		if (class_feature_sums)
			nfree(class_feature_sums);
		if (class_feature_counts)
			nfree(class_feature_counts);
		if (left_feature_sums_vec)
			nfree(left_feature_sums_vec);
		if (right_feature_sums_vec)
			nfree(right_feature_sums_vec);
		if (left_feature_counts_vec)
			nfree(left_feature_counts_vec);
		if (right_feature_counts_vec)
			nfree(right_feature_counts_vec);
		if (left_branch_means_vec)
			nfree(left_branch_means_vec);
		if (right_branch_means_vec)
			nfree(right_branch_means_vec);
		if (feature_order)
			nfree(feature_order);
		if (trees)
			nfree(trees);
		if (tree_majorities)
			nfree(tree_majorities);
		if (tree_majority_fractions)
			nfree(tree_majority_fractions);
		if (tree_seconds)
			nfree(tree_seconds);
		if (tree_second_fractions)
			nfree(tree_second_fractions);
		if (tree_oob_accuracy)
			nfree(tree_oob_accuracy);
		if (bootstrap_indices)
			nfree(bootstrap_indices);
		rf_dataset_free(&dataset);
	}

	/*
	 * Free query.data BEFORE ending SPI session (it's allocated in SPI
	 * context)
	 */
	if (query.data)
		nfree(query.data);
	NDB_SPI_SESSION_END(train_spi_session);

	if (table_name)
		nfree(table_name);
	if (feature_col)
		nfree(feature_col);
	if (label_col)
		nfree(label_col);
	PG_RETURN_INT32(model_id);
}

static double
rf_tree_predict_single(const GTree *tree,
					   const RFModel *model,
					   const Vector *vec,
					   double *left_dist,
					   double *right_dist,
					   int *leaf_out)
{
	const GTreeNode *nodes = NULL;
	int			idx;
	int			steps = 0;
	int			path_nodes[GTREE_MAX_DEPTH + 1];
	char		path_dir[GTREE_MAX_DEPTH];
	int			path_len = 0;
	int			leaf_idx = -1;
	double		result = 0.0;
	int			i;

	if (leaf_out)
		*leaf_out = -1;

	if (left_dist)
		*left_dist = -1.0;
	if (right_dist)
		*right_dist = -1.0;

	if (tree == NULL)
		return (model != NULL) ? model->majority_value : 0.0;

	if (tree->root < 0 || tree->count <= 0)
		return (model != NULL) ? model->majority_value : 0.0;

	nodes = gtree_nodes(tree);
	idx = tree->root;

	while (idx >= 0 && idx < tree->count)
	{
		const GTreeNode *node = &nodes[idx];

		if (path_len <= GTREE_MAX_DEPTH)
			path_nodes[path_len] = idx;

		if (node->is_leaf)
		{
			leaf_idx = idx;
			break;
		}

		if (vec == NULL || node->feature_idx < 0
			|| node->feature_idx >= vec->dim)
		{
			if (model)
				return model->majority_value;
			return 0.0;
		}

		if (vec->data[node->feature_idx] <= node->threshold)
		{
			if (path_len < GTREE_MAX_DEPTH)
				path_dir[path_len] = 'L';
			idx = node->left;
		}
		else
		{
			if (path_len < GTREE_MAX_DEPTH)
				path_dir[path_len] = 'R';
			idx = node->right;
		}

		path_len++;

		if (++steps > GTREE_MAX_DEPTH)
			break;
	}

	if (leaf_idx >= 0 && leaf_idx < tree->count && nodes[leaf_idx].is_leaf)
		result = nodes[leaf_idx].value;
	else
		result = (model != NULL) ? model->majority_value : 0.0;

	if (path_len > GTREE_MAX_DEPTH)
		path_len = GTREE_MAX_DEPTH;

	if (leaf_idx >= 0 && path_len <= GTREE_MAX_DEPTH)
		path_nodes[path_len] = leaf_idx;

	{
		StringInfoData path_log;
		int			edge_count = (leaf_idx >= 0) ? path_len : path_len - 1;

		initStringInfo(&path_log);
		appendStringInfo(&path_log, "[");
		for (i = 0; i <= edge_count && i <= GTREE_MAX_DEPTH; i++)
		{
			if (i > 0)
				appendStringInfoString(&path_log, ", ");
			appendStringInfo(&path_log, "%d", path_nodes[i]);
			if (i < edge_count && i < GTREE_MAX_DEPTH)
				appendStringInfo(&path_log, "%c", path_dir[i]);
		}
		appendStringInfoChar(&path_log, ']');
		nfree(path_log.data);
	}

	if (left_dist != NULL && right_dist != NULL && vec != NULL
		&& model != NULL && model->feature_limit > 0
		&& model->left_branch_means != NULL
		&& model->right_branch_means != NULL)
	{
		int			limit = model->feature_limit;
		int			f;
		const float *vec_data;
		double		lsum = 0.0;
		double		rsum = 0.0;

		if (vec->dim < limit)
			limit = vec->dim;
		if (model->n_features < limit)
			limit = model->n_features;

		vec_data = vec->data;
		for (f = 0; f < limit; f++)
		{
			double		val = (double) vec_data[f];
			double		ldiff = val - model->left_branch_means[f];
			double		rdiff = val - model->right_branch_means[f];

			lsum += ldiff * ldiff;
			rsum += rdiff * rdiff;
		}

		*left_dist = sqrt(lsum);
		*right_dist = sqrt(rsum);
	}

	if (leaf_out)
		*leaf_out = leaf_idx;

	return result;
}

Datum
predict_random_forest(PG_FUNCTION_ARGS)
{
	int32		model_id = 0;
	RFModel *model = NULL;

	Vector *feature_vec = NULL;
	double		result = 0.0;
	double		split_z = 0.0;
	bool		split_z_valid = false;
	const char *branch_name = "majority";
	double		branch_fraction = 0.0;
	double		branch_value = 0.0;
	double		left_mean_dist = -1.0;
	double		right_mean_dist = -1.0;
	int			mean_limit = 0;
	double		vote_majority = 0.0;
	double		best_vote_fraction = 0.0;
	double		second_vote_value = 0.0;
	double		second_vote_fraction = 0.0;
	double		vote_total_weight = 0.0;
	int			i = 0;
	double *vote_histogram = NULL;
	int			vote_classes = 0;
	double		fallback_value = 0.0;
	double		fallback_fraction = 0.0;
	int			top_feature_idx = -1;
	double		top_feature_importance = 0.0;
	MemoryContext oldcontext = NULL;

	(void) branch_name;		/* Reserved for future use */
	(void) split_z_valid;		/* Reserved for future use */
	(void) branch_value;		/* Reserved for future use */
	(void) top_feature_idx;		/* Reserved for future use */

	if (!PG_ARGISNULL(0))
		model_id = PG_GETARG_INT32(0);
	else
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: predict_random_forest: model ID is required"),
				 errdetail("First argument (model_id) cannot be NULL"),
				 errhint("Provide a valid model ID from the training function.")));

	if (!PG_ARGISNULL(1))
	{
		feature_vec = PG_GETARG_VECTOR_P(1);
		NDB_CHECK_VECTOR_VALID(feature_vec);
	}
	else
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: predict_random_forest: feature vector is required"),
				 errdetail("Second argument (feature vector) cannot be NULL"),
				 errhint("Provide a valid feature vector for prediction.")));

	if (!rf_lookup_model(model_id, &model))
	{
		bool		gpu_prediction_failed = false;
		
		/* Try GPU prediction first if available */
		if (NDB_SHOULD_TRY_GPU() && neurondb_gpu_is_available())
		{
			if (rf_try_gpu_predict_catalog(model_id, feature_vec, &result))
			{
				PG_RETURN_FLOAT8(result);
			}
			else
			{
				gpu_prediction_failed = true;
			}
		}
		
		/* If GPU prediction failed, check if model is GPU-only before trying CPU fallback */
		if (gpu_prediction_failed)
		{
			bytea *payload = NULL;
			Jsonb *metrics = NULL;
			bool		is_gpu_only = false;
			bool		fetch_succeeded = false;
			
			/* Check if model is GPU-only */
			{
				MemoryContext oldctx = MemoryContextSwitchTo(TopMemoryContext);
				fetch_succeeded = ml_catalog_fetch_model_payload(model_id, &payload, NULL, &metrics);
				MemoryContextSwitchTo(oldctx);
				
				if (fetch_succeeded)
				{
					if (metrics != NULL)
					{
						is_gpu_only = rf_metadata_is_gpu(metrics);
					}
					
					/* If metadata check didn't determine it, check payload format */
					if (!is_gpu_only && payload != NULL)
					{
						/* Check payload format - GPU models start with NdbCudaRfModelHeader */
						int payload_size = VARSIZE(payload) - VARHDRSZ;
						if (payload_size >= sizeof(int32))
						{
							const int32 *first_int = (const int32 *) VARDATA(payload);
							int32 first_value = *first_int;
							
							/* If first 4 bytes look like a reasonable tree_count, it's likely GPU format */
							if (first_value > 0 && first_value <= 10000)
							{
								if (payload_size >= sizeof(NdbCudaRfModelHeader))
								{
									const NdbCudaRfModelHeader *hdr = (const NdbCudaRfModelHeader *) VARDATA(payload);
									if (hdr->tree_count == first_value &&
										hdr->feature_dim > 0 && hdr->feature_dim <= 100000 &&
										hdr->class_count > 0 && hdr->class_count <= 10000)
									{
										is_gpu_only = true;
									}
								}
							}
						}
					}
				}
			}
			
			/* If model is GPU-only and GPU prediction failed, give specific error */
			if (fetch_succeeded && is_gpu_only)
			{
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: predict_random_forest: model %d is GPU-only and GPU prediction failed", model_id),
						 errdetail("Model %d was trained on GPU and requires GPU for prediction, but GPU prediction failed: %s", 
								   model_id, "GPU RF inference failed"),
						 errhint("Check GPU availability and configuration, or train a new model with CPU backend.")));
			}
		}
		
		/* Switch to TopMemoryContext before calling rf_load_model_from_catalog
		 * to ensure allocations happen in a stable context, not the PL/pgSQL context.
		 * The model will be stored in TopMemoryContext via rf_store_model, so it
		 * will persist after we switch back. */
		oldcontext = CurrentMemoryContext;
		if (TopMemoryContext != NULL)
			MemoryContextSwitchTo(TopMemoryContext);
		
		if (!rf_load_model_from_catalog(model_id, &model))
		{
			if (oldcontext != NULL)
				MemoryContextSwitchTo(oldcontext);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: predict_random_forest: model %d not found", model_id),
					 errdetail("Model with ID %d does not exist in the catalog", model_id),
					 errhint("Verify the model ID or train a new model.")));
		}
		
		/* Switch back to original context before returning.
		 * The model is stored in TopMemoryContext (via rf_store_model), so it's safe
		 * to switch back. PostgreSQL functions should return in the same context
		 * they were called in. */
		if (oldcontext != NULL)
			MemoryContextSwitchTo(oldcontext);
	}
	else
	{
	}

	fallback_value = model->second_value;
	fallback_fraction = model->second_fraction;
	branch_fraction = model->majority_fraction;
	branch_value = model->majority_value;

	if (model->n_features > 0 && feature_vec != NULL
		&& feature_vec->dim != model->n_features)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: predict_random_forest: feature dimension mismatch"),
				 errdetail("Model expects %d features but received %d", model->n_features, feature_vec->dim),
				 errhint("Ensure the feature vector has the same dimension as the training data.")));

	if (model->feature_means != NULL && feature_vec != NULL)
	{
		float	   *vec_data = feature_vec->data;
		double		dist = 0.0;
		int			j;
		double		max_z = 0.0;

		for (j = 0; j < model->n_features && j < feature_vec->dim; j++)
		{
			double		diff =
				(double) vec_data[j] - model->feature_means[j];

			dist += diff * diff;
			if (model->feature_variances != NULL)
			{
				double		var = model->feature_variances[j];
				double		z;

				if (var <= 0.0)
					continue;
				z = fabs(diff) / sqrt(var);
				if (z > max_z)
					max_z = z;
			}
		}
		dist = sqrt(dist);
		if (model->feature_variances != NULL
			&& model->second_fraction > 0.0 && max_z > 1.5)
			model->max_deviation = max_z;
	}
	else if (!PG_ARGISNULL(1))
		model->max_deviation = 0.0;

	if (feature_vec != NULL && model->split_feature >= 0)
	{
		int			sf = model->split_feature;

		if (sf < feature_vec->dim)
		{
			double		value = (double) feature_vec->data[sf];

			if (value <= model->split_threshold)
			{
				branch_name = "left";
				branch_fraction = model->left_branch_fraction;
				branch_value = model->left_branch_value;
			}
			else
			{
				branch_name = "right";
				branch_fraction = model->right_branch_fraction;
				branch_value = model->right_branch_value;
			}
			if (branch_fraction <= 0.0)
				branch_fraction = model->majority_fraction;

			if (model->feature_variances != NULL
				&& sf < model->n_features)
			{
				double		var = model->feature_variances[sf];

				if (var > 0.0)
				{
					split_z =
						(value - model->split_threshold)
						/ sqrt(var);
					if (fabs(split_z)
						> model->max_split_deviation)
						model->max_split_deviation =
							fabs(split_z);
					split_z_valid = true;
				}
			}
		}
	}

	if (model->n_classes > 0)
	{
		vote_classes = model->n_classes;
		nalloc(vote_histogram, double, vote_classes);
	}

	if (model->tree_count > 0 && model->trees != NULL)
	{
		for (i = 0; i < model->tree_count; i++)
		{
			const GTree *tree = model->trees[i];
			double		tree_left = -1.0;
			double		tree_right = -1.0;
			int			leaf_idx = -1;
			double		tree_result;
			double		vote_weight = 1.0;

			tree_result = rf_tree_predict_single(tree,
												 model,
												 feature_vec,
												 &tree_left,
												 &tree_right,
												 &leaf_idx);

			if (model->tree_oob_accuracy != NULL
				&& i < model->tree_count)
			{
				vote_weight = model->tree_oob_accuracy[i];
				if (vote_weight <= 0.0)
					vote_weight = 1.0;
			}

			if (vote_histogram != NULL)
			{
				int			cls = (int) rint(tree_result);

				if (cls >= 0 && cls < vote_classes)
				{
					vote_histogram[cls] += vote_weight;
					vote_total_weight += vote_weight;
				}
			}

			if (tree_left >= 0.0 && tree_right >= 0.0)
			{
				if (mean_limit <= 0)
				{
					left_mean_dist = tree_left;
					right_mean_dist = tree_right;
					mean_limit = model->feature_limit;
				}
			}
		}
	}
	else
	{
		result = rf_tree_predict_single(model->tree,
										model,
										feature_vec,
										&left_mean_dist,
										&right_mean_dist,
										NULL);
		if (vote_histogram != NULL)
		{
			int			cls = (int) rint(result);

			if (cls >= 0 && cls < vote_classes)
			{
				vote_histogram[cls] += 1.0;
				vote_total_weight += 1.0;
			}
		}
	}

	if (vote_histogram != NULL && vote_total_weight > 0.0)
	{
		int			best_idx = -1;
		int			second_idx = -1;
		double		best_weight = -1.0;
		double		second_weight = -1.0;

		for (i = 0; i < vote_classes; i++)
		{
			double		weight = vote_histogram[i];

			if (weight > best_weight)
			{
				if (best_idx >= 0)
				{
					second_idx = best_idx;
					second_weight = best_weight;
				}
				best_idx = i;
				best_weight = weight;
			}
			else if (weight > second_weight)
			{
				second_idx = i;
				second_weight = weight;
			}
		}

		if (best_idx >= 0 && best_weight > 0.0)
		{
			vote_majority = (double) best_idx;
			best_vote_fraction = best_weight / vote_total_weight;
			result = vote_majority;
			branch_name = "forest";
			branch_fraction = best_vote_fraction;
			branch_value = vote_majority;
		}

		if (second_idx >= 0 && second_weight > 0.0)
		{
			second_vote_value = (double) second_idx;
			second_vote_fraction =
				second_weight / vote_total_weight;
		}

		nfree(vote_histogram);
		vote_histogram = NULL;
	}

	if (best_vote_fraction <= 0.0)
		result = model->majority_value;

	if (second_vote_fraction > 0.0)
	{
		fallback_value = second_vote_value;
		fallback_fraction = second_vote_fraction;
	}

	if (mean_limit > 0 && left_mean_dist >= 0.0 && right_mean_dist >= 0.0)
	{

		if (right_mean_dist + 0.10 < left_mean_dist
			&& model->right_branch_fraction > 0.0)
		{
			result = model->right_branch_value;
			branch_name = "right-mean";
			branch_fraction = model->right_branch_fraction;
			branch_value = model->right_branch_value;
		}
		else if (left_mean_dist + 0.10 < right_mean_dist
				 && model->left_branch_fraction > 0.0)
		{
			result = model->left_branch_value;
			branch_name = "left-mean";
			branch_fraction = model->left_branch_fraction;
			branch_value = model->left_branch_value;
		}
	}

	if (model->feature_variances != NULL && fallback_fraction > 0.0
		&& !PG_ARGISNULL(1) && model->max_deviation > 2.0
		&& model->label_entropy > 0.1)
	{
		result = fallback_value;
		branch_name = "fallback";
		branch_fraction = fallback_fraction;
		branch_value = fallback_value;
	}

	if (result != model->majority_value)

	if (model->split_feature >= 0)
	{
		/* split_z_valid check removed */
	}

	if (model->feature_importance != NULL && model->n_features > 0)
	{
		for (i = 0; i < model->n_features; i++)
		{
			double		val = model->feature_importance[i];

			if (val > top_feature_importance)
			{
				top_feature_importance = val;
				top_feature_idx = i;
			}
		}
	}

	PG_RETURN_FLOAT8(result);
}

Datum
evaluate_random_forest(PG_FUNCTION_ARGS)
{
	Datum		result_datums[4];
	ArrayType *result_array = NULL;
	int32		model_id;

	RFModel *model = NULL;
	double		accuracy = 0.0;
	double		error_rate = 0.0;
	double		gini = 0.0;
	int			n_classes = 0;

	bytea *payload = NULL;
	Jsonb *metrics = NULL;

	if (PG_NARGS() < 1 || PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_random_forest: model_id is required"),
				 errdetail("First argument (model_id) cannot be NULL"),
				 errhint("Provide a valid model ID from the training function.")));

	model_id = PG_GETARG_INT32(0);

	if (!rf_lookup_model(model_id, &model))
	{
		if (!ml_catalog_fetch_model_payload(
											model_id, &payload, NULL, &metrics))
		{
			elog(WARNING,
				 "evaluate_random_forest: ml_catalog_fetch_model_payload returned false for model_id %d",
				 model_id);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("random_forest: model %d not found",
							model_id)));
		}

		if (rf_metadata_is_gpu(metrics) && metrics != NULL)
		{
			Datum		acc_datum;
			Datum		gini_datum;
			Datum		acc_numeric;
			Datum		gini_numeric;
			Numeric		acc_num;
			Numeric		gini_num;

			acc_datum = DirectFunctionCall2(
											jsonb_object_field,
											JsonbPGetDatum(metrics),
											CStringGetTextDatum("majority_fraction"));
			if (DatumGetPointer(acc_datum) == NULL)
			{
				acc_datum = DirectFunctionCall2(
												jsonb_object_field,
												JsonbPGetDatum(metrics),
												CStringGetTextDatum("oob_accuracy"));
			}
			if (DatumGetPointer(acc_datum) != NULL)
			{
				PG_TRY();
				{
					acc_numeric = DirectFunctionCall1(
													  jsonb_numeric,
													  acc_datum);
					if (DatumGetPointer(acc_numeric) != NULL)
					{
						acc_num = DatumGetNumeric(acc_numeric);
						accuracy = DatumGetFloat8(
												  DirectFunctionCall1(numeric_float8,
																	  NumericGetDatum(acc_num)));
					}
				}
				PG_CATCH();
				{
					{
						Datum		acc_text;

						acc_text = DirectFunctionCall1(
													   jsonb_extract_path_text,
													   acc_datum);
						if (DatumGetPointer(acc_text) != NULL)
						{
							char *acc_str = NULL;

							acc_str = TextDatumGetCString(acc_text);

							if (acc_str != NULL && strlen(acc_str) > 0)
							{
								accuracy = strtod(acc_str, NULL);
							}
							nfree(acc_str);
						}
					}
				}
				PG_END_TRY();
			}

			gini_datum = DirectFunctionCall2(
											 jsonb_object_field,
											 JsonbPGetDatum(metrics),
											 CStringGetTextDatum("gini"));
			if (DatumGetPointer(gini_datum) != NULL)
			{
				PG_TRY();
				{
					gini_numeric = DirectFunctionCall1(
													   jsonb_numeric,
													   gini_datum);
					if (DatumGetPointer(gini_numeric) != NULL)
					{
						gini_num = DatumGetNumeric(gini_numeric);
						gini = DatumGetFloat8(
											  DirectFunctionCall1(numeric_float8,
																  NumericGetDatum(gini_num)));
					}
				}
				PG_CATCH();
				{
					Datum		gini_text = DirectFunctionCall1(
																jsonb_extract_path_text,
																gini_datum);

					if (DatumGetPointer(gini_text) != NULL)
					{
						char	   *gini_str = TextDatumGetCString(gini_text);

						if (gini_str != NULL && strlen(gini_str) > 0)
							gini = strtod(gini_str, NULL);
						nfree(gini_str);
					}
				}
				PG_END_TRY();
			}

			/* n_classes is not in metrics, use default */
			n_classes = 2;

			if (metrics)
				nfree(metrics);
			if (payload)
				nfree(payload);

			error_rate = (accuracy > 1.0) ? 0.0 : (1.0 - accuracy);

			result_datums[0] = Float8GetDatum(accuracy);
			result_datums[1] = Float8GetDatum(error_rate);
			result_datums[2] = Float8GetDatum(gini);
			result_datums[3] = Float8GetDatum((double) n_classes);

			result_array = construct_array(
										   result_datums, 4, FLOAT8OID, sizeof(float8), true, 'd');

			PG_RETURN_ARRAYTYPE_P(result_array);
		}


		PG_TRY();
		{
			if (!rf_load_model_from_catalog(model_id, &model))
			{
				if (payload)
					nfree(payload);
				if (metrics)
					nfree(metrics);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("random_forest: model %d not found or failed to load",
								model_id)));
			}
		}
		PG_CATCH();
		{
			elog(WARNING, "evaluate_random_forest: exception during CPU model load for model_id %d", model_id);
			if (payload)
				nfree(payload);
			if (metrics)
				nfree(metrics);
			FlushErrorState();
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("random_forest: failed to load model %d (deserialization error)",
							model_id)));
		}
		PG_END_TRY();
	}

	if (model == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("random_forest: model %d not found",
						model_id)));

	if (model->oob_accuracy > 0.0)
		accuracy = model->oob_accuracy;
	else
		accuracy = model->majority_fraction;
	error_rate = (accuracy > 1.0) ? 0.0 : (1.0 - accuracy);
	gini = model->gini_impurity;
	n_classes = model->n_classes;

	result_datums[0] = Float8GetDatum(accuracy);
	result_datums[1] = Float8GetDatum(error_rate);
	result_datums[2] = Float8GetDatum(gini);
	result_datums[3] = Float8GetDatum((double) n_classes);

	result_array = construct_array(
								   result_datums, 4, FLOAT8OID, sizeof(float8), true, 'd');

	PG_RETURN_ARRAYTYPE_P(result_array);
}

/*
 * rf_predict_batch
 *
 * (removed unused function rf_predict_batch)
 */
#if 0
/* Removed unused function rf_predict_batch */
static void
rf_predict_batch_removed(const RFModel *model,
				 const float *features,
				 const double *labels,
				 int n_samples,
				 int feature_dim,
				 int *tp_out,
				 int *tn_out,
				 int *fp_out,
				 int *fn_out)
{
	int			i;
	int			tp = 0;
	int			tn = 0;
	int			fp = 0;
	int			fn = 0;
	int			n_classes = model->n_classes;

	double *vote_histogram = NULL;
	int			vote_classes = n_classes;

	if (model == NULL || features == NULL || labels == NULL || n_samples <= 0)
	{
		if (tp_out)
			*tp_out = 0;
		if (tn_out)
			*tn_out = 0;
		if (fp_out)
			*fp_out = 0;
		if (fn_out)
			*fn_out = 0;
		return;
	}

	/* Ensure n_classes is at least 2 for binary classification */
	if (vote_classes <= 0)
	{
		vote_classes = 2;
		n_classes = 2;
	}

	if (vote_classes > 0)
		nalloc(vote_histogram, double, vote_classes);
	NDB_CHECK_ALLOC(vote_histogram, "vote_histogram");

	for (i = 0; i < n_samples; i++)
	{
		const float *row = features + (i * feature_dim);
		double		y_true = labels[i];
		int			true_class;
		double		prediction = 0.0;
		int			pred_class;
		int			j;
		double		vote_total_weight = 0.0;

		if (!isfinite(y_true))
		{
			continue;
		}

		true_class = (int) rint(y_true);
		if (true_class < 0 || true_class >= n_classes)
		{
			continue;
		}

		if (vote_histogram != NULL)
		{
			for (j = 0; j < vote_classes; j++)
				vote_histogram[j] = 0.0;
		}

		if (model->tree_count > 0 && model->trees != NULL)
		{
			int			t;

			for (t = 0; t < model->tree_count; t++)
			{
				const GTree *tree = model->trees[t];
				double		tree_result;
				double		vote_weight = 1.0;

				tree_result = rf_tree_predict_row(tree, row, feature_dim);

				if (model->tree_oob_accuracy != NULL && t < model->tree_count)
				{
					vote_weight = model->tree_oob_accuracy[t];
					if (vote_weight <= 0.0)
						vote_weight = 1.0;
				}

				if (vote_histogram != NULL)
				{
					int			cls = (int) rint(tree_result);

					if (cls >= 0 && cls < vote_classes)
					{
						vote_histogram[cls] += vote_weight;
						vote_total_weight += vote_weight;
					}
				}
			}
		}
		else if (model->tree != NULL)
		{
			double		tree_result;

			tree_result = rf_tree_predict_row(model->tree, row, feature_dim);
			if (vote_histogram != NULL)
			{
				int			cls = (int) rint(tree_result);

				if (cls >= 0 && cls < vote_classes)
				{
					vote_histogram[cls] += 1.0;
					vote_total_weight += 1.0;
				}
			}
		}
		else
		{
			prediction = model->majority_value;
			pred_class = (int) rint(prediction);
		}

		if (vote_histogram != NULL && vote_total_weight > 0.0)
		{
			int			best_idx = -1;
			double		best_weight = -1.0;

			for (j = 0; j < vote_classes; j++)
			{
				if (vote_histogram[j] > best_weight)
				{
					best_idx = j;
					best_weight = vote_histogram[j];
				}
			}

			if (best_idx >= 0)
			{
				prediction = (double) best_idx;
				pred_class = best_idx;
			}
			else
			{
				prediction = model->majority_value;
				pred_class = (int) rint(prediction);
			}
		}
		else
		{
			prediction = model->majority_value;
			pred_class = (int) rint(prediction);
		}

		/* Ensure pred_class is in valid range for binary classification */
		if (n_classes == 2)
		{
			if (pred_class < 0)
				pred_class = 0;
			else if (pred_class > 1)
				pred_class = 1;

			if (true_class == 1 && pred_class == 1)
				tp++;
			else if (true_class == 0 && pred_class == 0)
				tn++;
			else if (true_class == 0 && pred_class == 1)
				fp++;
			else if (true_class == 1 && pred_class == 0)
				fn++;
			else
			{
			}
		}
		else
		{
			/* For multi-class, ensure pred_class is in valid range */
			if (pred_class < 0)
				pred_class = 0;
			else if (pred_class >= n_classes)
				pred_class = n_classes - 1;

			if (true_class == pred_class)
				tp++;
			else
				fn++;
		}
	}

	if (vote_histogram != NULL)
		nfree(vote_histogram);


	if (tp_out)
		*tp_out = tp;
	if (tn_out)
		*tn_out = tn;
	if (fp_out)
		*fp_out = fp;
	if (fn_out)
		*fn_out = fn;
}
#endif

/*
 * evaluate_random_forest_by_model_id
 *
 * Evaluates Random Forest model by model_id using optimized batch evaluation.
 * Supports both GPU and CPU models with GPU-accelerated batch evaluation when available.
 *
 * Returns jsonb with metrics: accuracy, precision, recall, f1_score, n_samples
 */
PG_FUNCTION_INFO_V1(evaluate_random_forest_by_model_id);

Datum
evaluate_random_forest_by_model_id(PG_FUNCTION_ARGS)
{
	int32		model_id;
	text *table_name = NULL;
	text *feature_col = NULL;
	text *label_col = NULL;

	char *tbl_str = NULL;
	char *feat_str = NULL;
	char *targ_str = NULL;
	int			ret;
	int			nvec = 0;
	int			i;
	Oid			feat_type_oid = InvalidOid;
	bool		feat_is_array = false;
	double		accuracy = 0.0;
	double		precision = 0.0;
	double		recall = 0.0;
	double		f1_score = 0.0;
	int			tp = 0;
	int			tn = 0;
	int			fp = 0;
	int			fn = 0;
	MemoryContext oldcontext = NULL;

	NdbSpiSession *eval_spi_session = NULL;
	StringInfoData query = {0};

	RFModel *model = NULL;
	Jsonb *result_jsonb = NULL;
	bytea *gpu_payload = NULL;
	Jsonb *gpu_metrics = NULL;
	bool		is_gpu_model = false;


	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_random_forest_by_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);

	/* Validate model_id before attempting to load */
	if (model_id <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_random_forest_by_model_id: model_id must be positive, got %d", model_id),
				 errdetail("Invalid model_id: %d", model_id),
				 errhint("Provide a valid model_id from neurondb.ml_models catalog.")));

	if (PG_ARGISNULL(1) || PG_ARGISNULL(2) || PG_ARGISNULL(3))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_random_forest_by_model_id: table_name, feature_col, and label_col are required")));

	table_name = PG_GETARG_TEXT_PP(1);
	feature_col = PG_GETARG_TEXT_PP(2);
	label_col = PG_GETARG_TEXT_PP(3);

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(label_col);


	oldcontext = CurrentMemoryContext;

	if (!rf_lookup_model(model_id, &model))
	{
		if (!rf_load_model_from_catalog(model_id, &model))
		{
			if (ml_catalog_fetch_model_payload(model_id, &gpu_payload, NULL, &gpu_metrics))
			{
				if (gpu_payload == NULL)
				{
					if (gpu_metrics)
						nfree(gpu_metrics);
					ereport(ERROR,
							(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							 errmsg("neurondb: evaluate_random_forest_by_model_id: model %d has no model_data (model may not have been trained or stored correctly)",
									model_id),
							 errhint("The model exists in the catalog but has no training data. Please retrain the model.")));
				}

				/* Check if this is a GPU model - either by metrics or by payload format */
				{
					uint32		payload_size;

					/* First check metrics for training_backend */
					if (gpu_metrics != NULL && rf_metadata_is_gpu(gpu_metrics))
					{
						is_gpu_model = true;
					}
					else
					{
						/* If metrics check didn't find GPU indicator, check payload format */
						/* GPU models start with NdbCudaRfModelHeader, CPU models start with uint8 training_backend */
						payload_size = VARSIZE(gpu_payload) - VARHDRSZ;
						
						/* CPU format: first byte is training_backend (uint8), then model_id (int32) */
						/* GPU format: first field is tree_count (int32) */
						/* Check if payload looks like GPU format (starts with int32, not uint8) */
						if (payload_size >= sizeof(int32))
						{
							const int32 *first_int = (const int32 *) VARDATA(gpu_payload);
							int32		first_value = *first_int;
							
							/* If first 4 bytes look like a reasonable tree_count, check for GPU format */
							if (first_value > 0 && first_value <= 10000)
							{
								/* Check if payload size matches GPU format */
								if (payload_size >= sizeof(NdbCudaRfModelHeader))
								{
									const NdbCudaRfModelHeader *hdr = (const NdbCudaRfModelHeader *) VARDATA(gpu_payload);
									
									/* Validate header fields */
									if (hdr->tree_count == first_value &&
										hdr->feature_dim > 0 && hdr->feature_dim <= 100000 &&
										hdr->class_count > 0 && hdr->class_count <= 10000 &&
										hdr->sample_count >= 0 && hdr->sample_count <= 1000000000)
									{
										/* Size check - GPU format has header + trees, minimum size check */
										if (payload_size >= sizeof(NdbCudaRfModelHeader) + 100)
										{
											is_gpu_model = true;
										}
									}
								}
							}
						}
					}
				}
				
				if (!is_gpu_model && gpu_payload != NULL)
				{
					if (gpu_payload)
						nfree(gpu_payload);
					if (gpu_metrics)
						nfree(gpu_metrics);
					gpu_payload = NULL;
					gpu_metrics = NULL;
					if (!rf_load_model_from_catalog(model_id, &model))
					{
						ereport(ERROR,
								(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
								 errmsg("neurondb: evaluate_random_forest_by_model_id: model %d not found",
										model_id)));
					}
				}
			}
			else
			{
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: evaluate_random_forest_by_model_id: model %d not found",
								model_id)));
			}
		}
	}

	if (model == NULL && !is_gpu_model && gpu_payload == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_random_forest_by_model_id: model %d not found",
						model_id)));

	oldcontext = CurrentMemoryContext;
	Assert(oldcontext != NULL);
	NDB_SPI_SESSION_BEGIN(eval_spi_session, oldcontext);

	initStringInfo(&query);
	appendStringInfo(&query,
					 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
					 quote_identifier(feat_str),
					 quote_identifier(targ_str),
					 quote_identifier(tbl_str),
					 quote_identifier(feat_str),
					 quote_identifier(targ_str));

	ret = ndb_spi_execute_safe(query.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_SELECT)
	{
		nfree(query.data);
		if (model != NULL)
			rf_free_deserialized_model(model);
		nfree(tbl_str);
		nfree(feat_str);
		nfree(targ_str);
		NDB_SPI_SESSION_END(eval_spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: evaluate_random_forest_by_model_id: query failed"),
				 errdetail("SPI execution returned code %d (expected %d)", ret, SPI_OK_SELECT),
				 errhint("Verify the table exists and contains valid feature and label columns.")));
	}

	nvec = SPI_processed;
	if (nvec < 1)
	{
		nfree(query.data);
		if (model != NULL)
			rf_free_deserialized_model(model);
		nfree(tbl_str);
		nfree(feat_str);
		nfree(targ_str);
		NDB_SPI_SESSION_END(eval_spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_random_forest_by_model_id: no valid rows found"),
				 errdetail("Dataset contains %d rows, minimum required is 10", nvec),
				 errhint("Add more data rows to the evaluation table.")));
	}

	if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL)
		feat_type_oid = SPI_gettypeid(SPI_tuptable->tupdesc, 1);
	if (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID)
		feat_is_array = true;

	/* Unified evaluation: Determine predict function based on compute mode */
	/* All metrics calculation is the same - only difference is predict function */
	{
		bool		use_gpu_predict = false;
		int			processed_count = 0;
		int			feat_dim = 0;
		const NdbCudaRfModelHeader *gpu_hdr = NULL;

		/* Determine if we should use GPU predict or CPU predict based on compute mode */
		if (is_gpu_model && neurondb_gpu_is_available() && !NDB_COMPUTE_MODE_IS_CPU())
		{
			/* GPU model and GPU mode: use GPU predict */
			if (gpu_payload != NULL && VARSIZE(gpu_payload) - VARHDRSZ >= sizeof(NdbCudaRfModelHeader))
			{
				gpu_hdr = (const NdbCudaRfModelHeader *) VARDATA(gpu_payload);
				feat_dim = gpu_hdr->feature_dim;
				use_gpu_predict = true;
			}
		}
		else if (model != NULL)
		{
			/* CPU model or CPU mode: use CPU predict */
			feat_dim = model->n_features;
			use_gpu_predict = false;
		}
		else if (is_gpu_model && gpu_payload != NULL)
		{
			/* GPU model but CPU mode: convert to CPU format for CPU predict */
			if (VARSIZE(gpu_payload) - VARHDRSZ >= sizeof(NdbCudaRfModelHeader))
			{
				gpu_hdr = (const NdbCudaRfModelHeader *) VARDATA(gpu_payload);
				feat_dim = gpu_hdr->feature_dim;

				/* Try to deserialize GPU model as CPU model */
				model = rf_model_deserialize(gpu_payload, NULL);
				if (model == NULL)
				{
					NDB_SPI_SESSION_END(eval_spi_session);
					nfree(gpu_payload);
					nfree(gpu_metrics);
					nfree(tbl_str);
					nfree(feat_str);
					nfree(targ_str);
					ndb_spi_stringinfo_free(eval_spi_session, &query);
					ereport(ERROR,
							(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							 errmsg("neurondb: evaluate_random_forest_by_model_id: failed to convert GPU model to CPU format"),
							 errdetail("GPU model conversion failed for model %d", model_id),
							 errhint("GPU model cannot be evaluated in CPU mode. Use GPU mode or retrain the model.")));
				}
				use_gpu_predict = false;
			}
		}

		/* Ensure we have a valid model or GPU payload */
		if (model == NULL && !use_gpu_predict)
		{
			NDB_SPI_SESSION_END(eval_spi_session);
			nfree(gpu_payload);
			nfree(gpu_metrics);
			nfree(tbl_str);
			nfree(feat_str);
			nfree(targ_str);
			ndb_spi_stringinfo_free(eval_spi_session, &query);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_random_forest_by_model_id: no valid model found"),
					 errdetail("Neither CPU model nor GPU payload is available"),
					 errhint("Verify the model exists in the catalog and is in the correct format.")));
		}

		if (feat_dim <= 0)
		{
			NDB_SPI_SESSION_END(eval_spi_session);
			if (model != NULL)
				rf_free_deserialized_model(model);
			nfree(gpu_payload);
			nfree(gpu_metrics);
			nfree(tbl_str);
			nfree(feat_str);
			nfree(targ_str);
			ndb_spi_stringinfo_free(eval_spi_session, &query);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_random_forest_by_model_id: invalid feature dimension %d",
							feat_dim)));
		}

		/* For GPU models in GPU mode, use batch evaluation for better performance and correct metrics */
#if defined(NDB_GPU_CUDA) || defined(NDB_GPU_METAL)
		if (use_gpu_predict && is_gpu_model && gpu_payload != NULL && nvec > 0)
		{
			float	   *features_array = NULL;
			int		   *labels_array = NULL;
			char	   *gpu_err = NULL;
			int			batch_rc;
			int			valid_samples = 0;
			int			j;

			/* Allocate arrays for batch evaluation */
			nalloc(features_array, float, (size_t) nvec * (size_t) feat_dim);
			nalloc(labels_array, int, nvec);

			/* Load features and labels from SPI results */
			for (i = 0; i < nvec; i++)
			{
				HeapTuple	tuple;
				TupleDesc	tupdesc;
				Datum		feat_datum;
				Datum		targ_datum;
				bool		feat_null;
				bool		targ_null;
				int			actual_dim;

				if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL ||
					i >= SPI_processed || SPI_tuptable->vals[i] == NULL)
					continue;

				tuple = SPI_tuptable->vals[i];
				tupdesc = SPI_tuptable->tupdesc;
				if (tupdesc == NULL)
					continue;

				feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
				if (tupdesc->natts < 2)
					continue;
				targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

				if (feat_null || targ_null)
					continue;

				/* Extract features and determine dimension */
				if (feat_is_array)
				{
					ArrayType *arr = DatumGetArrayTypeP(feat_datum);
					if (arr == NULL || ARR_NDIM(arr) != 1)
						continue;
					actual_dim = ARR_DIMS(arr)[0];
				}
				else
				{
					Vector *vec = DatumGetVector(feat_datum);
					if (vec == NULL)
						continue;
					actual_dim = vec->dim;
				}

				/* Validate feature dimension matches model */
				if (actual_dim != feat_dim)
					continue;

				/* Extract features to float array */
				if (feat_is_array)
				{
					ArrayType *arr = DatumGetArrayTypeP(feat_datum);
					if (feat_type_oid == FLOAT8ARRAYOID)
					{
						float8	   *data = (float8 *) ARR_DATA_PTR(arr);
						for (j = 0; j < feat_dim; j++)
							features_array[valid_samples * feat_dim + j] = (float) data[j];
					}
					else
					{
						float4	   *data = (float4 *) ARR_DATA_PTR(arr);
						memcpy(&features_array[valid_samples * feat_dim], data, sizeof(float) * feat_dim);
					}
				}
				else
				{
					Vector *vec = DatumGetVector(feat_datum);
					memcpy(&features_array[valid_samples * feat_dim], vec->data, sizeof(float) * feat_dim);
				}

				/* Extract label */
				labels_array[valid_samples] = (int) rint(DatumGetFloat8(targ_datum));
				valid_samples++;
			}

			/* Call GPU batch evaluation if we have valid samples */
			if (valid_samples > 0)
			{
#ifdef NDB_GPU_CUDA
				batch_rc = ndb_cuda_rf_evaluate_batch(gpu_payload,
													  features_array,
													  labels_array,
													  valid_samples,
													  feat_dim,
													  &accuracy,
													  &precision,
													  &recall,
													  &f1_score,
													  &gpu_err);
#else
				/* Metal backend: use CPU fallback for evaluation */
				batch_rc = -1;
				/* Note: gpu_err is passed as &gpu_err to CUDA functions, so we set it directly */
				gpu_err = pstrdup("Batch evaluation not yet implemented for Metal backend");
#endif

				if (batch_rc == 0)
				{
					/* Batch evaluation succeeded - metrics are already set */
					/* Verify metrics were actually set (should be non-zero if evaluation succeeded) */
					if (accuracy == 0.0 && precision == 0.0 && recall == 0.0 && f1_score == 0.0)
					{
						/* Metrics weren't set - batch evaluation may have failed silently */
						ereport(WARNING,
								(errmsg("evaluate_random_forest_by_model_id: batch evaluation reported success but metrics are zero - falling back to row-by-row")));
						/* Fall through to row-by-row evaluation */
						nfree(features_array);
						nfree(labels_array);
						if (gpu_err)
							nfree(gpu_err);
						/* Reset metrics for row-by-row calculation */
						accuracy = 0.0;
						precision = 0.0;
						recall = 0.0;
						f1_score = 0.0;
						tp = 0;
						tn = 0;
						fp = 0;
						fn = 0;
						processed_count = 0;
					}
					else
					{
						/* Metrics were set correctly - use them */
						processed_count = valid_samples;
						nfree(features_array);
						nfree(labels_array);
						if (gpu_err)
							nfree(gpu_err);
						/* Skip the row-by-row loop */
						/* Fall through to metrics calculation */
					}
				}
				else
				{
					/* Batch evaluation failed - fall back to row-by-row */
					/* Log error for debugging */
					ereport(DEBUG2,
							(errmsg("evaluate_random_forest_by_model_id: GPU batch evaluation failed: %s",
									gpu_err ? gpu_err : "unknown error")));
					if (gpu_err)
						nfree(gpu_err);
					nfree(features_array);
					nfree(labels_array);
					/* Reset metrics for row-by-row calculation */
					accuracy = 0.0;
					precision = 0.0;
					recall = 0.0;
					f1_score = 0.0;
					tp = 0;
					tn = 0;
					fp = 0;
					fn = 0;
					processed_count = 0;
				}
			}
			else
			{
				/* No valid samples - free arrays and fall back to row-by-row */
				nfree(features_array);
				nfree(labels_array);
			}
		}
#endif

		/* Unified evaluation loop - prediction based on compute mode */
		for (i = 0; i < nvec; i++)
		{
			HeapTuple	tuple;
			TupleDesc	tupdesc;
			Datum		feat_datum;
			Datum		targ_datum;
			bool		feat_null;
			bool		targ_null;
			int			y_true;
			int			y_pred = -1;
			int			j;
			int			actual_dim;
			float	   *feat_row = NULL;

			if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL ||
				i >= SPI_processed || SPI_tuptable->vals[i] == NULL)
				continue;

			tuple = SPI_tuptable->vals[i];
			tupdesc = SPI_tuptable->tupdesc;
			if (tupdesc == NULL)
				continue;

			feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
			if (tupdesc->natts < 2)
				continue;
			targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

			if (feat_null || targ_null)
				continue;

			/* Handle both integer and float label types */
			{
				Oid			label_type = SPI_gettypeid(tupdesc, 2);
				
				if (label_type == INT4OID || label_type == INT2OID || label_type == INT8OID)
					y_true = DatumGetInt32(targ_datum);
				else
					y_true = (int) rint(DatumGetFloat8(targ_datum));
			}


			/* Extract features and determine dimension */
			if (feat_is_array)
			{
				ArrayType *arr = DatumGetArrayTypeP(feat_datum);
				if (arr == NULL || ARR_NDIM(arr) != 1)
					continue;
				actual_dim = ARR_DIMS(arr)[0];
			}
			else
			{
				Vector *vec = DatumGetVector(feat_datum);
				if (vec == NULL)
					continue;
				actual_dim = vec->dim;
			}

			/* Validate feature dimension matches model */
			if (actual_dim != feat_dim)
				continue;

			/* Extract features to float array for prediction */
			nalloc(feat_row, float, feat_dim);
			if (feat_is_array)
			{
				ArrayType *arr = DatumGetArrayTypeP(feat_datum);
				if (feat_type_oid == FLOAT8ARRAYOID)
				{
					float8	   *data = (float8 *) ARR_DATA_PTR(arr);
					for (j = 0; j < feat_dim; j++)
						feat_row[j] = (float) data[j];
				}
				else
				{
					float4	   *data = (float4 *) ARR_DATA_PTR(arr);
					memcpy(feat_row, data, sizeof(float) * feat_dim);
				}
			}
			else
			{
				Vector *vec = DatumGetVector(feat_datum);
				memcpy(feat_row, vec->data, sizeof(float) * feat_dim);
			}

			/* Call appropriate predict function based on compute mode */
			if (use_gpu_predict)
			{
				/* GPU predict path - prediction based on compute mode */
#if defined(NDB_GPU_CUDA) || defined(NDB_GPU_METAL)
				int			predict_rc;
				char	   *gpu_err = NULL;

				predict_rc = ndb_gpu_rf_predict(gpu_payload,
												 feat_row,
												 feat_dim,
												 &y_pred,
												 &gpu_err);
				if (predict_rc != 0)
				{
					/* GPU predict failed - check compute mode */
					/* Log error for debugging */
					ereport(DEBUG2,
							(errmsg("evaluate_random_forest_by_model_id: GPU prediction failed for row %d: %s",
									i, gpu_err ? gpu_err : "unknown error")));
					if (NDB_REQUIRE_GPU())
					{
						/* Strict GPU mode: error out */
						if (gpu_err)
							nfree(gpu_err);
						nfree(feat_row);
						NDB_SPI_SESSION_END(eval_spi_session);
						if (model != NULL)
							rf_free_deserialized_model(model);
						nfree(gpu_payload);
						nfree(gpu_metrics);
						nfree(tbl_str);
						nfree(feat_str);
						nfree(targ_str);
						ndb_spi_stringinfo_free(eval_spi_session, &query);
						ereport(ERROR,
								(errcode(ERRCODE_INTERNAL_ERROR),
								 errmsg("neurondb: evaluate_random_forest_by_model_id: GPU prediction failed in GPU mode"),
								 errdetail("GPU prediction failed for row %d: %s", i, gpu_err ? gpu_err : "unknown error"),
								 errhint("GPU mode requires GPU prediction to succeed. Check GPU availability and model compatibility.")));
					}
					else
					{
						/* AUTO mode: fall back to CPU if available */
						if (gpu_err)
							nfree(gpu_err);
						if (model != NULL)
						{
							/* Use CPU model for prediction */
							double		prediction = 0.0;

							/* Call CPU predict function */
							if (model->tree_count > 0 && model->trees != NULL)
							{
								double		vote_histogram[256] = {0};
								double		vote_total_weight = 0.0;
								int			t;

								for (t = 0; t < model->tree_count && t < 256; t++)
								{
									const GTree *tree = model->trees[t];
									double		tree_result;
									double		vote_weight = 1.0;

									tree_result = rf_tree_predict_row(tree, feat_row, feat_dim);

									if (model->tree_oob_accuracy != NULL && t < model->tree_count)
									{
										vote_weight = model->tree_oob_accuracy[t];
										if (vote_weight <= 0.0)
											vote_weight = 1.0;
									}

									{
										int			cls = (int) rint(tree_result);

										if (cls >= 0 && cls < 256)
										{
											vote_histogram[cls] += vote_weight;
											vote_total_weight += vote_weight;
										}
									}
								}

								if (vote_total_weight > 0.0)
								{
									int			best_idx = -1;
									double		best_weight = -1.0;

									for (j = 0; j < 256; j++)
									{
										if (vote_histogram[j] > best_weight)
										{
											best_idx = j;
											best_weight = vote_histogram[j];
										}
									}

									if (best_idx >= 0)
										prediction = (double) best_idx;
								}
							}
							else if (model->tree != NULL)
							{
								prediction = rf_tree_predict_row(model->tree, feat_row, feat_dim);
							}
							else
							{
								prediction = model->majority_value;
							}

							y_pred = (int) rint(prediction);
						}
						else
						{
							/* No CPU model available - use majority class from GPU model header as fallback */
							if (gpu_err)
								nfree(gpu_err);
							/* Get majority class from GPU header if available */
							if (gpu_payload != NULL && VARSIZE(gpu_payload) - VARHDRSZ >= sizeof(NdbCudaRfModelHeader))
							{
								const NdbCudaRfModelHeader *hdr = (const NdbCudaRfModelHeader *) VARDATA(gpu_payload);
								if (hdr->majority_class >= 0)
								{
									y_pred = hdr->majority_class;
								}
								else
								{
									/* Default to 0 if no majority class */
									y_pred = 0;
								}
							}
							else
							{
								/* Default to 0 if no GPU payload */
								y_pred = 0;
							}
						}
					}
				}
				if (gpu_err)
					nfree(gpu_err);
#endif
			}
			else
			{
				/* CPU predict path - prediction based on compute mode */
				double		prediction = 0.0;

				if (model == NULL)
				{
					nfree(feat_row);
					continue;
				}

				/* Use CPU model for prediction */

				if (model->tree_count > 0 && model->trees != NULL)
				{
					double		vote_histogram[256] = {0};
					double		vote_total_weight = 0.0;
					int			t;

					for (t = 0; t < model->tree_count && t < 256; t++)
					{
						const GTree *tree = model->trees[t];
						double		tree_result;
						double		vote_weight = 1.0;

						tree_result = rf_tree_predict_row(tree, feat_row, feat_dim);

						if (model->tree_oob_accuracy != NULL && t < model->tree_count)
						{
							vote_weight = model->tree_oob_accuracy[t];
							if (vote_weight <= 0.0)
								vote_weight = 1.0;
						}

						{
							int			cls = (int) rint(tree_result);

							if (cls >= 0 && cls < 256)
							{
								vote_histogram[cls] += vote_weight;
								vote_total_weight += vote_weight;
							}
						}

					if (vote_total_weight > 0.0)
					{
						int			best_idx = -1;
						double		best_weight = -1.0;

						for (j = 0; j < 256; j++)
						{
							if (vote_histogram[j] > best_weight)
							{
								best_idx = j;
								best_weight = vote_histogram[j];
							}
						}

						if (best_idx >= 0)
							prediction = (double) best_idx;
					}
					}
				}
				else if (model->tree != NULL)
				{
					prediction = rf_tree_predict_row(model->tree, feat_row, feat_dim);
				}
				else
				{
					prediction = model->majority_value;
				}

				y_pred = (int) rint(prediction);
			}

			/* Clamp y_pred to valid binary classification range [0, 1] */
			if (y_pred < 0)
				y_pred = 0;
			else if (y_pred > 1)
				y_pred = 1;


			/* Compute confusion matrix (same for both CPU and GPU) */
			if (y_true == 1)
			{
				if (y_pred == 1)
					tp++;
				else
					fn++;
			}
			else
			{
				if (y_pred == 1)
					fp++;
				else
					tn++;
			}

			processed_count++;
			nfree(feat_row);
		}

		/* Calculate metrics from confusion matrix (same for both CPU and GPU) */
		/* Note: If GPU batch evaluation succeeded, it jumps here with metrics already set */
		/* Check if metrics need to be calculated from confusion matrix */
		if (processed_count > 0 && (tp + tn + fp + fn) > 0)
		{
			/* Row-by-row evaluation completed - calculate metrics from confusion matrix */
			int			total = tp + tn + fp + fn;
			
			accuracy = total > 0 ? (double) (tp + tn) / total : 0.0;
			
			/* Precision: tp / (tp + fp) - undefined if no positive predictions */
			if (tp + fp > 0)
				precision = (double) tp / (tp + fp);
			else
				precision = 0.0;
			
			/* Recall: tp / (tp + fn) - undefined if no positive labels */
			if (tp + fn > 0)
				recall = (double) tp / (tp + fn);
			else
				recall = 0.0;
			
			/* F1 score: harmonic mean of precision and recall */
			if (precision + recall > 0.0)
				f1_score = 2.0 * precision * recall / (precision + recall);
			else
				f1_score = 0.0;
		}
		else if (processed_count > 0 && (tp + tn + fp + fn) == 0)
		{
			/* Batch evaluation succeeded - metrics are already set, no need to recalculate */
			/* This case means GPU batch evaluation set the metrics directly */
			/* Verify metrics were actually set (should be non-zero if evaluation succeeded) */
			if (accuracy == 0.0 && precision == 0.0 && recall == 0.0 && f1_score == 0.0)
			{
				/* Metrics weren't set - this shouldn't happen, but handle gracefully */
				ereport(WARNING,
						(errmsg("evaluate_random_forest_by_model_id: batch evaluation reported success but metrics are zero")));
			}
		}
		else if (processed_count > 0)
		{
			int			total = tp + tn + fp + fn;
			
			accuracy = total > 0 ? (double) (tp + tn) / total : 0.0;
			
			/* Precision: tp / (tp + fp) - undefined if no positive predictions */
			/* For binary classification, if no positive predictions, precision is undefined (0.0) */
			if (tp + fp > 0)
				precision = (double) tp / (tp + fp);
			else
				precision = 0.0;  /* No positive predictions - precision undefined */
			
			/* Recall: tp / (tp + fn) - undefined if no positive labels */
			/* For binary classification, if no positive labels, recall is undefined (0.0) */
			if (tp + fn > 0)
				recall = (double) tp / (tp + fn);
			else
				recall = 0.0;  /* No positive labels - recall undefined */
			
			/* F1 score: harmonic mean of precision and recall */
			if (precision + recall > 0.0)
				f1_score = 2.0 * precision * recall / (precision + recall);
			else
				f1_score = 0.0;
		}
		else
		{
			accuracy = 0.0;
			precision = 0.0;
			recall = 0.0;
			f1_score = 0.0;
		}

		/* Cleanup */
		if (model != NULL)
			rf_free_deserialized_model(model);
		if (gpu_payload != NULL)
			nfree(gpu_payload);
		if (gpu_metrics != NULL)
			nfree(gpu_metrics);
	}

	/* Build JSONB result */
	ndb_spi_stringinfo_free(eval_spi_session, &query);
	NDB_SPI_SESSION_END(eval_spi_session);

	/* Switch to old context and build JSONB directly using JSONB API */
	MemoryContextSwitchTo(oldcontext);
	{
		JsonbParseState *state = NULL;
		JsonbValue	jkey;
		JsonbValue	jval;
		JsonbValue *final_value = NULL;
		Numeric		accuracy_num,
					precision_num,
					recall_num,
					f1_score_num,
					n_samples_num;

		PG_TRY();
		{
			(void) pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);

			jkey.type = jbvString;
			jkey.val.string.val = "accuracy";
			jkey.val.string.len = strlen("accuracy");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			accuracy_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(accuracy)));
			jval.type = jbvNumeric;
			jval.val.numeric = accuracy_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			jkey.val.string.val = "precision";
			jkey.val.string.len = strlen("precision");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			precision_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(precision)));
			jval.type = jbvNumeric;
			jval.val.numeric = precision_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			jkey.val.string.val = "recall";
			jkey.val.string.len = strlen("recall");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			recall_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(recall)));
			jval.type = jbvNumeric;
			jval.val.numeric = recall_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			jkey.val.string.val = "f1_score";
			jkey.val.string.len = strlen("f1_score");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			f1_score_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(f1_score)));
			jval.type = jbvNumeric;
			jval.val.numeric = f1_score_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			jkey.val.string.val = "n_samples";
			jkey.val.string.len = strlen("n_samples");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			n_samples_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(tp + tn + fp + fn)));
			jval.type = jbvNumeric;
			jval.val.numeric = n_samples_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);

			if (final_value == NULL)
			{
				elog(ERROR, "neurondb: evaluate_random_forest_by_model_id: pushJsonbValue(WJB_END_OBJECT) returned NULL");
			}

			result_jsonb = JsonbValueToJsonb(final_value);
		}
		PG_CATCH();
		{
			ErrorData  *edata = CopyErrorData();

			elog(ERROR, "neurondb: evaluate_random_forest_by_model_id: JSONB construction failed: %s", edata->message);
			FlushErrorState();
			result_jsonb = NULL;
		}
		PG_END_TRY();
	}

	if (result_jsonb == NULL)
	{
		nfree(tbl_str);
		nfree(feat_str);
		nfree(targ_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: evaluate_random_forest_by_model_id: failed to create JSONB result")));
	}

	nfree(tbl_str);
	nfree(feat_str);
	nfree(targ_str);
	PG_RETURN_JSONB_P(result_jsonb);
}

/* Old GPU evaluation kernel code removed - replaced with unified evaluation pattern above */

/*
 * rf_read_int_array
 *    Read an array of integers from a StringInfo buffer.
 */
static int *
rf_read_int_array(StringInfo buf, int count)
{
	int		   *arr = NULL;
	int			i;

	if (count <= 0)
		return NULL;

	nalloc(arr, int, count);
	NDB_CHECK_ALLOC(arr, "rf_read_int_array");

	for (i = 0; i < count; i++)
		arr[i] = pq_getmsgint(buf, 4);

	return arr;
}

/*
 * rf_read_double_array
 *    Read an array of doubles from a StringInfo buffer.
 */
static double *
rf_read_double_array(StringInfo buf, int count)
{
	double	   *arr = NULL;
	int			i;

	if (count <= 0)
		return NULL;

	nalloc(arr, double, count);
	NDB_CHECK_ALLOC(arr, "rf_read_double_array");

	for (i = 0; i < count; i++)
		arr[i] = pq_getmsgfloat8(buf);

	return arr;
}

/*
 * rf_write_int_array
 *    Write an array of integers to a StringInfo buffer.
 *    If arr is NULL, writes zeros to maintain buffer alignment.
 */
static void
rf_write_int_array(StringInfo buf, const int *arr, int count)
{
	int			i;

	if (count <= 0)
		return;

	if (arr == NULL)
	{
		/* Write zeros to maintain buffer alignment */
		for (i = 0; i < count; i++)
			pq_sendint32(buf, 0);
	}
	else
	{
		for (i = 0; i < count; i++)
			pq_sendint32(buf, arr[i]);
	}
}

/*
 * rf_write_double_array
 *    Write an array of doubles to a StringInfo buffer.
 *    If arr is NULL, writes zeros to maintain buffer alignment.
 */
static void
rf_write_double_array(StringInfo buf, const double *arr, int count)
{
	int			i;

	if (count <= 0)
		return;

	if (arr == NULL)
	{
		/* Write zeros to maintain buffer alignment */
		for (i = 0; i < count; i++)
			pq_sendfloat8(buf, 0.0);
	}
	else
	{
		for (i = 0; i < count; i++)
			pq_sendfloat8(buf, arr[i]);
	}
}

static void
rf_serialize_tree(StringInfo buf, const GTree *tree)
{
	const GTreeNode *nodes = NULL;
	int			i;

	if (tree == NULL)
	{
		pq_sendbyte(buf, 0);
		return;
	}

	pq_sendbyte(buf, 1);
	pq_sendint32(buf, tree->root);
	pq_sendint32(buf, tree->max_depth);
	pq_sendint32(buf, tree->count);

	nodes = gtree_nodes(tree);
	for (i = 0; i < tree->count; i++)
	{
		pq_sendint32(buf, nodes[i].feature_idx);
		pq_sendfloat8(buf, nodes[i].threshold);
		pq_sendint32(buf, nodes[i].left);
		pq_sendint32(buf, nodes[i].right);
		pq_sendbyte(buf, nodes[i].is_leaf ? 1 : 0);
		pq_sendfloat8(buf, nodes[i].value);
	}
}

static GTree *
rf_deserialize_tree(StringInfo buf)
{
	int			flag = pq_getmsgbyte(buf);
	int			count;
	int			i;
	int			root;
	int			max_depth;
	GTree *tree = NULL;
	MemoryContext oldctx = NULL;
	MemoryContext tree_create_ctx = NULL;

	if (flag == 0)
		return NULL;

	root = pq_getmsgint(buf, 4);
	max_depth = pq_getmsgint(buf, 4);
	count = pq_getmsgint(buf, 4);

	/* Create tree in TopMemoryContext so it survives transaction end */
	tree_create_ctx = MemoryContextSwitchTo(TopMemoryContext);
	tree = gtree_create("rf_model_tree", Max(count, 4));
	MemoryContextSwitchTo(tree_create_ctx);
	if (tree == NULL || tree->ctx == NULL)
		return NULL;
	{
		MemoryContext ctx = tree->ctx;
		oldctx = MemoryContextSwitchTo(ctx);
	}

	if (tree->nodes != NULL)
		nfree(tree->nodes);

	if (count > 0)
	{
		GTreeNode *nodes_tmp = NULL;
		nalloc(nodes_tmp, GTreeNode, count);
		NDB_CHECK_ALLOC(nodes_tmp, "nodes_tmp");
		tree->nodes = nodes_tmp;
		for (i = 0; i < count; i++)
		{
			tree->nodes[i].feature_idx = pq_getmsgint(buf, 4);
			tree->nodes[i].threshold = pq_getmsgfloat8(buf);
			tree->nodes[i].left = pq_getmsgint(buf, 4);
			tree->nodes[i].right = pq_getmsgint(buf, 4);
			tree->nodes[i].is_leaf = (pq_getmsgbyte(buf) != 0);
			tree->nodes[i].value = pq_getmsgfloat8(buf);
		}
	}

	tree->root = root;
	tree->max_depth = max_depth;
	MemoryContextSwitchTo(oldctx);

	return tree;
}

/*
 * rf_model_serialize
 *    Serialize an RFModel to a bytea for storage.
 */
static bytea *
rf_model_serialize(const RFModel *model, uint8 training_backend)
{
	StringInfoData buf;
	int			i;

	if (model == NULL)
		return NULL;

	/* Validate model before serialization */
	if (model->n_features <= 0 || model->n_features > 10000)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("rf_model_serialize: invalid n_features %d", model->n_features)));
	}

	if (model->n_classes <= 0 || model->n_classes > 1000)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("rf_model_serialize: invalid n_classes %d", model->n_classes)));
	}

	if (training_backend > 1)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("rf_model_serialize: invalid training_backend %d (must be 0 or 1)", training_backend)));
	}

	pq_begintypsend(&buf);

	/* Write training_backend first (0=CPU, 1=GPU) */
	pq_sendbyte(&buf, training_backend);

	/* Write header */
	pq_sendint32(&buf, model->model_id);
	pq_sendint32(&buf, model->n_features);
	pq_sendint32(&buf, model->n_samples);
	pq_sendint32(&buf, model->feature_limit);

	/* Write arrays */
	rf_write_int_array(&buf, model->class_counts, model->feature_limit);
	rf_write_double_array(&buf, model->feature_means, model->feature_limit);
	rf_write_double_array(&buf, model->feature_variances, model->feature_limit);
	rf_write_double_array(&buf, model->feature_importance, model->feature_limit);
	rf_write_double_array(&buf, model->left_branch_means, model->feature_limit);
	rf_write_double_array(&buf, model->right_branch_means, model->feature_limit);

	/* Write tree_count */
	pq_sendint32(&buf, model->tree_count);

	/* Write trees */
	if (model->tree_count > 0 && model->trees != NULL)
	{
		for (i = 0; i < model->tree_count; i++)
			rf_serialize_tree(&buf, model->trees[i]);
	}

	/* Write main tree */
	rf_serialize_tree(&buf, model->tree);

	/* Write tree arrays */
	rf_write_double_array(&buf, model->tree_majority, model->tree_count);
	rf_write_double_array(&buf, model->tree_majority_fraction, model->tree_count);
	rf_write_double_array(&buf, model->tree_second, model->tree_count);
	rf_write_double_array(&buf, model->tree_second_fraction, model->tree_count);
	rf_write_double_array(&buf, model->tree_oob_accuracy, model->tree_count);

	/* Note: Scalar values (majority_value, gini_impurity, etc.) are not serialized
	 * as they are not read by rf_model_deserialize. These values are set when
	 * the model is stored via rf_store_model().
	 */

	return pq_endtypsend(&buf);
}

static RFModel *
rf_model_deserialize(const bytea * data, uint8 * training_backend_out)
{
	RFModel *model = NULL;
	StringInfoData buf;
	uint8		training_backend = 0;

	if (data == NULL)
		return NULL;

	/* Initialize StringInfo from bytea */
	memset(&buf, 0, sizeof(StringInfoData));
	buf.data = VARDATA(data);
	buf.len = VARSIZE(data) - VARHDRSZ;
	buf.maxlen = buf.len;
	buf.cursor = 0;

	PG_TRY();
	{
		int			model_id;
		int			n_features;
		int			n_samples;
		int			feature_limit;
		int			i;

		/* Read training_backend first */
		training_backend = (uint8) pq_getmsgbyte(&buf);

		model_id = pq_getmsgint(&buf, 4);
		n_features = pq_getmsgint(&buf, 4);
		n_samples = pq_getmsgint(&buf, 4);
		feature_limit = pq_getmsgint(&buf, 4);

		nalloc(model, RFModel, 1);
		NDB_CHECK_ALLOC(model, "model");
		memset(model, 0, sizeof(RFModel));

		model->model_id = model_id;
		model->n_features = n_features;
		model->n_samples = n_samples;
		model->feature_limit = feature_limit;

		/* Read arrays */
		model->class_counts = rf_read_int_array(&buf, feature_limit);
		model->feature_means = rf_read_double_array(&buf, feature_limit);
		model->feature_variances = rf_read_double_array(&buf, feature_limit);
		model->feature_importance = rf_read_double_array(&buf, feature_limit);

		/* Only continue if model is still valid */
		if (model != NULL)
		{
			/* Continue with deserialization */
			model->left_branch_means = rf_read_double_array(&buf, model->feature_limit);

			if (model != NULL && model->left_branch_means == NULL && model->feature_limit > 0)
			{
							elog(ERROR, "rf_model_deserialize: failed to read left_branch_means (required field, cursor=%d/%d, feature_limit=%d)",
								 buf.cursor, buf.len, model->feature_limit);
							if (model->class_counts)
								nfree(model->class_counts);
							if (model->feature_means)
								nfree(model->feature_means);
							if (model->feature_variances)
								nfree(model->feature_variances);
							if (model->feature_importance)
								nfree(model->feature_importance);
							nfree(model);
							model = NULL;
			}
			else if (model != NULL)
			{
				model->right_branch_means = rf_read_double_array(&buf, model->feature_limit);

				if (model != NULL && model->right_branch_means == NULL && model->feature_limit > 0)
				{
					elog(ERROR, "rf_model_deserialize: failed to read right_branch_means (required field, cursor=%d/%d, feature_limit=%d)",
						 buf.cursor, buf.len, model->feature_limit);
					if (model->class_counts)
						nfree(model->class_counts);
					if (model->feature_means)
						nfree(model->feature_means);
					if (model->feature_variances)
						nfree(model->feature_variances);
					if (model->feature_importance)
						nfree(model->feature_importance);
					if (model->left_branch_means)
						nfree(model->left_branch_means);
					nfree(model);
					model = NULL;
				}
				else if (model != NULL)
				{
					model->tree_count = pq_getmsgint(&buf, 4);
					if (model->tree_count < 0 || model->tree_count > 10000)
					{
						elog(WARNING, "rf_model_deserialize: invalid tree_count %d", model->tree_count);
						if (model->class_counts)
							nfree(model->class_counts);
						if (model->feature_means)
							nfree(model->feature_means);
						if (model->feature_variances)
							nfree(model->feature_variances);
						if (model->feature_importance)
							nfree(model->feature_importance);
						if (model->left_branch_means)
							nfree(model->left_branch_means);
						if (model->right_branch_means)
							nfree(model->right_branch_means);
						nfree(model);
						model = NULL;
					}
					else if (model->tree_count > 0)
					{
						GTree **trees_tmp = NULL;
						int			j;
						nalloc(trees_tmp, GTree *, model->tree_count);
						NDB_CHECK_ALLOC(trees_tmp, "trees_tmp");
						/* Initialize all tree pointers to NULL */
						for (j = 0; j < model->tree_count; j++)
							trees_tmp[j] = NULL;
						model->trees = trees_tmp;
						if (model->trees == NULL)
						{
							elog(WARNING, "rf_model_deserialize: palloc failed for trees array");
							if (model->class_counts)
								nfree(model->class_counts);
							if (model->feature_means)
								nfree(model->feature_means);
							if (model->feature_variances)
								nfree(model->feature_variances);
							if (model->feature_importance)
								nfree(model->feature_importance);
							if (model->left_branch_means)
								nfree(model->left_branch_means);
							if (model->right_branch_means)
								nfree(model->right_branch_means);
							nfree(model);
							model = NULL;
						}
						else
						{
							for (i = 0; i < model->tree_count && model != NULL; i++)
							{
								model->trees[i] = rf_deserialize_tree(&buf);
								if (model->trees[i] == NULL)
								{
									elog(WARNING, "rf_model_deserialize: failed to deserialize tree %d", i);

									/*
									 * Free already deserialized
									 * trees
									 */
									for (i--; i >= 0; i--)
									{
										if (model->trees[i] != NULL)
											gtree_free(model->trees[i]);
									}
									nfree(model->trees);
									if (model->class_counts)
										nfree(model->class_counts);
									if (model->feature_means)
										nfree(model->feature_means);
									if (model->feature_variances)
										nfree(model->feature_variances);
									if (model->feature_importance)
										nfree(model->feature_importance);
									if (model->left_branch_means)
										nfree(model->left_branch_means);
									if (model->right_branch_means)
										nfree(model->right_branch_means);
									nfree(model);
									model = NULL;
								}
							}
						}
					}
					else
					{
						model->trees = NULL;
					}

					if (model != NULL)
					{
						model->tree = rf_deserialize_tree(&buf);
						if (model->tree == NULL)
						{
							elog(WARNING, "rf_model_deserialize: failed to deserialize main tree");
							if (model->trees != NULL)
							{
								for (i = 0; i < model->tree_count; i++)
								{
									if (model->trees[i] != NULL)
										gtree_free(model->trees[i]);
								}
								nfree(model->trees);
							}
							if (model->class_counts)
								nfree(model->class_counts);
							if (model->feature_means)
								nfree(model->feature_means);
							if (model->feature_variances)
								nfree(model->feature_variances);
							if (model->feature_importance)
								nfree(model->feature_importance);
							if (model->left_branch_means)
								nfree(model->left_branch_means);
							if (model->right_branch_means)
								nfree(model->right_branch_means);
							nfree(model);
							model = NULL;
						}
						else
						{
							model->tree_majority = rf_read_double_array(&buf, model->tree_count);
							model->tree_majority_fraction = rf_read_double_array(&buf, model->tree_count);
							model->tree_second = rf_read_double_array(&buf, model->tree_count);
							model->tree_second_fraction = rf_read_double_array(&buf, model->tree_count);
							model->tree_oob_accuracy = rf_read_double_array(&buf, model->tree_count);

							/* Final validation */
							if (buf.cursor > buf.len)
							{
								elog(WARNING, "rf_model_deserialize: buffer overrun (cursor=%d, len=%d)", buf.cursor, buf.len);
								rf_free_deserialized_model(model);
								model = NULL;
							}
						}
					}
				}
			}
		}
	}
	PG_CATCH();
	{
		elog(WARNING, "rf_model_deserialize: exception during deserialization");
		if (model != NULL)
		{
			rf_free_deserialized_model(model);
			model = NULL;
		}
		FlushErrorState();
	}
	PG_END_TRY();

	/* Return training_backend if output parameter provided */
	if (training_backend_out != NULL)
		*training_backend_out = training_backend;

	return model;
}

static void
rf_free_deserialized_model(RFModel *model)
{
	/* Don't free anything - everything is in TopMemoryContext and will be 
	 * cleaned up by PostgreSQL. The arrays are deep-copied by rf_store_model
	 * but freeing them here causes crashes. Trees are shallow-copied so 
	 * definitely can't be freed. */
	(void) model;
}

static void
rf_dataset_init(RFDataset * dataset)
{
	if (dataset == NULL)
		return;
	dataset->features = NULL;
	dataset->labels = NULL;
	dataset->n_samples = 0;
	dataset->feature_dim = 0;
}

static void
rf_dataset_free(RFDataset * dataset)
{
	if (dataset == NULL)
		return;
	if (dataset->features != NULL)
		nfree(dataset->features);
	if (dataset->labels != NULL)
		nfree(dataset->labels);
	rf_dataset_init(dataset);
}

static void
rf_dataset_load(const char *quoted_tbl,
				const char *quoted_feat,
				const char *quoted_label,
				RFDataset * dataset,
				StringInfo query)
{
	int			feature_dim = 0;
	int			n_samples = 0;
	int			i;

	if (dataset == NULL || query == NULL)
		elog(ERROR, "random_forest: invalid dataset load arguments");

	rf_dataset_free(dataset);

	/* Try to get feature dimension - handle both vector and array types */
	/* Use safe free/reinit to handle potential memory context changes */
	nfree(query->data);
	initStringInfo(query);
	appendStringInfo(query,
					 "SELECT %s FROM %s WHERE %s IS NOT NULL LIMIT 1",
					 quoted_feat,
					 quoted_tbl,
					 quoted_feat);

	if (ndb_spi_execute_safe(query->data, true, 1) == SPI_OK_SELECT
		&& SPI_processed > 0)
	{
		HeapTuple	tup;
		TupleDesc	tupdesc;

		Datum		feat_datum;
		bool		feat_null;
		Oid			feat_type;

		NDB_CHECK_SPI_TUPTABLE();
		/* Safe access for complex types - validate before access */
		if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL ||
			SPI_tuptable->vals[0] == NULL || SPI_tuptable->tupdesc == NULL)
		{
			/* Cannot determine feature dimension */
			feat_null = true;
		}
		else
		{
			tup = SPI_tuptable->vals[0];
			tupdesc = SPI_tuptable->tupdesc;
			feat_datum = SPI_getbinval(tup, tupdesc, 1, &feat_null);
		}
		if (!feat_null)
		{
			feat_type = SPI_gettypeid(tupdesc, 1);

			/* Check if it's an array type (double precision[]) */
			if (feat_type == FLOAT8ARRAYOID)
			{
				ArrayType  *arr = DatumGetArrayTypeP(feat_datum);

				if (arr != NULL)
					feature_dim = ArrayGetNItems(ARR_NDIM(arr), ARR_DIMS(arr));
			}
			/* Check if it's a float4 array */
			else if (feat_type == FLOAT4ARRAYOID)
			{
				ArrayType  *arr = DatumGetArrayTypeP(feat_datum);

				if (arr != NULL)
					feature_dim = ArrayGetNItems(ARR_NDIM(arr), ARR_DIMS(arr));
			}
			/* Try Vector type (check by attempting to cast) */
			else
			{
				Vector	   *vec = DatumGetVector(feat_datum);

				if (vec != NULL && vec->dim > 0)
					feature_dim = vec->dim;
			}
		}
	}

	dataset->feature_dim = feature_dim;

	/* Use safe free/reinit to handle potential memory context changes */
	nfree(query->data);
	initStringInfo(query);
	appendStringInfo(query,
					 "SELECT %s, (%s)::float8 FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
					 quoted_feat,
					 quoted_label,
					 quoted_tbl,
					 quoted_feat,
					 quoted_label);

	{
		int			ret = ndb_spi_execute_safe(query->data, true, 0);

		if (ret != SPI_OK_SELECT)
		{
			NDB_CHECK_SPI_TUPTABLE();
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("random_forest: failed to fetch training data"),
					 errdetail("SPI execution returned code %d (expected %d), query: %s", ret, SPI_OK_SELECT, query->data),
					 errhint("Verify the table '%s' exists and contains valid feature and label columns.", quoted_tbl)));
		}
	}

	n_samples = SPI_processed;
	dataset->n_samples = n_samples;

	if (n_samples <= 0)
		return;

	/* Defensive check: prevent excessive memory allocation */
	if (feature_dim > 100000)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("random_forest: feature dimension too large: feature_dim=%d",
						feature_dim),
				 errhint("Consider reducing feature dimension")));
	}

	/* Check for potential integer overflow in allocation size */
	if ((size_t) n_samples * (size_t) feature_dim > (SIZE_MAX / sizeof(float)))
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("random_forest: allocation size would overflow: n_samples=%d, feature_dim=%d",
						n_samples, feature_dim)));
	}

	/* Check for PostgreSQL allocation limit (typically ~1GB) */
	{
		size_t		alloc_size = sizeof(float) * (size_t) feature_dim * (size_t) n_samples;
		size_t		max_alloc = 1024UL * 1024UL * 1024UL;	/* 1GB limit */

		if (alloc_size > max_alloc)
		{
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("random_forest: dataset too large for single allocation: n_samples=%d, feature_dim=%d, size=%.2f GB",
							n_samples, feature_dim, (double) alloc_size / (1024.0 * 1024.0 * 1024.0)),
					 errhint("Consider using a smaller dataset, increasing work_mem, or using batch training")));
		}
	}

	{
		double *labels_tmp = NULL;
		nalloc(labels_tmp, double, (size_t) n_samples);
		NDB_CHECK_ALLOC(labels_tmp, "labels_tmp");
		dataset->labels = labels_tmp;
	}
	if (feature_dim > 0)
	{
		float *features_tmp = NULL;
		nalloc(features_tmp, float, (size_t) feature_dim * (size_t) n_samples);
		NDB_CHECK_ALLOC(features_tmp, "features_tmp");
		dataset->features = features_tmp;
	}

	for (i = 0; i < n_samples; i++)
	{
		HeapTuple	tup = SPI_tuptable->vals[i];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		Datum		feat_datum;
		Datum		label_datum;
		bool		feat_null;
		bool		label_null;

		feat_datum = SPI_getbinval(tup, tupdesc, 1, &feat_null);
		label_datum = SPI_getbinval(tup, tupdesc, 2, &label_null);

		if (feat_null || label_null)
		{
			dataset->labels[i] = NAN;
			continue;
		}

		dataset->labels[i] = DatumGetFloat8(label_datum);

		if (dataset->features != NULL)
		{
			Oid			feat_type = SPI_gettypeid(tupdesc, 1);

			/* Handle array types (double precision[] or float[]) */
			if (feat_type == FLOAT8ARRAYOID || feat_type == FLOAT4ARRAYOID)
			{
				ArrayType  *arr = DatumGetArrayTypeP(feat_datum);
				float	   *dest_row = dataset->features + (i * feature_dim);
				int			arr_len;
				int			j;

				if (arr == NULL || ARR_NDIM(arr) != 1)
				{
					dataset->labels[i] = NAN;
					continue;
				}

				arr_len = ArrayGetNItems(ARR_NDIM(arr), ARR_DIMS(arr));
				if (arr_len != feature_dim)
				{
					dataset->labels[i] = NAN;
					continue;
				}

				if (feat_type == FLOAT8ARRAYOID)
				{
					float8	   *fdat = (float8 *) ARR_DATA_PTR(arr);

					for (j = 0; j < feature_dim; j++)
						dest_row[j] = (float) fdat[j];
				}
				else
				{
					float4	   *fdat = (float4 *) ARR_DATA_PTR(arr);

					for (j = 0; j < feature_dim; j++)
						dest_row[j] = fdat[j];
				}
			}
			/* Handle Vector type */
			else
			{
				Vector	   *vec = DatumGetVector(feat_datum);
				float *vec_data = NULL;
				float *dest_row = NULL;
				int			j;

				if (vec == NULL || vec->dim != feature_dim)
				{
					dataset->labels[i] = NAN;
					continue;
				}

				vec_data = vec->data;
				dest_row = dataset->features + (i * feature_dim);
				for (j = 0; j < feature_dim; j++)
					dest_row[j] = vec_data[j];
			}
		}
	}
}

static bool
rf_load_model_from_catalog(int32 model_id, RFModel **out)
{
	bytea *payload = NULL;
	Jsonb *metrics = NULL;
	RFModel *decoded = NULL;
	bool		result = false;


	if (model_id <= 0)
	{
		elog(WARNING, "rf_load_model_from_catalog: invalid model_id %d", model_id);
		return false;
	}

	if (out == NULL)
	{
		elog(WARNING, "rf_load_model_from_catalog: out parameter is NULL");
		return false;
	}

	*out = NULL;

#ifdef MEMORY_CONTEXT_CHECKING
	/* Check memory context at entry */
	if (CurrentMemoryContext != NULL)
		MemoryContextCheck(CurrentMemoryContext);
#endif

	PG_TRY();
	{
		if (!ml_catalog_fetch_model_payload(
											model_id, &payload, NULL, &metrics))
		{
			result = false;
		}
		else if (payload == NULL)
		{
			if (metrics != NULL)
				nfree(metrics);
			result = false;
		}
		else if (VARSIZE(payload) < VARHDRSZ)
		{
			elog(WARNING, "rf_load_model_from_catalog: invalid payload size %d for model_id %d", VARSIZE(payload), model_id);
			nfree(payload);
			if (metrics != NULL)
				nfree(metrics);
			result = false;
		}
		else
		{
			/* Check if this is a GPU model - either by metrics or by payload format */
			bool		is_gpu_model = false;
			uint32		payload_size;

			/* First check metrics for training_backend */
			if (rf_metadata_is_gpu(metrics))
			{
				is_gpu_model = true;
			}
			else
			{
				/* If metrics check didn't find GPU indicator, check payload format */
				/* GPU models start with NdbCudaRfModelHeader, CPU models start with uint8 training_backend */
				payload_size = VARSIZE(payload) - VARHDRSZ;
				
				/* CPU format: first byte is training_backend (uint8), then model_id (int32) */
				/* GPU format: first field is tree_count (int32) */
				/* Check if payload looks like GPU format (starts with int32, not uint8) */
				if (payload_size >= sizeof(int32))
				{
					const int32 *first_int = (const int32 *) VARDATA(payload);
					int32		first_value = *first_int;
					
					/* If first 4 bytes look like a reasonable tree_count, check for GPU format */
					if (first_value > 0 && first_value <= 10000)
					{
						/* Check if payload size matches GPU format */
						if (payload_size >= sizeof(NdbCudaRfModelHeader))
						{
							const NdbCudaRfModelHeader *hdr = (const NdbCudaRfModelHeader *) VARDATA(payload);
							
							/* Validate header fields */
							if (hdr->tree_count == first_value &&
								hdr->feature_dim > 0 && hdr->feature_dim <= 100000 &&
								hdr->class_count > 0 && hdr->class_count <= 10000 &&
								hdr->sample_count >= 0 && hdr->sample_count <= 1000000000)
							{
								/* Size check - GPU format has header + trees, minimum size check */
								if (payload_size >= sizeof(NdbCudaRfModelHeader) + 100)
								{
									is_gpu_model = true;
								}
							}
						}
					}
				}
			}

			if (is_gpu_model)
			{
				if (payload != NULL)
					nfree(payload);
				if (metrics != NULL)
					nfree(metrics);
				result = false;
			}
			else
			{

			/* Deserialize with error handling */
			{
				uint8		training_backend = 0;

				decoded = rf_model_deserialize(payload, &training_backend);
			}

			if (decoded == NULL)
			{
				elog(WARNING, "rf_load_model_from_catalog: rf_model_deserialize returned NULL for model_id %d", model_id);
				nfree(payload);
				if (metrics != NULL)
					nfree(metrics);
				result = false;
			}
			else
			{
				/* Validate decoded model */
				if (decoded->n_features <= 0 || decoded->n_features > 1000000 ||
					decoded->n_classes <= 0 || decoded->n_classes > 10000 ||
					decoded->tree_count < 0 || decoded->tree_count > 10000)
				{
					elog(WARNING, "rf_load_model_from_catalog: invalid model parameters for model_id %d (n_features=%d, n_classes=%d, tree_count=%d)",
						 model_id, decoded->n_features, decoded->n_classes, decoded->tree_count);
					rf_free_deserialized_model(decoded);
					nfree(payload);
					if (metrics != NULL)
						nfree(metrics);
					result = false;
				}
				else
				{
					bool		store_succeeded = false;

					/* Validate memory context before attempting to store */
					if (TopMemoryContext == NULL)
					{
						elog(WARNING, "rf_load_model_from_catalog: TopMemoryContext is NULL, cannot store model %d", model_id);
						rf_free_deserialized_model(decoded);
						nfree(payload);
						if (metrics != NULL)
							nfree(metrics);
						result = false;
					}
					else
					{
						/*
						 * rf_store_model - errors will be caught by outer
						 * PG_TRY
						 */
						rf_store_model(model_id,
									   decoded->n_features,
									   decoded->n_samples,
									   decoded->n_classes,
									   decoded->majority_value,
									   decoded->majority_fraction,
									   decoded->gini_impurity,
									   decoded->label_entropy,
									   decoded->class_counts,
									   decoded->feature_means,
									   decoded->feature_variances,
									   decoded->feature_importance,
									   decoded->tree,
									   decoded->split_feature,
									   decoded->split_threshold,
									   decoded->second_value,
									   decoded->second_fraction,
									   decoded->left_branch_value,
									   decoded->left_branch_fraction,
									   decoded->right_branch_value,
									   decoded->right_branch_fraction,
									   decoded->max_deviation,
									   decoded->max_split_deviation,
									   decoded->feature_limit,
									   decoded->left_branch_means,
									   decoded->right_branch_means,
									   decoded->tree_count,
									   decoded->trees,
									   decoded->tree_majority,
									   decoded->tree_majority_fraction,
									   decoded->tree_second,
									   decoded->tree_second_fraction,
									   decoded->tree_oob_accuracy,
									   decoded->oob_accuracy);
						store_succeeded = true;
					}

					/* Only continue if rf_store_model succeeded */
					if (store_succeeded)
					{
						rf_free_deserialized_model(decoded);
						decoded = NULL;

						/* Don't free payload/metrics - they're in TopMemoryContext */
						/* nfree(payload); */
						payload = NULL;
						/* if (metrics != NULL)
						{
							nfree(metrics);
							metrics = NULL;
						} */

						if (out != NULL)
							result = rf_lookup_model(model_id, out);
						else
							result = true;
					}
				}
			}
			}
		}
	}
	PG_CATCH();
	{
		elog(WARNING, "rf_load_model_from_catalog: exception during model load for model_id %d", model_id);

		/* Cleanup on error */
		if (decoded != NULL)
			rf_free_deserialized_model(decoded);
		if (payload != NULL)
			nfree(payload);
		if (metrics != NULL)
			nfree(metrics);

		FlushErrorState();
		result = false;
	}
	PG_END_TRY();

	return result;
}

static bool
rf_metadata_is_gpu(Jsonb * metadata)
{
	bool		is_gpu = false;
	JsonbIterator *it = NULL;
	JsonbValue	v;
	JsonbIteratorToken r;

	if (metadata == NULL)
		return false;

	/* Check for training_backend integer in metrics */
	it = JsonbIteratorInit((JsonbContainer *) & metadata->root);
	while ((r = JsonbIteratorNext(&it, &v, true)) != WJB_DONE)
	{
		if (r == WJB_KEY && v.type == jbvString)
		{
			char	   *key = pnstrdup(v.val.string.val, v.val.string.len);

			if (strcmp(key, "training_backend") == 0)
			{
				r = JsonbIteratorNext(&it, &v, true);
				if (r == WJB_VALUE && v.type == jbvNumeric)
				{
					int			backend = DatumGetInt32(DirectFunctionCall1(numeric_int4, NumericGetDatum(v.val.numeric)));

					is_gpu = (backend == 1);
				}
			}
			nfree(key);
		}
	}

	return is_gpu;
}

static bool
rf_try_gpu_predict_catalog(int32 model_id,
						   const Vector *feature_vec,
						   double *result_out)
{
	bytea *payload = NULL;
	Jsonb *metrics = NULL;
	char *gpu_err = NULL;
	int			class_out = -1;
	bool		success = false;

	/* Check compute mode - only try GPU if compute mode allows it */
	if (!NDB_SHOULD_TRY_GPU())
		return false;

	if (!neurondb_gpu_is_available())
		return false;
	if (feature_vec == NULL)
		return false;
	if (feature_vec->dim <= 0)
		return false;
	
	/* Fetch payload in TopMemoryContext so it persists across multiple predictions */
	/* Fetch payload in TopMemoryContext so it persists across multiple predictions */
	{
		MemoryContext oldctx = MemoryContextSwitchTo(TopMemoryContext);
		bool fetch_ok = ml_catalog_fetch_model_payload(model_id, &payload, NULL, &metrics);
		MemoryContextSwitchTo(oldctx);
		
		if (!fetch_ok)
			return false;
	}

	if (payload == NULL)
		goto cleanup;

	/* Check if this is a GPU model - either by metrics or by payload format */
	{
		bool		is_gpu_model = false;
		uint32		payload_size;

		/* First check metrics for training_backend */
		if (rf_metadata_is_gpu(metrics))
		{
			is_gpu_model = true;
		}
		else
		{
			/* If metrics check didn't find GPU indicator, check payload format */
			/* GPU models start with NdbCudaRfModelHeader, CPU models start with uint8 training_backend */
			payload_size = VARSIZE(payload) - VARHDRSZ;
			
			/* CPU format: first byte is training_backend (uint8), then model_id (int32) */
			/* GPU format: first field is tree_count (int32) */
			/* Check if payload looks like GPU format (starts with int32, not uint8) */
			if (payload_size >= sizeof(int32))
			{
				const int32 *first_int = (const int32 *) VARDATA(payload);
				int32		first_value = *first_int;
				
				/* If first 4 bytes look like a reasonable tree_count, check for GPU format */
				if (first_value > 0 && first_value <= 10000)
				{
					/* Check if payload size matches GPU format */
					if (payload_size >= sizeof(NdbCudaRfModelHeader))
					{
						const NdbCudaRfModelHeader *hdr = (const NdbCudaRfModelHeader *) VARDATA(payload);
						
						/* Validate header fields */
						if (hdr->tree_count == first_value &&
							hdr->feature_dim > 0 && hdr->feature_dim <= 100000 &&
							hdr->class_count > 0 && hdr->class_count <= 10000 &&
							hdr->sample_count >= 0 && hdr->sample_count <= 1000000000)
						{
							/* Size check - GPU format has header + trees, minimum size check */
							if (payload_size >= sizeof(NdbCudaRfModelHeader) + 100)
							{
								is_gpu_model = true;
							}
						}
					}
				}
			}
		}

		if (!is_gpu_model)
			goto cleanup;
	}

	if (ndb_gpu_rf_predict(payload,
						   feature_vec->data,
						   feature_vec->dim,
						   &class_out,
						   &gpu_err)
		== 0)
	{
		if (result_out != NULL)
			*result_out = (double) class_out;
		success = true;
	}
	else if (gpu_err != NULL)
	{
		elog(WARNING,
			 "random_forest: GPU prediction failed for model %d (%s)",
			 model_id,
			 gpu_err);
	}

cleanup:

	/*
	 * Don't free anything - payload and metrics are allocated in TopMemoryContext
	 * and will persist for the lifetime of the session. They will be automatically
	 * freed when the memory context is destroyed. Manual nfree() causes crashes.
	 */
	(void) gpu_err;
	(void) payload;
	(void) metrics;

	return success;
}

typedef struct RFGpuModelState
{
	bytea	   *model_blob;
	Jsonb	   *metrics;
	int			feature_dim;
	int			class_count;
	int			sample_count;
}			RFGpuModelState;

static void
rf_gpu_release_state(RFGpuModelState * state)
{
	if (state == NULL)
		return;
	if (state->model_blob != NULL)
		nfree(state->model_blob);
	if (state->metrics != NULL)
		nfree(state->metrics);
	nfree(state);
}

static bool
rf_gpu_train(MLGpuModel *model, const MLGpuTrainSpec *spec, char **errstr)
{
	RFGpuModelState *state = NULL;
	bytea *payload = NULL;
	Jsonb *metrics = NULL;
	int			rc;


	if (errstr != NULL)
		*errstr = NULL;


	if (model == NULL || spec == NULL)
	{
		return false;
	}
	if (!neurondb_gpu_is_available())
	{
		return false;
	}
	if (spec->feature_matrix == NULL || spec->label_vector == NULL)
	{
		return false;
	}
	if (spec->sample_count <= 0 || spec->feature_dim <= 0)
	{
		return false;
	}
	if (spec->class_count <= 0)
	{
		return false;
	}


	payload = NULL;
	metrics = NULL;

	ereport(DEBUG2,
			(errmsg("rf_gpu_train: about to call ndb_gpu_rf_train"),
			 errdetail("feature_matrix=%p, label_vector=%p, sample_count=%d, feature_dim=%d, class_count=%d",
					   (void *) spec->feature_matrix, (void *) spec->label_vector,
					   spec->sample_count, spec->feature_dim, spec->class_count)));

	/* Log hyperparameters value before calling backend */
	if (spec->hyperparameters != NULL)
	{
		PG_TRY();
		{
			char	   *hyperparams_text = DatumGetCString(
											DirectFunctionCall1(jsonb_out,
																JsonbPGetDatum(spec->hyperparameters)));
			if (hyperparams_text)
				nfree(hyperparams_text);
		}
		PG_CATCH();
		{
			FlushErrorState();
		}
		PG_END_TRY();
	}
	else
	{
	}

	/* Pass NULL hyperparameters to avoid JSON parsing errors */
	/* The backends (CUDA, ROCm, Metal) will use defaults if hyperparameters are NULL */
	/* This avoids "Token 'X' is invalid" errors from corrupted JSONB */
	rc = ndb_gpu_rf_train(spec->feature_matrix,
						  spec->label_vector,
						  spec->sample_count,
						  spec->feature_dim,
						  spec->class_count,
						  NULL,  /* Always pass NULL to avoid JSON parsing errors */
						  &payload,
						  &metrics,
						  errstr);

	ereport(DEBUG2,
			(errmsg("rf_gpu_train: ndb_gpu_rf_train returned"),
			 errdetail("rc=%d, payload=%p, metrics=%p", rc, (void *) payload, (void *) metrics)));
	if (rc != 0 || payload == NULL)
	{
		if (payload != NULL)
			nfree(payload);
		if (metrics != NULL)
			nfree(metrics);
		return false;
	}

	if (model->backend_state != NULL)
	{
		rf_gpu_release_state((RFGpuModelState *) model->backend_state);
		model->backend_state = NULL;
	}

	nalloc(state, RFGpuModelState, 1);
	NDB_CHECK_ALLOC(state, "state");
	state->model_blob = payload;
	state->metrics = metrics;
	state->feature_dim = spec->feature_dim;
	state->class_count = spec->class_count;
	state->sample_count = spec->sample_count;

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = true;

	return true;
}

static bool
rf_gpu_predict(const MLGpuModel *model,
			   const float *input,
			   int input_dim,
			   float *output,
			   int output_dim,
			   char **errstr)
{
	const		RFGpuModelState *state;
	int			rc;
	int			class_id;

	if (errstr != NULL)
		*errstr = NULL;
	if (output != NULL && output_dim > 0)
		output[0] = -1.0f;
	if (model == NULL || input == NULL || output == NULL)
		return false;
	if (output_dim <= 0)
		return false;
	if (!model->gpu_ready || model->backend_state == NULL)
		return false;

	state = (const RFGpuModelState *) model->backend_state;
	class_id = -1;

	rc = ndb_gpu_rf_predict(state->model_blob,
							input,
							state->feature_dim > 0 ? state->feature_dim : input_dim,
							&class_id,
							errstr);
	if (rc != 0)
		return false;

	output[0] = (float) class_id;
	return true;
}

static bool
rf_gpu_evaluate(const MLGpuModel *model,
				const MLGpuEvalSpec *spec,
				MLGpuMetrics *out,
				char **errstr)
{
	const		RFGpuModelState *state;
	Jsonb	   *metrics_json = NULL;
	StringInfoData buf;

	if (errstr != NULL)
		*errstr = NULL;
	if (out != NULL)
		out->payload = NULL;
	if (model == NULL || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("rf_gpu_evaluate: invalid model or state");
		return false;
	}

	state = (const RFGpuModelState *) model->backend_state;

	/* Create metrics JSON with model information */
	initStringInfo(&buf);
	appendStringInfo(&buf,
					 "{\"algorithm\":\"random_forest\",\"storage\":\"gpu\","
					 "\"feature_dim\":%d,\"class_count\":%d,\"n_samples\":%d}",
					 state->feature_dim > 0 ? state->feature_dim : 0,
					 state->class_count > 0 ? state->class_count : 0,
					 state->sample_count > 0 ? state->sample_count : 0);

	/* Use ndb_jsonb_in_cstring for safe JSONB creation (avoids memory corruption from DirectFunctionCall) */
	metrics_json = ndb_jsonb_in_cstring(buf.data);
	nfree(buf.data);

	if (metrics_json == NULL)
	{
		elog(WARNING, "rf_gpu_evaluate: ndb_jsonb_in_cstring returned NULL for metrics JSON");
	}

	if (out != NULL)
		out->payload = metrics_json;


	return true;
}

static bool
rf_gpu_serialize(const MLGpuModel *model,
				 bytea * *payload_out,
				 Jsonb * *metadata_out,
				 char **errstr)
{
	const		RFGpuModelState *state;
	bytea	   *payload_copy = NULL;
	int			payload_size;

	if (errstr != NULL)
		*errstr = NULL;
	if (payload_out != NULL)
		*payload_out = NULL;
	if (metadata_out != NULL)
		*metadata_out = NULL;
	if (model == NULL || model->backend_state == NULL)
		return false;

	state = (const RFGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
		return false;

	payload_size = VARSIZE(state->model_blob);
	{
		char *payload_buf = NULL;
		nalloc(payload_buf, char, payload_size);
		payload_copy = (bytea *) payload_buf;
	}
	NDB_CHECK_ALLOC(payload_copy, "payload_copy");
	memcpy(payload_copy, state->model_blob, payload_size);

	if (payload_out != NULL)
		*payload_out = payload_copy;
	else
		nfree(payload_copy);

	if (metadata_out != NULL && state->metrics != NULL)
	{
		int			metadata_size;
		Jsonb *metadata_copy = NULL;

		metadata_size = VARSIZE(state->metrics);
		{
			char *metadata_buf = NULL;
			nalloc(metadata_buf, char, metadata_size);
			metadata_copy = (Jsonb *) metadata_buf;
		}
		NDB_CHECK_ALLOC(metadata_copy, "metadata_copy");
		memcpy(metadata_copy, state->metrics, metadata_size);
		*metadata_out = metadata_copy;
	}

	return true;
}

static bool
rf_gpu_deserialize(MLGpuModel *model,
				   const bytea * payload,
				   const Jsonb * metadata,
				   char **errstr)
{
	RFGpuModelState *state = NULL;
	bytea *payload_copy = NULL;
	int			payload_size;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || payload == NULL)
		return false;

	payload_size = VARSIZE(payload);
	{
		char *payload_buf = NULL;
		nalloc(payload_buf, char, payload_size);
		payload_copy = (bytea *) payload_buf;
	}
	NDB_CHECK_ALLOC(payload_copy, "payload_copy");
	memcpy(payload_copy, payload, payload_size);

	nalloc(state, RFGpuModelState, 1);
	NDB_CHECK_ALLOC(state, "state");
	state->model_blob = payload_copy;
	state->feature_dim = -1;
	state->class_count = -1;
	state->sample_count = -1;

	if (metadata != NULL)
	{
		int			metadata_size;
		Jsonb *metadata_copy = NULL;

		metadata_size = VARSIZE(metadata);
		{
			char *metadata_buf = NULL;
			nalloc(metadata_buf, char, metadata_size);
			metadata_copy = (Jsonb *) metadata_buf;
		}
		NDB_CHECK_ALLOC(metadata_copy, "metadata_copy");
		memcpy(metadata_copy, metadata, metadata_size);
		state->metrics = metadata_copy;
	}

	if (model->backend_state != NULL)
		rf_gpu_release_state((RFGpuModelState *) model->backend_state);

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = true;

	return true;
}

static void
rf_gpu_destroy(MLGpuModel *model)
{
	if (model == NULL)
		return;
	if (model->backend_state != NULL)
		rf_gpu_release_state((RFGpuModelState *) model->backend_state);
	model->backend_state = NULL;
	model->gpu_ready = false;
	model->is_gpu_resident = false;
}

void
neurondb_gpu_register_rf_model(void)
{
	static bool registered = false;
	static MLGpuModelOps rf_gpu_model_ops;

	if (registered)
		return;

	/* Initialize ops struct at runtime */
	rf_gpu_model_ops.algorithm = "random_forest";
	rf_gpu_model_ops.train = rf_gpu_train;
	rf_gpu_model_ops.predict = rf_gpu_predict;
	rf_gpu_model_ops.evaluate = rf_gpu_evaluate;
	rf_gpu_model_ops.serialize = rf_gpu_serialize;
	rf_gpu_model_ops.deserialize = rf_gpu_deserialize;
	rf_gpu_model_ops.destroy = rf_gpu_destroy;

	ndb_gpu_register_model_ops(&rf_gpu_model_ops);
	registered = true;
}

